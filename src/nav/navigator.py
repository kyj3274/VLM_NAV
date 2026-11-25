"""영상 길안내: 규칙 추적기 + Claude 인식 (VLM_NAV navigator.py의 규칙 흐름을 그대로 두고 입력만 Claude로 바꿈).

목적지(예: 314호)를 받아, 1초 구간마다 그 초의 안내 문장과 다음 행동을 만든다.
  인식  1초 구간에서 가장 선명한 장 1장을 Claude에 보냄(claude_perceiver.see): 문패, 층 표시, 장소, 엘리베이터(근거),
        화살표, 앞의 출입문. 중요한 문패 1개는 원본에서 잘라 확대해 다시 읽음(recheck).
  판단  규칙. 문패 확정(nav_memory 믿음 점수 3), 층 충돌은 2초 연속, 번호 증감으로 진행 방향, 엘리베이터는 근거가
        2초 연속, 하차 직후 4초는 뒤의 엘리베이터 무시, 10초 무안내면 한 번 말함, 시작 위치 파악 5초.
        엘리베이터 하차(버튼·자동 감지) 뒤 최대 6초 둘러보며 방향 근거를 모아 한 번 안내 (direction.py).
        층을 모르면 1층으로 가정하지 않는다(엘리베이터·계단 안내 없음, 15초 뒤 층을 물음, 말로 받은 층은 문패와 같은 근거).
  문장  상태로 틀에 맞춰 만든다. 10초 동안 할 말이 없으면 지금 행동을 되풀이하는 규칙 문장.
        (Claude say는 2단계에서 보조로 썼으나 3단계에서 사용자 지시로 스키마에서 뺌)
VLM_NAV 대비 바뀐 것은 docs/17_2026-11-17_Claude_인식_기준표.md에 표로 정리.

다음 행동 이름(ACTIONS)은 판단 지점 정답표(data/video/checkpoints.json)와 같다. 매 초 기록의 "action"이 지금 유효한 안내의 행동이다.

프레임: frames_8fps/<순서>_<영상>/i0000.jpg (1초에 8장, i = 초×8 + k).

  .venv/bin/python src/nav/navigator.py --route "314=1_start_졸프_영상3,2b_after_elevator_졸프_영상2" --dry-run --max-steps 3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys

import claude_perceiver as cp
import direction as dr
import nav_memory

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))  # 공통 판단 (src/nav/core, 로컬 버전과 같은 코드)
from arrival import STEP_M, TargetTracker, josa  # noqa: E402
from floor_graph import FloorGraph  # noqa: E402
import nav_graph  # noqa: E402
from claude_client import CreditLimit, DryRunClient, GatewayClient, Ledger, ReplayClient, ROOT

DENSE = 8
QUIET_SAY = 10    # 이 초 동안 새 안내가 없으면 한 번 말함
LOCATE_SEC = 5    # 경로 시작: 이 시간 동안 위치(층·장소)를 먼저 파악
LOCATE_ASKS = 2   # 층을 모르면 "주변을 더 보여 주세요"를 몇 번까지
EXIT_IGNORE = 4   # 엘리베이터에서 내린 뒤 이 초 동안은 (뒤에 있는) 엘리베이터를 무시
JUNC_SAY = 3.0    # 갈림길이 이 거리(m) 안에 보이면 어느 복도로 갈지 말함 (core/floor_graph.py)
JUNC_NEAR = 1.0   # 이보다 가까우면 말하지 않음 (갈림길에서 막 돌아선 때: 이미 고른 뒤)
SCAN_M = 12.0      # 넓은 곳(로비)에서 목적지를 못 보고 이만큼 걸을 때마다 제자리에서 한 바퀴 둘러보게 함 (시뮬레이터 로비 섬)
SCAN_SHOTS = 8    # 한 바퀴에 받는 사진 수 (45°씩)
GIVE_UP_M = 40.0  # 넓은 곳에서 가 본 곳만 이만큼 다니면(새 곳 없음) 못 찾았다고 알림 (더 알 것이 없음 = 안전한 실패)
VISIT_M = 4.0     # 가 본 자리에서 이보다 멀면 새 곳 (벽 문패는 5m 안에서 읽힘)
SCAN_SAY = "넓은 곳입니다. 제자리에서 휴대폰을 앞으로 들고 천천히 한 바퀴 돌아 주세요."
CLOCK_OFF = {"L": -20, "C": 0, "R": 20}  # 화면 왼쪽·가운데·오른쪽 → 카메라 방향에서 대략 몇 도
GRAPH_LEAN = 0.6  # 둘러보는 동안 복도 그래프 P(앞)이 이 이상(또는 1-이 이하)이면 약한 방향 근거 (2026-11-24 병합, docs/17 21절)
PASS_WALK_M = 1.5  # 옆을 비추라고 한 뒤 이만큼 걸었으면(걸음 수) 그 뒤 읽은 다른 문패를 '지나친' 근거로 씀 (멈춰 비추는 중에 읽은 옆 문패는 아님)
TURN_REPEAT = 8   # '뒤로 돌아 주세요'를 다시 말하기 전 최소 간격(초). 돌 시간을 주고, 그사이 문패로 돌았는지 판단
# 진행 방향은 공통 판단 core/nav_graph.py (확률). 전의 TREND_M(10m)·NEIGHBOR(±3) 규칙은 이 건물 번호 간격에 맞춘 것이라 뺐다 (2026-11-24)
                  # 2026-11-17 3층: 문패 순서가 317,316,309,315,307,313,314라 먼 번호(309→307 등)의 증감은 방향과 무관.
                  # 처음 규칙(5m 이내 가장 가까운 문패, 같은 벽 2개)은 315→307을 '멀어짐'으로 틀렸음
ARRIVE_M = 2.0    # 도착: 정면(가운데) 이 거리 이내가 2초 연속
EL_NEAR_M = 3.0   # 엘리베이터 '타세요'는 정면(가운데) 이 거리 이내이거나 어느 쪽이든 2m 이내일 때. 그 전에는 다가가라고만 함
                  # (2026-11-17 경로 A 1층 7초: 오른쪽 5m 엘리베이터에서 바로 '타세요' → 판단 지점 2(직진)를 틀려 추가.
                  #  claude_nav의 엘리베이터_찾기 → 호출_버튼_안내(약 2m)와 같은 두 단계)
DOOR_GONE = 2     # 시작 출입문이 이 초 동안 안 보이면 지나간 것으로 봄
STAIR_ON = 2      # 계단 장면이 이 장 연속이면 계단을 쓰는 중으로 봄 (2026-11-19)
STAIR_OFF = 4     # 계단이 아닌 장면이 이 장 연속이면 계단에서 나온 것으로 봄 (이번 시험 계단참에서 '복도'가 3장 연속 나와서 4)
EL_MISS = 2       # 엘리베이터 후보를 본 뒤 이 장 연속 안 보이면 지나침 또는 취소 (2026-11-19 사용자 지시 9)
EL_PASS_M = 2.5   # 지나침은 마지막에 옆(L/R) 이 거리 이내에서 보였을 때만. 더 멀리서 사라지면 그쪽으로 몸을 돌린 것으로 봄
                  # (영상 1층 11~13초: 오른쪽 4m 엘리베이터 쪽으로 돌 때 사라져 '지나쳤습니다'가 잘못 나와서)
QUICK_MAX = 2     # 빠른 확인(confirm_now)은 목적지 하나에 이 횟수(번마다 사진 2장)까지 (2026-11-23 사용자 지시 B5, 전에는 1장 × 3번)
QUICK_M = 5.0     # 목적지 후보를 이 거리(m) 안에서 읽으면 빠른 확인 (2026-11-23 B3, 전에는 2m 안 '멈춰 주세요' 때)
QUICK_SHOTS = 2   # 빠른 확인 한 번에 휴대폰이 찍어 보내는 사진 수 (0.4초 간격)
QUICK_WAIT = 6.0  # 빠른 확인 사진이 이 초 안에 다 안 오면(흐림·실패) 기다리지 않음
QUICK_SAY = "목적지 같습니다. 잠시 멈춰 주세요."
# 2026-11-23 사용자 지시: 1층 가정(VLM_NAV assume1)을 없앰. 층을 모르면 '모름'으로 두고 엘리베이터·계단 안내를 하지 않는다.
#   (4층에서 시작해도 위치 파악 5초 뒤 1층으로 보고 "엘리베이터를 타고 4층을 누르세요"라고 했음)
ASK_FLOOR_SEC = 15    # 구간 시작 뒤 이 초가 지나도 층을 모르면 "지금 몇 층인지 말씀해 주세요"
ASK_FLOOR_EVERY = 20  # 같은 질문은 이 초에 한 번까지
CHECK_SAY = "지금 층을 확인하고 있습니다. 휴대폰을 천천히 좌우로 비춰 주세요."
QUIET_GOAL = "복도를 따라 천천히 직진하세요. 문패를 확인하고 있습니다."  # 목적지 층 무안내 10초 (행동이 든 문장)
ASK_SAY = "지금 몇 층인지 말씀해 주세요."
FLOOR_MOVE_HOLD = 20  # '층 이동 가능'(계단·다른 층 표시)은 계단을 벗어나고 마지막 계단 단서 뒤 이 초가 지나면 끈다
                      # (PR #1 리뷰 1: 켜진 채로 남아 오독 한 번에 층이 바뀜. 2026-11-19 계단 시험에서 마지막 계단 장면 45.8초 →
                      #  4층 첫 문패 62.7초로 약 17초 걸려서 20초. 그 안에 첫 문패를 못 읽어도 층 충돌 2번 연속 규칙으로 바뀜)
EL_DISP_FRESH = 10    # 엘리베이터 안 층 표시창을 '내렸어요' 전 이 초 안에 읽었으면 내린 층의 임시 층으로 씀
# 엘리베이터에서 내린 뒤 (2026-11-23 사용자 지시 B5·B6·B8)
HALL = ("corridor", "lobby", "elev_hall", "junction", "room_door")  # 자동 하차 감지의 '복도·홀' 장면
EXIT_IN = 3       # 엘리베이터 안(in_elev) 장면이 이 장 연속이면 탄 것으로 봄
EXIT_OUT = 3      # 그 뒤 복도·홀 장면이 이 장 연속이면 내린 것으로 봄 (버튼 "내렸어요"와 같은 처리)
                  # 지시는 2장·2장. 휴대폰 기록 2개(실제 탑승 2번)에서 2·2는 잘못된 하차가 2번(밖에서 닫힌 문을 '안'으로 읽음,
                  # 칸 안에서 열린 문 너머 로비를 '복도'로 읽음), 3·3은 실제 하차 2번만 (판 4 기록·판 5 다시 인식 모두). 감지는 약 1.5초 늦음
ELEV_CLOSE_M = 2.0  # 탄 뒤 이 거리 안의 엘리베이터 문(근거 doors·panel)이 보이면 아직 안으로 봄
EXIT_STEPS = 3    # 복도·홀 장면 3장 동안 이 걸음 이상 걸어야 내림 (걸음 수를 받지 못하는 세션은 장면만)
EXIT_DUP = 20     # 내림 처리 뒤 이 초 안에 다시 오는 내림(버튼·자동)은 같은 하차로 보고 다시 처리하지 않음.
                  # 자동 감지는 내린 뒤 복도 사진 2장(약 3초) + 응답(2~4초) 뒤라, 그 무렵 사용자가 버튼을 눌러도 겹치지 않게 넉넉히.
SCAN_SEC = 6      # 내린 직후 둘러보기 최대 시간. 그동안은 진행 안내(직진 등)를 하지 않는다
EXIT_SAY = "잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."
OTHER_FLOOR_PHOTOS = 2   # 내린 뒤 가정한 층(엘리베이터로 이동한 층)은 다른 층 문패가 서로 다른 사진 이 장수에서 읽혀야 바꿈
OTHER_FLOOR_SEC = 20     # 그 사진들은 이 초 안에서
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착",
           "계단으로 올라가기", "계단으로 내려가기", "계단에서 나가기"]  # 계단 3개: 2026-11-19 사용자 지시 5
SIDE_FRONT = {"L": "왼쪽 앞에", "C": "정면에", "R": "오른쪽 앞에"}
BACK_CLOCK = {"L": 1, "C": 12, "R": 11}  # 지나친 뒤 뒤로 돌았을 때의 시계 방향 (왼쪽에 있던 것은 돌면 오른쪽 앞)


def fnum(f):
    return -int(f[1:]) if str(f).startswith("B") else int(f)


def new_stair():
    return {"run": 0, "off": 0, "on": False, "sign": None, "sign_text": "", "said": False, "exit_said": False}
ROOM = re.compile(r"^\d{3}$")


def blur(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class Runner:
    """1초 구간 관찰: 가장 선명한 장을 골라 Claude 인식 1번. (VLM_NAV Runner.observe 대신)"""

    def __init__(self, perceiver, per_call=1, two_segs=None, depth_model=None):
        """depth_model: core/depth.py의 깊이 모델(있으면 문패마다 벽 쪽·거리를 깊이로 잼, 없으면 Claude 값)."""
        self.p, self.per_call, self.depth = perceiver, per_call, depth_model
        self.two_segs = two_segs  # 2장 방식을 쓸 구간 이름 앞부분들 (None이면 모든 구간)

    @staticmethod
    def sharpest(frames):
        return frames[0] if len(frames) == 1 else max(frames, key=lambda p: blur(Image.open(p).convert("RGB").reduce(2)))

    def pick(self, frames):
        """1장: 구간에서 가장 선명한 장. 2장: 앞 절반과 뒤 절반에서 각각 가장 선명한 장(시간 순)."""
        seg = Path(frames[0]).parent.name
        two = self.per_call == 2 and (self.two_segs is None or any(seg.startswith(x) for x in self.two_segs))
        if two and len(frames) >= 2:
            h = len(frames) // 2
            return [self.sharpest(frames[:h]), self.sharpest(frames[h:])]
        return [self.sharpest(frames)]

    def observe(self, frames):
        tb = time.perf_counter()
        keys = self.pick(frames)
        pick = time.perf_counter() - tb
        r = self.p.see(keys if len(keys) > 1 else keys[0])
        out = {"key": keys[-1], "keys": keys, "pick_seconds": round(pick, 3), **r}
        if self.depth is not None and r.get("obs"):
            out["depth_seconds"] = self.add_depth(keys[-1], r["obs"])
        return out

    def add_depth(self, frame, obs):
        """대표 사진의 깊이로 문패마다 벽 쪽(복도 방향 기준 '왼쪽'/'오른쪽'/'?')과 거리(m)를 붙임 (core/depth.py).
        Claude의 L/C/R은 화면 위치, 거리는 눈대중이라 깊이 값을 우선 쓴다. 돌려줌: 걸린 초."""
        import depth
        tb = time.perf_counter()
        info, dep = depth.measure(self.depth, Image.open(frame).convert("RGB"))
        obs["depth"] = info
        for p in obs["plates"]:
            if p.get("box"):
                b = [v / 100 for v in p["box"]]
                p["wall"] = depth.wall_side(info, (b[0] + b[2]) / 2)
                p["dm"] = depth.box_distance(dep, b)
        return round(time.perf_counter() - tb, 3)

    def recheck(self, frame, plate):
        return self.p.recheck(frame, plate)


class Navigator:
    """길안내 한 번(목적지 하나)의 상태. 1초 구간 프레임을 step()에 넣으면 그 초의 기록(안내 문장 포함)을 돌려준다."""

    def __init__(self, runner, target, start_floor=None):
        """start_floor: 사용자가 말한 출발 층("4층에서 410호")이 있으면 그 층으로 시작 (문패와 같은 근거)."""
        self.runner, self.target = runner, target
        self.tf = target[0] if ROOM.match(target) else None  # 목적지 층 (3자리 호실의 첫 자리)
        self.st = {"floor": None, "wrong_floor": False, "arrived": False, "room": None, "floor_rooms": [],
                   "elevator_said": False, "el_run": 0, "el_seen": False, "el_ready": None, "button": None,
                   "pending": None, "fs_last": None, "near_d": [], "trend_said": None, "target_seen": None,
                   "arr_run": 0, "last_say": 0, "loc": None, "door_phase": False, "door_gone": 0,
                   "arrow_last": None, "el_approach": False, "act": "멈추고 둘러보기", "maybe_said": False,
                   "floor_move": None, "quick": None, "quick_n": 0,
                   "el_cand": None, "el_last": None, "el_miss": 0, "el_passed": False, "stair": new_stair(),
                   "floor_temp": False, "floor_src": None, "ask_t": None, "check_t": None, "el_disp": None,
                   "door_out": False, "elev_run": 0, "elev_in": False, "hall_run": 0, "exit_cand": None,
                   "scan": None, "dir": None, "dir_said": None, "dir_rooms": [], "turn_said_t": -99, "dir_last": None,
                   "lscan": 0, "scan_steps": None, "scan_hit": None, "scan_spots": [], "visited": [], "new_steps": None, "gave_up": False}
        self.records, self.clock, self.n = [], 0, 0  # n: 전체 걸음 번호 (연속 판단은 '바로 앞 걸음'으로, 2026-11-18)
        self.seg_n = 0
        self.mem = nav_memory.Memory(target)
        self.arrival = TargetTracker(target, lambda text: self.mem.event(self.clock, text))  # 도착 추정 (core/arrival.py)
        self.floor = FloorGraph(target, lambda text: self.mem.event(self.clock, text))  # 갈림길·복도 가지 (core/floor_graph.py)
        self.sensor = None  # 웹이 걸음마다 넣는 휴대폰 센서 {"steps", "heading"} (없으면 시간으로 걸은 거리 추정)
        # 가장 최근에 받은 사진의 센서 {"t", "steps", "heading"}. 지금 처리하는 사진(인식이 끝난 2초 전 사진)보다 새로울 수 있다.
        # 사진을 찍은 뒤 걸은 거리·돈 것을 보정하는 데 씀 (지연 보정, 2026-11-25 시뮬레이터)
        self.latest = None
        self.mem.event(0, f"안내 시작 (목적지 {target})")
        if start_floor:
            self.st.update(floor=str(start_floor), floor_src="말", wrong_floor=bool(self.tf) and str(start_floor) != self.tf)
            self.mem.event(0, f"출발 층 {start_floor}층 (사용자가 말함)")
        self.si, self.sec, self.seq, self.seg_start = 0, 0, "", 0
        self.quick_max = QUICK_MAX  # 문 앞 빠른 확인 횟수 (웹 서버가 0으로 끌 수 있음)
        self.announce = True  # 첫 걸음에 "○호로 안내를 시작합니다"를 말할지 (웹은 미리 말함)
        self.last_exit = None   # 마지막으로 내림을 처리한 시각 (같은 하차를 두 번 처리하지 않게)
        self.exit_say = False   # 다음 걸음에서 내림 문장을 말할지 (웹 버튼은 응답에서 바로 말하므로 False)
        self.steps_seen = False  # 휴대폰 걸음 수가 한 번이라도 0보다 컸나 (센서가 꺼져 있으면 걸음 조건을 쓰지 않음)
        self.plate_dirs = []    # 내린 뒤 걸으며 확정한 문패와 그때 나침반 방향 (다른 층 같은 끝자리 번호의 약한 근거, B8c)

    def start_head(self):
        """경로 시작 문장 앞부분. 웹은 '410호로 안내하겠습니다'를 이미 말했으므로 announce=False로 앞 문장을 뺀다."""
        t, tf = self.target, self.tf
        head = (f"{t}호로 안내를 시작합니다. " if ROOM.match(t) else f"{t}까지 안내를 시작합니다. ") if self.announce else ""
        return head + (f"{t}호는 {tf}층입니다. " if tf else "")

    def set_target(self, target):
        """안내 중 목적지 변경 (웹). 목적지와 도착·진행 방향 상태만 새로 하고, 층·확정 문패·엘리베이터 기억은 둔다.
        반환: 바뀐 목적지로 다음에 할 일 문장들."""
        st = self.st
        self.target, self.tf = target, target[0] if ROOM.match(target) else None
        self.mem.target = target
        st.update(arrived=False, arr_run=0, target_seen=None, near_d=[], trend_said=None, maybe_said=False,
                  elevator_said=False, el_approach=False, el_ready=None, quick=None, quick_n=0,
                  el_cand=None, el_passed=False, dir=None, dir_said=None, dir_rooms=[], turn_said_t=-99)
        self.arrival = TargetTracker(target, lambda text: self.mem.event(self.clock, text))
        self.floor.target = target
        # PR #1 리뷰 3: 이전 목적지로 가던 중 한 번의 호출로 확정된 문패가 새 목적지이면 확정을 풀고
        # '서로 다른 호출 2번' 기준으로 다시 검사한다 (FORGET초 넘게 안 보였으면 호출 수는 0으로 봄).
        p = self.mem.get(target)
        if p and p["ok"]:
            calls = p["calls"] if self.clock - p["last"] <= nav_memory.FORGET else 0
            if calls < nav_memory.TARGET_CALLS:
                p.update(wait_calls=True)
                self.mem.set_ok(target, False)
                self.mem.event(self.clock, f"{target}호 확정 풀림 (새 목적지: 서로 다른 호출 {nav_memory.TARGET_CALLS}번 필요, 지금 {calls}번)")
        self.mem.event(self.clock, f"목적지 변경: {target}")
        # PR #1 리뷰 4: 층을 모를 때도 행동을 다시 계산한다 (전에는 층을 알 때만 → '엘리베이터 타기'가 남았음)
        if st["floor"] is None:
            st["wrong_floor"], st["act"] = False, "멈추고 둘러보기"
            return [CHECK_SAY]
        return self.floor_next(head=f"지금 {st['floor']}층입니다. ")

    def floor_next(self, head=""):
        """층을 알 때 다음 행동 문장 (목적지 층이면 복도 직진, 아니면 엘리베이터). act·wrong_floor도 정한다."""
        st, target, tf = self.st, self.target, self.tf
        floor = st["floor"]
        st["wrong_floor"] = bool(tf) and floor != tf
        if not st["wrong_floor"]:
            if self.can_goal_scan():  # 목적지 층에서 시작·층을 앎: 내렸을 때와 같은 방향 정하기 (2026-11-23 시험 2)
                return [head + ("목적지 층입니다. " if tf else "") + self.goal_scan()]
            st["act"] = "직진"
            return [head + ("목적지 층입니다. " if tf else "") + "앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]
        if self.si > 0:
            st["act"] = "엘리베이터로 돌아가기"
            return [head + f"{target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요."]
        st["act"] = "직진"
        return [head + f"{target}호는 {tf}층에 있습니다. 엘리베이터를 찾아 {tf}층으로 가겠습니다. 앞으로 천천히 직진해 주세요."]

    def can_goal_scan(self, rooms=()):
        """목적지 층에서 방향 정하기(둘러보기)를 새로 시작할 수 있나: 호실 목적지, 이 구간에 둘러보기가 없었음,
        도착 전, 목적지 문패를 이미 보고 있지 않음."""
        st = self.st
        return (bool(ROOM.match(self.target)) and st["scan"] is None and not st["arrived"]
                and self.target not in rooms and st["target_seen"] is None)

    def new_scan(self, start):
        return {"t": start, "until": start + SCAN_SEC, "reads": [], "evs": [], "lw": [], "done": False, "why": None}

    def goal_scan(self):
        """목적지 층임을 알았을 때(시작 층이 목적지 층): 엘리베이터에서 내렸을 때와 같은 둘러보기 시작 (2026-11-23 사용자 지시).
        반환: '○호를 찾겠습니다. 잠시 멈춰서 …' 문장 뒷부분."""
        st = self.st
        st["scan"], st["act"] = self.new_scan(self.clock), "멈추고 둘러보기"
        self.mem.event(self.clock, "목적지 층: 방향을 정하려고 둘러보기 시작")
        return f"{self.target}호를 찾겠습니다. {EXIT_SAY}"

    def set_floor(self, floor, source="말"):
        """사용자가 말한 지금 층("여기 4층이야"). 문패와 같은 근거: 바로 층을 정하고, 그 뒤 다른 층 문패가
        연속 2번 보이면 문패를 따른다(층 충돌 규칙). 반환: 안내 문장들."""
        st = self.st
        floor = str(floor)
        old = st["floor"]
        st.update(floor=floor, floor_src=source, floor_temp=False, floor_move=None, pending=None, loc=None,
                  door_phase=False, elevator_said=False, el_approach=False, el_ready=None, el_cand=None)
        self.mem.event(self.clock, f"지금 층 {floor}층 ({'사용자가 말함' if source == '말' else source}"
                       + (f", 전에는 {old}층" if old and old != floor else "") + ")")
        st["last_say"] = self.clock
        return self.floor_next(head=f"{floor}층으로 알겠습니다. ")

    # ------------------------------------------------------------ 구간
    def new_segment(self, seq="", start=None, keep=None):
        """구간 시작. 첫 구간이 아니면 엘리베이터를 타고 내린 것으로 본다 (층·위치는 모름으로).
        내리기 직전(EL_DISP_FRESH초 안)에 엘리베이터 안 층 표시창을 읽었으면 그 층을 임시 층으로 둔다
        (임시 층은 다른 층 확정 문패 한 번이면 문패를 따름).
        start: 구간 시작 시각(자동 하차 감지는 첫 복도 사진 시각). keep: (층, 근거) 자동 감지 전 첫 복도 사진에서 이미 정한 층.
        내린 뒤에는 SCAN_SEC초 둘러보기(scan)를 시작한다 (2026-11-23 B6)."""
        if self.records or self.seg_n:
            self.si += 1
            self.mem.new_segment(self.clock)
        self.seq, self.sec, self.seg_start, self.seg_n = seq, 0, self.clock if start is None else start, 0
        if self.si > 0:
            disp = self.st.get("el_disp")
            carry = disp["floor"] if disp and self.clock - disp["t"] <= EL_DISP_FRESH else None
            src = "엘리베이터 표시창" if carry else None
            if keep:
                carry, src = keep
            if carry is None and self.tf:
                # 2026-11-23 사용자 지시: 내린 층은 기본으로 엘리베이터로 이동한 층("○층을 누르세요"라고 한 층 = 목적지 층).
                # 시험 1·3에서 4층에 내린 뒤 층을 20~40초 몰랐음. 다른 층 증거가 나오면 그때 고친다(OTHER_FLOOR_PHOTOS).
                carry, src = self.st.get("press_floor") or self.tf, "엘리베이터 이동"
            self.last_exit, self.exit_say = self.clock, True
            self.st.update(floor=carry, wrong_floor=bool(carry and self.tf) and carry != self.tf, room=None,
                           floor_rooms=[], elevator_said=False, el_run=0,
                           el_seen=False, el_ready=None, button=None, pending=None, fs_last=None, near_d=[],
                           trend_said=None, target_seen=None, arr_run=0, last_say=self.clock, loc=None,
                           door_phase=False, door_gone=0, arrow_last=None, el_approach=False, maybe_said=False,
                           floor_move=None, quick=None, quick_n=0, el_cand=None, el_last=None, el_miss=0,
                           el_passed=False, stair=new_stair(), floor_temp=src in ("엘리베이터 표시창", "엘리베이터 이동"),
                           floor_src=src, ask_t=None, check_t=None, el_disp=None,
                           assumed=src in ("엘리베이터 표시창", "엘리베이터 이동"), other_seen={}, exit_h=None,
                           elev_run=0, elev_in=False, hall_run=0, exit_cand=None, dir=None,
                           dir_said=None, dir_rooms=[], turn_said_t=-99, dir_last=None,
                           scan=self.new_scan(max(self.seg_start, self.clock)))  # 자동 감지: 내림을 알고 "멈춰서 비춰 주세요"라고 한 때부터
            self.arrival.reset()
            self.floor.reset()
            if carry and src == "엘리베이터 표시창":
                self.mem.event(self.clock, f"내린 층 임시 {carry}층 (엘리베이터 안 층 표시창)")
            elif carry and src == "엘리베이터 이동":
                self.mem.event(self.clock, f"내린 층 {carry}층으로 봄 (엘리베이터로 이동한 층, 다른 층 증거가 나오면 고침)")
            elif carry:
                self.mem.event(self.clock, f"내린 층 {carry}층 (내림을 알기 전 복도 사진의 {src})")

    def exit_sentence(self):
        """내린 직후 한 번: '엘리베이터에서 내렸습니다. 4층에 내렸습니다. 410호를 찾겠습니다. 잠시 멈춰서 …'.
        내린 층이 목적지 층이 아니면(표시창이 다른 층) 바로 엘리베이터로 돌아가라고 한다."""
        st, tf = self.st, self.tf
        f = st["floor"]
        if not f:
            return f"엘리베이터에서 내렸습니다. {EXIT_SAY}"
        head = f"엘리베이터에서 내렸습니다. {f}층에 내렸습니다. "
        if tf and f != tf:
            st["act"] = "엘리베이터로 돌아가기"
            return head + f"{self.target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요."
        return head + (f"{self.target}호를 찾겠습니다. " if ROOM.match(self.target) else "") + EXIT_SAY

    def elevator_way(self, heading):
        """내린 뒤 엘리베이터 쪽 (내릴 때 나침반 방향의 반대). 나침반이 없으면 '내린 곳 뒤쪽'."""
        h0 = self.st.get("exit_h")
        if h0 is None or heading is None:
            return "엘리베이터는 내린 곳 뒤쪽에 있습니다."
        rel = ((h0 + 180) - heading + 540) % 360 - 180
        if abs(rel) > 90:  # 뒤쪽: 돌고 난 뒤 기준 시계 방향
            c = round(((rel + 180 + 540) % 360 - 180) / 30) % 12 or 12
            return f"뒤로 돌아 {c}시 방향입니다."
        c = round(rel / 30) % 12 or 12
        return f"엘리베이터는 {c}시 방향입니다."

    def exit_elevator(self, source="버튼"):
        """엘리베이터에서 내림 (웹 버튼 "내렸어요"). 반환: 바로 말할 문장들. EXIT_DUP초 안에 이미 내림을 처리했으면
        (자동 감지 뒤에 버튼 등) 같은 하차로 보고 None."""
        if self.last_exit is not None and self.clock - self.last_exit < EXIT_DUP:
            self.mem.event(self.clock, f"내림 다시 받음 ({source}): {EXIT_DUP}초 안이라 같은 하차로 봄")
            return None
        if not self.records:  # 사진을 보내기 전에 눌러도 층이 바뀐 것으로 (새 구간)
            self.seg_n = 1
        self.new_segment(f"live_{self.si + 1}")
        self.mem.event(self.clock, f"엘리베이터에서 내림 ({source})")
        self.exit_say = False
        return [self.exit_sentence()]

    def watch_exit(self, obs, clock, steps=None):
        """자동 하차 감지 (2026-11-23 사용자 지시 B5): 엘리베이터 안 장면이 EXIT_IN장 이상 이어진 뒤 복도·홀 장면이
        EXIT_OUT장 연속이고 그동안 EXIT_STEPS걸음 이상 걸었으면(걸음 수가 있을 때) 버튼 "내렸어요"와 같은 처리.
        직전까지 엘리베이터 안이었으면 닫힌 엘리베이터 문만 보이는 장면(cd)·층 버튼 패널은 계속 안으로 본다
        (2026-11-23 시험 1: 칸 안에서 닫힌 문을 본 사진을 elev_hall로 읽어 잘못 내림). 반환: 감지했으면 기록용 dict."""
        st = self.st
        if steps is not None and steps > 0:
            self.steps_seen = True
        # 탄 뒤(안 장면 EXIT_IN장)에는 엘리베이터 문이 가까이(ELEV_CLOSE_M 안) 주인공인 장면·닫힌 문만 보이는 장면(cd)·
        # 층 버튼 패널은 Claude 장소와 상관없이 계속 안으로 본다. 판 6 프롬프트로도 칸 안에서 본 닫힌 문을 절반 넘게
        # elev_hall로 읽어서(시험 1: 17장 중 10장) 거리 규칙을 더함. 타기 전에는 쓰지 않음(밖에서 기다리며 본 닫힌 문,
        # 없는 패널을 본 사진이 '탔음'이 되지 않게).
        e = obs.get("elev") or {}
        close = e.get("why") in ("doors", "panel") and (e.get("m") is None or e["m"] <= ELEV_CLOSE_M)
        near = e.get("why") in ("doors", "panel", "button", "display") and (e.get("m") is None or e["m"] <= ELEV_CLOSE_M)
        if st["elev_in"] and e.get("why") == "button" and (e.get("m") or 99) <= ELEV_CLOSE_M and obs["place"] != "in_elev":
            # 탄 뒤 2m 안에 위·아래 호출 버튼 = 밖 (2026-11-23 시험 3: 1층에서 기다리며 닫힌 문을 '안'으로 3장 본 뒤
            # 호출 버튼 장면 3장이 하차로 잡힐 뻔함) → 탄 것을 취소
            st.update(elev_in=False, elev_run=0, hall_run=0, exit_cand=None, floor_move=None)
            self.mem.event(clock, "탄 뒤 호출 버튼이 가까이 보임: 탄 것이 아님 (취소)")
            return None
        inside = obs["place"] == "in_elev" or (st["elev_in"] and (obs.get("panel") or obs.get("closed_doors") or close))
        if inside and obs["place"] != "in_elev":
            self.mem.event(clock, "탄 뒤 엘리베이터 문이 가까이 보임(" + ("닫힌 문만" if obs.get("closed_doors") else
                           "층 버튼 패널" if obs.get("panel") else f"문 {e.get('m')}m") + "): 계속 안으로 봄")
        if inside:
            st["elev_run"] += 1
            st["hall_run"], st["exit_cand"] = 0, None
            if st["elev_run"] >= EXIT_IN and not st["elev_in"]:
                st["elev_in"] = True
                self.mem.event(clock, f"엘리베이터 안 장면 {EXIT_IN}장 연속: 탄 것으로 봄")
            if st["elev_in"]:  # 탔으면 층이 바뀔 수 있음: 계단처럼 다른 층 문패·층 표시 한 번이면 층을 바꾼다
                st["floor_move"] = {**(st["floor_move"] or {"t": clock, "why": "엘리베이터 안"}), "last": clock}
            return None
        st["elev_run"] = 0
        if not st["elev_in"] or obs["place"] not in HALL or near:  # 2m 안에 엘리베이터가 보이는 장면은 하차 근거가 아님
            st["hall_run"], st["exit_cand"] = 0, None
            return None
        st["hall_run"] += 1
        if st["hall_run"] == 1:
            st["exit_cand"] = {"t": clock, "floor": st["floor"], "steps": steps}
        if st["hall_run"] < EXIT_OUT:
            return None
        cand = st["exit_cand"]
        walked = None if (steps is None or cand.get("steps") is None or not self.steps_seen) else steps - cand["steps"]
        if walked is not None and walked < EXIT_STEPS:
            return None  # 밖 장면이지만 아직 걷지 않음 (문이 열린 채 서 있음 등): 더 기다림
        st_n = st["hall_run"]
        st.update(elev_in=False, hall_run=0, exit_cand=None)
        if self.last_exit is not None and clock - self.last_exit < EXIT_DUP:
            self.mem.event(clock, f"자동 하차 감지했지만 {EXIT_DUP}초 안에 이미 내림 처리: 같은 하차로 봄")
            return None
        # 첫 복도 사진에서 이미 다른 층 문패로 층을 바꿨으면 그 층을 이어받는다 (새 구간이 지우지 않게)
        keep = (st["floor"], st["floor_src"]) if st["floor"] and st["floor"] != cand["floor"] \
            and st["floor_src"] != "엘리베이터 표시창" else None
        seq = f"live_{self.si + 1}" if str(self.seq).startswith("live_") else f"{self.seq}+내림"  # 웹 기록 화면은 live_번호
        self.new_segment(seq, start=cand["t"], keep=keep)
        self.mem.event(clock, f"엘리베이터에서 내림 (자동 감지: 엘리베이터 안 {EXIT_IN}장 → 복도·홀 {st_n}장"
                       + (f", {walked}걸음)" if walked is not None else ", 걸음 수 없음)"))
        return {"t": cand["t"], "kept_floor": keep[0] if keep else None, "walked": walked}

    # ------------------------------------------------------------ 재확인 대상
    def recheck_pick(self, obs):
        """확대 재확인할 문패 1개 (없으면 None). 이미 확정한 번호는 다시 보지 않는다."""
        st, target, tf = self.st, self.target, self.tf
        done = set(self.mem.confirmed())
        plates = [p for p in obs["plates"] if p["n"] not in done and p.get("box")]
        if ROOM.match(target):
            near = [p for p in plates if ROOM.match(p["n"]) and p["n"][0] == target[0]
                    and abs(int(p["n"]) - int(target)) <= 2 and not self.quick_skips(p)]
            if near:
                return near[0], "목적지 근처 번호"
        unread = [p for p in obs.get("unread", []) if p.get("box")]
        if unread and (st["floor"] is None or st["floor"] == tf):
            return unread[0], "번호 안 읽힌 문패"
        # (2026-11-23 사용자 지시) 층이 정해진 뒤에는 '층 충돌 번호' 재확인을 하지 않는다: 목적지 후보와 번호를 못 읽은 문패만.
        # 재확인이 있던 걸음은 찍은 뒤 반영까지 중앙값 4.26초(없을 때 3.05초, 휴대폰 기록 200장).
        # 층을 모르는 동안은 일반 문패도 재확인한다 (판단 지점 A6: 317을 한 번의 호출로 확정해야 4초에 3층).
        if st["floor"] is None:
            first = [p for p in plates if ROOM.match(p["n"]) and not self.quick_skips(p)]
            if first:
                return first[0], "층을 정할 번호"
        return None, None

    def quick_skips(self, p):
        """빠른 확인이 곧 사진 2장으로 확인할 목적지 번호면 첫 사진의 재확인을 건너뛴다 (첫 응답 앞당기기, 2026-11-23)."""
        st = self.st
        return (p["n"] == self.target and self.quick_max > 0 and st["quick"] is None and st["quick_n"] < self.quick_max
                and not st["arrived"] and p.get("m") is not None and p["m"] <= QUICK_M
                and not (self.mem.get(self.target) or {}).get("ok"))

    @staticmethod
    def quick_obs(plates, base=None):
        """문 앞 빠른 확인 호출(가장 가까운 문패 번호만 읽음) 결과를 step()에 넣을 인식 형식으로.
        장소·앞 출입문은 바로 앞 인식(base)을 이어 쓰고, 엘리베이터·화살표·층 표시는 안 본 것으로 둔다."""
        b = base or {}
        return {"plates": [dict(p) for p in plates], "unread": [], "floor_sign": "", "floor": None,
                "place": b.get("place", "unknown"), "door_ahead": b.get("door_ahead", "none"),
                "elev": {"seen": False, "why": "none", "side": "none", "m": None, "btn": "none"},
                "arrows": [], "fixes": [], "quick": True}

    # ------------------------------------------------------------ 위치 파악 끝
    def locate_end(self, loc):
        """위치 파악 시간(LOCATE_SEC)이 끝났는데 층을 모름. 1층으로 가정하지 않는다 (2026-11-23 사용자 지시):
        층을 모르는 동안은 엘리베이터·계단 안내 없이 둘러보게 하고, ASK_FLOOR_SEC가 지나면 층을 묻는다."""
        st = self.st
        places = Counter(p for p in loc["places"][-LOCATE_SEC:] if p and p != "unknown")
        place = cp.PLACE_KO.get(places.most_common(1)[0][0]) if places else None
        # 2026-11-23 사용자 지시 7: 위치 파악이 끝났는데 층 근거가 없으면 바로 묻는다 (시험 3: 15초에 처음 물어 44초에 앎)
        st["loc"], st["act"], st["check_t"], st["ask_t"] = None, "멈추고 둘러보기", self.clock, self.clock
        return [(f"여기는 {place}인 것 같습니다. " if place else "") + ASK_SAY]

    def check_say(self, say, head=""):
        """층을 모를 때 '지금 층을 확인하고 있습니다…'를 말한다 (QUIET_SAY초 안에 이미 말했으면 되풀이하지 않음)."""
        st = self.st
        if self.scanning():  # 내린 직후 둘러보기 중: 이미 "좌우로 비춰 주세요"라고 함
            return
        if st["check_t"] is not None and self.clock - st["check_t"] < QUIET_SAY:
            if head:
                say.append(head.strip())
            return
        say.append(head + CHECK_SAY)
        st["check_t"], st["act"] = self.clock, "멈추고 둘러보기"

    # ------------------------------------------------------------ 한 걸음
    def scanning(self):
        sc = self.st.get("scan")
        return bool(sc) and not sc["done"]

    def step(self, frames=None, t=None, observed=None, quick=False, rechecked=None, heading=None, steps=None):
        """frames: 1초 구간의 사진 파일 경로들 (영상 실행). t: 사진 찍은 시각(초, 겹쳐 보내기·웹),
        observed: 이미 받은 인식 결과(겹쳐 보내기에서 다른 스레드가 부른 것).
        quick: 문 앞 빠른 확인 사진(quick_obs 형식, 확대 재확인 안 함).
        rechecked: (cand, why, 재확인 결과) — 웹 서버가 상태 잠금 밖에서 미리 부른 확대 재확인 (2026-11-19 사용자 지시 1).
        heading: 사진을 찍을 때 휴대폰 나침반 방향(도, 없으면 None). 내린 뒤 방향 정하기에 쓴다 (2026-11-23 B8).
        반환: 이 걸음의 기록."""
        st, target, tf = self.st, self.target, self.tf
        self._ev0 = len(self.mem.events)  # 이 걸음에서 생긴 사건(후보·확정·지나침·하차·방향…)을 기록에 함께 남김
        if t is not None:
            self.clock = t
        clock = self.clock
        say, note = [], {}
        o = observed or self.runner.observe(frames)
        calls = [] if o["cached"] else [o["call"]]
        latency = o["pick_seconds"] + (o["call"].get("seconds") or 0)
        note["credit_all"] = o["call"].get("credit") or 0.0  # 캐시에서 온 호출도 포함한 이 초의 크레딧
        obs = o["obs"]
        if obs is None:  # 호출 실패: 시간 초과면 멈춤 안내, 상태는 그대로
            if quick and st["quick"]:
                self._quick_result(None, say)
            err = o["call"].get("error")
            if o["call"].get("timed_out"):
                say.append("잠시 멈춰 주세요.")
            return self._finish(o, None, [], say, note, latency, calls, error=err, recheck=None)

        self.mem.place(clock, cp.PLACE_KO.get(obs["place"]))
        if not quick:  # 자동 하차 감지 (B5). 감지하면 이 사진부터 새 구간의 걸음
            auto = self.watch_exit(obs, clock, steps)
            if auto:
                note["auto_exit"] = auto
        # 휴대폰 센서 (core/arrival.py 걸은 거리): 걸음 수가 한 번도 0보다 크지 않았으면(센서 꺼짐) 시간으로 추정
        self.sensor = {"steps": steps if self.steps_seen else None, "heading": heading}
        si = self.si
        first = self.seg_n == 0  # 구간(경로 시작·엘리베이터 하차)의 첫 걸음
        heading = (self.sensor or {}).get("heading")
        self.mem.graph.sense_heading(clock, heading)  # 휴대폰 방위로 돌았는지 (core)
        self.mem.graph.move(clock, (self.sensor or {}).get("steps"))  # 걸음 수로 복도 위 위치 (없으면 시간)
        # 갈림길 (core/floor_graph.py): 걸음 수·방위로 위치를 대략 알고, 깊이로 옆이 트인 곳을 갈림길로 기억
        self.floor.move((self.sensor or {}).get("steps"), heading)
        if self.floor.corridor_turn:  # 다른 축 복도로 들어섬: 복도 하나의 번호 흐름 판단은 새로
            self.mem.graph.reset()
            if self.floor.edges[self.floor.edge]["dead"]:  # 들어선 가지는 이미 끝까지 가 봄: 걷는 쪽(다음 방위 기준 앞)은 가 본 끝
                self.mem.graph.ends.add(1)
            h0 = (self.arrival.mem or {}).get("heading")
            if h0 is None or heading is None or abs((heading - h0 + 180) % 360 - 180) > 45:
                self.arrival.cancel()  # 전 복도에서 본 목적지 위치는 버림 (이 복도에서 이미 본 것이면 둠)
            st.update(dir_said=None, dir_rooms=[], turn_said_t=-99)
        jn = obs.get("junction")
        st["junc"] = self.floor.see_junction(jn["m"], jn["open"], heading, jn.get("signs")) if jn else None
        if st["junc"] is None:
            st["junc_said"] = None  # 갈림길이 안 보이면 다음에 다가갈 때 다시 판단
        # 깊이로 막다른 곳 (옆을 비추는 중에는 안 봄). 앞에 갈림길이 보이면 앞 벽은 막다른 곳이 아니라 꺾을 곳이라 보지 않음
        # (막다른 곳으로 적으면 복도 판단이 '뒤로 돌아'로 갈림길 판단을 덮음, 시뮬레이터 로비)
        if obs.get("depth") and not self.arrival.asking() and st["junc"] is None and self.floor.at is None:
            ends = set(self.mem.graph.ends)
            self.mem.graph.see_depth(clock, obs["depth"].get("far_m"), heading)
            if self.mem.graph.ends - ends:
                self.floor.dead_end()
        # ---- 층 이동 가능 (2026-11-19 사용자 지시 2: 계단으로 층을 옮긴 경우)
        # 계단 장면이거나 층 표시가 보이면 표시해 두고, 그 뒤 다른 층 문패·층 표시가 또렷하면(확정 문패, 읽힌 층 표시)
        # 2번 연속을 기다리지 않고 바로 층을 바꾼다. 층이 바뀌면 지운다.
        # PR #1 리뷰 1: 지금 층과 같은 층 표시로는 켜지 않는다 (4층에서 '4F'를 본 뒤 215 오독 한 번에 2층으로 바뀌던 것).
        # 계단을 벗어나고 마지막 계단 단서(계단 장면, 다른 층 표시) 뒤 FLOOR_MOVE_HOLD초가 지나면 끈다.
        cue = "계단" if obs["place"] == "stairs" else (
            f"층 표시 '{obs['floor_sign']}'" if obs["floor"] and obs["floor"] != st["floor"] else None)
        if cue:
            if not st["floor_move"]:
                st["floor_move"] = {"t": clock, "why": cue, "last": clock}
                self.mem.event(clock, f"층 이동 가능 ({cue})")
            else:
                st["floor_move"]["last"] = clock
        elif st["floor_move"] and not st["stair"]["on"] and \
                clock - st["floor_move"].get("last", st["floor_move"]["t"]) >= FLOOR_MOVE_HOLD:
            self.mem.event(clock, f"층 이동 가능 끔 (계단 단서 {FLOOR_MOVE_HOLD}초 없음)")
            st["floor_move"] = None
        loc = st["loc"]
        if first and si == 0:
            if obs["door_ahead"] != "none":  # 출입문 앞에서 시작: 문을 지나 들어가는 것이 먼저 (층은 가정하지 않음, 2026-11-23)
                kind = "유리문" if obs["door_ahead"] == "glass" else "문"
                say.append(self.start_head() + (f"지금 {st['floor']}층입니다. " if st["floor"] else "")
                           + f"앞에 {kind}이 있습니다. 문을 지나 들어가세요.")
                # 밖에서 건물 출입문을 보고 시작했는가 (첫 사진의 장소가 '건물 출입문'): 지나가면 1층의 약한 근거 (2026-11-23 (2))
                st.update(door_phase=True, act="문 통과", door_out=obs["place"] == "entrance")
            elif st["floor"] is not None:  # 출발 층을 말로 받음: 위치 파악 없이 바로 다음 행동
                say.extend(self.floor_next(head=self.start_head() + f"지금 {st['floor']}층입니다. "))
            else:
                st["loc"] = loc = {"until": clock + LOCATE_SEC, "asks": 0, "places": []}
                say.append(self.start_head() + "지금 위치를 확인하겠습니다. 휴대폰을 앞으로 들고 천천히 좌우로 비춰 주세요.")
                st["act"] = "멈추고 둘러보기"
        if loc is not None:
            loc["places"].append(obs["place"])
        if first and si > 0 and st.get("exit_h") is None:
            st["exit_h"] = heading  # 내린 직후 바라보는 방향 (엘리베이터는 그 반대쪽)
        if first and si > 0:  # 내린 직후: 멈춰서 둘러보기 (B6). 웹 버튼은 응답에서 이미 말함
            if self.exit_say:
                say.append(self.exit_sentence())
                self.exit_say = False
            st["act"] = "멈추고 둘러보기"

        # ---- 출입문을 지났는가
        if st["door_phase"] and not (first and si == 0):
            st["door_gone"] = st["door_gone"] + 1 if obs["door_ahead"] == "none" else 0
            if st["door_gone"] >= DOOR_GONE:
                st["door_phase"] = False
                if st["floor"] is None and st.get("door_out"):
                    # 밖에서 건물 출입문을 지나 들어옴 = 지상층(1층)의 근거 (2026-11-23 사용자 지시 (2): 판단 지점 1~4가 실제로
                    # 건물 입구로 들어오는 1층). 문패·층 표시·말로 받은 층보다 약하게: 임시 층(floor_temp)으로 두어
                    # 다른 층 확정 문패·층 표시 한 번이면 그쪽을 따르고, 말로 받은 층은 바로 덮는다.
                    # 처음부터 실내에서 시작했으면(첫 사진이 출입문 앞이 아니면) 1층으로 가정하지 않는다.
                    st.update(floor="1", floor_temp=True, floor_src="출입문")
                    self.mem.event(clock, "건물 출입문을 지나 들어옴: 임시 1층 (문패·층 표시가 다르면 그쪽을 따름)")
                    say.extend(self.floor_next(head="문을 지났습니다. 건물 입구로 들어왔으니 1층으로 보고 안내하겠습니다. "))
                elif st["floor"] is None:  # 층을 모름: 엘리베이터를 찾으라고 하지 않는다 (2026-11-23, 전에는 1층 가정)
                    self.check_say(say, head="문을 지났습니다. ")
                else:
                    say.extend(self.floor_next(head="문을 지났습니다. "))

        # ---- 확대 재확인 (최대 1개)
        plates = [dict(p) for p in obs["plates"]]
        rflag, recheck = {}, None
        if quick:
            cand, why, rc = None, None, None
        elif rechecked is not None:
            cand, why, rc = rechecked
        else:
            cand, why = self.recheck_pick(obs)
            rc = self.runner.recheck(cand.get("frame") or o["key"], cand) if cand else None
        if cand:
            if rc:
                if not rc["cached"]:
                    calls.append(rc["call"])
                latency += rc["call"].get("seconds") or 0
                note["credit_all"] += rc["call"].get("credit") or 0.0
                recheck = {"why": why, "asked": cand["n"] or cand["name"], "read": rc["n"], "match": rc["match"]}
                if cand["n"]:
                    rflag[cand["n"]] = rc["match"]
                elif rc["n"] and rc["n"] in [p["n"] for p in plates]:
                    # 이름만 읽힌 문패를 확대해 읽은 번호가 이 사진에서 읽힌 번호와 같음 = 그 번호를 확대해서 다시 읽은 것
                    # (2026-11-23: q70 판단 지점 A6에서 '소재분석실' 확대가 317을 읽었는데 317 확정에 안 쓰여 층이 늦었음)
                    rflag.setdefault(rc["n"], True)
                    recheck["same_as"] = rc["n"]
                elif rc["n"] and rc["n"] not in [p["n"] for p in plates]:  # 이름만 읽힌 문패의 번호를 확대해서 읽음
                    plates.append({**cand, "n": rc["n"], "from_recheck": True})
                    plates.sort(key=lambda p: p["m"] if p["m"] is not None else 99)

        # ---- 문패 확정 (믿음 점수)
        rooms, src = [], {}
        # 내린 뒤 둘러보는 중 나침반이 있으면 몸을 돌리는 중이라 문패를 복도 그래프에 넣지 않는다.
        # 나침반이 없으면 넣는다 (전 규칙도 나침반이 없을 때만 둘러보며 읽은 번호 순서를 방향 근거로 썼다).
        scanning_now = self.scanning()
        for p in plates:
            rec = self.mem.see(clock, p["n"], p["side"], p["m"], p["on"], rflag.get(p["n"]), wall=p.get("wall"), dist=p.get("dm"),
                               graph=not quick and not (scanning_now and heading is not None))
            if rec["ok"] and rflag.get(p["n"]) is not False:
                rooms.append(p["n"])
                src[p["n"]] = p
        # 빠른 확인 사진이 목적지를 읽음 (2026-11-23 B4): 서로 다른 사진 2장(호출 2번)이면 믿음 점수와 상관없이 확정
        if quick and st["quick"] and target not in rooms:
            tp = next((p for p in plates if p["n"] == target), None)
            rec = self.mem.get(target)
            if tp and rec and not rec["ok"] and rec["calls"] >= nav_memory.TARGET_CALLS:
                rec.update(wait_calls=False)
                self.mem.set_ok(target, True)
                rec["ok_t"] = clock
                self.mem.event(clock, f"{target}호 문패 확정 (빠른 확인, 서로 다른 사진 {rec['calls']}장)")
                rooms.append(target)
                src[target] = tp
        rooms = [x for x in rooms if ROOM.match(x)] + [x for x in rooms if not ROOM.match(x)]
        # 내린 뒤 엘리베이터에서 멀어지며(돌아가는 중이 아닐 때) 확정한 문패와 그 사진의 나침반 방향 (B8c: 다른 층에서
        # 같은 끝자리 번호의 약한 근거). act는 이 걸음 전 것: 다른 층 문패를 처음 본 걸음도 넣는다.
        for x in rooms:
            if si > 0 and heading is not None and ROOM.match(x) and st["act"] != "엘리베이터로 돌아가기" \
                    and x not in [d["n"] for d in self.plate_dirs]:
                self.plate_dirs.append({"floor": x[0], "n": x, "h": heading, "t": clock})

        # ---- 계단 (2026-11-19 사용자 지시 3·4·5)
        stair = st["stair"]
        cur = st["floor"]  # 층을 모르면 계단 안내(올라/내려)를 하지 않는다 (2026-11-23, 전에는 1층 가정)
        up = None if not (tf and cur) or tf == cur else fnum(tf) > fnum(cur)
        sf = obs.get("stair_floors") or []
        if sf:
            stair["sign"], stair["sign_text"] = sf, obs.get("stair_sign", "")
            self.mem.event(clock, f"계단 층 표시 '{stair['sign_text']}' (" + "~".join(sf) + "층)")
        if obs["place"] == "stairs":
            stair["run"] += 1
            stair["off"] = 0
        else:
            stair["run"] = 0
            stair["off"] += 1
        if not stair["on"] and stair["run"] >= STAIR_ON:
            stair["on"] = True
            if up is not None and not stair["said"] and not st["arrived"]:
                way = "올라" if up else "내려"
                say.append(f"계단입니다. {tf}층까지 {way}가세요.")
                stair["said"] = True
                st["act"] = f"계단으로 {way}가기"
                # 계단을 고르면 엘리베이터 안내는 멈춘다
                st.update(elevator_said=False, el_approach=False, el_ready=None, el_cand=None, el_last=None, el_miss=0)

        # ---- 엘리베이터: 후보(근거 4종 아무거나)를 먼저 말하고, 다음 사진에서 버튼·층 표시창으로 확정 (지시 7)
        # 확정 = 앞 사진에 후보 + 지금 사진에 버튼·층 표시창. (전: 버튼·층 표시창 2번 연속. 이전 캐시는 후보=버튼·표시창이라 같음)
        el = obs["elev"]
        el_cand = el.get("cand", el["seen"])
        st["el_run"] = st["el_run"] + 1 if el_cand else 0
        el_ok = st["el_run"] >= 2 and el["seen"]
        if el["btn"] in ("L", "R"):
            st["button"] = {"L": "왼쪽", "R": "오른쪽"}[el["btn"]]
        ignore_el = si > 0 and clock - self.seg_start < EXIT_IGNORE
        need_el = bool(tf) and not stair["on"] and st["wrong_floor"] and st["floor"] is not None
        if el_cand and not ignore_el:
            st["el_last"], st["el_miss"], st["el_passed"] = {"t": clock, "side": el["side"], "m": el["m"]}, 0, False
            if (st["el_run"] == 1 and need_el and loc is None and not st["elevator_said"] and not st["el_approach"]
                    and not st["el_cand"] and el["side"] in SIDE_FRONT):
                say.append(f"{SIDE_FRONT[el['side']]} 엘리베이터가 있는 것 같습니다. 천천히 가세요.")
                st["el_cand"] = {"t": clock, "side": el["side"], "m": el["m"], "why": el["why"]}
                self.mem.event(clock, f"엘리베이터 후보 ({el['why']}, {el['side']}, {el['m']}m)")
        elif st["el_last"] is not None and not ignore_el:
            st["el_miss"] += 1
            last = st["el_last"]
            if st["el_miss"] >= EL_MISS:
                beside = last["side"] in ("L", "R") and (last["m"] is None or last["m"] <= EL_PASS_M)
                reached = st["elevator_said"] and last["m"] is not None and last["m"] <= 2.0
                said = st["el_cand"] or st["el_approach"] or st["elevator_said"]
                confirmed = st["el_approach"] or st["elevator_said"]
                if obs["place"] in ("stairs", "in_elev") or stair["on"] or not need_el or reached or not said:
                    pass  # 계단을 골랐거나 탔거나 이미 엘리베이터 앞까지 안내함: 말하지 않음
                elif beside:  # 지시 9: 옆에서 보이다 사라짐 = 지나침
                    say.append(f"엘리베이터를 지나쳤습니다. 뒤로 돌아 {BACK_CLOCK[last['side']]}시 방향입니다.")
                    st["act"], st["el_passed"] = "엘리베이터로 돌아가기", True
                    st.update(elevator_said=False, el_approach=False)
                    self.mem.event(clock, "엘리베이터 지나침")
                elif not confirmed:
                    self.mem.event(clock, "엘리베이터 후보 취소 (다시 안 보임)")
                st.update(el_cand=None if not confirmed else st["el_cand"], el_last=None, el_miss=0)
        if el_ok and not ignore_el:
            if not st["el_seen"]:
                self.mem.event(clock, "엘리베이터 확인 (" + ("호출 버튼" if el["why"] == "button" else "층 표시창") + ")")
            st["el_seen"] = True
        if el_ok and loc is None and not st["elevator_said"] and not ignore_el and tf and not stair["on"]:
            btn = f"호출 버튼은 엘리베이터 문 {st['button']} 벽에 있습니다. " if st["button"] else ""
            where = (f"엘리베이터가 {cp.CLOCK[el['side']]}시 방향" + (f" 약 {max(1, round(el['m']))}미터" if el["m"] else "")
                     + "에 있습니다. ") if el["side"] in cp.CLOCK else "앞에 엘리베이터가 있습니다. "
            near_el = el["m"] is None and el["side"] == "C" or el["m"] is not None and (
                el["m"] <= 2.0 or el["side"] == "C" and el["m"] <= EL_NEAR_M)
            seeking = need_el
            if seeking and not near_el:
                if not st["el_approach"]:
                    say.append(f"{where}그쪽으로 천천히 가세요.")
                    st["el_approach"] = True
                    if si == 0:
                        st["act"] = "직진"
            elif seeking:
                say.append(("" if st["el_approach"] else where) + f"{btn}엘리베이터를 타고 {tf}층을 누르세요.")
                st["elevator_said"] = True
                st["act"] = "엘리베이터 타기" if si == 0 else "엘리베이터로 돌아가기"
            elif st["floor"] is None and st["el_ready"] is None:
                # 층을 모르면 엘리베이터 안내를 하지 않는다 (2026-11-23). 엘리베이터는 기억에만 두고, 층을 알게 되면
                # 그때 보이는 엘리베이터로 안내한다. (전: 6초 기다린 뒤 층 모른 채 '엘리베이터를 타고 ○층을 누르세요')
                st["el_ready"] = clock
                self.check_say(say)

        # ---- 내린 뒤 가정한 층(엘리베이터로 이동한 층·표시창) (2026-11-23 사용자 지시 5)
        # 다른 층 층 표시가 또렷하게 한 번, 또는 다른 층 문패가 서로 다른 사진 2장에서 읽히면(확대 재확인이 다른 번호로
        # 읽은 것은 빼고) 고친다. 오독 한 번으로는 바꾸지 않는다. 같은 층 문패·층 표시가 보이면 확정 근거로 바꾼다.
        floor_change, wrong_assumed = None, None
        if st.get("assumed") and st["floor"]:
            seen = st.setdefault("other_seen", {})
            for p in plates:
                if ROOM.match(p["n"]) and p["n"][0] != st["floor"] and rflag.get(p["n"]) is not False:
                    lst = seen.setdefault(p["n"][0], [])
                    if not lst or lst[-1][0] != self.n:
                        lst.append([self.n, clock, p["n"]])
            for dg in list(seen):
                seen[dg] = [x for x in seen[dg] if clock - x[1] <= OTHER_FLOOR_SEC]
            fs0 = obs["floor"]
            if fs0 and fs0 != st["floor"]:
                wrong_assumed = (fs0, f"층 표시 '{obs['floor_sign']}'")
            else:
                hit = next((dg for dg, l in seen.items() if len(l) >= OTHER_FLOOR_PHOTOS), None)
                if hit:
                    wrong_assumed = (hit, "문패 " + ", ".join(dict.fromkeys(x[2] for x in seen[hit])) + f"호 (사진 {len(seen[hit])}장)")
            if wrong_assumed:
                floor_change = wrong_assumed[0]
                self.mem.event(clock, f"내린 층이 다름: {st['floor']}층 → {floor_change}층 ({wrong_assumed[1]})")
            else:
                rooms = [x for x in rooms if not ROOM.match(x) or x[0] == st["floor"]]  # 한 번 읽힌 다른 층 번호는 아직 안 씀
                same = [x for x in rooms if ROOM.match(x) and x[0] == st["floor"]]
                if same or (fs0 and fs0 == st["floor"]):
                    why = f"{same[0]}호" if same else f"층 표시 '{obs['floor_sign']}'"
                    st.update(assumed=False, floor_temp=False, floor_src=why)
                    self.mem.event(clock, f"내린 층 {st['floor']}층 확인 ({why})")
        # ---- 층 충돌: 한 번 정한 층과 첫 자리가 다른 번호는 2초 연속일 때만 인정
        other = [x for x in rooms if st["floor"] and ROOM.match(x) and x[0] != st["floor"]] if not floor_change else []
        one_shot = st["floor_move"] or st["floor_temp"]  # 계단·층 표시 뒤, 또는 임시 층(계단 표시·엘리베이터 표시창)
        if other:
            digit = other[0][0]
            if one_shot:  # 한 번이면 층 이동
                floor_change = digit
                self.mem.event(clock, f"층 이동 뒤 다른 층 문패 {other[0]}호: {st['floor']}층 → {digit}층 (연속 2번 안 기다림)")
            elif st["pending"] and st["pending"][0] == digit and st["pending"][1] == self.n - 1:
                floor_change = digit
                self.mem.event(clock, f"층 충돌 2번 연속: {st['floor']}층 → {digit}층")
            else:
                st["pending"] = (digit, self.n)
                rooms = [x for x in rooms if not ROOM.match(x) or x[0] == st["floor"]]
        # 층 표지: 또렷하게 읽힌 층 글자는 한 번으로 층 근거 (사용자 지시 2026-11-18, 전에는 2초 연속).
        # 이미 정한 층과 다르면 문패와 같은 층 충돌 규칙(2번 연속)
        fs = obs["floor"]
        fs_ok = None
        if fs and st["floor"] is None:
            fs_ok = fs
        elif fs and fs != st["floor"] and not floor_change and not st.get("assumed"):
            if one_shot:
                floor_change = fs
                self.mem.event(clock, f"층 이동 뒤 층 표시 '{obs['floor_sign']}': {st['floor']}층 → {fs}층 (연속 2번 안 기다림)")
            elif st["pending"] and st["pending"][0] == fs and st["pending"][1] == self.n - 1:
                floor_change = fs
                self.mem.event(clock, f"층 표지 충돌 2번 연속: {st['floor']}층 → {fs}층")
            else:
                st["pending"] = (fs, self.n)

        # ---- 층 판단과 안내
        say_before_floor = len(say)
        floor_rooms = [x for x in rooms if ROOM.match(x)]
        floor, basis = (floor_rooms[0][0], f"{floor_rooms[0]}호") if floor_rooms else (fs_ok, f"층 표지 '{obs['floor_sign']}'") \
            if fs_ok else (None, None)
        if floor_change:
            floor, basis = floor_change, (wrong_assumed[1] if wrong_assumed else "층 이동 뒤 한 번" if one_shot else "층 충돌 2번 연속")
        # 엘리베이터 안 층 표시창 (2026-11-23 사용자 지시 4, 프롬프트 판 4의 ed 칸): 엘리베이터 안에서 또렷하면 바로 그 층.
        # 타고 가는 동안 바뀌므로 다르면 바로 바꾸고, 다른 층 확정 문패 한 번이면 문패를 따르도록 임시 층으로 둔다.
        ed = obs.get("elev_floor") if obs["place"] == "in_elev" else None
        if ed and not floor:
            st["el_disp"] = {"floor": ed, "t": clock}
            if ed != st["floor"]:
                self.mem.event(clock, f"엘리베이터 안 층 표시창 {ed}층" + (f" (전에는 {st['floor']}층)" if st["floor"] else ""))
                st.update(floor=ed, floor_temp=True, floor_src="엘리베이터 표시창", loc=None, pending=None,
                          wrong_floor=bool(tf) and ed != tf)
                say.append(f"엘리베이터 표시창이 {ed}층입니다." + (" 목적지 층입니다." if ed == tf else ""))
        # 엘리베이터를 찾던 중(안내했거나 다가가는 중) 목적지 층에 와 있으면 엘리베이터 안내를 멈춘다 (지시 2)
        el_busy = st["elevator_said"] or st["el_approach"]
        if floor and floor != st["floor"]:
            st["floor_move"], st["floor_temp"] = None, False
            st["floor_src"] = basis
        if floor:
            new = [x for x in floor_rooms if x not in [r for _, r in st["floor_rooms"]]]
            where = ""
            if floor_rooms and floor_rooms[0] in src:
                where = f"{cp.SIDE_KO[src[floor_rooms[0]]['side']]}에 {floor_rooms[0]}호가 보입니다. "
            if st["floor"] is None and loc is not None:  # 위치 파악 중 층을 앎 → 위치를 말하고 다음 행동
                st["floor"], st["loc"] = floor, None
                places = Counter(p for p in loc["places"] if p and p != "unknown")
                place = cp.PLACE_KO.get(places.most_common(1)[0][0]) if places else None
                head = f"지금 위치는 {floor}층" + (f" {place}" if place else "") + f"입니다. {where}"
                if (floor == tf or not tf) and self.can_goal_scan(floor_rooms):
                    say.append(head + self.goal_scan())
                elif floor == tf or not tf:
                    say.append(head + "목적지 층입니다. 앞으로 천천히 직진해 주세요. 문패를 확인하며 안내하겠습니다.")
                    st["act"] = "직진"
                else:
                    st["wrong_floor"] = True
                    say.append(head + f"{target}호는 {tf}층에 있습니다. 엘리베이터로 {tf}층에 가야 합니다. "
                               "앞으로 천천히 직진하며 엘리베이터를 찾겠습니다.")
                    st["act"] = "직진"
            elif st["floor"] is None:
                st["floor"] = floor
                if (floor == tf or not tf) and not el_busy and self.can_goal_scan(floor_rooms):
                    # 시작 층이 목적지 층 (2026-11-23 시험 2: "4층이 맞습니다" 뒤 행동 안내가 없었음)
                    say.append(f"{floor}층입니다. {where}" + self.goal_scan())
                elif floor == tf or not tf:
                    say.append(f"{floor}층이 맞습니다. {where}"
                               + ("엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다." if el_busy else ""))
                    st.update(el_ready=None, elevator_said=False, el_approach=False)
                    if st["act"] not in ("목적지 접근", "도착", "문 통과") and not self.scanning():
                        st["act"] = "직진"
                else:
                    st["wrong_floor"] = True
                    if si > 0:
                        say.append(f"여기는 {floor}층입니다. {target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요.")
                        st["act"] = "엘리베이터로 돌아가기"
                    else:
                        say.append(f"여기는 {floor}층입니다. {target}호는 {tf}층에 있습니다. "
                                   + ("" if st["elevator_said"] else f"엘리베이터를 찾아 {tf}층으로 가겠습니다. 앞으로 천천히 직진해 주세요."))
                        if not st["elevator_said"]:
                            st["act"] = "직진"
            elif floor != st["floor"] and wrong_assumed:
                # 내린 층이 다름: 오류를 먼저 알리고 바로 행동 (2026-11-23 사용자 지시 5)
                st.update(floor=floor, wrong_floor=bool(tf) and floor != tf, assumed=False, floor_temp=False)
                if st["wrong_floor"]:
                    say[:] = [x for x in say if not x.startswith("엘리베이터에서 내렸습니다")]
                    say.append(f"층이 다릅니다. 여기는 {floor}층입니다. 엘리베이터로 돌아가 {tf}층을 누르세요. "
                               + self.elevator_way(heading))
                    st["act"] = "엘리베이터로 돌아가기"
                    if self.scanning():
                        st["scan"].update(done=True, why="다른 층")
                else:
                    say.append(f"층이 다릅니다. 여기는 {floor}층, 목적지 층입니다. 문패를 찾겠습니다.")
            elif floor != st["floor"]:
                say.append(f"층 판단이 바뀌었습니다: {st['floor']}층 → {floor}층.")
                st["floor"], st["wrong_floor"] = floor, bool(tf) and floor != tf
                if not st["wrong_floor"] and el_busy:
                    say.append("목적지 층입니다. 엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다.")
                    st.update(el_ready=None, elevator_said=False, el_approach=False)
                elif not st["wrong_floor"] and st["act"].startswith("계단"):
                    say.append("목적지 층입니다. 복도를 따라 문패를 찾겠습니다.")
                elif not st["wrong_floor"] and ROOM.match(target) and self.can_goal_scan(floor_rooms):
                    say.append("목적지 층입니다. " + self.goal_scan())
                if (st["act"] not in ("목적지 접근", "도착") and not self.scanning()) or st["wrong_floor"]:
                    st["act"] = ("엘리베이터로 돌아가기" if si > 0 else "직진") if st["wrong_floor"] else "직진"
            if floor_rooms:
                st["room"] = floor_rooms[0]
                if st["floor_temp"] and floor_rooms[0][0] == st["floor"] and not stair["on"]:
                    st["floor_temp"], st["floor_move"] = False, None  # 문패가 임시 층을 확인함
            for x in new:
                st["floor_rooms"].append((clock, x))

        # ---- 계단 층 표시: 계단 중 추정, 목적지 층이면 "계단에서 나가세요", 계단에서 나오면 마지막 표시를 임시 층으로 (지시 4)
        at = None
        if stair["on"]:
            if obs["floor"]:
                at = obs["floor"]
            elif sf and len(sf) == 1:
                at = sf[0]
            if at and at == tf and not stair["exit_said"] and not st["arrived"]:
                del say[say_before_floor:]  # 같은 걸음의 층 판단 문장 대신 이 한 문장
                say.append(f"{tf}층입니다. 계단에서 나가세요.")
                stair["exit_said"], st["act"] = True, "계단에서 나가기"
            if stair["off"] >= STAIR_OFF or (stair["off"] and stair["exit_said"]):
                pass
        if stair["on"] and stair["off"] >= STAIR_OFF:
            stair.update(on=False, said=False)
            temp = None
            if stair["sign"] and st["floor_move"]:  # 계단에 들어선 뒤 층을 문패로 다시 확인하지 않았으면
                temp = stair["sign"][-1] if up or up is None else stair["sign"][0]
            if stair["exit_said"]:
                temp = temp or tf
            stair["exit_said"] = False
            if temp and temp != st["floor"]:
                st.update(floor=temp, floor_temp=True, floor_src="계단 층 표시", wrong_floor=bool(tf) and temp != tf)
                st["floor_move"] = st["floor_move"] or {"t": clock, "why": "계단 층 표시"}
                self.mem.event(clock, f"계단에서 나옴: 임시 층 {temp}층 (계단 층 표시 '{stair['sign_text']}')")
                if temp == tf:
                    say.append(f"{tf}층으로 보입니다. 복도를 따라 문패를 찾겠습니다.")
                    st["act"] = "직진"
                else:
                    way = "올라" if fnum(tf) > fnum(temp) else "내려"
                    say.append(f"지금 약 {temp}층입니다. {target}호는 {tf}층에 있습니다. 계단으로 {tf}층까지 {way}가세요.")
                    st["act"] = f"계단으로 {way}가기"
            elif st["act"].startswith("계단"):
                st["act"] = "직진"
                self.mem.event(clock, "계단에서 나옴 (층 표시 없음)")

        # ---- 내린 뒤 방향 근거 모으기 (B7·B8): 호실 범위 안내판·안내도, 목적지가 든 화살표, 둘러보며 읽은 문패
        scanning = self.scanning()
        self.collect_direction(obs, heading, clock, scanning)

        # ---- 진행 방향: 공통 판단 core/nav_graph.py (복도 고정 그래프 + 확률, 뒤돌기 추정). 목적지를 본 뒤 지나친 것도 여기서
        same_floor = not st["wrong_floor"] and (st["floor"] == tf or not ROOM.match(target))
        for r in rooms:
            if ROOM.match(r) and r[0] == st["floor"]:
                self.floor.see_number(r, src[r].get("dm") or src[r].get("m"), heading)
        busy = st["arrived"] or st["lscan"] > 0  # 도착했거나 제자리에서 한 바퀴 둘러보는 중 (로비)
        jsay = (self.junction_say(target, jn, note)
                if same_floor and not busy and not quick and not scanning and st["junc"] is not None else [])
        say.extend(jsay)
        # (2026-11-24 병합: 전의 ±3 문패 증감·±3 밖 같은 쪽 벽 규칙 대신 팀원의 확률 판단을 쓴다. docs/17 21절)
        # 내린 직후 둘러보는 동안(B6)은 말하지 않고, 나침반이 없을 때만 그래프 판단(앞쪽/뒤쪽)을 방향 근거(보통)로 모은다
        # (전 규칙의 '둘러보며 읽은 번호 증감 = 보통 근거'와 같은 자리). 나침반이 있으면 그래프에 넣지도 않는다.
        # 빠른 확인 사진(멈춰서 가장 가까운 문패만 읽음, 맞은편 벽일 수 있음)도 진행 방향에 쓰지 않는다
        # (2026-11-23 시험 2: 405 앞에서 멈춘 사진이 맞은편 408을 읽어 "멀어지고 있습니다. 뒤로 돌아가세요")
        if ROOM.match(target) and not st["wrong_floor"] and st["floor"] == tf and not st["arrived"] and not quick:
            if not scanning:
                if not busy and not jsay:
                    say.extend(self.direction_say(target, note, seen_now=any(p["n"] == target for p in plates)))
            elif heading is None:
                d = self.mem.graph.direction(target)
                st["dir_last"] = {k: d.get(k) for k in ("direction", "p_ahead", "p_turned")}
                rel = {"앞쪽": "F", "뒤쪽": "B"}.get(d["direction"])
                # 아직 0.7에 못 미쳐도 한쪽으로 기울면(GRAPH_LEAN) 약한 근거로 둔다: 둘러보기가 시간으로 끝날 때 '긴 복도 쪽'으로
                # 보냈다가 1초 뒤 그래프가 반대로 말하던 것(경로 C 318: 6초 '앞쪽 복도로' → 7초 '뒤로 돌아 주세요')을 막음
                pa = d.get("p_ahead")
                lean = None if pa is None or d["direction"] != "모름" else "F" if pa >= GRAPH_LEAN else "B" if pa <= 1 - GRAPH_LEAN else None
                st["scan"]["lean"] = (dr.evidence(lean, None, dr.WEAK, f"복도 그래프 기울기 (P(앞)={pa})", clock) if lean else None)
                evs = st["scan"]["evs"]
                if rel and not (evs and evs[-1].get("graph") and evs[-1]["rel"] == rel):
                    evs.append({**dr.evidence(rel, None, dr.MEDIUM, f"복도 그래프 {d['direction']} (P(앞)={d.get('p_ahead')})", clock),
                                "graph": True, "walk": d.get("walk", [])})
                    st["trend_said"] = "toward" if rel == "F" else "away"
                    note["trend"] = f"복도 그래프 {d['direction']} (둘러보는 중, 방향 근거로): {d.get('why')}"

        # ---- 목적지
        # 목적지 안전장치 (2026-11-18 사용자 지시): 목적지 번호·이름은 서로 다른 호출 2번에서 읽혀야 확정.
        # 그 전에는 '확인 중'까지만 말한다 (3단계 겹쳐 보내기에서 415 오독이 한 번의 호출로 확정된 것을 보고 추가).
        # 확정과 도착을 같은 관찰로 함께 채운다 (2026-11-19 사용자 지시 3): 확정 전 읽기도 2m 이내면 도착 연속에 넣는다.
        #   2m 안에서 서로 다른 호출 2번이 목적지를 읽으면 두 번째에 확정과 도착이 함께 된다.
        tgt, maybe, pend_m = None, False, None
        if ROOM.match(target) and not st["wrong_floor"]:
            if target in rooms:
                tgt = src[target]
            else:
                tr = self.mem.get(target)
                maybe = bool(tr and tr["last"] == clock and not tr["ok"] and rflag.get(target) is not False
                             and (st["floor"] in (None, tf)))
                if maybe:
                    tp = next((p for p in plates if p["n"] == target), {})
                    pend_m = tp.get("dm") or tr["reads"][-1]["m"]
                    # 확정 전 읽기도 위치는 기억 (화면 밖으로 빠진 뒤 옆을 비추게 함, 확정이 아니므로 '가능성 높음'은 안 함)
                    self.arrival.seen(clock, pend_m, cp.SIDE_KO.get(tp.get("side"), "옆"), (self.sensor or {}).get("steps"), sure=False,
                                      heading=(self.sensor or {}).get("heading"))
        elif not ROOM.match(target) and obs.get("dest", {}).get("seen"):
            st["dest_calls"] = st.get("dest_calls", 0) + 1 if clock - st.get("dest_last", -99) <= nav_memory.FORGET else 1
            st["dest_last"] = clock
            if st["dest_calls"] >= nav_memory.TARGET_CALLS:
                tgt = obs["dest"]
            else:
                maybe, pend_m = True, obs["dest"].get("m")
                # 확정 전 읽기도 위치는 기억 (호실 목적지와 같음: 화면 밖으로 빠지면 옆을 비추게 함, 시뮬레이터 v3 이름 문패)
                self.arrival.seen(clock, pend_m, cp.SIDE_KO.get(obs["dest"].get("side"), "옆"), (self.sensor or {}).get("steps"),
                                  sure=False, heading=(self.sensor or {}).get("heading"))
        scan_now = st["lscan"] > 0
        if scan_now:  # 한 바퀴 둘러보는 중: 목적지를 보면 방위만 적고 말은 다 돈 뒤에
            seen = tgt or (({"side": (next((p for p in plates if p["n"] == target), {}) if ROOM.match(target) else obs.get("dest") or {}).get("side"),
                             "m": pend_m}) if maybe else None)
            h = (self.sensor or {}).get("heading")
            if seen and h is not None and (st["scan_hit"] is None or (seen.get("m") or 99) < (st["scan_hit"]["m"] or 99)):
                st["scan_hit"] = {"bearing": (h + CLOCK_OFF.get(seen.get("side"), 0)) % 360, "m": seen.get("dm") or seen.get("m")}
            tgt, maybe, pend_m = None, False, None
        if maybe and not st["arrived"]:
            note["target_pending"] = True
            close = pend_m is not None and pend_m <= ARRIVE_M
            st["arr_run"] = st["arr_run"] + 1 if close else 0
            if close and st["arr_run"] == 1:
                say.append(("" if st.get("maybe_said") else "목적지 같습니다. ") + "멈춰 주세요, 확인 중입니다.")
                st["maybe_said"], st["act"] = True, "목적지 접근"
            elif not st.get("maybe_said"):
                say.append("목적지 같습니다, 확인 중입니다.")
                st["maybe_said"] = True
                st["act"] = "목적지 접근"
        if tgt:
            side, m = tgt["side"], tgt.get("dm") or tgt["m"]  # 깊이로 잰 거리가 있으면 그것 (Claude 거리는 눈대중)
            self.arrival.seen(clock, m, cp.SIDE_KO.get(side, "앞"), (self.sensor or {}).get("steps"),
                              heading=(self.sensor or {}).get("heading"))
            asking = self.arrival.asking()
            close = m is not None and m <= self.arrival.arrive_m()  # 정면이 아니어도. 옆을 비춰 확인 중이면 조금 넉넉히
            st["arr_run"] = st["arr_run"] + 1 if close else 0
            first = st["target_seen"] is None
            prev = st["target_seen"] or {}
            st["target_seen"] = {"t": clock, "side": side, "m": m}
            note.update(target_side=side, target_distance_m=m)
            if st["arrived"]:
                pass  # 도착 안내는 한 번만 (겹쳐 보내기로 도착 뒤에 들어온 사진이 다시 말하지 않게, 2026-11-19)
            elif st["arr_run"] >= 2 or (quick and close):
                # 빠른 확인 사진(B4): 서로 다른 사진 2장으로 확정됐고 이 사진에서 2m 안이면 그 자리에서 도착
                where = {"C": "정면에", "L": "왼쪽 앞에", "R": "오른쪽 앞에"}[side]
                if self.arrival.asking() and self.arrival.ask_side in ("왼쪽", "오른쪽"):
                    where = f"{self.arrival.ask_side}에"  # 옆을 비추는 중: 화면 기준이 아니라 비추라고 한 몸 기준 쪽 (시뮬레이터 v2)
                # 도착하는 걸음에는 도착 문장만 (같은 걸음의 층 확인·'문패를 찾겠습니다'와 섞이지 않게, 2026-11-19)
                say[:] = [f"{where} {target}{'호' if ROOM.match(target) else ''} 문이 있습니다."]
                st["arrived"], st["act"] = True, "도착"
            elif st["arr_run"] == 1:
                say.append("멈춰 주세요, 확인 중입니다.")
                st["act"] = "목적지 접근"
            elif asking and m is not None and m > self.arrival.arrive_m():
                # 옆을 비췄더니 목적지가 아직 앞에 있음: 도착 대신 다시 걷게 함 (전에는 '가능성 높음'까지 그 자리에 세워 둠)
                say.append(f"{target}{'호' if ROOM.match(target) else ''} 문패가 {self.arrival.ask_side or '옆'} 약 {max(1, round(m))}미터 앞에 있습니다. 천천히 가세요.")
                self.arrival.retry()
                st["act"] = "목적지 접근"
            elif first or side != prev.get("side") or (m and prev.get("m") and prev["m"] - m >= 2):
                say.append(f"{target}{'호' if ROOM.match(target) else ''} 문패가 {cp.CLOCK[side]}시 방향"
                           + (f" 약 {max(1, round(m))}미터" if m else "") + "에 있습니다.")
                st["act"] = "목적지 접근"
        elif not maybe and not scan_now:
            st["arr_run"] = 0
            # 목적지 문패가 안 보임: 본 거리에서 걸은 만큼 빼서 남은 거리 추정 → 가까우면 멈추고 옆을 비추게 함 (core/arrival.py)
            if not st["wrong_floor"] and (st["floor"] == tf or not ROOM.match(target)) and not st["arrived"] and not quick:
                lt = self.fresh()
                ask_t0, s_now = self.arrival.ask_t, lt.get("steps")
                walked = s_now is None or st.get("ask_steps") is None or (s_now - st["ask_steps"]) * STEP_M >= PASS_WALK_M
                if ask_t0 is not None and ROOM.match(target) and walked and self.mem.graph.direction(target, clock).get("passed") and any(
                        seg == self.mem.seg and p["room"] != target and p["last"] > ask_t0 for (seg, _), p in self.mem.plates.items()):
                    # 옆을 비추라고 한 뒤 걸어서 다른 문패를 읽었고 그래프도 지나쳤다고 봄 → 옆 비추기를 풀고 '지나침'
                    # (PR #3 테스트: 410을 본 뒤 412까지 읽었는데 "정면 바로 앞의 문이 410호일 가능성" → 잘못 도착).
                    # 걸음 수를 알면 실제로 걸었을 때만: 멈춰서 비추다 옆 문패를 읽은 것은 지나친 것이 아님 (시뮬레이터)
                    self.arrival.cancel()
                    say.extend(self.direction_say(target, note))
                ask = self.arrival.unseen(max(clock, lt.get("t", clock)), lt.get("steps"), lt.get("heading"))
                if ask:
                    if self.arrival.ask_t == clock:
                        st["ask_steps"] = s_now  # 옆을 비추라고 한 때의 걸음 수
                    say.extend(ask)
                    st["act"] = "목적지 접근"
                    if self.arrival.ask_t == clock and st["quick"] is None and st["quick_n"] < self.quick_max:
                        note["confirm_now"] = True  # 옆을 비추는 순간 빠른 확인 사진 (웹 /confirm)
                        note["confirm_after"] = "end"  # 휴대폰은 이 문장이 끝나고 돌릴 시간을 준 뒤 찍음 (2026-11-24 병합)
                        st["quick"] = {"t": clock, "got": 0, "hit": False, "other": 0}
                        st["quick_n"] += 1
        # 빠른 확인 (2026-11-23 사용자 지시 B3~B5): 목적지 후보를 5m 안에서 읽으면 휴대폰이 곧바로 0.4초 간격으로
        # 사진 2장을 찍어 동시에 보낸다(웹 /confirm). 2장은 각각 다른 호출(다른 사진)로 친다. 하나라도 목적지를 읽으면
        # 서로 다른 사진 2장 기준으로 확정, 2m 안이면 도착. 둘 다 다른 번호를 읽으면 후보 취소. 목적지 하나에 최대 2번.
        q = st["quick"]
        if quick and q:
            self._quick_result(plates, say)
        elif q and clock - q["t"] > QUICK_WAIT:
            st["quick"] = None
        read_m = (tgt or {}).get("m") if tgt else pend_m
        if (not quick and (tgt or maybe) and not st["arrived"] and st["quick"] is None and st["quick_n"] < self.quick_max
                and read_m is not None and read_m <= QUICK_M
                and (st["quick_n"] == 0 or maybe or read_m <= ARRIVE_M)):
            note["confirm_now"] = True
            st["quick"] = {"t": clock, "got": 0, "hit": False, "other": 0}
            st["quick_n"] += 1
            if maybe:  # 아직 후보: 바로 멈추게 하고 그 말을 시작한 뒤 휴대폰이 2장을 찍는다 (2026-11-23)
                say[:] = [x for x in say if "확인 중입니다" not in x] + [QUICK_SAY]
                st["maybe_said"], st["act"] = True, "목적지 접근"

        # ---- 화살표 기록 (2초 연속 같은 방향일 때 사건으로)
        arrows = [a.get("to") for a in obs["arrows"]]
        if arrows:
            if st["arrow_last"] and st["arrow_last"][0] == arrows[0] and st["arrow_last"][1] == self.n - 1:
                lab = obs["arrows"][0].get("label") or ""
                self.mem.event(clock, f"화살표 표지 {arrows[0]}" + (f" ({lab})" if lab else ""))
            st["arrow_last"] = (arrows[0], self.n)

        # ---- 내린 직후 둘러보기 끝, 방향 안내 (B6~B8)
        say.extend(self.direction_step(heading, clock, bool(tgt or maybe), note))
        # ---- 목적지 층인데 둘러보기도 아니고 행동이 '멈추고 둘러보기'(=모름)로 남으면 직진 (2026-11-23 사용자 지시)
        if (ROOM.match(target) and st["floor"] == tf and not st["wrong_floor"] and not st["arrived"]
                and not self.scanning() and st["act"] in ("멈추고 둘러보기", "", None)):
            st["act"] = "직진"
            if not say:
                say.append(QUIET_GOAL)
        # ---- 위치 파악 시간 끝
        if st["loc"] is not None and clock + 1 >= st["loc"]["until"] and not st["arrived"]:
            say.extend(self.locate_end(st["loc"]))
            note["ask_floor"] = True
        # ---- 엘리베이터 앞에서 층을 모름: 층을 알게 되면 대기 표시를 지운다 (층 모른 채 타라고 하지 않음, 2026-11-23)
        if st["el_ready"] is not None and st["floor"] is not None:
            st["el_ready"] = None
        # ---- 층 묻기 (2026-11-23 사용자 지시 3): 구간 시작 뒤 ASK_FLOOR_SEC초가 지나도 층을 모르면, ASK_FLOOR_EVERY초에 한 번
        if (st["floor"] is None and not st["arrived"] and st["loc"] is None and clock - self.seg_start >= ASK_FLOOR_SEC
                and (st["ask_t"] is None or clock - st["ask_t"] >= ASK_FLOOR_EVERY)):
            say.append(ASK_SAY)
            st["ask_t"] = clock
            note["ask_floor"] = True
        say.extend(self.scan_say(obs, target, same_floor, say))
        # ---- 무안내 시간: 지금 행동을 되풀이하는 규칙 문장 (Claude say는 2026-11-18 사용자 지시로 스키마에서 뺌)
        # (2026-11-19 사용자 지시 3) 어떤 상태든 10초 동안 새 안내가 없으면 지금 상황을 짧게 말한다.
        if say:
            st["last_say"] = clock
        elif st["loc"] is None and not st["arrived"] and clock - st["last_say"] >= QUIET_SAY:
            say.append(self.situation(obs))
            st["last_say"] = clock
        return self._finish(o, obs, rooms, say, note, latency, calls, error=None, recheck=recheck)

    # ------------------------------------------------------------ 내린 뒤 방향 (direction.py)
    def collect_direction(self, obs, heading, clock, scanning):
        """방향 근거를 모은다. 안내판·화살표(강)는 구간 내내, 둘러보며 읽은 문패와 긴 복도 방향은 둘러보는 동안만."""
        st, target = self.st, self.target
        if not ROOM.match(target) or st["arrived"] or st["scan"] is None:
            return
        sc = st["scan"]
        evs = []
        hit = dr.board_hit(obs.get("ranges"), target)
        if hit:
            evs.append(dr.evidence(hit["to"], heading, dr.STRONG, f"호실 범위 안내판 {hit['lo']}~{hit['hi']}", clock))
        a = dr.arrow_hit(obs.get("arrows"), target)
        if a:
            evs.append(dr.evidence(a["to"], heading, dr.STRONG, f"화살표 표지 '{a.get('label')}'", clock))
        if sc is not None:
            sc["evs"].extend(evs)
        if scanning:
            for p in obs["plates"]:
                if ROOM.match(p["n"]) and p["n"][0] == self.tf:
                    sc["reads"].append({"h": heading, "side": p["side"], "m": p["m"], "n": p["n"], "photo": self.n})
            if obs.get("long_way"):
                sc["lw"].append(dr.evidence("F" if obs["long_way"] == "C" else obs["long_way"], heading, dr.NONE,
                                            "복도가 길게 보이는 쪽", clock))
        elif evs and st["floor"] in (None, self.tf) and not st["wrong_floor"]:
            st["late_evs"] = evs  # 둘러보기 뒤에 본 안내판·화살표: direction_step이 바뀐 방향이면 다시 안내

    def best_direction(self, heading):
        """모은 근거 중 가장 센 것 (같은 세기면 나중 것). 같은 쪽 벽 문패 비교는 2개 이상 같은 결론이면 강, 1개면 보통."""
        st, target = self.st, self.target
        sc = st["scan"] or {"evs": [], "reads": [], "lw": []}
        cands = list(sc["evs"])
        wv = dr.wall_votes(sc["reads"], target)
        if wv:
            strength = dr.STRONG if wv["votes"] >= 2 else dr.MEDIUM
            cands.append({"rel": wv["rel"], "abs": wv["abs"], "strength": strength,
                          "basis": "같은 쪽 벽 문패 비교 " + ", ".join(wv["pairs"]), "t": self.clock})
        if heading is not None and st["floor"] in (None, self.tf):  # 다른 층에서 같은 끝자리 번호를 본 방향 (약)
            tail = int(target[1:])
            prev = [d for d in self.plate_dirs if d["floor"] != self.tf and abs(int(d["n"][1:]) - tail) <= 2]
            if prev:
                d = min(prev, key=lambda d: (abs(int(d["n"][1:]) - tail), -d["t"]))
                cands.append({"rel": "F", "abs": d["h"], "strength": dr.WEAK, "t": d["t"],
                              "basis": f"{d['floor']}층에서 {d['n']}호를 본 방향 (끝자리 비교)"})
        if sc.get("lean"):  # 복도 그래프가 한쪽으로 기움 (약, 2026-11-24 병합)
            cands.append(sc["lean"])
        if sc["lw"]:
            cands.append(sc["lw"][-1])
        if not cands:
            return None
        best = max(cands, key=lambda e: (e["strength"], e["t"]))
        return {**best, "rel_now": dr.now_rel(best, heading)}

    def direction_step(self, heading, clock, target_seen, note):
        """둘러보기를 끝낼지 정하고, 끝나면 방향을 한 번 안내한다. 둘러보기 뒤에는 더 센(또는 다른 방향의 강한) 근거가
        오면 다시 안내한다. 반환: 말할 문장들."""
        st, target, tf = self.st, self.target, self.tf
        sc, out = st["scan"], []
        if sc is None or not ROOM.match(target) or st["arrived"]:
            return out
        if not sc["done"]:
            best = self.best_direction(heading)
            why = ("목적지 보임" if target_seen else "다른 층" if st["wrong_floor"] and st["floor"] else
                   "강한 방향 단서" if best and best["strength"] >= dr.STRONG else "시간" if clock >= sc["until"] else None)
            # 보통 근거(문패 비교 1개)는 더 모을 수 있게 시간 끝까지 기다린다 (2개가 되면 '○쪽입니다'로 올라감)
            if not why:
                if st["act"] not in ("목적지 접근", "도착", "엘리베이터로 돌아가기"):
                    st["act"] = "멈추고 둘러보기"
                return out
            sc.update(done=True, why=why)
            self.mem.event(clock, f"둘러보기 끝 ({why}, {round(clock - sc['t'], 1)}초)")
            if heading is not None:  # 몸을 돌리며 본 문패 순서는 진행 방향 판단에 쓰지 않음
                st["near_d"] = []
            if why in ("목적지 보임", "다른 층"):
                return out
            out.append(self.say_direction(best, floor_known=st["floor"] == tf, note=note))
            return out
        late = st.pop("late_evs", None)
        if late:
            ev = max(late, key=lambda e: e["t"])
            ev = {**ev, "rel_now": dr.now_rel(ev, heading)}
            cur = st["dir"]
            if not cur or cur["strength"] < dr.STRONG or cur["rel_now"] != ev["rel_now"] and cur["basis"] != ev["basis"]:
                out.append(self.say_direction(ev, floor_known=st["floor"] == tf, note=note))
        return out

    def say_direction(self, ev, floor_known, note):
        st, target = self.st, self.target
        if ev is None:
            ev = {"rel": "F", "abs": None, "strength": dr.NONE, "basis": "단서 없음 (앞으로)", "t": self.clock, "rel_now": "F"}
        rel = ev["rel_now"]
        text = dr.sentence(target, rel, ev["strength"], long_way=ev["basis"] == "복도가 길게 보이는 쪽",
                           floor_known=floor_known)
        st["dir"] = {"rel_now": rel, "abs": ev["abs"], "strength": ev["strength"], "basis": ev["basis"], "t": self.clock}
        st["act"] = dr.ACT[rel]
        # 복도 그래프(core/nav_graph.py)와 맞춤 (2026-11-24 병합): 뒤로 돌라고 했으면 '돌았다/안 돌았다' 가설을 열고,
        # 옆 복도로 가라고 했으면 다른 복도라 그래프를 새로 시작한다. 그래프 판단을 그대로 말한 것이면 같은 말을 되풀이하지 않게 표시.
        if rel == "B":
            self.mem.graph.turned(self.clock)
            self.arrival.cancel()
            st.update(dir_said="돌기", turn_said_t=self.clock, trend_said="away")
        elif rel in ("L", "R"):
            self.mem.graph.reset()
            st.update(dir_said=None, dir_rooms=[])
        elif ev.get("graph"):
            st.update(dir_said="앞쪽", dir_rooms=list(ev.get("walk") or []), trend_said="toward")
        note["direction"] = {"rel": rel, "strength": dr.STRENGTH_KO[ev["strength"]], "basis": ev["basis"]}
        self.mem.event(self.clock, f"방향 {dr.WORD[rel]} (근거 {dr.STRENGTH_KO[ev['strength']]}: {ev['basis']})")
        return text

    def _quick_result(self, plates, say):
        """빠른 확인 사진 한 장의 결과를 센다. 2장이 다 오면: 목적지를 못 읽고 둘 다 다른 번호면 후보 취소."""
        st, target = self.st, self.target
        q = st["quick"]
        q["got"] += 1
        nums = [p["n"] for p in (plates or []) if p.get("n")]
        if target in nums:
            q["hit"] = True
        elif nums:
            q["other"] += 1
        if q["got"] < QUICK_SHOTS:
            return
        st["quick"] = None
        rec = self.mem.get(target)
        if not q["hit"] and q["other"] >= QUICK_SHOTS and not st["arrived"] and not (rec and rec["ok"]):
            if rec:
                rec.update(n=0, calls=0, front=False, door=False, near=False, recheck=False, last=-99)
            st.update(maybe_said=False, arr_run=0, act="직진")
            self.mem.event(self.clock, f"빠른 확인 2장 모두 다른 번호: {target}호 후보 취소")
            say.append("목적지가 아닙니다. 계속 찾겠습니다.")

    def fresh(self):
        """지금 아는 가장 최근 센서 (처리 중인 사진의 센서보다 최근 것이 있으면 그것)."""
        lt = self.latest
        if lt and lt.get("t") is not None and lt["t"] >= self.clock:
            return lt
        return {"t": self.clock, **(self.sensor or {})}

    def turned_since(self):
        """이 사진을 찍은 뒤 사용자가 돌았나 (사진 때 방위와 가장 최근 방위가 135° 넘게 다름)."""
        h0, h1 = (self.sensor or {}).get("heading"), self.fresh().get("heading")
        if h0 is None or h1 is None:
            return False
        return abs((h1 - h0 + 180) % 360 - 180) > nav_graph.TURN_DEG

    def junction_say(self, target, jn, note):
        """갈림길이 JUNC_SAY 안에 보이면 어느 복도로 갈지 한 번 말한다 (갈림길에 들어올 때마다 한 번)."""
        st, j = self.st, self.st["junc"]
        heading = self.fresh().get("heading")
        key = (j, round((heading or 0) / 90) % 4)
        if not JUNC_NEAR <= jn["m"] <= JUNC_SAY or st.get("junc_said") == key or self.arrival.asking():
            return []
        flow = None
        if ROOM.match(target):
            flow = self.mem.graph.direction(target, self.clock).get("direction")
        c = self.floor.choose(j, heading, flow)
        st["junc_said"] = key
        if c is None:
            return ["이 층에서 목적지를 찾지 못했습니다. 주변 분께 물어봐 주세요."]
        note["junction"] = c["why"]
        self.mem.event(self.clock, f"갈림길 판단: {c['rel']} ({c['why']})")
        if c["rel"] == "back" and flow in ("앞쪽", "근처"):
            return []  # 갈림길 판단도 지금 복도를 고름: 복도 안에서 어느 쪽인지는 번호 흐름이 앎 (목적지가 갈림길 앞, 시뮬레이터 v3)
        if c["rel"] == "back":
            self.mem.graph.turned(self.clock)
            st.update(turn_said_t=self.clock)
            return ["갈림길입니다. 뒤로 돌아 주세요."]
        st["act"] = {"ahead": "직진", "left": "왼쪽", "right": "오른쪽"}[c["rel"]]
        return ["갈림길입니다. " + {"ahead": "직진하세요.", "left": "왼쪽 복도로 가세요.", "right": "오른쪽 복도로 가세요."}[c["rel"]]]

    def scan_say(self, obs, target, same_floor, say):
        """넓은 곳(로비)에서 목적지를 아직 못 봤으면 SCAN_M마다 한 바퀴 둘러보게 하고, 다 돌면 본 쪽을 지금 향한 쪽 기준 시계 방향으로.
        정면만 보고 벽을 따라 걸으면 로비 가운데 장소(안내데스크·카페)는 시야에 안 들어옴 (시뮬레이터 로비 섬)."""
        st, steps = self.st, (self.sensor or {}).get("steps")
        if st["lscan"] > 0:
            st["lscan"] -= 1
            st["last_say"] = self.clock  # 도는 동안 무안내 문장을 넣지 않음 (사용자가 돌기를 멈춤)
            if st["lscan"]:
                return []
            hit, st["scan_hit"] = st["scan_hit"], None
            st["scan_steps"] = steps
            if not hit:
                st["act"] = "직진"
                return ["주변에서 목적지가 보이지 않습니다. 계속 앞으로 천천히 가세요."]
            self.arrival.cancel()  # 도는 동안의 방위로 적힌 위치는 버림 (본 쪽은 아래 시계 방향으로 알려 줌)
            h1 = self.fresh().get("heading")
            k = round(((hit["bearing"] - (h1 or 0)) % 360) / 30) % 12 or 12
            st["act"] = "목적지 접근"
            label = f"{target}{'호' if ROOM.match(target) else ''}"
            return [f"{label}{josa(label, '이', '가')} {k}시 방향" + (f" 약 {max(1, round(hit['m']))}미터" if hit["m"] else "")
                    + "에 있습니다. 그쪽으로 천천히 가세요."]
        pos = self.floor.pos
        if obs.get("place") == "lobby" and pos is not None and steps is not None and all(math.dist(pos, q) > VISIT_M for q in st["visited"]):
            st["visited"].append(pos)
            st["new_steps"] = steps  # 새 곳에 옴
        if (obs.get("place") == "lobby" and st["new_steps"] is not None and steps is not None and not st["gave_up"]
                and same_floor and not st["arrived"] and st["target_seen"] is None and self.arrival.mem is None
                and (steps - st["new_steps"]) * STEP_M >= GIVE_UP_M):
            st.update(gave_up=True, act="멈춤")
            self.mem.event(self.clock, f"가 본 곳만 {GIVE_UP_M:.0f}m 넘게 다님 → 못 찾음")
            return ["이 층에서 목적지를 찾지 못했습니다. 주변 분께 물어봐 주세요."]
        if (say or obs.get("place") != "lobby" or not same_floor or st["arrived"] or st["target_seen"] is not None
                or self.arrival.mem is not None or self.arrival.asking() or steps is None or st["gave_up"]
                or (st["scan_steps"] is not None and (steps - st["scan_steps"]) * STEP_M < SCAN_M)):
            return []
        st["scan_steps"] = steps
        if pos is not None and any(math.dist(pos, q) < SCAN_M / 2 for q in st["scan_spots"]):
            return []  # 이미 둘러본 자리 근처: 다시 돌지 않음 (반경은 간격의 절반: 간판은 한 면에만 있어 다른 쪽에서 보면 보임)
        if pos is not None:
            st["scan_spots"].append(pos)
        st.update(lscan=SCAN_SHOTS, scan_hit=None, act="둘러보기")
        return [SCAN_SAY]

    def direction_say(self, target, note, seen_now=False):
        """복도 그래프의 방향 판단을 안내 문장으로. 판단이 바뀌거나 새 문패가 근거에 들어올 때만 말한다.
        '뒤로 돌아 주세요'는 TURN_REPEAT초 간격 (돌 시간을 주고, 그사이 문패로 돌았는지 판단).
        seen_now: 이 사진에서 목적지 문패를 읽음."""
        st, clock = self.st, self.clock
        flips = self.mem.graph.sense_flips
        if st["act"] == "뒤로 돌기" and flips and flips[-1] > st["turn_said_t"]:
            st["act"] = "직진"  # 방위로 돈 것을 확인: 10초 무안내 때 '뒤로 돌아'를 되풀이하지 않음 (시뮬레이터 v3, 끝에서 오감)
        if self.arrival.asking():  # 멈춰서 옆을 비추는 중: 그래프는 계속 걷는다고 보므로 '지나침'을 내지 않음 (2026-11-25)
            return []
        d = self.mem.graph.direction(target, clock)
        st["dir_last"] = {k: d.get(k) for k in ("direction", "p_ahead", "p_turned")}
        note["direction"] = d.get("why")
        kind, walk = d["direction"], d.get("walk", [])
        if d.get("passed") and seen_now:
            # 2026-11-24 병합: 그래프는 1초 = 1칸으로 걷는다고 보므로, "잠시 멈춰 주세요"(빠른 확인·확인 중)로 멈춘 동안
            # 목적지를 '지나쳤다'고 본다 (휴대폰 시험 1 66.6초·시험 2 99.6초, 3~4초 뒤 도착). 지금 사진에 목적지 문패가
            # 보이면 지나친 것이 아니므로 말하지 않는다.
            note["direction"] = d.get("why") + " (지금 사진에 목적지 문패가 보여 안 씀)"
            return []
        if st["act"] == "뒤로 돌기" and kind != "뒤쪽" and clock - st["turn_said_t"] >= TURN_REPEAT:
            # 2026-11-24 병합: 뒤로 돌라고 한 뒤 그래프가 더는 '뒤쪽'이 아니면(돌았는지 모름 포함) 행동을 '직진'으로.
            # 행동이 '뒤로 돌기'로 남으면 10초 무안내 문장이 "뒤로 돌아 주세요"를 되풀이해, 이미 돈 사용자를 다시 돌게 함
            # (413 세션 60.4·70.9초: 실제로 돌아서 412 → 413으로 가는 중)
            st["act"] = "직진"
        if kind not in ("앞쪽", "뒤쪽", "근처"):
            return []
        if kind in ("앞쪽", "뒤쪽") and self.turned_since():  # 사진을 찍은 뒤 이미 돌았음: 지금 향한 쪽 기준으로 바꿈
            kind = "앞쪽" if kind == "뒤쪽" else "뒤쪽"
            d = {**d, "passed": False, "why": d["why"] + " (사진 뒤 돌았음)"}
        if kind == "뒤쪽":
            # 뒤쪽이 이미 아는 갈림길로 이어지고, 그 갈림길에서는 지금 가는 쪽을 골랐다면 돌리지 않음 (복도 번호 흐름은
            # 갈림길 너머를 모름. 두 판단이 엇갈려 오가던 것, 시뮬레이터 v3)
            hb = self.fresh().get("heading")
            jb = self.floor.junction_behind(hb)
            if jb is not None:
                c = self.floor.choose(jb, (hb + 180) % 360, "앞쪽")
                if c and c["rel"] == "back":
                    return []
        new_room = walk[-1] if walk and walk[-1] not in st["dir_rooms"] else None
        if kind == st["dir_said"] and not new_room:
            return []
        if kind == "뒤쪽" and clock - st["turn_said_t"] < TURN_REPEAT:
            return []
        st["dir_rooms"] = list(walk)
        out = []
        if kind == "앞쪽":
            out.append((f"{new_room}호를 지났습니다. " if new_room else "") + f"{target}호는 앞쪽입니다. 계속 직진하세요.")
            st["act"], st["trend_said"] = "직진", "toward"
        elif kind == "뒤쪽":
            out.append(f"{target}호를 지나친 것 같습니다. 뒤로 돌아 주세요." if d.get("passed")
                       else f"{target}호는 뒤쪽인 것 같습니다. 뒤로 돌아 주세요.")
            st["act"], st["trend_said"] = "뒤로 돌기", "away"
            self.mem.graph.turned(clock)
            self.arrival.cancel()
            st.update(dir_said="돌기", turn_said_t=clock)
        elif kind == "근처" and st["dir_said"] != "근처":
            out.append(f"{target}호 근처입니다. 문패를 확인하며 천천히 가세요.")
            st["act"] = "목적지 접근"
            self.arrival.near(clock)
        if out:
            self.mem.event(clock, f"방향 판단: {kind} ({d['why']})")
            if kind != "뒤쪽":
                st["dir_said"] = kind
        return out

    def situation(self, obs):
        """무안내 10초 때 말할 지금 상황 한 문장 (규칙 문장, 숫자는 상태에서)."""
        st, target, tf = self.st, self.target, self.tf
        act, floor, stair = st["act"], st["floor"], st["stair"]
        place = cp.PLACE_KO.get(obs.get("place")) if obs.get("place") not in ("unknown", "stairs", None) else None
        seeking = st["wrong_floor"] and floor is not None
        if act == "뒤로 돌기":
            return f"{target}호는 뒤쪽인 것 같습니다. 뒤로 돌아 주세요."
        if act == "엘리베이터로 돌아가기":
            return f"{target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요."
        if act == "계단에서 나가기":
            return f"{tf}층입니다. 계단에서 나가세요."
        if act.startswith("계단으로"):
            sign = stair["sign"]
            est = ""
            if sign:
                est = f"지금 약 {sign[0]}층입니다. " if len(sign) == 1 else f"지금 약 {sign[0]}층과 {sign[-1]}층 사이입니다. "
            return f"계단입니다. {est}{tf}층까지 {'올라' if act == '계단으로 올라가기' else '내려'}가세요."
        if act == "엘리베이터 타기":
            # 상태에 맞는 문장 (2026-11-23 시험 3: 같은 문장 되풀이를 휴대폰이 읽지 않아 88초 무음)
            if st["elev_in"]:
                return f"엘리베이터 안입니다. {tf}층에 도착하면 내리세요."
            if obs.get("place") in ("elev_hall", "in_elev") or (obs.get("elev") or {}).get("cand"):
                return f"엘리베이터를 기다리고 있습니다. 문이 열리면 타고 {tf}층을 누르세요."
            return f"엘리베이터를 타고 {tf}층을 누르세요."
        if act == "목적지 접근":
            return f"{target}호를 확인하고 있습니다. 천천히 다가가세요."
        if act == "문 통과":
            return "앞의 문을 지나 들어가세요."
        if floor is None:  # 층을 모름: 엘리베이터·계단 말 없이 (2026-11-23, 전에는 1층 가정 때 '엘리베이터를 찾고 있습니다')
            st["check_t"] = self.clock
            return CHECK_SAY
        if seeking and not st["elevator_said"]:
            return "엘리베이터를 찾고 있습니다. 천천히 직진하며 주변을 비춰 주세요."
        if floor and floor == tf:
            # 목적지 층: 늘 행동이 들어간 문장 (2026-11-23 시험 2: "4층 복도입니다. 문패를 찾고 있습니다"만 되풀이)
            if act not in ("직진", "왼쪽", "오른쪽"):
                st["act"] = "직진"
            return QUIET_GOAL
        return "계속 앞으로 천천히 가세요."

    def _finish(self, o, obs, rooms, say, note, latency, calls, error, recheck):
        st = self.st
        say[:] = [x.strip() for x in say if x and x.strip()]
        for x in say:
            self.mem.event(self.clock, "안내: " + x)
        live = isinstance(self.clock, float)
        sec = round(self.clock - self.seg_start, 2) if live else self.sec
        rec = {**note, "route_target": self.target, "seq": self.seq, "sec": sec, "t": self.clock,
               "key_frame": Path(o["key"]).name, "key_frames": [Path(k).name for k in o.get("keys", [o["key"]])],
               "obs": obs, "recheck": recheck, "rooms": rooms, "action": st["act"], "say": say,
               "state": json.loads(json.dumps({k: v for k, v in st.items() if k != "floor_rooms"})),
               "claude_calls": len(calls), "cached": o["cached"], "credit": round(sum(c.get("credit") or 0 for c in calls), 5),
               "tokens": {"in": sum(c.get("input_tokens", 0) for c in calls), "out": sum(c.get("output_tokens", 0) for c in calls),
                          "cache_read": sum(c.get("cache_read", 0) for c in calls)},
               "latency_seconds": round(latency, 3), "error": error,
               "events": [e["text"] for e in self.mem.events[getattr(self, "_ev0", len(self.mem.events)):]]}
        print(f"[{self.target}] {self.seq[:2]} {sec:05.1f}s {rec['latency_seconds']:.2f}s {rooms} "
              f"{st['act']} | {' / '.join(say)}" + (f" [오류 {error}]" if error else ""), flush=True)
        self.records.append(rec)
        if not live:
            self.clock += 1
        self.sec += 1
        self.n += 1
        self.seg_n += 1
        return rec


def run_route(runner, root, target, seq_names, out, max_steps=None, start_floor=None):
    nav = Navigator(runner, target, start_floor=start_floor)
    for seq in seq_names:
        nav.new_segment(seq)
        files = sorted((Path(root) / seq).glob("i*.jpg"))
        groups = [files[i:i + DENSE] for i in range(0, len(files), DENSE)]
        for frames in groups[:max_steps]:
            if nav.st["arrived"]:
                break
            nav.step(frames)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in nav.records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out.with_name(out.stem + "_memory.json"), "w", encoding="utf-8") as f:
        json.dump(nav.mem.dump(), f, ensure_ascii=False, indent=1)
    lat = [r["latency_seconds"] for r in nav.records if not r["error"]] or [0]
    print(f"[{target}] {len(nav.records)}초, Claude 새 호출 {sum(r['claude_calls'] for r in nav.records)}번, "
          f"크레딧 {sum(r['credit'] for r in nav.records):.3f}, 시간 중앙값 {statistics.median(lat):.2f}s 최대 {max(lat):.2f}s",
          flush=True)
    return nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data/video/frames_8fps"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/routes"))
    ap.add_argument("--route", action="append", required=True, help="목적지=순서1,순서2 …")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--send-side", type=int, default=None, help="Claude에 보낼 사진 긴 변 (기본: 원본)")
    ap.add_argument("--per-call", type=int, default=1, choices=[1, 2], help="1초 구간에서 보낼 사진 수(한 번의 호출)")
    ap.add_argument("--two-segs", default=None, help="2장 방식을 쓸 구간 앞부분(쉼표). 예: 2a,2b (나머지는 1장)")
    ap.add_argument("--cache", default=str(ROOT / "outputs/cache/claude_v2.jsonl"),
                    help="인식 캐시 (2단계 결과는 outputs/cache/claude.jsonl: say 칸이 있던 프롬프트)")
    ap.add_argument("--name", default="claude")
    ap.add_argument("--prompt-version", type=int, default=6,
                    help="인식 프롬프트 판 (6: 2026-11-23 엘리베이터 안·밖 기준, 5: 2026-11-23 호실 범위 안내판·긴 복도 방향·엘리베이터 안 기준, "
                         "4: 2026-11-23 엘리베이터 안 층 표시창, 3: 2026-11-19 계단 층 표시·엘리베이터 근거 4종, "
                         "2: 이전 캐시 재생)")
    ap.add_argument("--start-floor", default=None, help="사용자가 말한 출발 층 (예: 1). 없으면 층 모름으로 시작")
    ap.add_argument("--send-quality", type=int, default=70,
                    help="Claude에 보내는 JPEG 품질 (70: 2026-11-23부터. 이전 캐시 재생은 90)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", action="store_true", help="캐시만 씀 (Claude를 부르지 않음, 크레딧 0). 규칙만 바꿨을 때")
    ap.add_argument("--max-steps", type=int, default=None, help="구간마다 앞에서부터 이 초만")
    ap.add_argument("--credit-limit", type=float, default=60.0, help="이번 실행 크레딧 한도")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--depth", action="store_true", help="깊이 모델로 문패 벽 쪽·거리 재기 (core/depth.py, torch·transformers 필요)")
    a = ap.parse_args()
    if a.dry_run:
        client = DryRunClient(a.model)
    elif a.replay:
        client = ReplayClient(a.model)
    else:
        client = GatewayClient(a.model, timeout=a.timeout, ledger=Ledger(a.credit_limit))
    before = None if a.dry_run or a.replay else client.credits_used()
    depth_model = None
    if a.depth:
        import depth
        depth_model = depth.load()
        print("깊이 모델: " + ("켬" if depth_model is not None else "불러오지 못해 끔"), flush=True)
    runner = Runner(cp.Perceiver(client, send_side=a.send_side, cache_path=None if a.dry_run else a.cache,
                                 version=a.prompt_version, quality=a.send_quality), per_call=a.per_call,
                    two_segs=a.two_segs.split(",") if a.two_segs else None, depth_model=depth_model)
    try:
        for spec in a.route:
            target, seqs = spec.split("=")
            seq_names = seqs.split(",")
            name = "_".join(s.split("_")[0] for s in seq_names)
            run_route(runner, a.frames, target, seq_names, Path(a.out_dir) / f"guide_{target}_{name}_{a.name}.jsonl",
                      a.max_steps, start_floor=a.start_floor)
    except CreditLimit as e:
        print(f"크레딧 한도로 멈춤: {e}", flush=True)
    if before is not None:
        after = client.credits_used()
        print(f"이번 실행 토큰 계산 {client.ledger.total:.3f}cr ({client.ledger.calls}번), 잔액 차이 "
              f"{(after - before) if after is not None else '?'}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
