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
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import claude_perceiver as cp
import direction as dr
import nav_memory
from claude_client import CreditLimit, DryRunClient, GatewayClient, Ledger, ReplayClient, ROOT

DENSE = 8
QUIET_SAY = 10    # 이 초 동안 새 안내가 없으면 한 번 말함
LOCATE_SEC = 5    # 경로 시작: 이 시간 동안 위치(층·장소)를 먼저 파악
LOCATE_ASKS = 2   # 층을 모르면 "주변을 더 보여 주세요"를 몇 번까지
EXIT_IGNORE = 4   # 엘리베이터에서 내린 뒤 이 초 동안은 (뒤에 있는) 엘리베이터를 무시
TREND_M = 10.0    # 진행 방향 판단에 쓰는 문패 거리(m) 상한 (Claude 거리는 대략값이라 넉넉히)
NEIGHBOR = 3      # 진행 방향은 목적지 번호 ±3 안의 문패로만 판단 (목적지 근처 번호는 차례대로 붙어 있다고 봄)
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
EXIT_DUP = 20     # 내림 처리 뒤 이 초 안에 다시 오는 내림(버튼·자동)은 같은 하차로 보고 다시 처리하지 않음.
                  # 자동 감지는 내린 뒤 복도 사진 2장(약 3초) + 응답(2~4초) 뒤라, 그 무렵 사용자가 버튼을 눌러도 겹치지 않게 넉넉히.
SCAN_SEC = 6      # 내린 직후 둘러보기 최대 시간. 그동안은 진행 안내(직진 등)를 하지 않는다
EXIT_SAY = "잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."
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

    def __init__(self, perceiver, per_call=1, two_segs=None):
        self.p, self.per_call = perceiver, per_call
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
        return {"key": keys[-1], "keys": keys, "pick_seconds": round(pick, 3), **r}

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
                   "scan": None, "dir": None, "far": {}}
        self.records, self.clock, self.n = [], 0, 0  # n: 전체 걸음 번호 (연속 판단은 '바로 앞 걸음'으로, 2026-11-18)
        self.seg_n = 0
        self.mem = nav_memory.Memory(target)
        self.mem.event(0, f"안내 시작 (목적지 {target})")
        if start_floor:
            self.st.update(floor=str(start_floor), floor_src="말", wrong_floor=bool(self.tf) and str(start_floor) != self.tf)
            self.mem.event(0, f"출발 층 {start_floor}층 (사용자가 말함)")
        self.si, self.sec, self.seq, self.seg_start = 0, 0, "", 0
        self.quick_max = QUICK_MAX  # 문 앞 빠른 확인 횟수 (웹 서버가 0으로 끌 수 있음)
        self.announce = True  # 첫 걸음에 "○호로 안내를 시작합니다"를 말할지 (웹은 미리 말함)
        self.last_exit = None   # 마지막으로 내림을 처리한 시각 (같은 하차를 두 번 처리하지 않게)
        self.exit_say = False   # 다음 걸음에서 내림 문장을 말할지 (웹 버튼은 응답에서 바로 말하므로 False)
        self.plate_dirs = []    # 내린 뒤 걸으며 확정한 문패와 그때 나침반 방향 (다른 층 같은 끝자리 번호의 약한 근거, B8c)

    def start_head(self):
        """경로 시작 문장 앞부분. 웹은 '410호로 안내하겠습니다'를 이미 말했으므로 announce=False로 앞 문장을 뺀다."""
        t, tf = self.target, self.tf
        head = f"{t}호로 안내를 시작합니다. " if self.announce else ""
        return head + (f"{t}호는 {tf}층입니다. " if tf else "")

    def set_target(self, target):
        """안내 중 목적지 변경 (웹). 목적지와 도착·진행 방향 상태만 새로 하고, 층·확정 문패·엘리베이터 기억은 둔다.
        반환: 바뀐 목적지로 다음에 할 일 문장들."""
        st = self.st
        self.target, self.tf = target, target[0] if ROOM.match(target) else None
        self.mem.target = target
        st.update(arrived=False, arr_run=0, target_seen=None, near_d=[], trend_said=None, maybe_said=False,
                  elevator_said=False, el_approach=False, el_ready=None, quick=None, quick_n=0,
                  el_cand=None, el_passed=False, dir=None, far={})
        # PR #1 리뷰 3: 이전 목적지로 가던 중 한 번의 호출로 확정된 문패가 새 목적지이면 확정을 풀고
        # '서로 다른 호출 2번' 기준으로 다시 검사한다 (FORGET초 넘게 안 보였으면 호출 수는 0으로 봄).
        p = self.mem.get(target)
        if p and p["ok"]:
            calls = p["calls"] if self.clock - p["last"] <= nav_memory.FORGET else 0
            if calls < nav_memory.TARGET_CALLS:
                p.update(ok=False, ok_t=None, wait_calls=True)
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
            st["act"] = "직진"
            return [head + ("목적지 층입니다. " if tf else "") + "앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]
        if self.si > 0:
            st["act"] = "엘리베이터로 돌아가기"
            return [head + f"{target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요."]
        st["act"] = "직진"
        return [head + f"{target}호는 {tf}층에 있습니다. 엘리베이터를 찾아 {tf}층으로 가겠습니다. 앞으로 천천히 직진해 주세요."]

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
            self.last_exit, self.exit_say = self.clock, True
            self.st.update(floor=carry, wrong_floor=bool(carry and self.tf) and carry != self.tf, room=None,
                           floor_rooms=[], elevator_said=False, el_run=0,
                           el_seen=False, el_ready=None, button=None, pending=None, fs_last=None, near_d=[],
                           trend_said=None, target_seen=None, arr_run=0, last_say=self.clock, loc=None,
                           door_phase=False, door_gone=0, arrow_last=None, el_approach=False, maybe_said=False,
                           floor_move=None, quick=None, quick_n=0, el_cand=None, el_last=None, el_miss=0,
                           el_passed=False, stair=new_stair(), floor_temp=src == "엘리베이터 표시창",
                           floor_src=src, ask_t=None, check_t=None, el_disp=None,
                           elev_run=0, elev_in=False, hall_run=0, exit_cand=None, dir=None, far={},
                           scan={"t": self.seg_start, "until": self.seg_start + SCAN_SEC, "reads": [], "evs": [],
                                 "lw": [], "done": False, "why": None})
            if carry and src == "엘리베이터 표시창":
                self.mem.event(self.clock, f"내린 층 임시 {carry}층 (엘리베이터 안 층 표시창)")
            elif carry:
                self.mem.event(self.clock, f"내린 층 {carry}층 (내림을 알기 전 복도 사진의 {src})")

    def exit_sentence(self):
        st = self.st
        disp = f"표시창으로는 {st['floor']}층입니다. " if st["floor"] and st["floor_src"] == "엘리베이터 표시창" else ""
        return f"엘리베이터에서 내렸습니다. {disp}{EXIT_SAY}"

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

    def watch_exit(self, obs, clock):
        """자동 하차 감지 (2026-11-23 사용자 지시 B5): 엘리베이터 안 장면이 EXIT_IN장 이상 이어진 뒤 복도·홀 장면이
        EXIT_OUT장 연속이면 버튼 "내렸어요"와 같은 처리. 반환: 감지했으면 기록용 dict."""
        st = self.st
        if obs["place"] == "in_elev":
            st["elev_run"] += 1
            st["hall_run"], st["exit_cand"] = 0, None
            if st["elev_run"] >= EXIT_IN and not st["elev_in"]:
                st["elev_in"] = True
                self.mem.event(clock, f"엘리베이터 안 장면 {EXIT_IN}장 연속: 탄 것으로 봄")
            if st["elev_in"]:  # 탔으면 층이 바뀔 수 있음: 계단처럼 다른 층 문패·층 표시 한 번이면 층을 바꾼다
                st["floor_move"] = {**(st["floor_move"] or {"t": clock, "why": "엘리베이터 안"}), "last": clock}
            return None
        st["elev_run"] = 0
        if not st["elev_in"] or obs["place"] not in HALL:
            st["hall_run"], st["exit_cand"] = 0, None
            return None
        st["hall_run"] += 1
        if st["hall_run"] == 1:
            st["exit_cand"] = {"t": clock, "floor": st["floor"]}
        if st["hall_run"] < EXIT_OUT:
            return None
        cand = st["exit_cand"]
        st.update(elev_in=False, hall_run=0, exit_cand=None)
        if self.last_exit is not None and clock - self.last_exit < EXIT_DUP:
            self.mem.event(clock, f"자동 하차 감지했지만 {EXIT_DUP}초 안에 이미 내림 처리: 같은 하차로 봄")
            return None
        # 첫 복도 사진에서 이미 다른 층 문패로 층을 바꿨으면 그 층을 이어받는다 (새 구간이 지우지 않게)
        keep = (st["floor"], st["floor_src"]) if st["floor"] and st["floor"] != cand["floor"] \
            and st["floor_src"] != "엘리베이터 표시창" else None
        seq = f"live_{self.si + 1}" if str(self.seq).startswith("live_") else f"{self.seq}+내림"  # 웹 기록 화면은 live_번호
        self.new_segment(seq, start=cand["t"], keep=keep)
        self.mem.event(clock, f"엘리베이터에서 내림 (자동 감지: 엘리베이터 안 {EXIT_IN}장 → 복도·홀 {EXIT_OUT}장)")
        return {"t": cand["t"], "kept_floor": keep[0] if keep else None}

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
        st["loc"], st["act"], st["check_t"] = None, "멈추고 둘러보기", self.clock
        return [(f"여기는 {place}인 것 같습니다. " if place else "") + CHECK_SAY]

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

    def step(self, frames=None, t=None, observed=None, quick=False, rechecked=None, heading=None):
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
            auto = self.watch_exit(obs, clock)
            if auto:
                note["auto_exit"] = auto
        si = self.si
        first = self.seg_n == 0  # 구간(경로 시작·엘리베이터 하차)의 첫 걸음
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
        for p in plates:
            rec = self.mem.see(clock, p["n"], p["side"], p["m"], p["on"], rflag.get(p["n"]))
            if rec["ok"] and rflag.get(p["n"]) is not False:
                rooms.append(p["n"])
                src[p["n"]] = p
        # 빠른 확인 사진이 목적지를 읽음 (2026-11-23 B4): 서로 다른 사진 2장(호출 2번)이면 믿음 점수와 상관없이 확정
        if quick and st["quick"] and target not in rooms:
            tp = next((p for p in plates if p["n"] == target), None)
            rec = self.mem.get(target)
            if tp and rec and not rec["ok"] and rec["calls"] >= nav_memory.TARGET_CALLS:
                rec.update(ok=True, ok_t=clock, wait_calls=False)
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

        # ---- 층 충돌: 한 번 정한 층과 첫 자리가 다른 번호는 2초 연속일 때만 인정
        floor_change = None
        other = [x for x in rooms if st["floor"] and ROOM.match(x) and x[0] != st["floor"]]
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
        elif fs and fs != st["floor"] and not floor_change:
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
            floor, basis = floor_change, ("층 이동 뒤 한 번" if one_shot else "층 충돌 2번 연속")
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
                if floor == tf or not tf:
                    say.append(head + "목적지 층입니다. 앞으로 천천히 직진해 주세요. 문패를 확인하며 안내하겠습니다.")
                    st["act"] = "직진"
                else:
                    st["wrong_floor"] = True
                    say.append(head + f"{target}호는 {tf}층에 있습니다. 엘리베이터로 {tf}층에 가야 합니다. "
                               "앞으로 천천히 직진하며 엘리베이터를 찾겠습니다.")
                    st["act"] = "직진"
            elif st["floor"] is None:
                st["floor"] = floor
                if floor == tf or not tf:
                    say.append(f"{floor}층이 맞습니다. {where}"
                               + ("엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다." if el_busy else ""))
                    st.update(el_ready=None, elevator_said=False, el_approach=False)
                    if st["act"] not in ("목적지 접근", "도착", "문 통과"):
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
            elif floor != st["floor"]:
                say.append(f"층 판단이 바뀌었습니다: {st['floor']}층 → {floor}층.")
                st["floor"], st["wrong_floor"] = floor, bool(tf) and floor != tf
                if not st["wrong_floor"] and el_busy:
                    say.append("목적지 층입니다. 엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다.")
                    st.update(el_ready=None, elevator_said=False, el_approach=False)
                elif not st["wrong_floor"] and st["act"].startswith("계단"):
                    say.append("목적지 층입니다. 복도를 따라 문패를 찾겠습니다.")
                if st["act"] not in ("목적지 접근", "도착") or st["wrong_floor"]:
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

        # ---- 진행 방향: 확정한 가까운(TREND_M 이내) 문패 번호가 목적지 번호에 가까워지는가
        # 내린 직후 둘러보는 동안(B6)은 말하지 않고 방향 근거(보통)로만 모은다. 나침반이 있으면 몸을 돌리는 중이라 쓰지 않음.
        if ROOM.match(target) and not st["wrong_floor"] and st["floor"] == tf:
            # 목적지 문패 자신도 위치 표시(거리 0)로 넣는다: 목적지를 지나친 뒤 옆 번호가 보이면 '멀어짐'
            # (2026-11-17 410 경로: 410을 본 뒤 407→412에서 '방향이 맞습니다'라고 한 것을 고침)
            near = [x for x in floor_rooms if x in src and src[x]["m"] is not None and src[x]["m"] <= TREND_M
                    and abs(int(x) - int(target)) <= NEIGHBOR]
            if near and (not st["near_d"] or st["near_d"][-1][0] != near[0]):
                n0 = near[0]
                d, side = abs(int(n0) - int(target)), src[n0]["side"]
                prev = st["near_d"][-1] if st["near_d"] else None
                st["near_d"].append((n0, d, side, clock))
                if n0 == target or prev is None:
                    pass  # 목적지 자신은 목적지 안내가 따로 말함
                elif scanning:
                    if heading is None and d != prev[1]:
                        rel = "F" if d < prev[1] else "B"
                        st["scan"]["evs"].append(dr.evidence(rel, None, dr.MEDIUM, f"문패 번호 {prev[0]}→{n0}", clock))
                        st["trend_said"] = "toward" if d < prev[1] else "away"
                        note["trend"] = f"{prev[0]}→{n0} " + ("가까워짐" if d < prev[1] else "멀어짐") + " (둘러보는 중, 방향 근거로)"
                elif d < prev[1] and st["target_seen"] and st["trend_said"] == "away":
                    note["trend"] = f"{prev[0]}→{n0} 가까워짐이지만 목적지를 이미 지나쳐 말하지 않음"
                elif d < prev[1]:
                    say.append(f"{n0}호를 지나고 있습니다. 목적지 방향이 맞습니다. 계속 직진하세요.")
                    st["trend_said"], st["act"] = "toward", "직진"
                    note["trend"] = f"{prev[0]}→{n0} 가까워짐"
                elif d > prev[1] and st["trend_said"] != "away":
                    say.append(f"{n0}호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요.")
                    st["trend_said"], st["act"] = "away", "뒤로 돌기"
                    note["trend"] = f"{prev[0]}→{n0} 멀어짐"
            # 먼 문패(목적지 ±3 밖) (B8d): 같은 쪽 벽 문패가 2번 연속 목적지에서 멀어지면 뒤로 (번호가 띄엄띄엄이라 한 번으로는 안 함)
            if not scanning and not st["target_seen"]:
                for x in floor_rooms:
                    if x not in src or abs(int(x) - int(target)) <= NEIGHBOR or src[x]["side"] not in ("L", "R"):
                        continue
                    lst = st["far"].setdefault(src[x]["side"], [])
                    if lst and lst[-1][0] == x:
                        continue
                    d = abs(int(x) - int(target))
                    run = (lst[-1][2] + 1 if d > lst[-1][1] else 0) if lst else 0
                    lst.append([x, d, run])
                    if run >= 2 and st["trend_said"] != "away":
                        say.append(f"{x}호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요.")
                        st["trend_said"], st["act"], st["far"] = "away", "뒤로 돌기", {}
                        note["trend"] = "같은 쪽 벽 먼 문패 " + "→".join(y[0] for y in lst[-3:]) + " 멀어짐"
                        break

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
                    pend_m = tr["reads"][-1]["m"]
        elif not ROOM.match(target) and obs.get("dest", {}).get("seen"):
            st["dest_calls"] = st.get("dest_calls", 0) + 1 if clock - st.get("dest_last", -99) <= nav_memory.FORGET else 1
            st["dest_last"] = clock
            if st["dest_calls"] >= nav_memory.TARGET_CALLS:
                tgt = obs["dest"]
            else:
                maybe, pend_m = True, obs["dest"].get("m")
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
            side, m = tgt["side"], tgt["m"]
            close = m is not None and m <= ARRIVE_M  # 정면이 아니어도 (사진 속 L/C/R은 모두 9~3시 안)
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
                # 도착하는 걸음에는 도착 문장만 (같은 걸음의 층 확인·'문패를 찾겠습니다'와 섞이지 않게, 2026-11-19)
                say[:] = [f"{where} {target}{'호' if ROOM.match(target) else ''} 문이 있습니다."]
                st["arrived"], st["act"] = True, "도착"
            elif st["arr_run"] == 1:
                say.append("멈춰 주세요, 확인 중입니다.")
                st["act"] = "목적지 접근"
            elif first or side != prev.get("side") or (m and prev.get("m") and prev["m"] - m >= 2):
                say.append(f"{target}{'호' if ROOM.match(target) else ''} 문패가 {cp.CLOCK[side]}시 방향"
                           + (f" 약 {max(1, round(m))}미터" if m else "") + "에 있습니다.")
                st["act"] = "목적지 접근"
        elif not maybe:
            st["arr_run"] = 0
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
        # ---- 위치 파악 시간 끝
        if st["loc"] is not None and clock + 1 >= st["loc"]["until"] and not st["arrived"]:
            say.extend(self.locate_end(st["loc"]))
        # ---- 엘리베이터 앞에서 층을 모름: 층을 알게 되면 대기 표시를 지운다 (층 모른 채 타라고 하지 않음, 2026-11-23)
        if st["el_ready"] is not None and st["floor"] is not None:
            st["el_ready"] = None
        # ---- 층 묻기 (2026-11-23 사용자 지시 3): 구간 시작 뒤 ASK_FLOOR_SEC초가 지나도 층을 모르면, ASK_FLOOR_EVERY초에 한 번
        if (st["floor"] is None and not st["arrived"] and st["loc"] is None and clock - self.seg_start >= ASK_FLOOR_SEC
                and (st["ask_t"] is None or clock - st["ask_t"] >= ASK_FLOOR_EVERY)):
            say.append(ASK_SAY)
            st["ask_t"] = clock
            note["ask_floor"] = True
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
        if not ROOM.match(target) or st["arrived"] or self.si == 0 and not scanning:
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

    def situation(self, obs):
        """무안내 10초 때 말할 지금 상황 한 문장 (규칙 문장, 숫자는 상태에서)."""
        st, target, tf = self.st, self.target, self.tf
        act, floor, stair = st["act"], st["floor"], st["stair"]
        place = cp.PLACE_KO.get(obs.get("place")) if obs.get("place") not in ("unknown", "stairs", None) else None
        seeking = st["wrong_floor"] and floor is not None
        if act == "뒤로 돌기":
            return "목적지와 멀어지고 있습니다. 뒤로 돌아가세요."
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
            return f"{floor}층 {place or ''}입니다. 문패를 찾고 있습니다.".replace("층 입니다", "층입니다")
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
    ap.add_argument("--prompt-version", type=int, default=5,
                    help="인식 프롬프트 판 (5: 2026-11-23 호실 범위 안내판·긴 복도 방향·엘리베이터 안 기준, "
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
    a = ap.parse_args()
    if a.dry_run:
        client = DryRunClient(a.model)
    elif a.replay:
        client = ReplayClient(a.model)
    else:
        client = GatewayClient(a.model, timeout=a.timeout, ledger=Ledger(a.credit_limit))
    before = None if a.dry_run or a.replay else client.credits_used()
    runner = Runner(cp.Perceiver(client, send_side=a.send_side, cache_path=None if a.dry_run else a.cache,
                                 version=a.prompt_version, quality=a.send_quality), per_call=a.per_call,
                    two_segs=a.two_segs.split(",") if a.two_segs else None)
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
