"""영상 길안내: 규칙 추적기 + Claude 인식 (VLM_NAV navigator.py의 규칙 흐름을 그대로 두고 입력만 Claude로 바꿈).

목적지(예: 314호)를 받아, 1초 구간마다 그 초의 안내 문장과 다음 행동을 만든다.
  인식  1초 구간에서 가장 선명한 장 1장을 Claude에 보냄(claude_perceiver.see): 문패, 층 표시, 장소, 엘리베이터(근거),
        화살표, 앞의 출입문. 중요한 문패 1개는 원본에서 잘라 확대해 다시 읽음(recheck).
  판단  규칙. 문패 확정(nav_memory 믿음 점수 3), 층 충돌은 2초 연속, 번호 증감으로 진행 방향, 엘리베이터는 근거가
        2초 연속, 엘리베이터 대기 6초, 하차 직후 4초는 뒤의 엘리베이터 무시, 10초 무안내면 한 번 말함, 시작 위치 파악 5초.
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
import nav_memory
from claude_client import CreditLimit, DryRunClient, GatewayClient, Ledger, ReplayClient, ROOT

DENSE = 8
QUIET_SAY = 10    # 이 초 동안 새 안내가 없으면 한 번 말함
LOCATE_SEC = 5    # 경로 시작: 이 시간 동안 위치(층·장소)를 먼저 파악
LOCATE_ASKS = 2   # 층을 모르면 "주변을 더 보여 주세요"를 몇 번까지
EL_WAIT = 6       # 엘리베이터 앞에서 층 단서를 기다리는 초
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
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착"]
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

    def __init__(self, runner, target):
        self.runner, self.target = runner, target
        self.tf = target[0] if ROOM.match(target) else None  # 목적지 층 (3자리 호실의 첫 자리)
        self.st = {"floor": None, "wrong_floor": False, "arrived": False, "room": None, "floor_rooms": [],
                   "elevator_said": False, "el_run": 0, "el_seen": False, "el_ready": None, "button": None,
                   "pending": None, "fs_last": None, "near_d": [], "trend_said": None, "target_seen": None,
                   "arr_run": 0, "last_say": 0, "loc": None, "assume1": False, "door_phase": False, "door_gone": 0,
                   "arrow_last": None, "el_approach": False, "act": "멈추고 둘러보기", "maybe_said": False}
        self.records, self.clock, self.n = [], 0, 0  # n: 전체 걸음 번호 (연속 판단은 '바로 앞 걸음'으로, 2026-11-18)
        self.seg_n = 0
        self.mem = nav_memory.Memory(target)
        self.mem.event(0, f"안내 시작 (목적지 {target})")
        self.si, self.sec, self.seq, self.seg_start = 0, 0, "", 0
        self.announce = True  # 첫 걸음에 "○호로 안내를 시작합니다"를 말할지 (웹은 미리 말함)

    def start_head(self):
        """경로 시작 문장 앞부분. 웹은 '410호로 안내하겠습니다'를 이미 말했으므로 announce=False로 앞 문장을 뺀다."""
        t, tf = self.target, self.tf
        head = f"{t}호로 안내를 시작합니다. " if self.announce else ""
        return head + (f"{t}호는 {tf}층입니다. " if tf else "")

    def set_target(self, target):
        """안내 중 목적지 변경 (웹). 목적지와 도착·진행 방향 상태만 새로 하고, 층·확정 문패·엘리베이터 기억은 둔다."""
        st = self.st
        self.target, self.tf = target, target[0] if ROOM.match(target) else None
        self.mem.target = target
        st.update(arrived=False, arr_run=0, target_seen=None, near_d=[], trend_said=None, maybe_said=False,
                  elevator_said=False, el_approach=False, el_ready=None)
        if st["floor"] is not None:
            st["wrong_floor"] = bool(self.tf) and st["floor"] != self.tf
            st["act"] = ("엘리베이터로 돌아가기" if self.si > 0 else "직진") if st["wrong_floor"] else "직진"
        self.mem.event(self.clock, f"목적지 변경: {target}")

    # ------------------------------------------------------------ 구간
    def new_segment(self, seq=""):
        """구간 시작. 첫 구간이 아니면 엘리베이터를 타고 내린 것으로 본다 (층·위치는 모름으로)."""
        if self.records or self.seg_n:
            self.si += 1
            self.mem.new_segment(self.clock)
        self.seq, self.sec, self.seg_start, self.seg_n = seq, 0, self.clock, 0
        if self.si > 0:
            self.st.update(floor=None, wrong_floor=False, room=None, floor_rooms=[], elevator_said=False, el_run=0,
                           el_seen=False, el_ready=None, button=None, pending=None, fs_last=None, near_d=[],
                           trend_said=None, target_seen=None, arr_run=0, last_say=self.clock, loc=None, assume1=False,
                           door_phase=False, door_gone=0, arrow_last=None, el_approach=False, maybe_said=False)

    # ------------------------------------------------------------ 재확인 대상
    def recheck_pick(self, obs):
        """확대 재확인할 문패 1개 (없으면 None). 이미 확정한 번호는 다시 보지 않는다."""
        st, target, tf = self.st, self.target, self.tf
        done = set(self.mem.confirmed())
        plates = [p for p in obs["plates"] if p["n"] not in done and p.get("box")]
        if ROOM.match(target):
            near = [p for p in plates if ROOM.match(p["n"]) and p["n"][0] == target[0]
                    and abs(int(p["n"]) - int(target)) <= 2]
            if near:
                return near[0], "목적지 근처 번호"
        unread = [p for p in obs.get("unread", []) if p.get("box")]
        if unread and (st["floor"] is None or st["floor"] == tf):
            return unread[0], "번호 안 읽힌 문패"
        if st["floor"] is None:
            first = [p for p in plates if ROOM.match(p["n"])]
            if first:
                return first[0], "층을 정할 번호"
        elif plates:
            other = [p for p in plates if ROOM.match(p["n"]) and p["n"][0] != st["floor"]]
            if other:
                return other[0], "층 충돌 번호"
        return None, None

    # ------------------------------------------------------------ 위치 파악 끝
    def locate_end(self, loc):
        st, tf = self.st, self.tf
        places = Counter(p for p in loc["places"][-LOCATE_SEC:] if p and p != "unknown")
        place = cp.PLACE_KO.get(places.most_common(1)[0][0]) if places else None
        if self.si == 0:  # 경로 시작은 1층으로 가정하고 바로 다음 행동 (문패를 읽으면 문패가 우선)
            st["loc"], st["assume1"] = None, True
            head = (f"여기는 {place}인 것 같습니다. " if place else "") + "1층으로 보고 안내하겠습니다. "
            if tf == "1":
                st["act"] = "직진"
                return [head + "목적지 층입니다. 앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]
            if st["el_seen"]:
                st["elevator_said"], st["act"] = True, "엘리베이터 타기"
                return [head + f"앞에 엘리베이터가 있습니다. 엘리베이터를 타고 {tf}층을 누르세요."]
            st["act"] = "직진"
            return [head + f"{self.target}호는 {tf}층이니 엘리베이터를 찾겠습니다. 앞으로 천천히 직진해 주세요."]
        st["loc"] = None
        st["act"] = "직진"
        return ["앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]

    # ------------------------------------------------------------ 한 걸음
    def step(self, frames=None, t=None, observed=None):
        """frames: 1초 구간의 사진 파일 경로들 (영상 실행). t: 사진 찍은 시각(초, 겹쳐 보내기·웹),
        observed: 이미 받은 인식 결과(겹쳐 보내기에서 다른 스레드가 부른 것). 반환: 이 걸음의 기록."""
        st, target, tf = self.st, self.target, self.tf
        if t is not None:
            self.clock = t
        si, clock = self.si, self.clock
        first = self.seg_n == 0  # 구간(경로 시작·엘리베이터 하차)의 첫 걸음
        say, note = [], {}
        o = observed or self.runner.observe(frames)
        calls = [] if o["cached"] else [o["call"]]
        latency = o["pick_seconds"] + (o["call"].get("seconds") or 0)
        note["credit_all"] = o["call"].get("credit") or 0.0  # 캐시에서 온 호출도 포함한 이 초의 크레딧
        obs = o["obs"]
        if obs is None:  # 호출 실패: 시간 초과면 멈춤 안내, 상태는 그대로
            err = o["call"].get("error")
            if o["call"].get("timed_out"):
                say.append("잠시 멈춰 주세요.")
            return self._finish(o, None, [], say, note, latency, calls, error=err, recheck=None)

        self.mem.place(clock, cp.PLACE_KO.get(obs["place"]))
        loc = st["loc"]
        if first and si == 0:
            if obs["door_ahead"] != "none":  # 출입문 앞에서 시작: 문을 지나 들어가는 것이 먼저 (건물 입구 = 1층으로 가정)
                kind = "유리문" if obs["door_ahead"] == "glass" else "문"
                say.append(self.start_head() + f"앞에 {kind}이 있습니다. 문을 지나 들어가세요.")
                st.update(door_phase=True, assume1=True, act="문 통과")
            else:
                st["loc"] = loc = {"until": clock + LOCATE_SEC, "asks": 0, "places": []}
                say.append(self.start_head() + "지금 위치를 확인하겠습니다. 휴대폰을 앞으로 들고 천천히 좌우로 비춰 주세요.")
                st["act"] = "멈추고 둘러보기"
        if loc is not None:
            loc["places"].append(obs["place"])
        if first and si > 0:
            say.append("엘리베이터에서 내렸습니다. 문패를 확인하며 천천히 이동하세요.")
            st["act"] = "직진"

        # ---- 출입문을 지났는가
        if st["door_phase"] and not (first and si == 0):
            st["door_gone"] = st["door_gone"] + 1 if obs["door_ahead"] == "none" else 0
            if st["door_gone"] >= DOOR_GONE:
                st["door_phase"] = False
                if tf and tf != "1" and st["floor"] is None:
                    say.append(f"문을 지났습니다. {target}호는 {tf}층이니 엘리베이터를 찾겠습니다. 앞으로 천천히 직진해 주세요.")
                else:
                    say.append("문을 지났습니다. 앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다.")
                st["act"] = "직진"

        # ---- 확대 재확인 (최대 1개)
        plates = [dict(p) for p in obs["plates"]]
        rflag, recheck = {}, None
        cand, why = self.recheck_pick(obs)
        if cand:
            rc = self.runner.recheck(cand.get("frame") or o["key"], cand)
            if rc:
                if not rc["cached"]:
                    calls.append(rc["call"])
                latency += rc["call"].get("seconds") or 0
                note["credit_all"] += rc["call"].get("credit") or 0.0
                recheck = {"why": why, "asked": cand["n"] or cand["name"], "read": rc["n"], "match": rc["match"]}
                if cand["n"]:
                    rflag[cand["n"]] = rc["match"]
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
        rooms = [x for x in rooms if ROOM.match(x)] + [x for x in rooms if not ROOM.match(x)]

        # ---- 엘리베이터 (근거가 2초 연속)
        el = obs["elev"]
        st["el_run"] = st["el_run"] + 1 if el["seen"] else 0
        if el["btn"] in ("L", "R"):
            st["button"] = {"L": "왼쪽", "R": "오른쪽"}[el["btn"]]
        ignore_el = si > 0 and clock - self.seg_start < EXIT_IGNORE
        if st["el_run"] >= 2 and not ignore_el:
            if not st["el_seen"]:
                self.mem.event(clock, "엘리베이터 확인 (" + ("호출 버튼" if el["why"] == "button" else "층 표시창") + ")")
            st["el_seen"] = True
        if st["el_run"] >= 2 and loc is None and not st["elevator_said"] and not ignore_el and tf:
            btn = f"호출 버튼은 엘리베이터 문 {st['button']} 벽에 있습니다. " if st["button"] else ""
            where = (f"엘리베이터가 {cp.CLOCK[el['side']]}시 방향" + (f" 약 {max(1, round(el['m']))}미터" if el["m"] else "")
                     + "에 있습니다. ") if el["side"] in cp.CLOCK else "앞에 엘리베이터가 있습니다. "
            near_el = el["m"] is None and el["side"] == "C" or el["m"] is not None and (
                el["m"] <= 2.0 or el["side"] == "C" and el["m"] <= EL_NEAR_M)
            seeking = st["wrong_floor"] or (st["floor"] is None and st["assume1"] and tf != "1")
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
                say.append("앞에 엘리베이터가 있습니다. 지금 몇 층인지 먼저 확인하겠습니다. 주변을 천천히 비춰 주세요.")
                st["el_ready"], st["act"] = clock, "멈추고 둘러보기"

        # ---- 층 충돌: 한 번 정한 층과 첫 자리가 다른 번호는 2초 연속일 때만 인정
        floor_change = None
        other = [x for x in rooms if st["floor"] and ROOM.match(x) and x[0] != st["floor"]]
        if other:
            digit = other[0][0]
            if st["pending"] and st["pending"][0] == digit and st["pending"][1] == self.n - 1:
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
            if st["pending"] and st["pending"][0] == fs and st["pending"][1] == self.n - 1:
                floor_change = fs
                self.mem.event(clock, f"층 표지 충돌 2번 연속: {st['floor']}층 → {fs}층")
            else:
                st["pending"] = (fs, self.n)

        # ---- 층 판단과 안내
        floor_rooms = [x for x in rooms if ROOM.match(x)]
        floor, basis = (floor_rooms[0][0], f"{floor_rooms[0]}호") if floor_rooms else (fs_ok, f"층 표지 '{obs['floor_sign']}'") \
            if fs_ok else (None, None)
        if floor_change:
            floor, basis = floor_change, "층 충돌 2번 연속"
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
                st["floor"], assumed = floor, st["assume1"]
                st["assume1"] = False
                if assumed and floor != "1":
                    say.append(f"{basis}를 보니 1층이 아니라 {floor}층입니다.")
                if floor == tf or not tf:
                    say.append(f"{floor}층이 맞습니다. {where}"
                               + ("엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다." if st["el_ready"] is not None else ""))
                    st["el_ready"] = None
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
                st["act"] = ("엘리베이터로 돌아가기" if si > 0 else "직진") if st["wrong_floor"] else "직진"
            if floor_rooms:
                st["room"] = floor_rooms[0]
            for x in new:
                st["floor_rooms"].append((clock, x))

        # ---- 진행 방향: 확정한 가까운(TREND_M 이내) 문패 번호가 목적지 번호에 가까워지는가
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

        # ---- 목적지
        # 목적지 안전장치 (2026-11-18 사용자 지시): 목적지 번호·이름은 서로 다른 호출 2번에서 읽혀야 확정.
        # 그 전에는 '확인 중'까지만 말한다 (3단계 겹쳐 보내기에서 415 오독이 한 번의 호출로 확정된 것을 보고 추가).
        tgt, maybe = None, False
        if ROOM.match(target) and not st["wrong_floor"]:
            if target in rooms:
                tgt = src[target]
            else:
                tr = self.mem.get(target)
                maybe = bool(tr and tr["last"] == clock and not tr["ok"] and rflag.get(target) is not False
                             and (st["floor"] in (None, tf)))
        elif not ROOM.match(target) and obs.get("dest", {}).get("seen"):
            st["dest_calls"] = st.get("dest_calls", 0) + 1 if clock - st.get("dest_last", -99) <= nav_memory.FORGET else 1
            st["dest_last"] = clock
            if st["dest_calls"] >= nav_memory.TARGET_CALLS:
                tgt = obs["dest"]
            else:
                maybe = True
        if maybe and not st["arrived"]:
            note["target_pending"] = True
            if not st.get("maybe_said"):
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
            if st["arr_run"] >= 2:
                where = {"C": "정면에", "L": "왼쪽 앞에", "R": "오른쪽 앞에"}[side]
                say.append(f"{where} {target}{'호' if ROOM.match(target) else ''} 문이 있습니다.")
                st["arrived"], st["act"] = True, "도착"
            elif st["arr_run"] == 1:
                say.append("멈춰 주세요, 확인 중입니다.")
                st["act"] = "목적지 접근"
            elif first or side != prev.get("side") or (m and prev.get("m") and prev["m"] - m >= 2):
                say.append(f"{target}{'호' if ROOM.match(target) else ''} 문패가 {cp.CLOCK[side]}시 방향"
                           + (f" 약 {max(1, round(m))}미터" if m else "") + "에 있습니다.")
                st["act"] = "목적지 접근"
        else:
            st["arr_run"] = 0

        # ---- 화살표 기록 (2초 연속 같은 방향일 때 사건으로)
        arrows = [a.get("to") for a in obs["arrows"]]
        if arrows:
            if st["arrow_last"] and st["arrow_last"][0] == arrows[0] and st["arrow_last"][1] == self.n - 1:
                lab = obs["arrows"][0].get("label") or ""
                self.mem.event(clock, f"화살표 표지 {arrows[0]}" + (f" ({lab})" if lab else ""))
            st["arrow_last"] = (arrows[0], self.n)

        # ---- 위치 파악 시간 끝
        if st["loc"] is not None and clock + 1 >= st["loc"]["until"] and not st["arrived"]:
            say.extend(self.locate_end(st["loc"]))
        # ---- 엘리베이터 앞 대기
        if not st["arrived"] and st["el_ready"] is not None and not st["elevator_said"]:
            if st["floor"] is not None:
                st["el_ready"] = None
            elif clock - st["el_ready"] >= EL_WAIT:
                say.append("앞에 엘리베이터가 있습니다. 지금 층을 알 수 있는 표시는 보이지 않습니다. "
                           + (f"호출 버튼은 엘리베이터 문 {st['button']} 벽에 있습니다. " if st["button"] else "")
                           + f"엘리베이터를 타고 {tf}층을 누르세요.")
                st["elevator_said"], st["act"] = True, "엘리베이터 타기"
        # ---- 무안내 시간: 지금 행동을 되풀이하는 규칙 문장 (Claude say는 2026-11-18 사용자 지시로 스키마에서 뺌)
        if say:
            st["last_say"] = clock
        elif st["loc"] is None and not st["arrived"] and clock - st["last_say"] >= QUIET_SAY and st["el_ready"] is None:
            seeking = st["wrong_floor"] or (st["floor"] is None and st["assume1"] and tf not in (None, "1"))
            if st["act"] == "뒤로 돌기":
                say.append("목적지와 멀어지고 있습니다. 뒤로 돌아가세요.")
            elif st["act"] == "엘리베이터로 돌아가기":
                say.append(f"{target}호는 {tf}층에 있습니다. 엘리베이터로 돌아가 {tf}층을 누르세요.")
            elif st["act"] == "직진" and seeking and not st["elevator_said"]:
                say.append("엘리베이터를 찾고 있습니다. 천천히 직진하며 주변을 비춰 주세요.")
            elif st["act"] == "직진":
                say.append("계속 앞으로 천천히 가세요.")
            st["last_say"] = clock
        return self._finish(o, obs, rooms, say, note, latency, calls, error=None, recheck=recheck)

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
               "latency_seconds": round(latency, 3), "error": error}
        print(f"[{self.target}] {self.seq[:2]} {sec:05.1f}s {rec['latency_seconds']:.2f}s {rooms} "
              f"{st['act']} | {' / '.join(say)}" + (f" [오류 {error}]" if error else ""), flush=True)
        self.records.append(rec)
        if not live:
            self.clock += 1
        self.sec += 1
        self.n += 1
        self.seg_n += 1
        return rec


def run_route(runner, root, target, seq_names, out, max_steps=None):
    nav = Navigator(runner, target)
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
    runner = Runner(cp.Perceiver(client, send_side=a.send_side, cache_path=None if a.dry_run else a.cache), per_call=a.per_call,
                    two_segs=a.two_segs.split(",") if a.two_segs else None)
    try:
        for spec in a.route:
            target, seqs = spec.split("=")
            seq_names = seqs.split(",")
            name = "_".join(s.split("_")[0] for s in seq_names)
            run_route(runner, a.frames, target, seq_names, Path(a.out_dir) / f"guide_{target}_{name}_{a.name}.jsonl",
                      a.max_steps)
    except CreditLimit as e:
        print(f"크레딧 한도로 멈춤: {e}", flush=True)
    if before is not None:
        after = client.credits_used()
        print(f"이번 실행 토큰 계산 {client.ledger.total:.3f}cr ({client.ledger.calls}번), 잔액 차이 "
              f"{(after - before) if after is not None else '?'}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
