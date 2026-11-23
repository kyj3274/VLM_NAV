"""로컬 버전 길안내 기억: 시간 순서로 쌓는 문패 기록, 장소 흐름, 사건 기록 (navigator.Navigator가 situ 모드에서 씀).

문패는 번호마다 근거를 쌓아 믿음 점수를 매긴다 (11/16 시연 126걸음 모의: 3점 기준이면 진짜 10개 중 9개, 오독 1종, 문패 아님 0종).
  읽힌 횟수(최대 3) + 크고 또렷함 1 + 문 옆 1 + 이미 확정한 같은 층 번호와 가까움 1  →  3점 이상이면 확정
사건 기록은 글 한 줄씩이라 나중에 텍스트 모델에 그대로 넣을 수 있다 (text()).

방향(direction)·뒤돌기(turned)는 공통 판단 core/nav_graph.py (Claude 버전과 같은 코드)에 맡긴다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))  # 공통 모듈 (src/nav/core)
from nav_graph import SIDE_SURE, CorridorGraph  # noqa: E402

CONFIRM = 3       # 확정에 필요한 믿음 점수
BIG_SCORE = 0.95  # '크고 또렷함': 점수가 이 이상이고 글자 높이가 big_h 이상
NEAR_DIFF, NEAR_SCORE = 10, 0.8  # '가까움': 확정한 같은 층 번호와 차이 10 이내, 점수 0.8 이상
FORGET = 6        # 이 걸음 동안 안 보이면 읽힌 횟수를 다시 셈
PLACE_MIN = 0.35


class Memory:
    def __init__(self, target, big_h=0.012):
        self.target, self.big_h = target, big_h
        self.seg = 0
        self.plates = {}   # (구간, 번호) → 기록
        self.places = []   # [{"from", "to", "place"}] 같은 장소가 이어진 구간
        self.events = []   # [{"t", "seg", "text"}]
        self.graph = CorridorGraph(self.event)  # 복도 고정 그래프 (구간마다 새로)

    def new_segment(self, t):
        self.seg += 1
        self.graph.reset()
        self.event(t, "엘리베이터에서 내림 (층·문패 기록을 새로 시작)")

    def event(self, t, text):
        self.events.append({"t": t, "seg": self.seg, "text": text})

    def place(self, t, groups):
        """Places365 묶음 확률 {복도: 0.8, …}. 확실한 장소가 바뀌면 사건으로 남김."""
        best = max(groups, key=groups.get) if groups else None
        name = best if best and groups[best] >= PLACE_MIN else None
        if self.places and self.places[-1]["place"] == name:
            self.places[-1]["to"] = t
            return
        self.places.append({"from": t, "to": t, "place": name})
        if name and (len(self.places) < 2 or t - self.places[-2]["to"] <= 1):
            run = [p for p in self.places if p["place"] == name]
            if len(run) == 1:  # 이 장소가 처음 나왔을 때만
                self.event(t, f"장소 인식: {name}")

    def confirmed(self):
        return [p["room"] for (seg, _), p in self.plates.items() if seg == self.seg and p["ok"]]

    def see(self, t, room, score, h, cx, door, side=None, dist=None):
        """문패 번호를 한 번 읽음. 기록을 갱신하고 돌려준다 (rec["ok"]가 확정 여부).
        side: 깊이로 정한 벽 쪽('왼쪽'/'오른쪽'), 없으면 화면 위치로. dist: 깊이로 잰 거리(m)."""
        p = self.plates.setdefault((self.seg, room), {"room": room, "first": t, "last": -99, "n": 0, "total": 0, "big": False,
                                                       "door": False, "near": False, "ok": False, "ok_t": None, "max_h": 0,
                                                       "side": None, "reads": [], "side_votes": [], "dist": None,
                                                       "best": 0, "dmin": None, "near_t": None})
        if t - p["last"] > FORGET:
            p["n"] = 0
        p["n"] += 1
        p["total"] += 1
        p["last"] = t
        p["big"] |= score >= BIG_SCORE and h >= self.big_h
        p["door"] |= bool(door)
        ok = self.confirmed()
        p["near"] |= score >= NEAR_SCORE and "-" not in room and any(
            o[0] == room[0] and "-" not in o and abs(int(o) - int(room)) <= NEAR_DIFF for o in ok if o != room)
        p["best"] = max(p["best"], score)
        vote = None if side == "?" else side or ("왼쪽" if cx < 0.4 else "오른쪽" if cx > 0.6 else "정면")
        if vote:  # side '?' = 깊이는 있지만 이 읽기로는 벽 쪽을 못 정함
            p["side_votes"].append(vote)
        if p["side_votes"]:
            top = max(set(p["side_votes"]), key=p["side_votes"].count)
            p["side"], p["side_sure"] = top, top != "정면" and p["side_votes"].count(top) / len(p["side_votes"]) >= SIDE_SURE
        else:
            p["side"], p["side_sure"] = None, False
        self.graph.see(t, room, score, vote, dist)
        if dist is not None:
            p["dist"] = dist
            if p["dmin"] is None or dist <= p["dmin"]:
                p["dmin"], p["near_t"] = dist, t  # 가장 가까이 지난 때 = 지나간 순서
        if h >= p["max_h"]:
            p["max_h"] = h
        p["reads"] = (p["reads"] + [{"t": t, "score": round(score, 2), "h": round(h, 4), "cx": round(cx, 2), "door": bool(door),
                                     "side": side if side != "?" else None, "dist": dist}])[-8:]
        p["trust"] = min(p["n"], 3) + p["big"] + p["door"] + p["near"]
        if not p["ok"] and p["trust"] >= CONFIRM:
            p["ok"], p["ok_t"] = True, t
            self.graph.set_ok(room)
            why = [f"{p['n']}번 읽음"] + (["크고 또렷함"] if p["big"] else []) + (["문 옆"] if p["door"] else []) + (["본 번호와 가까움"] if p["near"] else [])
            self.event(t, f"{room}호 문패 확정 ({p['side'] or '벽 미정'}, {', '.join(why)})")
        return p

    def turned(self, t):
        """뒤로 돌라고 안내함 (공통 판단 core/nav_graph.py)."""
        self.graph.turned(t)

    def direction(self, target, since=-1):
        """복도 고정 그래프 위의 확률 방향 판단 (공통 판단 core/nav_graph.py). since는 예전 호환용."""
        return self.graph.direction(target)

    def dump(self):
        return {"target": self.target, "segment": self.seg, "events": self.events, "places": self.places,
                "plates": [{"seg": seg, **p} for (seg, _), p in self.plates.items()]}

    def text(self, last=30):
        """텍스트 모델에 넣을 기억 요약."""
        lines = [f"목적지: {self.target}호", "지금까지 일어난 일:"]
        lines += [f"- {e['t']}초: {e['text']}" for e in self.events[-last:]]
        maybe = [p for (seg, _), p in self.plates.items() if seg == self.seg and not p["ok"]]
        if maybe:
            lines.append("아직 확정하지 않은 번호: " + ", ".join(f"{p['room']}({p['total']}번)" for p in maybe[-6:]))
        return "\n".join(lines)
