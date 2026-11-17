"""길안내 한 번의 기억: 시간 순서로 쌓는 문패 기록, 장소 흐름, 사건 기록 (navigator.Navigator가 situ 모드에서 씀).

문패는 번호마다 근거를 쌓아 믿음 점수를 매긴다 (11/16 시연 126걸음 모의: 3점 기준이면 진짜 10개 중 9개, 오독 1종, 문패 아님 0종).
  읽힌 횟수(최대 3) + 크고 또렷함 1 + 문 옆 1 + 이미 확정한 같은 층 번호와 가까움 1  →  3점 이상이면 확정
사건 기록은 글 한 줄씩이라 나중에 텍스트 모델에 그대로 넣을 수 있다 (text()).

방향(direction): 확실한 문패만 쓰는 장면 그래프 방식 (src/eval/scene_graph.py와 같은 판단).
  벽 쪽이 일관된(2/3 이상) 확정 문패를 벽별로 지나간 순서대로 놓고, 목적지와 번호가 가장 가까운 문패의 벽에서
  "앞으로 갈수록 번호가 커지나/작아지나"를 정해 목적지가 앞인지 뒤인지 판단한다. 목적지와 번호 차이가 1이면 '근처'.
  11/17 영상 판단 지점(하차 후 13개): 12개 맞음 (예전 규칙 11개). 같은 영상으로 만든 규칙이라 다른 건물 확인 필요.
"""

CONFIRM = 3       # 확정에 필요한 믿음 점수
BIG_SCORE = 0.95  # '크고 또렷함': 점수가 이 이상이고 글자 높이가 big_h 이상
NEAR_DIFF, NEAR_SCORE = 10, 0.8  # '가까움': 확정한 같은 층 번호와 차이 10 이내, 점수 0.8 이상
FORGET = 6        # 이 걸음 동안 안 보이면 읽힌 횟수를 다시 셈
SIDE_SURE = 2 / 3  # 벽 쪽 표가 이 비율 이상 같아야 그래프(방향 판단)에 넣음
ANCHOR_DIST = 7.0  # 깊이가 있을 때, 이 거리 안까지 가까이 지나간 문패만 그래프 기준점으로 (멀리서 읽힌 번호는 순서가 틀리기 쉬움)
FLOW_SCORE = 0.95  # 한 번만 읽혔어도 이 점수 이상이면 번호 흐름 계산에는 씀
PLACE_MIN = 0.35


class Memory:
    def __init__(self, target, big_h=0.012):
        self.target, self.big_h = target, big_h
        self.seg = 0
        self.plates = {}   # (구간, 번호) → 기록
        self.places = []   # [{"from", "to", "place"}] 같은 장소가 이어진 구간
        self.events = []   # [{"t", "seg", "text"}]

    def new_segment(self, t):
        self.seg += 1
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
        if side != "?":  # '?' = 깊이는 있지만 이 읽기로는 벽 쪽을 못 정함
            p["side_votes"].append(side or ("왼쪽" if cx < 0.4 else "오른쪽" if cx > 0.6 else "정면"))
        if p["side_votes"]:
            top = max(set(p["side_votes"]), key=p["side_votes"].count)
            p["side"], p["side_sure"] = top, top != "정면" and p["side_votes"].count(top) / len(p["side_votes"]) >= SIDE_SURE
        else:
            p["side"], p["side_sure"] = None, False
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
            why = [f"{p['n']}번 읽음"] + (["크고 또렷함"] if p["big"] else []) + (["문 옆"] if p["door"] else []) + (["본 번호와 가까움"] if p["near"] else [])
            self.event(t, f"{room}호 문패 확정 ({p['side']}, {', '.join(why)})")
        return p

    def direction(self, target, since=-1):
        """확실한 문패로 만든 그래프에서 목적지 방향. since 이후에 본 문패만 씀 (뒤로 돈 뒤에는 순서를 새로 셈).
        돌려줌: {"direction": 앞쪽/뒤쪽/근처/보임/다른 층/모름, "why": …, "walk": [번호…]}"""
        mine = [p for (sg, _), p in self.plates.items() if sg == self.seg and p["last"] > since]
        sure = [p for p in mine if p["ok"]]
        floors = {}
        for p in sure:
            floors[p["room"][0]] = floors.get(p["room"][0], 0) + 1
        if not floors:
            return {"direction": "모름", "why": "확실한 문패 없음"}
        fl = max(floors, key=floors.get)
        if fl != target[0]:
            return {"direction": "다른 층", "why": f"확실한 문패의 첫 자리 {fl}"}
        if any(p["room"] == target for p in sure):
            return {"direction": "보임", "why": f"{target}호 문패 확정"}
        anchors = [p for p in mine if p.get("side_sure") and p["room"][0] == fl and "-" not in p["room"]
                   and (p["ok"] or p["best"] >= FLOW_SCORE)
                   and (p["dmin"] is None or p["dmin"] <= ANCHOR_DIST)]
        order = lambda p: p["near_t"] if p["near_t"] is not None else p["last"]  # noqa: E731  지나간 순서
        if not anchors:
            return {"direction": "모름", "why": "벽 쪽이 확실한 같은 층 문패 없음"}
        key = min(anchors, key=lambda p: abs(int(p["room"]) - int(target)))
        wall = sorted([p for p in anchors if p["side"] == key["side"]], key=order)
        nums = [int(p["room"]) for p in wall]
        steps = [b - a for a, b in zip(nums, nums[1:]) if b != a]
        dec, inc = sum(d < 0 for d in steps), sum(d > 0 for d in steps)
        if dec == inc:  # 이 벽만으로 못 정하면 양쪽 벽 전체로
            for side in ("왼쪽", "오른쪽"):
                ns = [int(p["room"]) for p in sorted((p for p in anchors if p["side"] == side), key=order)]
                ds = [b - a for a, b in zip(ns, ns[1:]) if b != a]
                dec, inc = dec + sum(d < 0 for d in ds), inc + sum(d > 0 for d in ds)
            dec -= sum(d < 0 for d in steps); inc -= sum(d > 0 for d in steps)  # 위에서 key 벽을 두 번 더한 것 빼기
        walk = [p["room"] for p in wall]
        if dec == inc:
            return {"direction": "모름", "why": "번호 흐름을 정할 근거 부족", "walk": walk}
        flow = -1 if dec > inc else 1
        t, last = int(target), nums[-1]
        if abs(t - last) <= 1:
            return {"direction": "근처", "why": f"마지막 문패 {last}와 목적지가 바로 옆 번호", "walk": walk, "side": key["side"]}
        ahead = (t - last) * flow > 0
        return {"direction": "앞쪽" if ahead else "뒤쪽", "why": f"{key['side']} 벽 {' → '.join(walk)} (앞으로 갈수록 번호 {'감소' if flow < 0 else '증가'})",
                "walk": walk, "side": key["side"]}

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
