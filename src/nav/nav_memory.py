"""길안내 한 번의 기억: 시간 순서로 쌓는 문패 기록, 장소 흐름, 사건 기록 (navigator.Navigator가 씀).

문패는 번호마다 근거를 쌓아 믿음 점수를 매긴다. 3점 이상이면 확정 (VLM_NAV의 3점 기준을 그대로 쓰고 항목만 Claude 인식에 맞게 바꿈).
  여러 호출에서 같은 번호(최대 2) + 정면 가까이(가운데, 3m 이내) 1 + 문 옆 1 + 확정한 같은 층 번호와 10 이내 1
  + 확대 재확인 일치 1
  (VLM_NAV: 읽힌 횟수 최대 3 + 크고 또렷함 + 문 옆 + 가까운 번호. Claude는 한 장을 한 번만 보므로 횟수는 2까지만 센다.)
확대 재확인이 다른 번호를 읽으면 그 걸음의 읽기는 세지 않는다.
사건 기록은 글 한 줄씩이라 나중에 텍스트 모델에 그대로 넣을 수 있다 (text()).
"""

CONFIRM = 3      # 확정에 필요한 믿음 점수
READS_MAX = 2    # 같은 번호를 여러 호출에서 읽은 횟수는 2점까지
FRONT_M = 3.0    # '정면 가까이': 화면 가운데이고 이 거리(m) 이내
NEAR_DIFF = 10   # '가까운 번호': 확정한 같은 층 번호와 차이 10 이내
FORGET = 6       # 이 초 동안 안 보이면 읽힌 횟수를 다시 셈


class Memory:
    def __init__(self, target):
        self.target = target
        self.seg = 0
        self.plates = {}   # (구간, 번호) → 기록
        self.places = []   # [{"from", "to", "place"}] 같은 장소가 이어진 구간
        self.events = []   # [{"t", "seg", "text"}]

    def new_segment(self, t):
        self.seg += 1
        self.event(t, "엘리베이터에서 내림 (층·문패 기록을 새로 시작)")

    def event(self, t, text):
        self.events.append({"t": t, "seg": self.seg, "text": text})

    def place(self, t, name):
        """Claude가 본 장소 이름(한국어, 모르면 None). 장소가 바뀌면 처음 나왔을 때만 사건으로 남김."""
        if self.places and self.places[-1]["place"] == name:
            self.places[-1]["to"] = t
            return
        self.places.append({"from": t, "to": t, "place": name})
        if name and sum(p["place"] == name for p in self.places) == 1:
            self.event(t, f"장소 인식: {name}")

    def confirmed(self):
        return [p["room"] for (seg, _), p in self.plates.items() if seg == self.seg and p["ok"]]

    def get(self, room):
        return self.plates.get((self.seg, room))

    def see(self, t, room, side, m, on, recheck=None):
        """문패 번호를 한 번 읽음. recheck: True 일치 / False 다른 번호 / None 안 함·못 읽음.
        기록을 갱신하고 돌려준다 (rec["ok"]가 확정 여부)."""
        p = self.plates.setdefault((self.seg, room), {"room": room, "first": t, "last": -99, "n": 0, "total": 0,
                                                       "front": False, "door": False, "near": False, "recheck": False,
                                                       "ok": False, "ok_t": None, "min_m": None, "side": None,
                                                       "rejected": 0, "reads": []})
        p["reads"] = (p["reads"] + [{"t": t, "side": side, "m": m, "on": on, "recheck": recheck}])[-8:]
        if recheck is False:
            p["rejected"] += 1
            return p
        if t - p["last"] > FORGET:
            p["n"] = 0
        if p["last"] != t:
            p["n"] += 1
        p["total"] += 1
        p["last"] = t
        p["front"] |= side == "C" and m is not None and m <= FRONT_M
        p["door"] |= on == "door"
        p["recheck"] |= recheck is True
        ok = self.confirmed()
        p["near"] |= "-" not in room and any(
            o[0] == room[0] and "-" not in o and abs(int(o) - int(room)) <= NEAR_DIFF for o in ok if o != room)
        if m is not None and (p["min_m"] is None or m <= p["min_m"]):
            p["min_m"], p["side"] = m, side
        elif p["side"] is None:
            p["side"] = side
        p["trust"] = min(p["n"], READS_MAX) + p["front"] + p["door"] + p["near"] + p["recheck"]
        if not p["ok"] and p["trust"] >= CONFIRM:
            p["ok"], p["ok_t"] = True, t
            why = [f"{p['n']}번 읽음"] + (["정면 가까이"] if p["front"] else []) + (["문 옆"] if p["door"] else []) + \
                  (["본 번호와 가까움"] if p["near"] else []) + (["확대 재확인"] if p["recheck"] else [])
            self.event(t, f"{room}호 문패 확정 ({p['side']}, {', '.join(why)})")
        return p

    def dump(self):
        return {"target": self.target, "segment": self.seg, "events": self.events, "places": self.places,
                "plates": [{"seg": seg, **p} for (seg, _), p in self.plates.items()]}

    def text(self, last=30):
        """텍스트 모델에 넣을 기억 요약."""
        lines = [f"목적지: {self.target}", "지금까지 일어난 일:"]
        lines += [f"- {e['t']}초: {e['text']}" for e in self.events[-last:]]
        maybe = [p for (seg, _), p in self.plates.items() if seg == self.seg and not p["ok"]]
        if maybe:
            lines.append("아직 확정하지 않은 번호: " + ", ".join(f"{p['room']}({p['total']}번)" for p in maybe[-6:]))
        return "\n".join(lines)
