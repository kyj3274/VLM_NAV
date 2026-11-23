"""길안내 한 번의 기억: 시간 순서로 쌓는 문패 기록, 장소 흐름, 사건 기록 (navigator.Navigator가 situ 모드에서 씀).

문패는 번호마다 근거를 쌓아 믿음 점수를 매긴다 (11/16 시연 126걸음 모의: 3점 기준이면 진짜 10개 중 9개, 오독 1종, 문패 아님 0종).
  읽힌 횟수(최대 3) + 크고 또렷함 1 + 문 옆 1 + 이미 확정한 같은 층 번호와 가까움 1  →  3점 이상이면 확정
사건 기록은 글 한 줄씩이라 나중에 텍스트 모델에 그대로 넣을 수 있다 (text()).

방향(direction): 확실한 문패만 쓰는 장면 그래프 방식 (src/eval/scene_graph.py와 같은 판단).
  벽 쪽이 일관된(2/3 이상) 확정 문패를 벽별로 지나간 순서대로 놓고, 목적지와 번호가 가장 가까운 문패의 벽에서
  "앞으로 갈수록 번호가 커지나/작아지나"를 정해 목적지가 앞인지 뒤인지 판단한다. 목적지와 번호 차이가 1이면 '근처'.
  11/17 영상 판단 지점(하차 후 13개): 12개 맞음 (예전 규칙 11개). 같은 영상으로 만든 규칙이라 다른 건물 확인 필요.
"""

import math

CONFIRM = 3       # 확정에 필요한 믿음 점수
BIG_SCORE = 0.95  # '크고 또렷함': 점수가 이 이상이고 글자 높이가 big_h 이상
NEAR_DIFF, NEAR_SCORE = 10, 0.8  # '가까움': 확정한 같은 층 번호와 차이 10 이내, 점수 0.8 이상
FORGET = 6        # 이 걸음 동안 안 보이면 읽힌 횟수를 다시 셈
SIDE_SURE = 2 / 3  # 벽 쪽 표가 이 비율 이상 같아야 그래프(방향 판단)에 넣음
FLOW_EPS = 0.1   # 방향 판단: 같은 벽 두 문패의 증감이 실제 흐름과 반대일 확률 (번호 건너뜀·오독)
DIR_SURE = 0.7   # 방향 판단: P(앞)이 이 이상이면 앞쪽, 1-이 이하면 뒤쪽, 그 사이는 모름
VOTE_OK = 0.85   # 사용자 방향 판단: 한 번의 벽 표(깊이)가 맞을 확률
POS_SCALE = 4    # 방향 판단: 지금 위치에서 이 걸음(초)만큼 떨어진 문패는 무게가 반
PLACE_MIN = 0.35


class Memory:
    def __init__(self, target, big_h=0.012):
        self.target, self.big_h = target, big_h
        self.seg = 0
        self.plates = {}   # (구간, 번호) → 기록
        self.places = []   # [{"from", "to", "place"}] 같은 장소가 이어진 구간
        self.events = []   # [{"t", "seg", "text"}]
        # 복도 고정 그래프: 문패 위치·벽은 복도 기준(처음 걷던 방향 = +, 그 왼쪽 벽 = A)으로 두고,
        # 사용자가 어느 쪽을 보는지(돌았는지)는 따로 추정한다. 돌아도 그래프는 지우지 않는다.
        self.log = []      # [(t, 구간, 번호, 사용자 기준 벽 '왼쪽'/'오른쪽'/None, 깊이 거리)] 문패 읽기 전부
        self.flips = []    # 돌았다고 결론 낸 때 (이 뒤부터 사용자 방향이 반대)
        self.turn_t = None # 돌라고 안내했지만 실제로 돌았는지 아직 모르는 때

    def new_segment(self, t):
        self.seg += 1
        self.flips, self.turn_t = [], None
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
        self.log.append((t, self.seg, room, vote if vote in ("왼쪽", "오른쪽") else None, dist))
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
            self.event(t, f"{room}호 문패 확정 ({p['side'] or '벽 미정'}, {', '.join(why)})")
        return p

    def turned(self, t):
        """뒤로 돌라고 안내함. 그래프는 그대로 두고 '이때 돌았다/안 돌았다' 두 가설을 연다.
        앞서 열어 둔 가설이 있으면 지금까지의 근거로 결론 내고 닫는다."""
        if self.turn_t is not None:
            p1 = self._heading_belief()[1]
            if p1 > 0.5:
                self.flips.append(self.turn_t)
            self.event(t, f"{self.turn_t}초에 {'돌았던' if p1 > 0.5 else '돌지 않았던'} 것으로 봄 (P={p1:.2f})")
        self.turn_t = t
        self.event(t, "뒤로 돌라고 안내함: 돌았는지는 이후 문패로 판단 (그래프는 유지)")

    # ---- 복도 고정 그래프
    def _view(self, turned):
        """사용자 방향 가설 하나(마지막 안내 때 돌았나)로 본 복도 그래프.
        돌려줌: (문패 목록 [{room, wall A/B, sure, pos, trust}], 지금 방향 ±1, 지금 위치)"""
        flips = self.flips + ([self.turn_t] if turned and self.turn_t is not None else [])

        def heading(t):
            return -1 if sum(t > f for f in flips) % 2 else 1

        pos_at, s_ = {}, 0.0
        tmax = max([x[0] for x in self.log] + [0])
        for t in range(tmax + 1):  # 걸음(초)마다 지금 방향으로 한 칸 (속도 일정 가정, 자이로·걸음 수가 오면 교체)
            pos_at[t] = s_
            s_ += heading(t + 1)
        reads = {}
        for t, sg, room, side, dist in self.log:
            if sg == self.seg:
                reads.setdefault(room, []).append((t, side, dist))
        out = []
        for room, rs in reads.items():
            p = self.plates[(self.seg, room)]
            votes = ["A" if (sd == "왼쪽") == (heading(t) == 1) else "B" for t, sd, _ in rs if sd]
            top = max(set(votes), key=votes.count) if votes else None
            sure = bool(votes) and votes.count(top) / len(votes) >= SIDE_SURE
            near = min((r for r in rs if r[2] is not None), key=lambda r: r[2], default=rs[-1])  # 가장 가까이 지난 때
            agree = votes.count(top) if votes else 0
            out.append({"room": room, "wall": top, "sure": sure, "pos": pos_at[near[0]], "ok": p["ok"],
                        "agree": agree, "disagree": len(votes) - agree,
                        "trust": p["best"] * (1 - 0.5 ** p["total"])})
        return out, heading(tmax + 1), pos_at[tmax]

    @staticmethod
    def _flow(plates):
        """복도 + 방향(위치가 커지는 쪽)으로 번호가 증가/감소하는 가설의 로그 근거 {1: …, -1: …}."""
        logw = {1: 0.0, -1: 0.0}
        for wall in ("A", "B"):
            wl = sorted((p for p in plates if p["wall"] == wall), key=lambda p: p["pos"])
            for a, b in zip(wl, wl[1:]):
                diff = int(b["room"]) - int(a["room"])
                if not diff or a["pos"] == b["pos"]:
                    continue
                rel = a["trust"] * b["trust"] * (1.0 if abs(diff) <= 3 else 0.5)  # 가까운 번호 = 같은 벽 이웃일 가능성 큼
                for d in (1, -1):  # 0.5(정보 없음) 대비: 맞는 쌍은 +, 어긋난 쌍은 −
                    logw[d] += rel * math.log(((1 - FLOW_EPS) if (diff > 0) == (d > 0) else FLOW_EPS) / 0.5)
        return logw

    def _same_floor(self, plates, fl):
        return [p for p in plates if p["room"][0] == fl and "-" not in p["room"] and p["sure"]]

    def _heading_belief(self, fl=None):
        """{False: P(안 돎), True: P(돎)}. 마지막 돌기 안내 뒤의 문패가 어느 가설에서 번호 흐름과 더 잘 맞나."""
        if self.turn_t is None:
            return {False: 1.0, True: 0.0}
        fl = fl or self.target[0]
        ev = {}
        for h in (False, True):
            plates = self._view(h)[0]
            logw = self._flow(self._same_floor(plates, fl))
            m = max(logw.values())
            ev[h] = m + math.log(sum(math.exp(v - m) for v in logw.values()) / 2)  # 흐름 가설에 대해 평균
            # 같은 문패를 돌기 전후로 다시 봤을 때 벽 표가 이 가설에서 한쪽으로 모이는가
            ev[h] += sum(p["trust"] * (p["agree"] * math.log(VOTE_OK / 0.5) + p["disagree"] * math.log((1 - VOTE_OK) / 0.5))
                         for p in plates)
        m = max(ev.values())
        z = sum(math.exp(v - m) for v in ev.values())
        return {h: math.exp(v - m) / z for h, v in ev.items()}

    def direction(self, target, since=-1):
        """복도 고정 그래프 위에서 목적지 방향을 확률로 판단. (since는 예전 호환용, 쓰지 않음)
        돌려줌: {"direction": 앞쪽/뒤쪽/근처/보임/다른 층/모름, "why": …, "walk": [번호…], "p_ahead": …}

        - 문패: 복도 기준 벽(A/B)과 위치(가장 가까이 지난 때의 복도 위 위치). 믿음 = 최고 OCR 점수 × (1 - 0.5^읽힌 횟수).
        - 번호 흐름 가설(복도 + 방향으로 증가/감소)을 같은 벽 이웃 문패의 증감으로 갱신 (예외 확률 FLOW_EPS).
        - 사용자 방향: 돌라고 안내한 뒤에는 '돎/안 돎' 두 가설을 반반으로 열고, 그 뒤 문패가 맞는 쪽으로 기움.
        - P(앞) = Σ 가설 · 지금 위치 가까운 같은 벽 문패마다 [목적지가 그 문패에서 사용자가 향한 쪽에 있나]."""
        mine = [p for (sg, _), p in self.plates.items() if sg == self.seg]
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
        hb = self._heading_belief(fl)
        t = int(target)
        p_ahead, near_votes, best = 0.0, 0.0, None
        for h, ph in hb.items():
            if ph < 1e-3:
                continue
            plates, head, s_now = self._view(h)
            same = self._same_floor(plates, fl)
            if not same:
                continue
            logw = self._flow(same)
            m = max(logw.values())
            pd = {d: math.exp(v - m) for d, v in logw.items()}
            z = sum(pd.values())
            pd = {d: v / z for d, v in pd.items()}
            key = min(same, key=lambda p: abs(int(p["room"]) - t))
            wall = [p for p in same if p["wall"] == key["wall"]]
            w = [p["trust"] * 0.5 ** (abs(p["pos"] - s_now) / POS_SCALE) for p in wall]  # 지금 위치에 가까울수록 무게
            pa = sum(pd[d] * sum(wi * (1.0 if (t - int(p["room"])) * d * head > 0 else 0.5 if t == int(p["room"]) else 0.0)
                                 for wi, p in zip(w, wall)) / sum(w) for d in pd)
            p_ahead += ph * pa
            nearest = sorted(wall, key=lambda p: abs(p["pos"] - s_now))[:3]
            nums = [int(p["room"]) for p in nearest]
            near_votes += ph * (len(nums) >= 2 and min(nums) <= t <= max(nums))
            if best is None or ph > best[0]:
                passed = sorted((p for p in wall if (p["pos"] - s_now) * head <= 0), key=lambda p: p["pos"] * head)
                best = (ph, [p["room"] for p in passed] or [p["room"] for p in wall], key["wall"], pd[1] if head == 1 else pd[-1])
        if best is None:
            return {"direction": "모름", "why": "벽 쪽이 확실한 같은 층 문패 없음"}
        walk = best[1]
        info = {"walk": walk, "side": best[2], "p_ahead": round(p_ahead, 2), "p_turned": round(hb[True], 2)}
        why = (f"{best[2]}벽 {' → '.join(walk)}, P(앞)={p_ahead:.2f}, 향한 쪽으로 번호 증가 P={best[3]:.2f}"
               + (f", 돌았을 확률 {hb[True]:.2f}" if self.turn_t is not None else ""))
        if near_votes >= 0.5:
            return {"direction": "근처", "why": why + ", 목적지가 지금 근처 번호 사이", **info}
        if p_ahead >= DIR_SURE:
            return {"direction": "앞쪽", "why": why, **info}
        if p_ahead <= 1 - DIR_SURE:
            return {"direction": "뒤쪽", "why": why, **info}
        return {"direction": "모름", "why": why + " (아직 확신 부족)", **info}

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
