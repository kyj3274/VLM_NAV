"""복도 고정 그래프와 확률 방향 판단 (인식과 무관한 공통 판단, Claude 버전과 로컬 버전이 같이 쓴다).

문패 읽기 하나하나를 (시각, 번호, 사용자 기준 벽, 거리, 읽기 점수)로 넣으면,
  - 문패를 복도 기준 벽(A = 처음 걷던 방향의 왼쪽, B = 오른쪽)과 복도 위 위치에 놓고
  - '걷는 방향으로 번호가 증가/감소' 가설을 같은 벽 이웃 문패의 증감으로 갱신하고
  - 사용자가 돌았는지('돌았다/안 돌았다' 가설)를 그 뒤 문패로 판단해서
  - 목적지가 앞쪽/뒤쪽/근처인지 확률(P(앞))로 낸다.
규칙(±1, 다수결) 대신 확률이라, 근거가 부족하면 틀리는 대신 '모름'으로 늦어진다.
11/18~20 실험·시연: docs/20 (판단 지점 12/13, 11/20 105호 실제 U턴을 P=0.82로 알아챔).

알려진 가정 (다른 건물에서 확인 필요): 방 번호는 숫자 3자리 안팎이고 첫 자리가 층, 위치 한 칸 = 걷는 속도로 1초.
위치는 휴대폰 걸음 수가 있으면 걸음으로, 없으면 시간으로 센다 (move).
"""

import math

from arrival import STEP_M, TURN_DEG, WALK_MPS  # 같은 core 폴더 (걸음 길이·걷는 속도 가정, 돈 것으로 볼 방위 변화)

SIDE_SURE = 2 / 3  # 벽 표가 이 비율 이상 같아야 그래프에 넣음
FLOW_EPS = 0.1     # 같은 벽 두 문패의 증감이 실제 흐름과 반대일 확률 (번호 건너뜀·오독)
DIR_SURE = 0.7     # P(앞)이 이 이상이면 앞쪽, 1-이 이하면 뒤쪽, 그 사이는 모름
VOTE_OK = 0.85     # 한 번의 벽 표(깊이)가 맞을 확률 (돌았는지 판단에 씀)
POS_SCALE = 4      # 지금 위치에서 이 칸만큼 떨어진 문패는 무게가 반
PASS_POS = 2       # 확정된 목적지 문패가 지금 위치보다 이 칸 넘게 뒤에 있으면 '지나침'(뒤쪽)
END_M = 4.0        # 깊이로 앞의 가장 먼 곳이 이 안이면 막다른 곳(복도 끝)
END_RUN = 2        # 막다른 곳이 이만큼 연속 사진에서 보여야 인정 (사람·물건이 잠깐 가린 것 제외)
END_AXIS = 45      # 휴대폰이 걷는 방향에서 이 각도 넘게 돌아가 있으면(옆을 비춤) 막다른 곳 판단 안 함
NEAR_POS = 5       # 목적지 추정 위치가 지금 위치에서 이 칸(걷는 속도로 약 4m) 안이면 근처
BRACKET = 8        # 목적지 번호를 사이에 둔 같은 벽 이웃 문패의 번호 차이가 이 이하일 때만 그 사이로 위치를 추정
EXPLORED = 0.1     # 끝까지 가 본 쪽에 목적지가 있을 가능성을 이만큼 낮춤 (문패를 놓쳤을 수 있어 0은 아님)
ELSEWHERE = 0.1   # 끝을 본 뒤: 목적지가 이 복도가 아닌 곳(갈림길 너머 다른 복도)에 있을 가능성 (앞·뒤·다른 곳으로 나눔)


class CorridorGraph:
    def __init__(self, event=None):
        self.event = event or (lambda t, text: None)  # 사건 기록 함수 (기억의 event)
        self.reset()

    def reset(self):
        """새 구간(엘리베이터에서 내림 등): 그래프를 새로 시작."""
        self.log = []      # [(t, 번호, 사용자 기준 벽 '왼쪽'/'오른쪽'/None, 거리 m 또는 None)]
        self.rooms = {}    # 번호 → {"best": 최고 읽기 점수, "total": 읽힌 횟수, "ok": 확정}
        self.flips = []    # 돌았다고 결론 낸 때
        self.turn_t = None # 돌라고 안내했지만 실제로 돌았는지 아직 모르는 때
        self.ref = None    # 휴대폰 방위(도)를 받으면: 처음 걷던 방향의 방위. 있으면 돌았는지를 방위로 정함 (가설 안 씀)
        self.sense_flips, self.sense_dir = [], 1
        self.ends = set()  # 끝까지 가 본 쪽 (복도 기준 방향 ±1). 끝을 봤으면 거기까지 걸어온 것이라 그 사이는 다 지나옴
        self.end_run = 0
        self.walk = []     # [(시각, 걸은 칸, 걸음 수)] move()로 받음. 없으면 시간 = 걸은 칸

    def see(self, t, room, score, side=None, dist=None):
        """문패 읽기 하나. side: 사용자 기준 '왼쪽'/'오른쪽'(모르면 None), dist: 문패까지 거리(m), score: 0~1 읽기 믿음."""
        self.log.append((t, room, side if side in ("왼쪽", "오른쪽") else None, dist))
        r = self.rooms.setdefault(room, {"best": 0.0, "total": 0, "ok": False})
        r["best"], r["total"] = max(r["best"], score), r["total"] + 1

    def set_ok(self, room, ok=True):
        """문패 확정 여부 (확정은 각 버전의 믿음 점수 규칙이 정함). 층 판단과 '보임'에만 씀."""
        self.rooms.setdefault(room, {"best": 0.0, "total": 0, "ok": False})["ok"] = ok

    def move(self, t, steps):
        """사진마다 휴대폰 걸음 수(없으면 None). 걸음 수가 있으면 그만큼만 위치를 옮긴다 (멈춰 서 있으면 그대로).
        전에는 시간만큼 옮겨, "멈춰 주세요"로 서서 확인하는 동안 목적지를 '지나쳤다'고 보고 되돌려 보냈다
        (2026-11-25 시뮬레이터 인식 오류 약함, 목적지 2.6m 앞에서 왕복)."""
        if self.walk and t <= self.walk[-1][0]:
            return
        if not self.walk:
            self.walk.append((t, t, steps))
            return
        pt, pw, ps = self.walk[-1]
        dw = (steps - ps) * STEP_M / WALK_MPS if steps is not None and ps is not None else t - pt
        self.walk.append((t, pw + max(0.0, dw), steps))

    def _w(self, t):
        """시각 t까지 걸은 칸. 걸음 수를 받은 뒤로는 다음 사진 전까지 그대로, 시간으로 세던 중이면 시간만큼 더함."""
        last = next((r for r in reversed(self.walk) if r[0] <= t), None)
        if last is None:
            return t
        return last[1] + (0.0 if last[2] is not None else t - last[0])

    def sense_heading(self, t, deg):
        """사진을 찍을 때의 휴대폰 방위(도, 나침반). 처음 방위를 기준으로, 지금 걷는 방향에서 TURN_DEG 넘게 바뀌면
        그 사진 시각에 돈 것으로 기록한다. 안내와 상관없이 스스로 돈 것(복도 끝 등)도 잡는다 (2026-11-25 시뮬레이터 v1)."""
        if deg is None:
            return
        if self.ref is None:
            self.ref, self.turn_t = deg, None  # 방위가 오면 '돌았나' 가설 대신 방위를 씀
            return
        walk = self.ref + (0 if self.sense_dir > 0 else 180)
        if abs((deg - walk + 180) % 360 - 180) > TURN_DEG:
            self.sense_flips.append(t - 0.01)  # 이 사진은 이미 돈 뒤 (돌기 시각은 바로 앞)
            self.sense_dir = -self.sense_dir
            self.event(t, f"휴대폰 방위로 돈 것을 확인 ({deg:.0f}°)")

    def see_depth(self, t, far_m, deg=None):
        """깊이: 앞에서 가장 먼 곳까지 거리(m). END_M 안이 END_RUN장 연속이면 지금 걷는 쪽 끝까지 왔다고 기록.
        휴대폰 방위가 걷는 방향에서 END_AXIS 넘게 돌아가 있으면(옆을 비춤) 세지 않는다.
        방위가 없으면 쓰지 않는다: 어느 쪽을 보고 있는지 모르면 끝이 어느 쪽인지도 모름 (스스로 돈 것을 모르면
        끝을 본 쪽을 계속 앞쪽으로 착각해 반대로 보냄, 시뮬레이터)."""
        if far_m is None or self.ref is None or deg is None:
            return
        walk = self.ref + (0 if self.sense_dir > 0 else 180)
        if abs((deg - walk + 180) % 360 - 180) > END_AXIS:
            self.end_run = 0
            return
        self.end_run = self.end_run + 1 if far_m < END_M else 0
        if self.end_run >= END_RUN:
            head = self.sense_dir
            if head not in self.ends:
                self.ends.add(head)
                self.event(t, f"막다른 곳 (앞 {far_m:.1f}m): 이쪽 끝까지 왔음")

    def turned(self, t):
        """뒤로 돌라고 안내함. 그래프는 그대로 두고 '이때 돌았다/안 돌았다' 두 가설을 연다.
        앞서 열어 둔 가설이 있으면 지금까지의 근거로 결론 내고 닫는다.
        휴대폰 방위가 있으면 가설을 열지 않는다 (돌았는지는 sense_heading이 정함)."""
        if self.ref is not None:
            self.event(t, "뒤로 돌라고 안내함 (돌았는지는 휴대폰 방위로 확인)")
            return
        if self.turn_t is not None:
            p1 = self.heading_belief()[True]
            if p1 > 0.5:
                self.flips.append(self.turn_t)
            self.event(t, f"{self.turn_t}초에 {'돌았던' if p1 > 0.5 else '돌지 않았던'} 것으로 봄 (P={p1:.2f})")
        self.turn_t = t
        self.event(t, "뒤로 돌라고 안내함: 돌았는지는 이후 문패로 판단 (그래프는 유지)")

    # ---- 그래프 보기
    def _view(self, turned, now=None):
        """사용자 방향 가설 하나(마지막 안내 때 돌았나)로 본 복도 그래프. now: 지금 시각(없으면 마지막으로 문패를 본 때).
        돌려줌: (문패 목록 [{room, wall A/B, sure, pos, trust, ok, agree, disagree}], 지금 방향 ±1, 지금 위치)"""
        flips = self.flips + self.sense_flips + ([self.turn_t] if turned and self.turn_t is not None else [])

        fl_sorted = sorted(flips)

        def heading(t):
            return -1 if sum(t > f for f in fl_sorted) % 2 else 1

        w = self._w

        def pos(t):  # 걸은 칸(걸음 수, 없으면 시간)만큼 지금 방향으로 움직인다고 봄
            s_, prev, sign = 0.0, 0.0, 1
            for f in fl_sorted:
                if t <= f:
                    break
                s_, prev, sign = s_ + sign * (w(f) - w(prev)), f, -sign
            return s_ + sign * (w(t) - w(prev))

        tmax = max([x[0] for x in self.log] + [0] + ([now] if now is not None else []))
        reads = {}
        for t, room, side, dist in self.log:
            reads.setdefault(room, []).append((t, side, dist))
        out = []
        for room, rs in reads.items():
            r = self.rooms[room]
            votes = ["A" if (sd == "왼쪽") == (heading(t) == 1) else "B" for t, sd, _ in rs if sd]
            top = max(set(votes), key=votes.count) if votes else None
            sure = bool(votes) and votes.count(top) / len(votes) >= SIDE_SURE
            near = min((x for x in rs if x[2] is not None), key=lambda x: x[2], default=rs[-1])  # 가장 가까이 지난 때
            agree = votes.count(top) if votes else 0
            # 문패는 본 자리가 아니라 본 거리만큼 앞에 있다 (3m 앞에서 본 문패를 그때 위치에 두면 지나치기 전에 '지나침'이 됨,
            # 2026-11-25 시뮬레이터). 거리 m → 걷는 속도로 위치 칸(초)으로 바꿈
            ahead = (near[2] or 0.0) / WALK_MPS
            out.append({"room": room, "wall": top, "sure": sure, "pos": pos(near[0]) + heading(near[0]) * ahead, "ok": r["ok"],
                        "agree": agree, "disagree": len(votes) - agree,
                        "trust": r["best"] * (1 - 0.5 ** r["total"])})
        return out, heading(tmax + 1e-6), pos(tmax)

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

    @staticmethod
    def _same_floor(plates, fl):
        return [p for p in plates if p["room"][0] == fl and p["room"].isdigit() and p["sure"]]

    def heading_belief(self, fl=None, target=None):
        """{False: P(안 돎), True: P(돎)}. 마지막 돌기 안내 뒤의 문패가 어느 가설에서 더 잘 맞나."""
        if self.turn_t is None:
            return {False: 1.0, True: 0.0}
        fl = fl or (target[0] if target else None) or self._floor() or ""
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

    def _floor(self):
        floors = {}
        for room, r in self.rooms.items():
            if r["ok"]:
                floors[room[0]] = floors.get(room[0], 0) + 1
        return max(floors, key=floors.get) if floors else None

    @staticmethod
    def _bracket(plates, t):
        """같은 벽에서 위치가 이웃한 두 문패의 번호가 t를 사이에 두면, 번호 비율로 t의 위치를 추정. 없으면 None."""
        best = None
        for wall in ("A", "B"):
            wl = sorted((p for p in plates if p["wall"] == wall), key=lambda p: p["pos"])
            for a, b in zip(wl, wl[1:]):
                na, nb = int(a["room"]), int(b["room"])
                if na != nb and min(na, nb) < t < max(na, nb) and abs(nb - na) <= BRACKET:
                    if best is None or abs(nb - na) < best[0]:
                        best = (abs(nb - na), a["pos"] + (b["pos"] - a["pos"]) * (t - na) / (nb - na))
        return best[1] if best else None

    def _explored(self, pa, head):
        """끝까지 가 본 쪽을 낮춘 (P(앞), P(뒤)). 끝을 본 뒤에는 '이 복도가 아닌 곳'도 하나의 경우로 둔다:
        번호 흐름이 이미 끝을 본 쪽을 가리키면 앞·뒤 모두 낮아져 '모름'이 되고, 갈림길 판단(floor_graph)에 맡긴다
        (2026-11-25 시뮬레이터 v3: 갈림길 너머 복도의 목적지를 막다른 곳 너머로 보고 끝에서 오갔음)."""
        if not self.ends:
            return pa, 1 - pa
        a = pa * (EXPLORED if head in self.ends else 1)
        b = (1 - pa) * (EXPLORED if -head in self.ends else 1)
        return a / (a + b + ELSEWHERE), b / (a + b + ELSEWHERE)

    # ---- 판단
    def direction(self, target, now=None):
        """목적지 방향. now: 지금 시각 (문패가 안 보이는 동안에도 걸은 만큼 위치를 옮김, 2026-11-25). 돌려줌: {"direction": 앞쪽/뒤쪽/근처/보임/다른 층/모름, "why", "walk": [지나온 번호…], "p_ahead", "side"}"""
        fl = self._floor()
        if fl is None:
            return {"direction": "모름", "why": "확실한 문패 없음"}
        if fl != target[0]:
            return {"direction": "다른 층", "why": f"확실한 문패의 첫 자리 {fl}"}
        if self.rooms.get(target, {}).get("ok"):
            # 목적지를 본 뒤 지나쳤나: 가장 그럴듯한 방향 가설에서 목적지 문패가 사용자 뒤로 PASS_POS 넘게 있음
            hb = self.heading_belief(fl)
            if self.turn_t is not None and 0.3 < hb[True] < 0.7:  # 돌라고 한 뒤 돌았는지 아직 모름: 되풀이하지 않음
                return {"direction": "모름", "why": f"{target}호를 지나친 뒤 돌았는지 확인 중 (P={hb[True]:.2f})",
                        "p_turned": round(hb[True], 2)}
            plates, head, s_now = self._view(max(hb, key=hb.get), now)
            tp = next((p for p in plates if p["room"] == target), None)
            if tp is not None and (tp["pos"] - s_now) * head < -PASS_POS:
                return {"direction": "뒤쪽", "why": f"{target}호 문패를 본 뒤 지나침", "walk": [target], "passed": True,
                        "p_ahead": 0.0, "p_turned": round(hb[True], 2)}
            return {"direction": "보임", "why": f"{target}호 문패 확정"}
        hb = self.heading_belief(fl)
        t = int(target)
        p_ahead, p_back, near_votes, best = 0.0, 0.0, 0.0, None
        for h, ph in hb.items():
            if ph < 1e-3:
                continue
            plates, head, s_now = self._view(h, now)
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
            # 목적지 번호를 사이에 둔 같은 벽 이웃 문패 두 개가 있으면 그 사이로 목적지 위치를 추정 (405·409 → 407은 가운데).
            # 지금 위치와 비교해 근처/앞/뒤. 전에는 '가장 가까운 문패 3개의 번호 사이'로 근처를 정해서, 목적지를 놓치고
            # 지나친 뒤에도 이미 지나온 문패 때문에 계속 근처라고 했다 (2026-11-25 시뮬레이터 v2, 인식 놓침).
            tpos = self._bracket(same, t)
            if tpos is not None:
                rel = (tpos - s_now) * head
                if abs(rel) <= NEAR_POS:
                    near_votes += ph
                else:
                    pa = 0.95 if rel > 0 else 0.05
            pa, pb = self._explored(pa, head)
            p_ahead += ph * pa
            p_back += ph * pb
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
            return {"direction": "근처", "why": why + ", 이웃 번호 사이로 추정한 목적지 위치가 가까움", **info}
        if p_ahead >= DIR_SURE:
            return {"direction": "앞쪽", "why": why, **info}
        if p_back >= DIR_SURE:
            return {"direction": "뒤쪽", "why": why, **info}
        return {"direction": "모름", "why": why + " (아직 확신 부족)", **info}
