"""층 그래프: 갈림길에서 어느 복도로 갈지 (인식과 무관한 공통 판단, 2026-11-25 시뮬레이터 v3).

복도 하나 안의 방향은 nav_graph.py(번호 흐름)가 정한다. 이 파일은 그 위 층으로, 갈림길과 복도 가지를 기억한다.
  위치: 휴대폰 걸음 수 × 걸음 길이를 방위 쪽으로 더해 대략 안다 (센서가 없으면 이 판단은 하지 않음 = '모름').
  갈림길: 깊이로 옆이 트인 곳을 보면 그 위치에 갈림길을 둔다 (JUNC_MATCH 안이면 같은 갈림길).
  가지: 갈림길에서 나가는 방향(방위 90° 단위)마다 복도 가지. 가지마다 거기서 본 번호, 끝까지 가 봤는지를 적는다.

갈림길에서 고르기: 아직 다 안 가 본 가지(경계)마다 '목적지가 거기 있을 가능성'을 근거로 곱하고, 거기까지 거리로 나눈다.
  근거가 없으면 모든 가지가 같아서 가까운 안 가 본 곳부터 빠짐없이 둘러보게 된다 (번호가 없는 건물).
  - 표지판: 이 갈림길에서 목적지(이름 또는 번호 범위)를 가리키면 그 가지 × SIGN, 다른 가지는 ÷ SIGN
  - 번호: 그 가지에서 본 번호 범위에 목적지가 있으면 × IN_RANGE, 아니면 번호 차이만큼 줄임 (GAP)
  - 지금 복도의 번호 흐름(nav_graph)이 앞/뒤를 가리키면 그 가지 × FLOW
  - 끝까지 가 본 가지 × EXPLORED
"""

import math

STEP_M = 0.6       # 걸음 하나 거리 (arrival.py와 같음)
JUNC_MATCH = 4.0   # 새로 본 갈림길이 기억한 갈림길과 이 거리(m) 안이면 같은 곳 (걸음 수 추정 오차 여유)
AT_M = 1.5         # 갈림길 위치에서 이 안이면 '갈림길에 있음'
SIGN = 20.0        # 표지판이 목적지를 그 가지로 가리킬 때
IN_RANGE = 3.0     # 가지에서 본 번호 범위 안에 목적지 번호가 있을 때 (못 보고 지나쳤을 수 있음)
GAP = 10.0         # 본 번호와 목적지 번호 차이가 이만큼이면 가능성 1/e
FLOW = 3.0         # 지금 복도 번호 흐름이 그 쪽을 가리킬 때
EXPLORED = 0.05    # 끝까지 가 봤는데 목적지를 못 본 가지
DIST = 20.0        # 거리 무게: 이만큼 멀면 가능성을 반으로 봄


def _c4(deg):
    """방위를 0·90·180·270 중 가까운 것으로."""
    return int(round((deg % 360) / 90.0)) % 4 * 90


def _vec(deg):
    """방위(나침반: 0 북, 90 동, 시계 방향) → 평면 (x 동, y 북)."""
    r = math.radians(deg)
    return math.sin(r), math.cos(r)


def mentions(item, target):
    """표지판 항목이 목적지를 말하나: 같은 이름, 또는 '401~420' 범위에 목적지 번호가 들어감."""
    item = item.strip()
    if item == target:
        return True
    if "~" in item and target.isdigit():
        a, b = (s.strip() for s in item.split("~", 1))
        return a.isdigit() and b.isdigit() and int(a) <= int(target) <= int(b)
    return False


class FloorGraph:
    def __init__(self, target, event=None):
        self.target = target
        self.event = event or (lambda text: None)
        self.reset()

    def reset(self):
        self.pos = None             # 걸음 수로 추정한 지금 위치 (x, y) m
        self.steps = None
        self.junctions = []         # [{"p": (x, y), "open": set(방위), "signs": {방위: [항목]}}]
        self.branch = {}            # (갈림길 번호, 방위) → 가지 번호
        self.edges = []             # [{"nums": set, "dead": bool, "ends": set(갈림길 번호)}]
        self.edge = self._new_edge()  # 지금 걷는 가지
        self.at = None              # 지금 서 있는 갈림길
        self.corridor_turn = False  # 방금 다른 축으로 걷기 시작함: 갈림길에서 다른 복도로, 또는 꺾인 곳·넓은 곳의 벽에서 (복도 그래프를 새로)
        self.came = None            # 갈림길에 들어올 때 방위
        self.axis = None            # 마지막으로 걸은 방위 (90° 단위)

    def _new_edge(self):
        self.edges.append({"nums": set(), "dead": False, "ends": set()})
        return len(self.edges) - 1

    # ------------------------------------------------------------ 관찰
    def move(self, steps, heading):
        """휴대폰 걸음 수·방위 (사진마다). 걸은 만큼 위치를 옮기고 갈림길에 들고 나는 것을 적는다."""
        self.corridor_turn = False
        if steps is None or heading is None:
            return
        if self.pos is None:
            self.pos, self.steps, self.axis = (0.0, 0.0), steps, _c4(heading)  # 첫 사진 방위 = 복도 그래프가 처음 잡는 방향
            return
        d = (steps - self.steps) * STEP_M
        self.steps = steps
        if d <= 0:
            return
        vx, vy = _vec(_c4(heading))
        self.pos = (self.pos[0] + vx * d, self.pos[1] + vy * d)
        near = self._nearest(self.pos, AT_M)
        if near is not None and self.at is None:  # 갈림길에 들어섬: 온 가지를 그 갈림길의 뒤쪽 가지로
            self.at, self.came = near, self.axis if self.axis is not None else _c4(heading)  # 들어오기 전 걷던 축
            self._link(near, (self.came + 180) % 360, self.edge)
        elif near is None and self.at is not None:  # 갈림길을 나섬
            j, c = self.at, _c4(heading)
            e = self.branch.get((j, c))
            if e is None:
                e = self._new_edge()
                self._link(j, c, e)
            self.edge, self.at = e, None
            self.corridor_turn = self.came is not None and (c - self.came) % 180 != 0
            if self.corridor_turn:
                self.event(f"갈림길에서 다른 복도로 들어섬 (방위 {c}°)")
        elif near is None and self.at is None and self.axis is not None and (_c4(heading) - self.axis) % 180 != 0:
            # 갈림길이 아닌 곳에서 다른 축으로 걸음 (꺾인 복도, 로비 벽을 따라 돎): 번호가 이어지는 직선이 바뀜.
            # 앞 직선의 '끝까지 가 봄'은 이 직선과 상관없으므로 새 가지로 (시뮬레이터 로비)
            self.edge, self.corridor_turn = self._new_edge(), True
            self.event(f"갈림길 아닌 곳에서 다른 축으로 걸음 (방위 {_c4(heading)}°)")
        self.axis = _c4(heading)

    def _link(self, j, c, e):
        if (j, c) in self.branch and self.branch[(j, c)] != e:  # 한 바퀴 돌아 같은 가지를 다른 쪽에서 만남: 합침
            old = self.edges[self.branch[(j, c)]]
            new = self.edges[e]
            new["nums"] |= old["nums"]
            new["ends"] |= old["ends"]
            new["dead"] = new["dead"] or old["dead"]
            for k, v in list(self.branch.items()):
                if v == self.branch[(j, c)]:
                    self.branch[k] = e
        self.branch[(j, c)] = e
        self.edges[e]["ends"].add(j)
        self.junctions[j]["open"].add(c)

    def _nearest(self, p, within):
        best = None
        for i, j in enumerate(self.junctions):
            d = math.dist(p, j["p"])
            if d <= within and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    def see_junction(self, m, opens, heading, signs=None):
        """앞 m미터에 갈림길. opens: 몸 기준 트인 쪽 {'ahead','left','right'}. signs: [{"dir": 몸 기준 쪽, "items": [...], "match": 선택}].
        match: 인식(Claude)이 목적지를 알고 보고 '이 표지판 항목이 목적지'라고 한 것 (영어 표지판처럼 글자가 달라도).
        돌려줌: 갈림길 번호 (위치를 모르면 None)."""
        if self.pos is None or heading is None:
            return None
        h = _c4(heading)
        vx, vy = _vec(h)
        p = (self.pos[0] + vx * m, self.pos[1] + vy * m)
        j = self._nearest(p, JUNC_MATCH)
        if j is None:
            self.junctions.append({"p": p, "open": set(), "signs": {}})
            j = len(self.junctions) - 1
            self.event(f"갈림길 발견 ({m:.0f}m 앞, 트인 쪽 {', '.join(sorted(opens))})")
        rel = {"ahead": 0, "right": 90, "left": 270}
        self.junctions[j]["open"] |= {(h + rel[o]) % 360 for o in opens}
        if m > AT_M and (j, (h + 180) % 360) not in self.branch:  # 다가가는 중: 지금 걷는 가지가 그 갈림길의 뒤쪽 가지
            self._link(j, (h + 180) % 360, self.edge)
        for s in signs or []:
            c = (h + rel.get(s["dir"], 0)) % 360
            self.junctions[j]["signs"].setdefault(c, set()).update(s["items"] + ([self.target] if s.get("match") else []))
        return j

    def see_number(self, n, m, heading):
        """확정 문패 번호. 사이에 아는 갈림길이 있으면 그 너머 가지에 적는다."""
        if self.pos is None or heading is None:
            self.edges[self.edge]["nums"].add(int(n))
            return
        h = _c4(heading)
        vx, vy = _vec(h)
        e = self.edge
        for i, j in enumerate(self.junctions):
            ahead = (j["p"][0] - self.pos[0]) * vx + (j["p"][1] - self.pos[1]) * vy
            side = abs((j["p"][0] - self.pos[0]) * vy - (j["p"][1] - self.pos[1]) * vx)
            if side <= AT_M and AT_M < ahead < (m or 0) - 1.0:
                e = self.branch.get((i, h))
                if e is None:
                    e = self._new_edge()
                    self._link(i, h, e)
                break
        self.edges[e]["nums"].add(int(n))

    def dead_end(self):
        """지금 가지의 끝까지 왔음 (깊이로 앞이 막힘, nav_graph와 같은 판단)."""
        if not self.edges[self.edge]["dead"]:
            self.edges[self.edge]["dead"] = True
            self.event("이 복도 가지는 끝까지 가 봄")

    def junction_behind(self, heading):
        """지금 복도에서 등 뒤쪽에 있는 가장 가까운 아는 갈림길 (없으면 None)."""
        if self.pos is None or heading is None:
            return None
        vx, vy = _vec(_c4(heading))
        best = None
        for i, j in enumerate(self.junctions):
            dx, dy = j["p"][0] - self.pos[0], j["p"][1] - self.pos[1]
            back, side = -(dx * vx + dy * vy), abs(dx * vy - dy * vx)
            if side <= AT_M and back > 0 and (best is None or back < best[0]):
                best = (back, i)
        return best[1] if best else None

    # ------------------------------------------------------------ 판단
    def _score(self, j, c, flow):
        """갈림길 j의 방위 c 가지에 목적지가 있을 가능성 (곱한 근거, 근거가 없으면 1)."""
        why, s = [], 1.0
        signs = self.junctions[j]["signs"]
        hit = {k for k, items in signs.items() if any(mentions(i, self.target) for i in items)}
        if hit:
            s *= SIGN if c in hit else 1 / SIGN
            why.append("표지판" + ("이 가리킴" if c in hit else " 다른 쪽"))
        e = self.branch.get((j, c))
        e = self.edges[e] if e is not None else None
        if e and e["nums"] and self.target.isdigit():
            t = int(self.target)
            if min(e["nums"]) <= t <= max(e["nums"]):
                s *= IN_RANGE
                why.append(f"본 번호 {min(e['nums'])}~{max(e['nums'])} 사이")
            else:
                gap = min(abs(t - n) for n in e["nums"])
                s *= math.exp(-gap / GAP)
                why.append(f"본 번호와 {gap} 차이")
        if flow and c == flow:
            s *= FLOW
            why.append("지금 복도 번호 흐름")
        if e and e["dead"]:
            s *= EXPLORED
            why.append("끝까지 가 봄")
        return s, why

    def _frontier(self, j, c):
        e = self.branch.get((j, c))
        if e is None:
            return True
        e = self.edges[e]
        return not e["dead"] and len(e["ends"]) < 2

    def _paths(self, start):
        """갈림길 start에서 다른 갈림길까지 거리와 첫 방위 (양 끝을 아는 가지만 길로)."""
        dist, first, todo = {start: 0.0}, {start: None}, [start]
        while todo:
            todo.sort(key=lambda k: dist[k])
            a = todo.pop(0)
            for (ja, c), e in self.branch.items():
                if ja != a or len(self.edges[e]["ends"]) < 2:
                    continue
                for b in self.edges[e]["ends"] - {a}:
                    nd = dist[a] + math.dist(self.junctions[a]["p"], self.junctions[b]["p"])
                    if nd < dist.get(b, math.inf):
                        dist[b], first[b] = nd, first[a] if first[a] is not None else c
                        todo.append(b)
        return dist, first

    def choose(self, j, heading, flow_rel=None):
        """갈림길 j에서 갈 방위와 그 이유. flow_rel: 지금 복도 번호 흐름 '앞쪽'/'뒤쪽' (nav_graph).
        돌려줌 {"rel": 'ahead'/'left'/'right'/'back', "why": str} 또는 None (근거도 경계도 없음)."""
        h = _c4(heading)
        flow = {"앞쪽": h, "뒤쪽": (h + 180) % 360}.get(flow_rel)
        dist, first = self._paths(j)
        best = None
        for k, dk in dist.items():
            for c in sorted(self.junctions[k]["open"]):
                if not self._frontier(k, c):
                    continue
                s, why = self._score(k, c, flow if k == j else None)
                u = s / (1 + dk / DIST)
                go = c if k == j else first[k]
                turn = {0: 0, 90: 1, 270: 1, 180: 2}[(go - h) % 360]  # 같으면 덜 도는 쪽 (곧장 > 옆 > 뒤)
                if best is None or u > best[0] + 1e-9 or (abs(u - best[0]) <= 1e-9 and turn < best[5]):
                    best = (u, go, k, c, why, turn)
        if best is None:
            return None
        rel = {0: "ahead", 90: "right", 180: "back", 270: "left"}[(best[1] - h) % 360]
        where = "이 갈림길" if best[2] == j else f"{dist[best[2]]:.0f}m 떨어진 갈림길"
        return {"rel": rel, "why": f"{where}의 {best[3]}° 가지 (가능성 {best[0]:.2f}"
                + (": " + ", ".join(best[4]) if best[4] else ": 근거 없음, 안 가 본 곳") + ")"}
