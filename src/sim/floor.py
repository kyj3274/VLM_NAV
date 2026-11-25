"""시뮬레이터 v3: 복도 여러 개가 갈림길로 이어진 층.

좌표: 평면 (x 동, y 북) m. 복도는 시작점 p0, 방위 ang(0 북, 90 동), 길이 length. 복도 안 위치는 시작점에서 잰 x.
복도 하나는 world.py의 복도와 같은 모양이라 가상 카메라(world.observe)를 그대로 쓴다 (벽 A는 +방향 왼쪽).
갈림길: 수직인 두 복도가 만나는 점. 깊이로 '앞 m미터에 왼쪽/오른쪽/앞이 트임'을 본다고 둔다 (obs["junction"]).
  실제 인식(claude_perceiver)에는 아직 이 칸이 없다. 깊이 지도에서 옆이 트인 곳을 찾는 인식이 따로 필요하다.
문패: 'number' 번호 (복도마다 번호가 이어짐, 복도마다 증가 방향은 무작위), 'name' 이름만 (번호로 방향을 알 수 없음),
  'english' 영어 이름만 (사용자는 한국어로 '세미나실'이라 말함. 같은 실인지는 Claude가 목적지를 알고 보므로 dest·표지판 match로 판단한다고 둠).
표지판: 갈림길마다 signs 확률로, 가지마다 그 복도의 번호 범위(번호) 또는 실 이름들(이름)을 화살표로 보여 줌.
"""

import math
import random

import world

JUNC_SEE = 6.0     # 갈림길을 이 거리(m) 안에서 앞쪽에 보면 obs["junction"]
DOOR_GAP = 4.0     # 문 간격
OPEN_M = 2.5       # 갈림길에서 이 거리 안에는 문이 없음 (트인 곳)
FLOORS = {         # 복도들: (시작점, 방위, 길이)
    "L": [((0, 0), 90, 30), ((30, 0), 0, 24)],
    "T": [((0, 0), 90, 40), ((20, 0), 0, 24)],
    "plus": [((0, 0), 90, 40), ((20, -16), 0, 32)],
    "H": [((0, 0), 90, 36), ((0, 20), 90, 36), ((18, 0), 0, 20)],
    "comb": [((0, 0), 90, 48), ((12, 0), 0, 16), ((36, 0), 0, 16)],
    "loop": [((0, 0), 90, 30), ((30, 0), 0, 20), ((0, 20), 90, 30), ((0, 0), 0, 20)],
}
NAMES = ["학과사무실", "세미나실", "회의실", "자료실", "휴게실", "전산실", "창고", "기계실", "강의실", "실습실", "상담실",
         "행정실", "독서실", "탕비실", "방송실", "문서고", "준비실", "연구실", "교수실", "대학원실", "샤워실", "장비실"]
NAMES_EN = ["Department Office", "Seminar Room", "Meeting Room", "Archive", "Lounge", "Computer Room", "Storage", "Machine Room",
            "Lecture Room", "Lab", "Counseling Room", "Admin Office", "Reading Room", "Pantry", "Broadcast Room", "Records Room",
            "Prep Room", "Research Lab", "Faculty Office", "Graduate Room", "Shower Room", "Equipment Room"]


def vec(c):
    r = math.radians(c)
    return math.sin(r), math.cos(r)


def point(cor, x):
    vx, vy = vec(cor["ang"])
    return cor["p0"][0] + vx * x, cor["p0"][1] + vy * x


def make_floor(kind, mode="number", signs=0.0, rng=None, floor=4):
    rng = rng or random.Random(0)
    cors = [{"id": i, "p0": p0, "ang": ang, "length": L, "doors": [], "junc": []} for i, (p0, ang, L) in enumerate(FLOORS[kind])]
    # 갈림길: 수직인 두 복도의 교점이 둘 다의 위(끝 포함)에 있으면
    for a in cors:
        for b in cors:
            if a["id"] >= b["id"] or (a["ang"] - b["ang"]) % 180 == 0:
                continue
            ax = (b["p0"][0] - a["p0"][0]) * vec(a["ang"])[0] + (b["p0"][1] - a["p0"][1]) * vec(a["ang"])[1]
            bx = (a["p0"][0] - b["p0"][0]) * vec(b["ang"])[0] + (a["p0"][1] - b["p0"][1]) * vec(b["ang"])[1]
            if -0.01 <= ax <= a["length"] + 0.01 and -0.01 <= bx <= b["length"] + 0.01:
                a["junc"].append({"x": ax, "to": b["id"], "tx": bx, "signs": None})
                b["junc"].append({"x": bx, "to": a["id"], "tx": ax, "signs": None})
    for c in cors:
        xs = [x for x in [3.0 + DOOR_GAP * k for k in range(int(c["length"] // DOOR_GAP) + 1)] if x <= c["length"] - 2]
        xs = [x for x in xs if all(abs(x - j["x"]) > OPEN_M for j in c["junc"])]
        if rng.random() < 0.5:
            xs = xs[::-1]  # 번호가 커지는 방향
        c["doors"] = [{"x": x, "wall": w} for x in xs for w in "AB"]
    names = [(n + s, e + s) for s in ["", " 2", " 3", " 4"] for n, e in zip(NAMES, NAMES_EN)]
    rng.shuffle(names)
    k = 1
    for c in cors:
        for d in c["doors"]:
            ko, en = (str(floor * 100 + k),) * 2 if mode == "number" else names[k - 1]
            d["n"], d["ko"] = (en if mode == "english" else ko), ko
            k += 1
    for c in cors:
        for j in c["junc"]:
            if rng.random() < signs:
                j["signs"] = True
    return {"kind": kind, "mode": mode, "corridors": cors}


def branch_items(fl, c, x, sign):
    """복도 c의 x에서 sign(±1) 쪽 문들의 표지판 항목: 번호면 '최소~최대', 이름이면 이름들 (영어 모드는 영어)."""
    ds = [d["n"] for d in fl["corridors"][c]["doors"] if (d["x"] - x) * sign > 0]
    if not ds:
        return []
    if fl["mode"] == "number":
        ns = sorted(int(n) for n in ds)
        return [f"{ns[0]}~{ns[-1]}"]
    return sorted(set(ds))


def opens(fl, c, j, h):
    """갈림길 j(복도 c 위)에서 몸 방향 h로 섰을 때 트인 쪽: {'ahead','left','right'} → (복도, x, 방향)."""
    cor, other = fl["corridors"][c], fl["corridors"][j["to"]]
    body = (cor["ang"] + (0 if h > 0 else 180)) % 360
    out = {}
    if (h > 0 and j["x"] < cor["length"] - 0.5) or (h < 0 and j["x"] > 0.5):
        out["ahead"] = (c, j["x"], h)
    for rel, d in (("left", (body - 90) % 360), ("right", (body + 90) % 360)):
        s = 1 if d == other["ang"] else -1 if d == (other["ang"] + 180) % 360 else 0
        if (s > 0 and j["tx"] < other["length"] - 0.5) or (s < 0 and j["tx"] > 0.5):
            out[rel] = (other["id"], j["tx"], s)
    return out


def observe(fl, w, target):
    """가상 사용자 w가 보는 것 (Claude 인식 형식). 이름 문패는 목적지일 때만 dest로 (번호처럼 흐름을 주지 않음)."""
    cor = fl["corridors"][w.c]
    obs = world.observe(cor, w.x, w.h, w.yaw)
    plates = [p for p in obs["plates"] if p["n"].isdigit()]
    ko = {d["n"]: d["ko"] for d in cor["doors"]}
    hit = [p for p in obs["plates"] if ko[p["n"]] == target and not p["n"].isdigit()]
    obs["plates"] = plates
    if hit:
        obs["dest"] = {"seen": True, "side": hit[0]["side"], "m": hit[0]["m"], "why": "plate"}
    elif not target.isdigit():
        obs["dest"] = {"seen": False, "side": "none", "m": None, "why": "none"}
    if not w.yaw:
        ahead = [j for j in cor["junc"] if -0.3 <= (j["x"] - w.x) * w.h <= JUNC_SEE]
        if ahead:
            j = min(ahead, key=lambda j: abs(j["x"] - w.x))
            op = opens(fl, w.c, j, w.h)
            signs = []
            if j["signs"]:
                for rel, (c2, x2, s2) in op.items():
                    kos = [d["ko"] for d in fl["corridors"][c2]["doors"] if (d["x"] - x2) * s2 > 0]
                    signs.append({"dir": rel, "items": branch_items(fl, c2, x2, s2), "match": target in kos})
            obs["junction"] = {"m": round(abs(j["x"] - w.x), 1), "open": sorted(op), "signs": signs}
    return obs


class FloorWalker:
    """world.Walker와 같은 사용자에 갈림길을 더함: '왼쪽/오른쪽 복도로' → 다음 갈림길에서 그쪽으로, '갈림길 … 직진' → 곧장.
    안내가 없으면 갈림길에서 곧장, 곧장이 막혔으면 트인 쪽 (둘이면 무작위)."""

    def __init__(self, fl, c, x, h, rng=None):
        self.fl, self.c, self.x, self.h = fl, c, x, h
        self.rng = rng or random.Random(0)
        self.yaw, self.sweep, self.hold, self.plan = 0.0, [], False, None
        self.walked, self.turns, self.end_turns = 0.0, 0, 0

    def hear(self, say):
        text = " ".join(say)
        if not text:
            return
        self.hold = "멈춰" in text
        if "뒤로 돌아" in text:
            self.h, self.turns, self.sweep, self.plan = -self.h, self.turns + 1, [], None
        for k, p in (("왼쪽 복도", "left"), ("오른쪽 복도", "right")):
            if k in text:
                self.plan = p
        if "갈림길" in text and "직진" in text:
            self.plan = "ahead"
        if "비춰" in text and "좌우로" in text:  # 정면 근처: 오른쪽·왼쪽 번갈아
            self.sweep = [s * a for a in world.SWEEP[:2] for s in (-1, 1)]
        elif "비춰" in text and ("왼쪽으로" in text or "오른쪽으로" in text):
            self.sweep = [(1 if "왼쪽으로" in text else -1) * a for a in world.SWEEP]
        elif not self.hold:
            self.sweep = []

    def move(self, dt):
        if self.sweep:
            self.yaw = float(self.sweep.pop(0))
            return
        self.yaw = 0.0
        if self.hold:
            return
        d = world.WALK_MPS * dt
        cor = self.fl["corridors"][self.c]
        js = [j for j in cor["junc"] if 0 <= (j["x"] - self.x) * self.h <= d]
        if js:
            j = min(js, key=lambda j: abs(j["x"] - self.x))
            d -= abs(j["x"] - self.x)
            self.walked += abs(j["x"] - self.x)
            self.x = j["x"]
            op = opens(self.fl, self.c, j, self.h)
            p = self.plan if self.plan in op else "ahead" if "ahead" in op else self.rng.choice(sorted(op)) if op else None
            self.plan = None
            if p is None:  # 막힌 곳
                self.h, self.end_turns = -self.h, self.end_turns + 1
                return
            self.c, self.x, self.h = op[p]
            cor = self.fl["corridors"][self.c]
        nx = self.x + self.h * d
        if nx < 0 or nx > cor["length"]:
            nx = min(max(nx, 0.0), cor["length"])
            self.h, self.end_turns = -self.h, self.end_turns + 1
        self.walked += abs(nx - self.x)
        self.x = nx

    def sensor(self):
        body = (self.fl["corridors"][self.c]["ang"] + (0 if self.h > 0 else 180)) % 360
        return {"steps": int(self.walked / world.STEP_M), "heading": (body - self.yaw) % 360 if world.USE_HEADING else None}


def shortest(fl, a, b):
    """복도를 따라 (복도, x) a에서 b까지 최단 거리."""
    nodes = {}  # (복도, x) → 점 번호. 갈림길은 두 복도에서 같은 점
    def nid(c, x):
        p = point(fl["corridors"][c], x)
        for k, q in nodes.items():
            if math.dist(q, p) < 0.05:
                return k
        nodes[len(nodes)] = p
        return len(nodes) - 1
    on = {c["id"]: [] for c in fl["corridors"]}
    for c in fl["corridors"]:
        for j in c["junc"]:
            on[c["id"]].append((j["x"], nid(c["id"], j["x"])))
    sa, sb = nid(*a), nid(*b)
    on[a[0]].append((a[1], sa))
    on[b[0]].append((b[1], sb))
    adj = {}
    for c, pts in on.items():
        pts.sort()
        for (x1, n1), (x2, n2) in zip(pts, pts[1:]):
            adj.setdefault(n1, []).append((n2, x2 - x1))
            adj.setdefault(n2, []).append((n1, x2 - x1))
    dist, todo = {sa: 0.0}, [sa]
    while todo:
        todo.sort(key=lambda k: dist[k])
        u = todo.pop(0)
        for v, w in adj.get(u, []):
            if dist[u] + w < dist.get(v, math.inf):
                dist[v] = dist[u] + w
                todo.append(v)
    return dist.get(sb, math.inf)
