"""시뮬레이터 v0: 직선 복도 하나, 완벽한 인식, 가상 사용자.

인식은 완벽하다고 두고 '보이는 것으로 잘 찾아가는가'(판단·안내)만 시험한다.
가상 카메라는 Claude 인식 결과와 같은 형식(claude_perceiver.clean의 출력)을 만들어 Navigator.step(observed=…)에 넣는다.

좌표: 복도는 x축 (0 ~ length m), 폭 WIDTH. 벽 A는 y = +WIDTH/2, 벽 B는 y = −WIDTH/2.
사용자는 복도 가운데(y = 0)를 걷고, 방향은 +x(heading 1) 또는 −x(heading −1).
"""

import math
import random
from pathlib import Path

WIDTH = 2.4        # 복도 폭 (m)
FOV = 60           # 카메라 가로 시야각 (도)
RANGE = 8.0        # 문패를 읽을 수 있는 최대 거리 (m). 실험 건물 영상(720p)과 맞춘 값은 5 (check_video.py)
OBLIQUE = 80       # 벽에 붙은 문패를 이 각도(벽의 법선 기준)보다 비스듬히 보면 못 읽음 (도)
MAX_PLATES = 4     # 한 장에서 받는 문패 수 (claude_perceiver.clean과 같음)
WALK_MPS = 0.8     # 걷는 속도
STEP_M = 0.6       # 걸음 하나
# 인식 오류 (v2). 0이면 완벽한 인식. --err 1(약함)·2(심함) 또는 하나씩 지정
ERR = {"miss": 0.0,      # 보이는 문패를 못 읽을 확률 (사진마다)
       "misread": 0.0,   # 문마다 정해진 '잘못 읽히는 번호'로 읽을 확률 (413→418처럼 같은 문은 같은 식으로 틀림)
       "fake": 0.0,      # 사진마다 가짜 문패(포스터·안내도의 숫자)가 하나 섞일 확률
       "wall": 0.0,      # 깊이로 정한 왼쪽/오른쪽이 반대일 확률
       "dist": 0.0,      # 거리 오차 (표준편차 비율)
       "recheck_ok": 1.0}  # 확대 재확인이 실제 번호를 읽을 확률
ERR_PRESETS = {0: {}, 1: {"miss": 0.2, "misread": 0.05, "fake": 0.02, "wall": 0.1, "dist": 0.15, "recheck_ok": 0.95},
               2: {"miss": 0.4, "misread": 0.15, "fake": 0.05, "wall": 0.2, "dist": 0.3, "recheck_ok": 0.9}}
USE_HEADING = True  # 휴대폰 방위를 보냄 (--no-heading으로 끔)
SWEEP = (30, 60, 90, 120)  # "○쪽으로 천천히 돌려 비춰 주세요": 사진마다 이만큼 돌림 (도, 앞에서 옆을 지나 조금 뒤까지)


# ------------------------------------------------------------ 건물
LAYOUTS = ("odd_even", "one_side", "u_shape", "skip", "uneven")
BUILDINGS = Path(__file__).resolve().parent / "buildings"
REAL = tuple(sorted(p.stem for p in BUILDINGS.glob("*.json")))  # 손으로 옮긴 실제 건물


def make_corridor(layout, n=10, floor=4, rng=None):
    """복도 하나를 만든다. 돌려줌: {"length", "layout", "doors": [{"x", "wall", "n"}]}.
    odd_even: 벽 A 홀수·벽 B 짝수, 같은 방향으로 증가 (흔한 배치)
    one_side: 한쪽 벽에만 연속 번호
    u_shape:  벽 A는 +x로 증가, 벽 B는 끝에서 돌아오며 이어짐 (복도를 한 바퀴 도는 번호)
    skip:     odd_even에서 번호를 1~4씩 건너뜀
    uneven:   odd_even에서 문 간격이 2.5~7m로 제각각"""
    rng = rng or random.Random(0)
    base = floor * 100
    gap = [4.0] * n if layout != "uneven" else [rng.uniform(2.5, 7.0) for _ in range(n)]
    xs = [3.0 + sum(gap[:i]) for i in range(n)]
    length = xs[-1] + 3.0
    doors = []
    if layout in ("odd_even", "uneven"):
        for i, x in enumerate(xs):
            doors += [{"x": x, "wall": "A", "n": str(base + 2 * i + 1)}, {"x": x, "wall": "B", "n": str(base + 2 * i + 2)}]
    elif layout == "one_side":
        doors = [{"x": x, "wall": "A", "n": str(base + i + 1)} for i, x in enumerate(xs)]
    elif layout == "u_shape":
        doors = [{"x": x, "wall": "A", "n": str(base + i + 1)} for i, x in enumerate(xs)]
        doors += [{"x": x, "wall": "B", "n": str(base + 2 * n - i)} for i, x in enumerate(xs)]
    elif layout == "skip":
        k = 1
        for x in xs:
            doors += [{"x": x, "wall": "A", "n": str(base + k)}, {"x": x, "wall": "B", "n": str(base + k + 1)}]
            k += 2 * rng.randint(1, 4)
    else:
        raise ValueError(layout)
    return {"length": length, "layout": layout, "doors": doors}


def load_building(path):
    """손으로 옮긴 실제 건물 (buildings/*.json). make_corridor와 같은 모양에 start_x·speed·name을 더함."""
    import json
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"length": b["length"], "layout": Path(path).stem, "name": b["name"], "start_x": b["start_x"],
            "speed": b["speed"], "video": b.get("video"),
            "doors": [{"x": d["x"], "wall": d["wall"], "n": d["n"]} for d in b["doors"]]}


def wall_y(wall):
    return WIDTH / 2 if wall == "A" else -WIDTH / 2


# ------------------------------------------------------------ 가상 카메라
def _wrap(a):
    return (a + 180) % 360 - 180


def far_m(corr, x, cam, cap=20.0):
    """깊이 모델의 far_m 흉내: 시야 안 광선들이 벽·복도 끝에 닿는 거리 중 가장 먼 것."""
    best = 0.0
    for k in range(int(FOV) + 1):
        a = math.radians(cam - FOV / 2 + k)
        dx, dy = math.cos(a), math.sin(a)
        ts = [cap]
        if dy:
            ts.append((WIDTH / 2 if dy > 0 else -WIDTH / 2) / dy)
        if dx:
            ts.append(((corr["length"] if dx > 0 else 0.0) - x) / dx)
        best = max(best, min(t for t in ts if t > 0))
    return round(best, 1)


def misreading(n, rng):
    """이 문패를 잘못 읽으면 나오는 번호 (끝 두 자리 중 하나가 바뀜, 가끔 첫 자리도)."""
    i = rng.choices([0, 1, 2], weights=[0.1, 0.45, 0.45])[0]
    d = rng.choice([c for c in "0123456789" if c != n[i] and not (i == 0 and c == "0")])
    return n[:i] + d + n[i + 1:]


def add_errors(corr, obs, rng):
    """완벽한 인식 결과에 ERR대로 오류를 섞음. 각 문패에 진짜 번호(true)를 남겨 확대 재확인이 씀."""
    alt = corr.setdefault("_alt", {d["n"]: misreading(d["n"], random.Random(f"{d['n']}{d['x']}")) for d in corr["doors"]})
    out = []
    for p in obs["plates"]:
        if rng.random() < ERR["miss"]:
            continue
        q = {**p, "true": p["n"]}
        if rng.random() < ERR["misread"]:
            q["n"] = alt[p["n"]]
        if ERR["dist"]:
            q["m"] = round(max(0.5, p["m"] * rng.lognormvariate(0, ERR["dist"])), 1)
            q["dm"] = q["m"]
        if q["wall"] in ("왼쪽", "오른쪽") and rng.random() < ERR["wall"]:
            q["wall"] = "오른쪽" if q["wall"] == "왼쪽" else "왼쪽"
        out.append(q)
    if rng.random() < ERR["fake"]:
        fl = corr["doors"][0]["n"][0] if rng.random() < 0.5 else str(rng.randint(1, 9))
        cx = rng.uniform(10, 90)
        out.append({"img": None, "n": f"{fl}{rng.randint(1, 30):02d}", "name": "", "side": "L" if cx < 33 else "R" if cx > 67 else "C",
                    "m": round(rng.uniform(2, 6), 1), "on": rng.choice(["wall", "door"]), "box": [round(cx - 3), 30, round(cx + 3), 36],
                    "wall": rng.choice(["왼쪽", "오른쪽"]), "dm": None, "true": None})
    obs["plates"] = sorted(out, key=lambda p: p["m"])[:MAX_PLATES]
    return obs


def observe(corr, x, heading, yaw=0.0, all_plates=False):
    """위치 x, 몸 방향 heading(±1), 카메라를 몸 기준 yaw도(왼쪽 +) 돌린 상태에서 보이는 것 → Claude 인식 결과 형식.
    완벽한 인식: 시야·거리·각도 안의 문패는 번호·거리를 정확히 읽는다.
    wall(깊이로 정하는 벽 쪽)은 몸 방향 기준 '왼쪽'/'오른쪽', 옆을 비출 때는 깊이로 복도 방향을 못 잡으므로 '?'.
    all_plates: 시야·거리를 따지지 않고 앞쪽 모든 문패 (실제 영상에 보인 문패의 거리·위치를 지도로 잴 때)."""
    body = 0.0 if heading > 0 else 180.0
    cam = body + yaw
    plates = []
    for d in corr["doors"]:
        dx, dy = d["x"] - x, wall_y(d["wall"])
        dist = math.hypot(dx, dy)
        rel = _wrap(math.degrees(math.atan2(dy, dx)) - cam)
        oblique = math.degrees(math.atan2(abs(dx), abs(dy)))  # 벽 법선과 시선 사이 각도
        if all_plates:
            if abs(rel) >= 90:
                continue
        elif dist > RANGE or abs(rel) > FOV / 2 or oblique > OBLIQUE:
            continue
        cx = min(max(50 - rel / FOV * 100, 3), 97)  # 왼쪽(+)이 화면 왼쪽
        side = "L" if cx < 100 / 3 else "R" if cx > 200 / 3 else "C"
        left = math.sin(math.radians(math.degrees(math.atan2(dy, dx)) - body)) > 0
        plates.append({"img": None, "n": d["n"], "name": "", "side": side, "m": round(dist, 1), "on": "door",
                       "box": [round(cx - 3), 30, round(cx + 3), 36],
                       "wall": "?" if yaw else ("왼쪽" if left else "오른쪽"), "dm": round(dist, 2)})
    plates.sort(key=lambda p: p["m"])
    return {"plates": plates if all_plates else plates[:MAX_PLATES], "depth": {"far_m": far_m(corr, x, cam), "far_x": 0.5},
            "unread": [], "floor_sign": "", "floor": None, "place": "corridor",
            "stair_sign": "", "stair_floors": [], "door_ahead": "none",
            "elev": {"seen": False, "cand": False, "why": "none", "side": "none", "m": None, "btn": "none"},
            "arrows": [], "fixes": [], "elev_floor": None}


class SimRunner:
    """Navigator가 부르는 runner 자리. 인식 결과는 바깥에서 observed로 넣고, 확대 재확인은 완벽히 일치."""

    def observe(self, frames):
        raise RuntimeError("시뮬레이터는 step(observed=…)로만 부름")

    def __init__(self, rng=None):
        self.rng = rng or random.Random(0)

    def recheck(self, frame, plate):
        """원본 해상도로 잘라 다시 읽기: ERR['recheck_ok'] 확률로 진짜 번호(가짜 문패면 읽을 번호 없음)."""
        true = plate.get("true", plate["n"])
        n = true if self.rng.random() < ERR["recheck_ok"] else plate["n"]
        return {"n": n or None, "name": "", "match": (n == plate["n"]) if plate["n"] and n else None, "cached": True,
                "call": {"seconds": 0.0, "credit": 0.0}}


def wrap_obs(obs):
    return {"key": "sim.jpg", "keys": ["sim.jpg"], "pick_seconds": 0.0, "obs": obs, "cached": True,
            "call": {"seconds": 0.0, "credit": 0.0}}


# ------------------------------------------------------------ 가상 사용자
class Walker:
    """안내 문장·행동을 글자 그대로 따르는 사용자.
    '뒤로 돌아' → 180° 돎, '멈춰 주세요' → 다음 안내까지 서 있음, '○쪽으로 … 비춰' → 그 자리에서 SWEEP대로 천천히 돌려 비춤,
    그 밖에는 직진. 복도 끝에 닿으면 스스로 돌아섬(사람이 벽 앞에서 하는 행동).
    spont: 사진마다 이 확률로 이유 없이 돌아섬 (헷갈림·피하기 등. 판단이 '돎 = 복도 끝'처럼 행동에 기대지 않는지 시험)."""

    def __init__(self, corr, x, heading, spont=0.0, rng=None):
        self.corr, self.x, self.heading = corr, x, heading
        self.spont, self.rng = spont, rng or random.Random(0)
        self.spont_turns = 0
        self.yaw, self.sweep, self.hold = 0.0, [], False
        self.walked, self.turns, self.end_turns = 0.0, 0, 0

    def hear(self, say):
        text = " ".join(say)
        if not text:
            return
        self.hold = "멈춰" in text  # 새 안내가 오면 그 안내를 따름 (멈춤은 다음 안내까지)
        if "뒤로 돌아" in text:
            self.heading, self.turns = -self.heading, self.turns + 1
            self.sweep = []
        if "비춰" in text and "좌우로" in text:  # 정면 근처: 오른쪽·왼쪽 번갈아
            self.sweep = [s * a for a in SWEEP[:2] for s in (-1, 1)]
        elif "비춰" in text and ("왼쪽으로" in text or "오른쪽으로" in text):
            sign = 1 if "왼쪽으로" in text else -1
            self.sweep = [sign * a for a in SWEEP]
        elif not self.hold:
            self.sweep = []  # 다른 안내(예: '천천히 가세요')를 들으면 비추기를 멈추고 따름

    def move(self, dt):
        """다음 사진까지 dt초 동안 움직임. 비추는 중이면 카메라만 다음 각도로."""
        if self.sweep:
            self.yaw = float(self.sweep.pop(0))
            return
        self.yaw = 0.0
        if self.hold:
            return
        if self.spont and self.rng.random() < self.spont:
            self.heading, self.spont_turns = -self.heading, self.spont_turns + 1
            return
        nx = self.x + self.heading * WALK_MPS * dt
        if nx < 0 or nx > self.corr["length"]:
            nx = min(max(nx, 0.0), self.corr["length"])
            self.heading, self.end_turns = -self.heading, self.end_turns + 1
        self.walked += abs(nx - self.x)
        self.x = nx

    def sensor(self):
        """휴대폰 센서. 방위는 휴대폰이 향한 쪽(몸 방향 + 옆을 비춘 각도)이라 비출 때도 바뀐다. use_heading=False면 방위 없음."""
        h = ((0 if self.heading > 0 else 180) - self.yaw) % 360 if USE_HEADING else None
        return {"steps": int(self.walked / STEP_M), "heading": h}
