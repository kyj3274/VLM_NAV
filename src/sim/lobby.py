"""시뮬레이터 로비: 복도가 아니라 넓은 홀. 벽 네 면에 문이 둘러 있고, 사용자는 남쪽 출입문 안쪽에서 북쪽을 보고 출발.

  .venv/bin/python src/sim/lobby.py                    # 번호 문패, 크기 3종 × 30
  .venv/bin/python src/sim/lobby.py --mode name        # 이름만 (english도 됨)
  .venv/bin/python src/sim/lobby.py --show 24 3        # 24m 로비, 시드 3 한 번 자세히
  .venv/bin/python src/sim/lobby.py --islands 3        # 로비 가운데에 장소 3개 (안내데스크·카페 같은 섬)

좌표: (x 동, y 북) m, 로비는 [0, W] × [0, H] (H = W × 0.7). 문은 벽을 따라 DOOR_GAP마다, 번호는 벽을 따라 한 바퀴 이어짐.
가상 카메라는 world.py와 같은 기준 (시야 FOV, 읽는 거리 RANGE, 비스듬함 OBLIQUE).
깊이: 화면 열마다 벽까지 거리를 재서 가장 먼 열(far_x)과 거리(far_m), 문패 벽 쪽은 실제와 같은 core/depth.wall_side
  (복도와 달리 로비는 가장 먼 곳이 구석이라, 정면 벽 문패도 왼쪽/오른쪽으로 갈릴 수 있음).
가상 사용자: 안내대로 걷고, 안내가 없으면 직진. 벽 앞(WALL_M)에 닿으면 벽을 따라 왼쪽이나 오른쪽으로 돎
  (시각장애인이 넓은 곳에서 벽을 따라 가는 방법, shorelining). '왼쪽/오른쪽 복도로' 안내가 있었으면 그쪽, 없으면 무작위.
섬(--islands): 로비 가운데 사각 덩어리(안내데스크, 카페, 편의점…). 면 하나나 둘에 이름표가 있고, 시야를 가리며,
  사용자에게는 벽처럼 부딪혀 돌아가는 곳. 번호 모드에서도 섬은 이름 (실제 로비의 데스크·매장에 호실 번호는 드묾).
  목적지는 섬이 있으면 반은 섬 장소. 최단 거리는 곧은 거리로 잼 (섬을 돌아가는 길은 안 셈 → SPL이 조금 낮게 나옴).
실제 인식은 이 장소를 'lobby'로 볼 것이라 place='lobby'.
갈림길 칸(junction): 앞 벽이 JUNC_SEE 안이면 '앞은 막히고 옆이 트인 곳'(T자·구석)으로 floor.py와 같은 형식으로 줌
  (복도 끝 T자와 같은 모양. 실제 인식에는 아직 없음, floor.py와 같은 한계). --no-junction으로 끄면 복도 판단만.
"""

import argparse
import contextlib
import io
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import floor as fw  # noqa: E402
import bench_floor  # noqa: E402  (재생 화면용 머릿속 지도)
import navigator as nv  # noqa: E402  (core 경로를 넣음)
import depth  # noqa: E402  core: 깊이로 문패 벽 쪽 정하기
import world  # noqa: E402

DOOR_GAP = 4.0
WALL_M = 1.0       # 벽에서 이만큼 떨어져 걷고 돎
ENTRY_M = 3.0      # 출입문 양옆 이 거리 안에는 문 없음
SIZES = (16, 24, 36)
DT, MAX_T, SUCCESS_M = 1.5, 600, 3.0
JUNCTION = True
ISLAND_NAMES = [("안내데스크", "Information Desk"), ("카페", "Cafe"), ("편의점", "Convenience Store"), ("무인민원발급기", "Civil Service Kiosk"),
                ("ATM", "ATM"), ("분실물센터", "Lost and Found"), ("서점", "Bookstore"), ("휴대폰 충전소", "Charging Station")]
ISLAND_GAP = 3.0   # 섬과 벽·다른 섬 사이 최소 통로 (m)
ISLAND_RANGE = 12.0  # 섬 간판(카페·안내데스크)은 크게 걸려 이 거리까지 읽힌다고 둠 (가정, 잰 값 아님. 문패는 RANGE 5)


def make_lobby(W, mode="number", rng=None, floor=4, islands=0):
    rng = rng or random.Random(0)
    H = round(W * 0.7)
    # 벽을 시계 방향으로: 남(서→동)… 대신 둘레 거리 s로 놓고 점·안쪽 법선을 구함
    walls = [((0, 0), (1, 0), (0, 1), W), ((W, 0), (0, 1), (-1, 0), H), ((W, H), (-1, 0), (0, -1), W), ((0, H), (0, -1), (1, 0), H)]
    doors = []
    for (p0, d, n, L) in walls:
        for k in range(int((L - 2) // DOOR_GAP)):
            t = 2 + DOOR_GAP * (k + 0.5)
            p = (p0[0] + d[0] * t, p0[1] + d[1] * t)
            if p[1] == 0 and abs(p[0] - W / 2) < ENTRY_M:
                continue
            doors.append({"p": p, "n_in": n})
    k0 = rng.randrange(len(doors))
    doors = doors[k0:] + doors[:k0]
    if rng.random() < 0.5:
        doors = doors[::-1]
    names = [(n + s, e + s) for s in ["", " 2", " 3"] for n, e in zip(fw.NAMES, fw.NAMES_EN)]
    rng.shuffle(names)
    for k, d in enumerate(doors):
        ko, en = (str(floor * 100 + k + 1),) * 2 if mode == "number" else names[k]
        d["n"], d["ko"] = (en if mode == "english" else ko), ko
    blocks, inames = [], rng.sample(ISLAND_NAMES, len(ISLAND_NAMES)) if islands else []
    for _ in range(islands * 50):  # 겹치지 않게 자리 찾기
        if len(blocks) == islands:
            break
        bw, bh = rng.uniform(2, 5), rng.uniform(2, 4)
        x0, y0 = rng.uniform(ISLAND_GAP, W - ISLAND_GAP - bw), rng.uniform(ISLAND_GAP + 1, H - ISLAND_GAP - bh)
        b = (x0, y0, x0 + bw, y0 + bh)
        if all(b[0] > o[2] + ISLAND_GAP or b[2] < o[0] - ISLAND_GAP or b[1] > o[3] + ISLAND_GAP or b[3] < o[1] - ISLAND_GAP for o in blocks):
            blocks.append(b)
    for b in blocks:
        faces = [((b[0] + b[2]) / 2, b[1], (0, -1)), (b[2], (b[1] + b[3]) / 2, (1, 0)),
                 ((b[0] + b[2]) / 2, b[3], (0, 1)), (b[0], (b[1] + b[3]) / 2, (-1, 0))]
        ko, en = inames.pop()
        for x, y, n in rng.sample(faces, rng.choice([1, 2])):  # 같은 장소 이름표가 붙은 면 1~2개
            doors.append({"p": (x, y), "n_in": n, "island": True, "ko": ko, "n": en if mode == "english" else ko})
    return {"W": W, "H": H, "mode": mode, "doors": doors, "blocks": blocks}


def _ray_box(p, v, b):
    """p에서 방향 v로 사각형 b에 닿는 거리 (안 닿으면 None). 슬랩 방법."""
    t0, t1 = 0.0, math.inf
    for i in (0, 1):
        lo, hi = b[i], b[i + 2]
        if abs(v[i]) < 1e-9:
            if not lo <= p[i] <= hi:
                return None
            continue
        a, c = (lo - p[i]) / v[i], (hi - p[i]) / v[i]
        t0, t1 = max(t0, min(a, c)), min(t1, max(a, c))
    return t0 if t0 <= t1 else None


def _wrap(a):
    return (a + 180) % 360 - 180


def bearing(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def wall_dist(lb, p, c):
    """p에서 방위 c로 벽이나 섬까지 거리."""
    vx, vy = fw.vec(c)
    ts = []
    if vx > 1e-9: ts.append((lb["W"] - p[0]) / vx)
    if vx < -1e-9: ts.append(-p[0] / vx)
    if vy > 1e-9: ts.append((lb["H"] - p[1]) / vy)
    if vy < -1e-9: ts.append(-p[1] / vy)
    ts += [t for b in lb.get("blocks", []) if (t := _ray_box(p, (vx, vy), b)) is not None]
    return min(ts)


def hidden(lb, a, q):
    """a에서 q까지 사이에 섬이 가로막나."""
    v, L = (q[0] - a[0], q[1] - a[1]), math.dist(a, q)
    return any((t := _ray_box(a, v, b)) is not None and t < 1 - 0.05 / max(L, 0.1) for b in lb.get("blocks", []))


def observe(lb, w, target):
    cam, plates, hit = (w.b - w.yaw) % 360, [], None
    cols = [(wall_dist(lb, w.p, (cam + a) % 360), 0.5 + a / world.FOV) for a in range(-30, 31, 2)]  # 오른쪽(+방위)이 화면 오른쪽
    far_m, far_x = max(cols)
    info = {"far_m": round(far_m, 1), "far_x": round(far_x, 3)}
    for d in lb["doors"]:
        dist = math.dist(w.p, d["p"])
        brg = bearing(w.p, d["p"])
        rel = _wrap(cam - brg)  # 왼쪽 +
        back = bearing(d["p"], w.p)
        oblique = abs(_wrap(back - bearing((0, 0), d["n_in"])))
        if dist > (ISLAND_RANGE if d.get("island") else world.RANGE) or abs(rel) > world.FOV / 2 or oblique > world.OBLIQUE:
            continue
        if hidden(lb, w.p, (d["p"][0] + d["n_in"][0] * 0.05, d["p"][1] + d["n_in"][1] * 0.05)):
            continue
        cx = min(max(50 - rel / world.FOV * 100, 3), 97)
        side = "L" if cx < 100 / 3 else "R" if cx > 200 / 3 else "C"
        q = {"img": None, "n": d["n"], "name": "", "side": side, "m": round(dist, 1), "on": "door",
             "box": [round(cx - 3), 30, round(cx + 3), 36],
             "wall": depth.wall_side(info, cx / 100), "dm": round(dist, 2)}
        if d["ko"] == target and not d["n"].isdigit():
            hit = q
        plates.append(q)
    plates.sort(key=lambda p: p["m"])
    obs = world.observe({"doors": [], "length": 1}, 0, 1)  # 빈 칸 모양만 빌림
    obs.update(plates=[p for p in plates if p["n"].isdigit()][:world.MAX_PLATES], place="lobby",
               depth=info)
    ahead = wall_dist(lb, w.p, w.b) - WALL_M
    if JUNCTION and not w.yaw and ahead <= fw.JUNC_SEE:
        vx, vy = fw.vec(w.b)
        q = (w.p[0] + vx * ahead, w.p[1] + vy * ahead)
        opens = [r for r, t in (("left", -90), ("right", 90)) if wall_dist(lb, q, (w.b + t) % 360) - WALL_M > 0.5]
        obs["junction"] = {"m": round(ahead, 1), "open": opens, "signs": []}
    if hit:
        obs["dest"] = {"seen": True, "side": hit["side"], "m": hit["m"], "why": "plate"}
    elif not target.isdigit():
        obs["dest"] = {"seen": False, "side": "none", "m": None, "why": "none"}
    return obs


class LobbyWalker:
    def __init__(self, lb, p, b, rng):
        self.lb, self.p, self.b, self.rng = lb, p, b, rng
        self.yaw, self.sweep, self.hold, self.plan = 0.0, [], False, None
        self.walked, self.end_turns = 0.0, 0

    def hear(self, say):
        text = " ".join(say)
        if not text:
            return
        self.hold = "멈춰" in text
        if "뒤로 돌아" in text:
            self.b, self.sweep, self.plan = (self.b + 180) % 360, [], None
        for k, p in (("왼쪽 복도", -90), ("오른쪽 복도", 90)):
            if k in text:
                self.plan = p
        if "한 바퀴" in text:  # 제자리에서 휴대폰을 들고 한 바퀴 (왼쪽으로 45°씩, 마지막 사진은 처음 방향)
            self.sweep = list(range(45, 360, 45))
            return
        clock = re.search(r"(\d+)시 방향", text)
        if clock and "비춰" not in text:  # 'N시 방향 약 M미터': 그쪽으로 몸을 돌림 (넓은 곳에서는 정면이 아닐 수 있음)
            self.b = (self.b + (int(clock.group(1)) % 12) * 30) % 360
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
        free = wall_dist(self.lb, self.p, self.b) - WALL_M
        if free <= 0.05:  # 벽 앞: 벽을 따라 돎 (막히지 않은 쪽). 비스듬히 걷다 닿았으면 먼저 벽과 나란한 축으로
            self.b = round(self.b / 90) * 90 % 360
            ok = [t for t in (-90, 90) if wall_dist(self.lb, self.p, (self.b + t) % 360) - WALL_M > 0.5]
            t = self.plan if self.plan in ok else self.rng.choice(ok) if ok else 180
            self.b, self.plan, self.end_turns = (self.b + t) % 360, None, self.end_turns + 1
            free = wall_dist(self.lb, self.p, self.b) - WALL_M
        step = min(d, max(free, 0.0))
        vx, vy = fw.vec(self.b)
        self.p = (self.p[0] + vx * step, self.p[1] + vy * step)
        self.walked += step

    def sensor(self):
        return {"steps": int(self.walked / world.STEP_M), "heading": (self.b - self.yaw) % 360}


def said_door(lb, w, text):
    """도착 문장이 가리키는 문: 말한 쪽(왼쪽/오른쪽, 없으면 아무 쪽)에서 가장 가까운 문 (양옆 문이 같은 거리일 때 쪽으로 가림)."""
    side = 1 if "왼쪽" in text else -1 if "오른쪽" in text else 0
    ds = [d for d in lb["doors"] if side == 0 or _wrap(w.b - bearing(w.p, d["p"])) * side > 0]
    return min(ds or lb["doors"], key=lambda d: math.dist(d["p"], w.p))


def shortest(lb, a, d):
    """출발점에서 목적지 문 앞(벽에서 WALL_M)까지 곧은 거리 (넓은 곳이라 장애물 없음)."""
    return math.dist(a, (d["p"][0] + d["n_in"][0] * WALL_M, d["p"][1] + d["n_in"][1] * WALL_M))


def episode(W, seed, mode="number", show=False, trace=None, islands=0):
    rng = random.Random(seed)
    lb = make_lobby(W, mode, rng, islands=islands)
    isl = [d for d in lb["doors"] if d.get("island")]
    tgt = rng.choice(isl) if isl and rng.random() < 0.5 else rng.choice([d for d in lb["doors"] if not d.get("island")])
    start = (W / 2, WALL_M)
    w = LobbyWalker(lb, start, 0, random.Random(seed + 1000))
    nav = nv.Navigator(world.SimRunner(random.Random(seed + 2000)), tgt["ko"], start_floor="4")
    nav.quick_max = 0
    nav.new_segment("1")
    sp = shortest(lb, start, tgt)
    out = {"layout": f"{W}m" + (f" 섬{islands}" if islands else ""), "island": bool(tgt.get("island")), "seed": seed, "target": tgt["ko"], "success": False, "wrong_arrival": False, "end": "시간 초과"}
    t = 0.0
    if trace is not None:
        trace.append({"lobby": lb, "target": tgt, "kind": f"로비 {W}m" + (f" 섬{islands}" if islands else ""), "seed": seed})
    while t <= MAX_T:
        obs = observe(lb, w, tgt["ko"])
        nav.latest = {"t": t, **w.sensor()}
        with contextlib.redirect_stdout(io.StringIO()):
            rec = nav.step(t=t, observed=world.wrap_obs(obs), **w.sensor())  # 웹처럼 찍을 때의 걸음 수·방위
        say, act = rec["say"], rec["action"]
        text = " ".join(say)
        seen = [p["n"] for p in obs["plates"]] + (["★"] if obs.get("dest", {}).get("seen") else [])
        events = [e["text"] for e in nav.mem.events if e["t"] == nav.clock and not e["text"].startswith("안내:")]
        if trace is not None:
            trace.append({"t": t, "p": w.p, "b": w.b, "yaw": w.yaw, "say": say, "act": act, "seen": seen, "events": events,
                          "walked": w.walked, "junction": obs.get("junction"), "mind": bench_floor.mind(nav, w.sensor()["heading"])})
        if show:
            print(f"{t:6.1f}s ({w.p[0]:5.1f},{w.p[1]:5.1f}) {w.b:3.0f}°{'↺' if w.yaw else ' '} [{','.join(seen)}] {act} | {' / '.join(say)}")
        if act == "도착" or "문패를 확인하지 못했습니다" in text:
            d = said_door(lb, w, text)
            ok = d["ko"] == tgt["ko"] and math.dist(d["p"], w.p) <= SUCCESS_M  # 섬은 면마다 같은 장소 이름표
            out.update(success=ok, wrong_arrival=not ok, end="도착" if act == "도착" else "가능성 높음")
            break
        if "찾지 못했습니다" in text:
            out["end"] = "못 찾음"
            break
        w.hear(say)
        w.move(DT)
        t = round(t + DT, 2)
    out.update(t=t, walked=round(w.walked, 1), shortest=round(sp, 1), extra=max(0.0, w.walked - sp),
               spl=sp / max(w.walked, sp) if out["success"] else 0.0, end_turns=w.end_turns)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    ap.add_argument("--mode", default="number", choices=["number", "name", "english"])
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--show", nargs=2, type=int, metavar=("W", "SEED"))
    ap.add_argument("--no-junction", action="store_true", help="벽 앞을 갈림길로 주지 않음 (복도 판단만)")
    ap.add_argument("--islands", type=int, default=0, help="로비 가운데 장소(섬) 수")
    a = ap.parse_args()
    world.RANGE = a.range
    global JUNCTION
    JUNCTION = not a.no_junction
    if a.show:
        print(episode(a.show[0], a.show[1], a.mode, show=True, islands=a.islands))
        return
    import bench_floor
    rows = [episode(W, s, a.mode, islands=a.islands) for W in a.sizes for s in range(a.n)]
    bench_floor.table(rows)
    if a.islands:
        for k, name in ((True, "섬 목적지"), (False, "벽 목적지")):
            rs = [r for r in rows if r["island"] == k]
            ok = [r for r in rs if r["success"]]
            print(f"{name}: {len(ok)}/{len(rs)} 성공, 잘못 {sum(r['wrong_arrival'] for r in rs)}, "
                  f"시간(중) {statistics.median(r['t'] for r in ok) if ok else float('nan'):.1f}초")


if __name__ == "__main__":
    main()
