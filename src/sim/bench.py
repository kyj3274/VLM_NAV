"""시뮬레이터 v0 실행: 지금 Claude 버전 Navigator(src/nav)를 그대로 넣고 성공률·틀린 방향 등을 잰다.

  .venv/bin/python src/sim/bench.py                    # 배치 5종 × 40회
  .venv/bin/python src/sim/bench.py --n 100 --lat 2    # 인식 결과가 2초 늦게 도착
  .venv/bin/python src/sim/bench.py --show u_shape 3   # 한 번을 걸음마다 출력

지표 (한 번 = 시작 위치·방향·목적지 무작위)
  성공: 도착(또는 '가능성 높음')을 말한 곳에서 목적지 벽의 가장 가까운 문이 목적지 (SUCCESS_M 넘게 떨어지면 실패)
  잘못된 도착: 다른 곳에서 도착을 말함 (가장 나쁜 실패)
  틀린 방향: '앞쪽입니다'인데 실제로 뒤, '뒤쪽/지나친'인데 실제로 앞 (1m 넘게). 표에는 한 번이라도 있었던 실행 수
    틀린방향 = 사용자가 들을 때 기준, 판단오류 = 안내가 기준으로 삼은 때 기준 (지연 보정을 끄면 사진 찍은 때; 둘의 차이는 늦은 안내)
  SPL: 성공 × 최단 거리 ÷ 걸은 거리
"""

import argparse
import contextlib
import io
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import navigator as nv  # noqa: E402
import world  # noqa: E402

DT = 1.5          # 사진 간격 (겹쳐 보내기와 같음)
MAX_T = 240       # 이 초 안에 도착을 못 말하면 실패
SUCCESS_M = 4.0   # 도착을 말한 곳과 목적지 문의 x 거리가 이보다 멀면 실패
MIN_START = 4.0   # 시작 위치와 목적지 문 사이 최소 거리
FRESH = True      # 가장 최근 센서로 지연 보정 (--no-fresh로 끔)


def episode(layout, seed, lat=0.0, show=False, trace=None, spont=0.0):
    """trace: 리스트를 주면 걸음마다 화면용 기록을 넣음 (view.py)."""
    rng = random.Random(seed)
    corr = world.load_building(world.BUILDINGS / f"{layout}.json") if layout in world.REAL else world.make_corridor(layout, rng=rng)
    x0 = rng.uniform(0.5, corr["length"] - 0.5)
    tgt = rng.choice([d for d in corr["doors"] if abs(d["x"] - x0) >= MIN_START])
    walker = world.Walker(corr, x0, rng.choice([1, -1]), spont=spont, rng=random.Random(seed + 1000))
    nav = nv.Navigator(world.SimRunner(random.Random(seed + 2000)), tgt["n"], start_floor=tgt["n"][0])
    erng = random.Random(seed + 3000)
    nav.quick_max = 0  # 빠른 확인(휴대폰이 바로 2장 찍기)은 v0에서 흉내 내지 않음
    nav.new_segment("1")
    hist, t = [], 0.0
    out = {"layout": layout, "seed": seed, "target": tgt["n"], "success": False, "wrong_arrival": False,
           "wrong_dir": 0, "wrong_judge": 0, "end": "시간 초과", "says": 0}
    while t <= MAX_T:
        hist.append((t, walker.x, walker.heading, walker.yaw, walker.sensor()))
        # 지금 도착한 인식 결과는 lat초 전에 찍은 사진. 웹 서버처럼 찍은 시각과 그때의 걸음 수를 넘긴다
        cap = max((h for h in hist if h[0] <= t - lat + 1e-9), key=lambda h: h[0], default=hist[0])
        obs = world.add_errors(corr, world.observe(corr, cap[1], cap[2], cap[3]), erng)
        nav.latest = {"t": t, **walker.sensor()} if FRESH else None  # 방금 찍어 보낸 사진의 센서 (지연 보정)
        with contextlib.redirect_stdout(io.StringIO()):
            rec = nav.step(t=float(cap[0]), observed=world.wrap_obs(obs), **cap[4])  # 웹처럼 찍을 때의 걸음 수·방위
        say, act = rec["say"], rec["action"]
        out["says"] += len(say)
        ahead = (tgt["x"] - walker.x) * walker.heading  # 지금 몸 방향 기준 목적지까지 (+ 앞)
        text = " ".join(say)
        wrong = lambda a: ("앞쪽입니다" in text and a < -1) or (("뒤쪽인 것 같" in text or "지나친" in text) and a > 1)
        out["wrong_dir"] += wrong(ahead)  # 사용자가 들을 때 기준
        # 판단 자체가 틀림: 안내가 기준으로 삼은 때(지연 보정이면 가장 최근 센서 = 지금, 아니면 사진 찍은 때)의 방향으로 채점
        out["wrong_judge"] += wrong(ahead if FRESH else (tgt["x"] - cap[1]) * cap[2])
        if trace is not None:
            if not trace:
                trace.append({"corridor": corr, "target": tgt, "width": world.WIDTH, "fov": world.FOV, "range": world.RANGE,
                              "layout": layout, "seed": seed, "lat": lat})
            d = rec["state"].get("dir_last") or {}
            trace.append({"t": t, "x": round(walker.x, 2), "heading": walker.heading, "yaw": walker.yaw,
                          "cap_x": round(cap[1], 2), "cap_heading": cap[2], "cap_yaw": cap[3],
                          "seen": [p["n"] for p in obs["plates"]], "action": act, "say": say, "ahead": round(ahead, 1),
                          "p_ahead": d.get("p_ahead"), "direction": rec.get("direction"),
                          "events": [e["text"] for e in nav.mem.events if e["t"] == nav.clock and not e["text"].startswith("안내:")]})
        if show:
            seen = ",".join(p["n"] for p in obs["plates"])
            print(f"{t:6.1f}s x={walker.x:5.1f} {'→' if walker.heading > 0 else '←'}{'↺' if walker.yaw else ' '} "
                  f"목적지까지 {ahead:+5.1f}m [{seen}] {act} | {' / '.join(say)}")
        claim = act == "도착" or "문패를 확인하지 못했습니다" in text
        if claim:
            same = [d for d in corr["doors"] if d["wall"] == tgt["wall"]]
            ok = min(same, key=lambda d: abs(d["x"] - walker.x)) is tgt and abs(tgt["x"] - walker.x) <= SUCCESS_M
            out.update(success=ok, wrong_arrival=not ok, end="도착" if act == "도착" else "가능성 높음")
            break
        if "찾지 못했습니다" in text:
            out["end"] = "못 찾음"
            break
        walker.hear(say)
        walker.move(DT)
        t = round(t + DT, 2)
    shortest = abs(tgt["x"] - x0)
    out.update(t=t, walked=round(walker.walked, 1), shortest=round(shortest, 1), turns=walker.turns,
               end_turns=walker.end_turns, spl=shortest / max(walker.walked, shortest) if out["success"] else 0.0)
    return out


def table(rows):
    def line(name, rs):
        n = len(rs)
        ok = [r for r in rs if r["success"]]
        med_t = statistics.median(r["t"] for r in ok) if ok else float("nan")
        return (f"{name:9s} {n:4d}  {len(ok) / n:6.0%}  {sum(r['wrong_arrival'] for r in rs):5d}  "
                f"{sum(r['wrong_dir'] > 0 for r in rs):5d}  {sum(r['wrong_judge'] > 0 for r in rs):5d}  {statistics.mean(r['spl'] for r in rs):5.2f}  {med_t:6.1f}  "
                f"{sum(r['end_turns'] for r in rs) / n:5.2f}  {sum(r['says'] for r in rs) / n:5.1f}")
    print(f"{'배치':9s} {'횟수':>4s}  {'성공':>6s}  {'잘못도착':>5s}  {'틀린방향':>5s}  {'판단오류':>5s}  {'SPL':>5s}  {'시간(중)':>6s}  "
          f"{'끝에서돎':>5s}  {'안내수':>5s}")
    for lay in dict.fromkeys(r["layout"] for r in rows):
        print(line(lay, [r for r in rows if r["layout"] == lay]))
    print(line("전체", rows))
    ends = {}
    for r in rows:
        if not r["success"]:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
    print("실패 이유:", ends or "없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="배치마다 실행 수")
    ap.add_argument("--lat", type=float, default=0.0, help="인식 결과가 늦게 도착하는 초")
    ap.add_argument("--layouts", nargs="*", default=list(world.LAYOUTS), help=f"생성 {world.LAYOUTS}, 실제 {world.REAL}")
    ap.add_argument("--range", type=float, default=world.RANGE, help="문패를 읽는 최대 거리 (실제 영상과 맞춘 값 5)")
    ap.add_argument("--show", nargs=2, metavar=("LAYOUT", "SEED"))
    ap.add_argument("--no-heading", action="store_true", help="휴대폰 방위를 보내지 않음 (돌았는지를 문패로만 판단)")
    ap.add_argument("--no-fresh", action="store_true", help="가장 최근 센서로 지연 보정하지 않음")
    ap.add_argument("--spont", type=float, default=0.0, help="사진마다 이유 없이 돌아설 확률 (예: 0.03)")
    ap.add_argument("--err", type=int, default=0, choices=sorted(world.ERR_PRESETS), help="인식 오류: 0 없음, 1 약함, 2 심함")
    for k in world.ERR:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, help=f"인식 오류 {k} 직접 지정")
    a = ap.parse_args()
    world.RANGE, world.USE_HEADING = a.range, not a.no_heading
    world.ERR.update(world.ERR_PRESETS[a.err])
    global FRESH
    FRESH = not a.no_fresh
    world.ERR.update({k: getattr(a, k) for k in world.ERR if getattr(a, k) is not None})
    if a.show:
        print(episode(a.show[0], int(a.show[1]), a.lat, show=True, spont=a.spont))
        return
    rows = [episode(lay, s, a.lat, spont=a.spont) for lay in a.layouts for s in range(a.n)]
    table(rows)


if __name__ == "__main__":
    main()
