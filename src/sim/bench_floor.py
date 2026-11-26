"""시뮬레이터 v3 실행: 갈림길이 있는 층에서 Claude 버전 Navigator(갈림길 판단 core/floor_graph.py 포함).

  .venv/bin/python src/sim/bench_floor.py                       # 번호 문패, 표지판 없음
  .venv/bin/python src/sim/bench_floor.py --mode name           # 이름 문패만 (번호로 방향을 모름 = 빠짐없이 둘러보기)
  .venv/bin/python src/sim/bench_floor.py --mode name --signs 1 # 이름 + 갈림길 표지판
  .venv/bin/python src/sim/bench_floor.py --mode english        # 영어 이름 문패, 목적지는 한국어로 말함
  .venv/bin/python src/sim/bench_floor.py --show T 3

지표는 bench.py와 같음 (성공, 잘못 도착, SPL, 걸린 시간) + 헛걸음 = 걸은 거리 − 최단 거리.
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
import floor as fw  # noqa: E402
import navigator as nv  # noqa: E402
import world  # noqa: E402

DT = 1.5
MAX_T = 600
SUCCESS_M = 4.0
MIN_PATH = 8.0     # 시작과 목적지 사이 최소 길 거리


def episode(kind, seed, mode="number", signs=0.0, show=False, trace=None):
    rng = random.Random(seed)
    fl = fw.make_floor(kind, mode, signs, rng)
    cors = fl["corridors"]
    while True:
        c = rng.choice(cors)
        x0 = rng.uniform(0.5, c["length"] - 0.5)
        tc = rng.choice(cors)
        tgt = rng.choice(tc["doors"])
        sp = fw.shortest(fl, (c["id"], x0), (tc["id"], tgt["x"]))
        if sp >= MIN_PATH:
            break
    w = fw.FloorWalker(fl, c["id"], x0, rng.choice([1, -1]), rng=random.Random(seed + 1000))
    nav = nv.Navigator(world.SimRunner(random.Random(seed + 2000)), tgt["ko"], start_floor="4")
    nav.quick_max = 0
    nav.new_segment("1")
    t = 0.0
    out = {"layout": kind, "seed": seed, "target": tgt["ko"], "success": False, "wrong_arrival": False, "end": "시간 초과", "says": 0}
    while t <= MAX_T:
        obs = fw.observe(fl, w, tgt["ko"])
        nav.latest = {"t": t, **w.sensor()}
        with contextlib.redirect_stdout(io.StringIO()):
            rec = nav.step(t=t, observed=world.wrap_obs(obs), **w.sensor())  # 웹처럼 찍을 때의 걸음 수·방위
        say, act = rec["say"], rec["action"]
        out["says"] += len(say)
        text = " ".join(say)
        if trace is not None:
            if not trace:
                trace.append({"floor": fl, "target": {**tgt, "c": tc["id"]}, "kind": kind, "seed": seed})
            trace.append({"t": t, "c": w.c, "x": round(w.x, 2), "h": w.h, "yaw": w.yaw, "say": say, "act": act,
                          "seen": [p["n"] for p in obs["plates"]] + (["★"] if obs.get("dest", {}).get("seen") else []),
                          "junction": obs.get("junction"), "mind": nav.mind_map(),
                          "events": [e["text"] for e in nav.mem.events if e["t"] == nav.clock and not e["text"].startswith("안내:")]})
        if show:
            j = obs.get("junction")
            print(f"{t:6.1f}s 복도{w.c} x={w.x:5.1f} {'+' if w.h > 0 else '-'}{'↺' if w.yaw else ' '} "
                  f"[{','.join(p['n'] for p in obs['plates'])}{'★' if obs.get('dest', {}).get('seen') else ''}]"
                  f"{' 갈림길 ' + str(j['m']) + 'm ' + '/'.join(j['open']) if j else ''} {act} | {' / '.join(say)}")
            for e in nav.mem.events:
                if e["t"] == nav.clock and ("갈림길" in e["text"] or "복도" in e["text"] or "끝" in e["text"]):
                    print(f"{'':10s}· {e['text']}")
        if act == "도착" or "문패를 확인하지 못했습니다" in text:
            same = [d for d in fl["corridors"][w.c]["doors"] if d["wall"] == tgt["wall"]]
            ok = w.c == tc["id"] and min(same, key=lambda d: abs(d["x"] - w.x)) is tgt and abs(tgt["x"] - w.x) <= SUCCESS_M
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


def table(rows):
    def line(name, rs):
        ok = [r for r in rs if r["success"]]
        md = lambda k: statistics.median(r[k] for r in ok) if ok else float("nan")
        return (f"{name:7s} {len(rs):4d}  {len(ok) / len(rs):5.0%}  {sum(r['wrong_arrival'] for r in rs):4d}  "
                f"{statistics.mean(r['spl'] for r in rs):5.2f}  {md('t'):6.1f}  {md('extra'):6.1f}  {sum(r['end_turns'] for r in rs) / len(rs):5.2f}")
    print(f"{'층':7s} {'횟수':>4s}  {'성공':>5s}  {'잘못':>4s}  {'SPL':>5s}  {'시간(중)':>6s}  {'헛걸음(중)':>6s}  {'끝에서돎':>5s}")
    for k in dict.fromkeys(r["layout"] for r in rows):
        print(line(k, [r for r in rows if r["layout"] == k]))
    print(line("전체", rows))
    ends = {}
    for r in rows:
        if not r["success"]:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
    print("실패 이유:", ends or "없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--layouts", nargs="*", default=list(fw.FLOORS))
    ap.add_argument("--mode", default="number", choices=["number", "name", "english"])
    ap.add_argument("--signs", type=float, default=0.0, help="갈림길마다 표지판이 있을 확률")
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--show", nargs=2, metavar=("LAYOUT", "SEED"))
    a = ap.parse_args()
    world.RANGE = a.range
    if a.show:
        print(episode(a.show[0], int(a.show[1]), a.mode, a.signs, show=True))
        return
    table([episode(k, s, a.mode, a.signs) for k in a.layouts for s in range(a.n)])


if __name__ == "__main__":
    main()
