"""시뮬레이터 v1 검증: 실제 영상(3층 2b, 4층 2a)과 같은 길을 걸려서 비교한다.

  .venv/bin/python src/sim/check_video.py            # 시야 60°
  .venv/bin/python src/sim/check_video.py --fov 50   # 시야를 바꿔 보기

1) 카메라: 초마다 시뮬레이터가 보는 문패 ↔ 영상 정답(frame_gt.json)에 보인 문패
2) 판단: 같은 길에서 '영상에 실제로 보인 문패'와 '시뮬레이터가 본 문패'를 각각 Navigator(Claude 버전)에 넣고
   3층 판단 지점(경로 A 314호: A6~A10, 경로 C 318호: C6~C8)의 행동 정답과 비교
   영상 정답에는 거리가 없어서, 실제로 보인 문패의 거리·화면 위치는 시뮬레이터 지도에서 잰다.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "nav"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import navigator as nv  # noqa: E402
import world  # noqa: E402

BUILD = Path(__file__).resolve().parent / "buildings"


def gt_frames(seq):
    d = json.loads((ROOT / "data/video/frame_gt.json").read_text(encoding="utf-8"))
    fr = [f for f in d["frames"] if f["seq"] == seq]
    return [{p["번호"] for p in f["정답"]["호실_표지판"]} for f in sorted(fr, key=lambda f: f["frame"])]


def pos(b, sec):
    return b["start_x"] + b["speed"] * sec


def camera_check(b, real):
    print(f"\n== {b['name']}: 카메라 (초마다 보인 문패, 실제 | 시뮬레이터)")
    rows, agree = {}, 0
    for sec, rs in enumerate(real):
        ss = {p["n"] for p in world.observe(b, pos(b, sec), 1)["plates"]}
        agree += rs == ss
        for n in rs | ss:
            r = rows.setdefault(n, {"real": [], "sim": []})
            if n in rs:
                r["real"].append(sec)
            if n in ss:
                r["sim"].append(sec)
    span = lambda xs: f"{xs[0]:2d}~{xs[-1]:2d}초 ({len(xs)})" if xs else "     -      "
    for n in sorted(rows, key=lambda n: (rows[n]["real"] or rows[n]["sim"])[0]):
        print(f"  {n}: 실제 {span(rows[n]['real'])} | 시뮬 {span(rows[n]['sim'])}")
    print(f"  초마다 같은 문패 집합: {agree}/{len(real)}초")


def run_nav(b, target, real, source):
    """source 'real': 영상에 보인 문패만 (거리·위치는 지도로), 'sim': 시뮬레이터 카메라."""
    nav = nv.Navigator(world.SimRunner(), target)
    nav.quick_max = 0
    nav.new_segment("1")
    nav.seg_n = 1
    nav.new_segment("2")  # 엘리베이터에서 내린 상태 (판단 지점과 같음)
    out = []
    for sec, rs in enumerate(real):
        obs = world.observe(b, pos(b, sec), 1, all_plates=(source == "real"))
        if source == "real":
            obs["plates"] = [p for p in obs["plates"] if p["n"] in rs]
        with contextlib.redirect_stdout(io.StringIO()):
            rec = nav.step(t=float(sec), observed=world.wrap_obs(obs))
        out.append(rec)
    return out


def judge_check(b, real, route, target, cps):
    print(f"\n== 경로 {route} ({target}호): 판단 지점 행동 (정답 | 실제 문패 | 시뮬레이터 문패)")
    recs = {s: run_nav(b, target, real, s) for s in ("real", "sim")}
    score = {"real": 0, "sim": 0}
    for c in cps:
        acts = {s: recs[s][c["sec"]]["action"] for s in recs}
        for s in recs:
            score[s] += acts[s] in c["action"]
        mark = lambda s: ("✓" if acts[s] in c["action"] else "✗") + " " + acts[s]
        print(f"  {c['id']:>3} {c['sec']:2d}초 {'/'.join(c['action']):12s} | {mark('real'):12s} | {mark('sim')}")
    print(f"  점수: 실제 문패 {score['real']}/{len(cps)}, 시뮬레이터 {score['sim']}/{len(cps)}")
    for s in recs:
        says = [f"{r['t']:.0f}초 {x}" for r in recs[s] for x in r["say"]]
        print(f"  안내 ({'실제 문패' if s == 'real' else '시뮬레이터'}): " + " / ".join(says[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=float, default=world.FOV)
    ap.add_argument("--range", type=float, default=world.RANGE)
    a = ap.parse_args()
    world.FOV, world.RANGE = a.fov, a.range
    cps = json.loads((ROOT / "data/video/checkpoints.json").read_text(encoding="utf-8"))["checkpoints"]
    b3, b4 = world.load_building(BUILD / "cnu_3f.json"), world.load_building(BUILD / "cnu_4f.json")
    r3, r4 = gt_frames(b3["video"]), gt_frames(b4["video"])
    print(f"시야 {world.FOV:.0f}°, 읽는 거리 {world.RANGE:.0f}m, 비스듬함 한계 {world.OBLIQUE}°")
    camera_check(b3, r3)
    camera_check(b4, r4)
    judge_check(b3, r3, "A", "314", [c for c in cps if c["route"] == "A" and c["seg"] == 1 and c["id"] != "5"])
    judge_check(b3, r3, "C", "318", [c for c in cps if c["route"] == "C"])


if __name__ == "__main__":
    main()
