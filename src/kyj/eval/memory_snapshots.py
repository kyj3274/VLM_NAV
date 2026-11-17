"""영상 경로를 지금 추적기(situ)로 돌리면서 걸음마다 그 시점까지의 기억(nav_memory)을 글로 남긴다 (서버).
판단 지점 평가(reason_series.py)의 입력. 경로 A·C는 1층→3층, B는 1층→4층.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python memory_snapshots.py <결과.json>
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))  # 서비스 모듈 (src/kyj/nav)

import navigator as vt

FRAMES = Path("/workspace/capstone/outputs/kyj/video/frames_8fps")
ROUTES = {"A": ("314", ["1_start_졸프_영상3", "2b_after_elevator_졸프_영상2"]),
          "B": ("314", ["1_start_졸프_영상3", "2a_after_elevator_졸프_영상1"]),
          "C": ("318", ["1_start_졸프_영상3", "2b_after_elevator_졸프_영상2"])}


def main():
    runner = vt.Runner(types.SimpleNamespace(url="http://127.0.0.1:11436", model="qwen3-vl:4b-instruct", key="sharp"))
    out = []
    for name, (target, seqs) in ROUTES.items():
        nav = vt.Navigator(runner, target, "situ")
        for seg, seq in enumerate(seqs):
            nav.new_segment(seq)
            files = sorted((FRAMES / seq).glob("i*.jpg"))
            for frames in [files[i:i + vt.DENSE] for i in range(0, len(files), vt.DENSE)]:
                rec = nav.step(frames)
                pg = rec["raw"].get("place_groups") or {}
                best = max(pg, key=pg.get) if pg else None
                out.append({"route": name, "target": target, "seg": seg, "sec": rec["sec"], "t": rec["t"], "rooms": rec["rooms"],
                            "floor": rec["state"].get("floor"), "assume1": rec["state"].get("assume1"),
                            "place": best if best and pg[best] >= 0.35 else None, "say": rec["say"],
                            "events": list(nav.mem.events), "confirmed": nav.mem.confirmed(),
                            "plates": [{k: p[k] for k in ("room", "first", "last", "total", "side", "ok", "ok_t")}
                                       for (sg, _), p in nav.mem.plates.items() if sg == nav.mem.seg]})
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
    print(len(out), "걸음 저장")


if __name__ == "__main__":
    main()
