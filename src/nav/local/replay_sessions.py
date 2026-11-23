"""웹 세션의 사진을 지금 추적기에 그대로 다시 넣어 본다 (서버). 고친 규칙이 실제 시연 기록에서 어떻게 달라지는지 확인용.
VLM은 부르지 않는다(주소를 비워 실패하게 둠). 결과: 세션마다 초별 확정 호실과 안내 문장.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python replay_sessions.py <결과.json> <세션 이름>[:목적지] [...]
  목적지를 붙이면(예: 20261116_075501_314:318) 그 목적지로 바꿔 다시 돌린다
"""
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 로컬 버전 모듈 (src/nav/local)

import navigator as vt

S = Path("/workspace/capstone/outputs/kyj/web_sessions")


def obs_rooms(rec):
    return [o["text"] for o in rec["raw"]["ocr"] if vt.room_rules.ROOM.match(o["text"]) and o["score"] >= 0.5]


def main():
    args = types.SimpleNamespace(url="http://127.0.0.1:9", model="none", key="sharp")  # VLM 없음
    runner = vt.Runner(args)
    out = {}
    for spec in sys.argv[2:]:
        name, _, target = spec.partition(":")
        target = target or name.rsplit("_", 1)[1]
        groups = defaultdict(list)
        for p in sorted((S / name).glob("s*_*_*.jpg")):
            seg, sec, _ = p.stem[1:].split("_")
            groups[(int(seg), int(sec))].append(p)
        nav = vt.Navigator(runner, target, "situ")
        seg_now, rows = None, []
        for (seg, sec), paths in sorted(groups.items()):
            if seg != seg_now:
                nav.new_segment(f"live_{seg}")
                seg_now = seg
            rec = nav.step(paths)
            runner.cache.clear()
            rows.append({"seg": seg, "sec": sec, "rooms": rec["rooms"], "say": rec["say"], "raw_rooms": obs_rooms(rec)})
        out[spec] = rows
        out[spec + "/memory"] = nav.mem.dump()
        print("\n" + nav.mem.text(60), flush=True)
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
