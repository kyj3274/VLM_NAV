"""실험 D 진단: 호실 후보 전부를 VLM에 물었을 때(--diag-rooms) 답과 보정 기준 라벨(정답/오답)의 일치 표."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_d_analyze import refs, OUT  # noqa: E402

ref, adj = refs()
for v in sys.argv[1:]:
    tab = Counter()
    for l in open(OUT / f"exp_d_diagrooms_{v}_dev.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r["vlm_called"]:
            continue
        truth = {x["room"]: x["near"] for x in adj[r["image_id"]]["rooms"]}
        ign = set(adj[r["image_id"]].get("ignore_rooms", []))
        for k, c in enumerate(r["questions"]["rooms"], 1):
            if c["room"] in ign:
                continue
            lab = ("near" if truth[c["room"]] else "true") if c["room"] in truth else "false"
            a = (r["vlm"] or {}).get(f"R{k}", "err")
            tab[(lab, a)] += 1
    labs = ["near", "true", "false"]
    print(f"{v}: " + " | ".join(f"{lab}: 예 {tab[(lab, '예')]} / 아니요 {tab[(lab, '아니요')]} / 판단불가 {tab[(lab, '판단 불가')]}" for lab in labs))
