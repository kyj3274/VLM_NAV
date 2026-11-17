"""실험 E 진단: --diag-rooms 결과(호실 후보 전부를 물음)에서 보정 기준 라벨별 답 분포.
사용: python3 src/kyj/exp_e_diag.py outputs/kyj/exp_e_diag_*.jsonl"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_d_analyze import refs  # noqa: E402

ref, adj = refs()
print("| 결과 | 가까운 정답 예/아니요/불가 | 먼 정답 예/아니요/불가 | 오답 예/아니요/불가 | 근거 숫자=후보 & 답 일치 |")
print("|---|---|---|---|---|")
for p in sys.argv[1:]:
    tab, agree, ev_n = Counter(), 0, 0
    for l in open(p, encoding="utf-8"):
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
            e = (r.get("evidence") or {}).get(f"R{k}")
            if e is not None:
                ev_n += 1
                agree += (c["room"] in e.replace(" ", "")) == (a == "예")
    cell = lambda lab: f"{tab[(lab, '예')]}/{tab[(lab, '아니요')]}/{tab[(lab, '판단 불가')]}"
    print(f"| {Path(p).stem} | {cell('near')} | {cell('true')} | {cell('false')} | {f'{agree}/{ev_n}' if ev_n else '–'} |")
