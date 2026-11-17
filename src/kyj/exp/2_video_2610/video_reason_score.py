"""video_reason.py 결과(reason.jsonl)를 정답표(checkpoints.json)로 채점한다.

사용: python3 src/kyj/video_reason_score.py 결과1.jsonl [결과2.jsonl …] data/video/checkpoints.json
"""

import json
import sys
from collections import defaultdict

FIELDS = [("current_floor", "floor", "층"), ("on_goal_floor", "on_goal", "목적지 층인가"),
          ("progress", "progress", "진행 방향"), ("next_action", "action", "다음 행동")]
LABELS = {"rule": "① 규칙", "img": "② 사진+기록", "text": "③ 기록만", "img_hint": "④ 사진+기록+힌트",
          "a_img": "a 선택지 섞음 (사진)", "a_text": "a 선택지 섞음 (기록만)", "b_img": "b 생각 칸 (사진)",
          "b_text": "e 생각 칸 (기록만)", "c_decomp": "c 질문 쪼개기",
          "d_text:qwen2.5vl:7b": "d qwen2.5vl 7B (기록만)", "d_text_think:qwen2.5vl:7b": "d qwen2.5vl 7B (기록만+생각)",
          "d_text:gemma3:12b": "d gemma3 12B (기록만)", "d_text_think:gemma3:12b": "d gemma3 12B (기록만+생각)"}


def main():
    rows = [json.loads(l) for f in sys.argv[1:-1] for l in open(f, encoding="utf-8")]
    CONDS = []
    for r in rows:
        if r["cond"] not in [c for c, _ in CONDS]:
            CONDS.append((r["cond"], LABELS.get(r["cond"], r["cond"])))
    gt = {c["id"]: c for c in json.load(open(sys.argv[-1], encoding="utf-8"))["checkpoints"]}
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["cond"]] = r

    print("| 조건 | " + " | ".join(f[2] for f in FIELDS) + " | 근거 번호가 실제 기록 | 응답 시간 평균 |")
    print("|---|" + "---|" * (len(FIELDS) + 2))
    for cond, label in CONDS:
        cells, ev_ok, ev_n, sec = [], 0, 0, []
        for key, gkey, _ in FIELDS:
            ok = n = 0
            for i, g in gt.items():
                a = (by[i].get(cond) or {}).get("answer") or {}
                n += 1
                ok += a.get(key) in g[gkey]
            cells.append(f"{ok}/{n}")
        for i in gt:
            r = by[i].get(cond) or {}
            a = r.get("answer") or {}
            if "evidence" in a:
                ev_n += 1
                ev_ok += all(0 <= int(e) <= len(r["log"]) for e in a["evidence"]) and bool(a["evidence"])
            sec.append(r.get("vlm_seconds", 0))
        ev = f"{ev_ok}/{ev_n}" if ev_n else "-"
        print(f"| {label} | " + " | ".join(cells) + f" | {ev} | {sum(sec) / len(sec):.1f}초 |")

    print("\n| 지점 | 장면 | 정답 행동 | " + " | ".join(l for _, l in CONDS) + " |")
    print("|---|---|---|" + "---|" * len(CONDS))
    for i, g in gt.items():
        cells = []
        for cond, _ in CONDS:
            a = (by[i].get(cond) or {}).get("answer") or {}
            act = a.get("next_action", "실패")
            mark = "✅" if act in g["action"] else "❌"
            extra = []
            for key, gkey, name in FIELDS[:3]:
                if a.get(key) not in g[gkey]:
                    extra.append(f"{name}={a.get(key)}")
            cells.append(f"{mark} {act}" + (f" ({', '.join(extra)})" if extra else ""))
        print(f"| {i} | {g['scene']} | {' / '.join(g['action'])} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
