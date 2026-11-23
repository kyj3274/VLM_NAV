"""판단 지점 17개 채점: 새 방식(Claude 인식 + 규칙) 경로 실행 기록 vs VLM_NAV 결과.

새 방식: outputs/routes/guide_<목적지>_<구간>_<이름>.jsonl 의 그 초 기록
  다음 행동 = 기록의 action (그 초에 유효한 안내의 행동)
  층       = state.floor ("3층", 없으면 "모름")
  목적지 층인가 = 층이 목적지 층이면 "예", 다른 층이면 "아니요", 모르면 "모름"
  진행 방향 = state.trend_said: toward → 가까워짐, away → 멀어짐, 없으면 "모름"
  (도착해서 실행이 일찍 끝난 경로는 마지막 기록의 상태를 쓴다)
VLM_NAV: data/video/results/results_reason.jsonl (cond=rule: 규칙만 15/17, img·text: qwen3-vl 4B)
  Gemini 3.5 Flash(15/17)는 지점별 답이 저장소에 없어 문서(docs/11) 숫자만 적는다.

  .venv/bin/python src/eval/score_checkpoints.py --name claude
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = next(d for d in Path(__file__).resolve().parents if (d / "data" / "video").is_dir())
COLS = [("floor", "층"), ("on_goal", "목적지 층인가"), ("progress", "진행 방향"), ("action", "다음 행동")]


def ours(rec, target):
    st = rec["state"]
    f = st.get("floor")
    return {"floor": f"{f}층" if f else "모름",
            "on_goal": "모름" if not f else ("예" if f == target[0] else "아니요"),
            "progress": {"toward": "가까워짐", "away": "멀어짐"}.get(st.get("trend_said"), "모름"),
            "action": rec["action"]}


def load_route(name, route, spec):
    segs = spec["segments"]
    tag = "_".join(s.split("_")[0] for s in segs)
    path = ROOT / f"outputs/routes/guide_{spec['target']}_{tag}_{name}.jsonl"
    recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    return path, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="claude")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gt = json.load(open(ROOT / "data/video/checkpoints.json", encoding="utf-8"))
    base = defaultdict(dict)
    for line in open(ROOT / "data/video/results/results_reason.jsonl", encoding="utf-8"):
        r = json.loads(line)
        base[r["cond"]][r["id"]] = r.get("answer") or {}
    routes = {k: load_route(a.name, k, v) for k, v in gt["routes"].items()}
    rows, score = [], defaultdict(int)
    for cp in gt["checkpoints"]:
        spec = gt["routes"][cp["route"]]
        _, recs = routes[cp["route"]]
        seq = spec["segments"][cp["seg"]]
        hit = [r for r in recs if r["seq"] == seq and r["sec"] == cp["sec"]]
        rec = hit[0] if hit else recs[-1]
        o = ours(rec, spec["target"])
        rule = base["rule"].get(cp["id"], {})
        rule = {"floor": rule.get("current_floor"), "on_goal": rule.get("on_goal_floor"),
                "progress": rule.get("progress"), "action": rule.get("next_action")}
        row = {"id": cp["id"], "sec": cp["sec"], "scene": cp["scene"], "gold": cp["action"],
               "ours": o, "rule": rule, "say": rec["say"], "rooms": rec["rooms"],
               "plates_seen": [p["n"] for p in (rec.get("obs") or {}).get("plates", [])],
               "exact_record": bool(hit)}
        for k, _ in COLS:
            gold = cp["on_goal"] if k == "on_goal" else cp[k]
            score[("ours", k)] += o[k] in gold
            score[("rule", k)] += rule[k] in gold
            row[f"ok_{k}"] = o[k] in gold
            row[f"rule_ok_{k}"] = rule[k] in gold
        rows.append(row)
    n = len(rows)
    print("| 조건 | " + " | ".join(c for _, c in COLS) + " |\n|---|" + "---|" * len(COLS))
    print("| **새 방식 (Claude Haiku 인식 + 규칙)** | " + " | ".join(f"**{score[('ours', k)]}/{n}**" for k, _ in COLS) + " |")
    print("| VLM_NAV 규칙 (OCR·YOLOE·Places365) | " + " | ".join(f"{score[('rule', k)]}/{n}" for k, _ in COLS) + " |")
    print("| VLM_NAV Gemini 3.5 Flash 사진+기록 (문서 숫자) | 15/17 | 15/17 | 5/17 | 15/17 |")
    print("\n| 지점 | 장면 | 정답 | 새 방식 | VLM_NAV 규칙 |\n|---|---|---|---|---|")
    for r in rows:
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        print(f"| {r['id']} | {r['scene']} | {'/'.join(r['gold'])} | {mark(r['ok_action'])} {r['ours']['action']}"
              f"{'' if r['exact_record'] else ' (실행 끝난 뒤 상태)'} | {mark(r['rule_ok_action'])} {r['rule']['action']} |")
    out = Path(a.out or ROOT / f"outputs/eval/checkpoints_{a.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"score": {f"{w}_{k}": v for (w, k), v in score.items()}, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
