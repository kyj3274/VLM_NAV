"""파이프라인(YOLOE+OCR, +VLM)과 VLM 단독 fields 결과를 사진 조사 기준으로 비교. 문·화장실은 합친 이름.

사용법: python3 score_pipeline.py 이름=결과.jsonl [...]
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare_fields  # noqa: E402
from score_vs_survey import load_survey, near  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
MERGE = {"출입문": "문", "유리문": "문", "미닫이문": "문", "화장실표시": "화장실", "화장실입구": "화장실"}
CLASSES = ["문", "엘리베이터", "계단", "화장실", "비상구표시"]
EMPTY = {"rooms": [], "floors": [], "spaces": [], "landmarks": [], "directions": []}


def merged(names):
    return {MERGE.get(n, n) for n in names} & set(CLASSES)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def fmt(values):
    return " / ".join(f"{v * 100:.0f}" for v in values[:2]) + f" / **{values[2] * 100:.0f}**"


def score(survey, records):
    answers = {r["image_id"]: (r["result"] if r.get("success") else EMPTY) for r in records}
    ids = [i for i in survey if i in answers]
    out = {}

    def count(get_truth, get_guess):
        tp = fp = fn = 0
        for i in ids:
            t, g = get_truth(i), get_guess(i)
            tp += len(t & g)
            fp += len(g - t)
            fn += len(t - g)
        return tp, fp, fn

    rooms = count(lambda i: set(survey[i]["rooms"]), lambda i: {x.replace("호", "").strip() for x in answers[i]["rooms"]})
    out["rooms"] = prf(*rooms)
    out["rooms_raw"] = rooms
    floors = count(lambda i: set(survey[i]["floor_text"]), lambda i: set(answers[i]["floors"]))
    out["floors_raw"] = floors
    out["landmarks"] = prf(*count(lambda i: merged(survey[i]["landmarks"]), lambda i: merged(answers[i]["landmarks"])))
    for name in CLASSES:
        out[name] = prf(*count(lambda i: {name} & merged(survey[i]["landmarks"]), lambda i: {name} & merged(answers[i]["landmarks"])))

    arrow_hit = arrow_dir = arrow_fp = 0
    for i in ids:
        truth = survey[i]["arrows"]
        guess = [d for d in answers[i]["directions"] if d["type"] == "SIGN_ARROW"]
        if truth and guess:
            arrow_hit += 1
            arrow_dir += any(near(t["value"], g["value"]) for t in truth for g in guess)
        elif guess:
            arrow_fp += 1
    out["arrows"] = (arrow_hit, arrow_dir, arrow_fp, sum(1 for i in ids if survey[i]["arrows"]))

    latencies = [r["latency_seconds"] for r in records if r["image_id"] in survey and r.get("latency_seconds")]
    out["latency"] = (statistics.median(latencies), statistics.mean(latencies), sum(latencies) / 60)
    out["vlm_calls"] = sum(1 for r in records if r.get("vlm_reasons")) if any("vlm" in r for r in records) else None
    out["ids"] = len(ids)
    return out


def main(pairs):
    survey = load_survey(ROOT / "review" / "survey")
    skip = compare_fields.dup_ids()
    results = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        records = [r for r in (json.loads(l) for l in open(path, encoding="utf-8")) if r["image_id"] not in skip]
        results[name] = score(survey, records)
    names = list(results)
    print(f"사진 {results[names[0]]['ids']}장 기준 (P / R / F1)\n")
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| **호실** | " + " | ".join(fmt(results[n]["rooms"]) for n in names) + " |")
    print("| └ 맞음 / 틀림 / 놓침 | " + " | ".join("{} / {} / {}".format(*results[n]["rooms_raw"]) for n in names) + " |")
    print("| **층 표시** 맞음 / 틀림 / 놓침 | " + " | ".join("{} / {} / {}".format(*results[n]["floors_raw"]) for n in names) + " |")
    print("| **랜드마크 5종** | " + " | ".join(fmt(results[n]["landmarks"]) for n in names) + " |")
    for c in CLASSES:
        print(f"| └ {c} | " + " | ".join(fmt(results[n][c]) for n in names) + " |")
    a = results[names[0]]["arrows"][3]
    print(f"| **화살표** 찾음 / 방향 맞음 (조사 {a}장) | " + " | ".join("{} / {}".format(*results[n]["arrows"][:2]) for n in names) + " |")
    print("| └ 화살표 없는데 있다고 함 | " + " | ".join(str(results[n]["arrows"][2]) for n in names) + " |")
    print("| **시간** 중앙값 / 평균 (초) | " + " | ".join("{:.2f} / {:.2f}".format(*results[n]["latency"][:2]) for n in names) + " |")
    print("| └ 176장 합계 (분) | " + " | ".join(f"{results[n]['latency'][2]:.1f}" for n in names) + " |")
    print("| VLM 부른 사진 | " + " | ".join("-" if results[n]["vlm_calls"] is None else str(results[n]["vlm_calls"]) for n in names) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
