"""채점 기준 v2로 fields 결과를 채점한다 (조정용 dev / 평가용 test 분리).

기준: review/survey_v2.json
  {image_id: {"rooms": [{"room": "404", "near": true}], "landmarks": [...], "arrows": [...], "floor_text": [...]}}
  near = 문패 숫자 글자 높이가 사진 높이의 1.5% 이상 (가까운 호실)

사용법: python3 score_v2.py --split dev 이름=결과.jsonl [...]
         python3 score_v2.py --split dev --json 이름=결과.jsonl   (에이전트용 기계 판독 출력)
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
# Mac: 프로젝트 review/ 폴더, 서버: 이 스크립트 옆 eval/ 폴더
_EVAL = Path(__file__).resolve().parent / "eval"
REFERENCE = ROOT / "review" / "survey_v2.json" if (ROOT / "review" / "survey_v2.json").exists() else _EVAL / "survey_v2.json"
SPLIT = ROOT / "review" / "split.json" if (ROOT / "review" / "split.json").exists() else _EVAL / "split.json"
MERGE = {"출입문": "문", "유리문": "문", "미닫이문": "문", "화장실표시": "화장실", "화장실입구": "화장실"}
CLASSES = ["문", "엘리베이터", "계단", "화장실", "비상구표시"]
EMPTY = {"rooms": [], "floors": [], "spaces": [], "landmarks": [], "directions": []}
HOURS = {f"{h}시": h % 12 for h in range(1, 13)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def near_clock(a, b):
    if a not in HOURS or b not in HOURS:
        return False
    gap = abs(HOURS[a] - HOURS[b]) % 12
    return min(gap, 12 - gap) <= 1


def merged(names):
    return {MERGE.get(n, n) for n in names} & set(CLASSES)


def score(reference, records, ids):
    answers = {}
    for r in records:
        answers[r["image_id"]] = r["result"] if r.get("success", True) and r.get("result") else EMPTY
    ids = [i for i in ids if i in answers]
    missing = [i for i in ids if i not in answers]

    tp = fp = fn = 0
    near_tp = near_fn = 0
    for i in ids:
        truth = {x["room"] for x in reference[i]["rooms"]}
        near = {x["room"] for x in reference[i]["rooms"] if x["near"]}
        guess = {x.replace("호", "").strip() for x in answers[i]["rooms"]} - set(reference[i].get("ignore_rooms", []))
        tp += len(guess & truth)
        fp += len(guess - truth)
        fn += len(truth - guess)
        near_tp += len(guess & near)
        near_fn += len(near - guess)
    rooms = prf(tp, fp, fn)
    near_recall = near_tp / (near_tp + near_fn) if near_tp + near_fn else 0.0

    def count(name=None):
        a = b = c = 0
        for i in ids:
            t = merged(reference[i]["landmarks"])
            g = merged(answers[i]["landmarks"])
            if name:
                t, g = t & {name}, g & {name}
            a += len(t & g)
            b += len(g - t)
            c += len(t - g)
        return prf(a, b, c)

    floors_tp = floors_fp = floors_fn = 0
    for i in ids:
        t, g = set(reference[i]["floor_text"]), set(answers[i]["floors"])
        floors_tp += len(t & g)
        floors_fp += len(g - t)
        floors_fn += len(t - g)

    arrow_hit = arrow_dir = arrow_fp = 0
    for i in ids:
        truth = reference[i]["arrows"]
        guess = [d for d in answers[i]["directions"] if d.get("type") == "SIGN_ARROW"]
        if truth and guess:
            arrow_hit += 1
            arrow_dir += any(near_clock(t["value"], g["value"]) for t in truth for g in guess)
        elif guess:
            arrow_fp += 1

    latencies = [r["latency_seconds"] for r in records if r["image_id"] in ids and r.get("latency_seconds") is not None]
    return {
        "images": len(ids),
        "missing": len(missing),
        "rooms": rooms,
        "near_rooms_recall": round(near_recall, 3),
        "near_rooms_found": near_tp,
        "near_rooms_total": near_tp + near_fn,
        "landmarks": count(),
        "per_landmark": {c: count(c) for c in CLASSES},
        "floors": prf(floors_tp, floors_fp, floors_fn),
        "arrows": {"found": arrow_hit, "direction_ok": arrow_dir, "false_alarm_images": arrow_fp,
                   "truth_images": sum(1 for i in ids if reference[i]["arrows"])},
        "latency_median": round(statistics.median(latencies), 2) if latencies else None,
        "latency_mean": round(statistics.mean(latencies), 2) if latencies else None,
        "vlm_calls": sum(1 for r in records if r["image_id"] in ids and r.get("vlm_called")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    reference = json.load(open(REFERENCE, encoding="utf-8"))
    split = json.load(open(SPLIT))
    ids = sorted(reference) if args.split == "all" else split[args.split]

    results = {}
    for pair in args.runs:
        name, path = pair.split("=", 1)
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        results[name] = score(reference, records, ids)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return

    names = list(results)
    f = lambda d: f"{d['p'] * 100:.0f} / {d['r'] * 100:.0f} / **{d['f1'] * 100:.0f}**"
    print(f"split={args.split}, 사진 {results[names[0]]['images']}장 (P / R / F1)\n")
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| **호실** | " + " | ".join(f(results[n]["rooms"]) for n in names) + " |")
    print("| └ 맞음 / 틀림 / 놓침 | " + " | ".join(f"{results[n]['rooms']['tp']} / {results[n]['rooms']['fp']} / {results[n]['rooms']['fn']}" for n in names) + " |")
    print("| **가까운 호실 재현율** | " + " | ".join(f"{results[n]['near_rooms_recall'] * 100:.0f}% ({results[n]['near_rooms_found']}/{results[n]['near_rooms_total']})" for n in names) + " |")
    print("| **랜드마크 5종** | " + " | ".join(f(results[n]["landmarks"]) for n in names) + " |")
    for c in CLASSES:
        print(f"| └ {c} | " + " | ".join(f(results[n]["per_landmark"][c]) for n in names) + " |")
    print("| 층 표시 맞음 / 틀림 / 놓침 | " + " | ".join(f"{results[n]['floors']['tp']} / {results[n]['floors']['fp']} / {results[n]['floors']['fn']}" for n in names) + " |")
    print("| 화살표 찾음 / 방향 맞음 / 오탐 사진 | " + " | ".join(f"{results[n]['arrows']['found']} / {results[n]['arrows']['direction_ok']} / {results[n]['arrows']['false_alarm_images']}" for n in names) + " |")
    print("| 시간 중앙값 / 평균 (초) | " + " | ".join(f"{results[n]['latency_median']} / {results[n]['latency_mean']}" for n in names) + " |")
    print("| VLM 부른 사진 | " + " | ".join(str(results[n]["vlm_calls"]) for n in names) + " |")


if __name__ == "__main__":
    main()
