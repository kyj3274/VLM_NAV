"""프레임별 정답표(frame_gt.json)로 영상 호실 읽기를 채점한다.

1초 1장(frames/fNNN)과 8장(frames_8fps/i)의 시각이 최대 1초 어긋나서, ±1초 안에 있으면 맞은 것으로 본다.
  맞음  = 그 초에 낸 번호가 ±1초 정답 안에 있음
  잘못 냄 = 그 초에 낸 번호가 ±1초 정답 어디에도 없음
  놓침  = 그 초의 정답 번호를 ±1초 안에서 한 번도 내지 않음

사용: python3 src/kyj/video_frame_score.py 이름=결과.jsonl [이름=결과.jsonl …]
  결과 형식: 줄마다 {"seq", "sec", "rooms": [...]} (video_tracker.py 결과) 또는 1단계 {"seq", "frame", "result": {"rooms"}}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

GT = Path(__file__).resolve().parents[2] / "data" / "video" / "frame_gt.json"
TOL = 1


def load_pred(path):
    pred = defaultdict(set)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["seq"].startswith("1_"):
            continue
        if "sec" in r:
            pred[(r["seq"], r["sec"])] |= set(r["rooms"])
        else:
            pred[(r["seq"], int(r["frame"][1:4]))] |= {x if isinstance(x, str) else x["number"] for x in r["result"]["rooms"]}
    return pred


def window(d, seq, s):
    out = set()
    for k in range(s - TOL, s + TOL + 1):
        out |= d.get((seq, k), set())
    return out


def score(gt, pred):
    tp = fp = fn = 0
    wrong, missed = [], []
    for (seq, s), g in gt.items():
        p = pred.get((seq, s), set())
        gw, pw = window(gt, seq, s), window(pred, seq, s)
        tp += len(p & gw)
        fp += len(p - gw)
        fn += len(g - pw)
        wrong += [f"{seq[:2]} {s}초 {x}" for x in sorted(p - gw)]
        missed += [f"{seq[:2]} {s}초 {x}" for x in sorted(g - pw)]
    return tp, fp, fn, wrong, missed


def main():
    gt = {(r["seq"], int(r["frame"][1:4])): {x["번호"] for x in r["정답"]["호실_표지판"]}
          for r in json.load(open(GT, encoding="utf-8"))["frames"] if not r["seq"].startswith("1_")}
    n = sum(len(v) for v in gt.values())
    print(f"정답 호실 (초 단위 합): {n}\n")
    print("| 방식 | 맞음 | 잘못 냄 | 놓침 | 정답 중 읽은 비율 |\n|---|---|---|---|---|")
    details = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        tp, fp, fn, wrong, missed = score(gt, load_pred(path))
        print(f"| {name} | {tp} | {fp} | {fn} | {(n - fn) / n:.0%} |")
        details.append((name, wrong, missed))
    # 문패 단위: 실제 문패마다 한 번이라도 맞게 읽었나, 없는 번호를 몇 개 냈나
    plates = defaultdict(set)
    for (seq, _), g in gt.items():
        plates[seq] |= g
    print("\n| 방식 | 문패 읽음 (한 번 이상) | 못 읽은 문패 | 없는 번호를 낸 것 |\n|---|---|---|---|")
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        pred = load_pred(path)
        seen = defaultdict(set)
        for (seq, _), p in pred.items():
            seen[seq] |= p
        ok = sum(len(plates[s] & seen[s]) for s in plates)
        miss = [f"{s[:2]} {x}" for s in plates for x in sorted(plates[s] - seen[s])]
        ghost = [f"{s[:2]} {x}" for s in seen for x in sorted(seen[s] - plates[s])]
        print(f"| {name} | {ok}/{sum(len(v) for v in plates.values())} | {', '.join(miss) or '-'} | {', '.join(ghost) or '-'} |")
    for name, wrong, missed in details:
        print(f"\n[{name}] 잘못 냄: {', '.join(wrong) or '-'}\n[{name}] 놓침: {', '.join(missed) or '-'}")


if __name__ == "__main__":
    main()
