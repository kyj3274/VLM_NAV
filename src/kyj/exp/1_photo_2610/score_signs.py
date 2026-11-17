"""signs(글자 읽기) 결과를 채점 기준 v2로 비교한다. signs는 자유 문장이라 그 안에서 단서를 뽑아 본다.

  호실   글자에서 3자리 번호(예 404, 310-1)를 뽑아 기준 rooms와 비교 → P / R / F1, 가까운 호실 재현율
  표지   EXIT·비상구 → 비상구표시, TOILET·화장실 → 화장실표시 로 보고 기준 landmarks와 사진 단위 비교
  기타   성공·시간 초과·'글자 없음', 시간, 프롬프트 예시(410) 베끼기, 층이 안 맞는 번호, 같은 말 반복

사진: 3층 중복 50장을 뺀 176장 (review/dups.json). 기준 v2는 사람 정답이 아니다(모델끼리 비교용).

사용법: python3 score_signs.py 이름=결과.jsonl [...]
  결과 파일은 run_task.py 형식(image_id) 또는 팀원 형식(image만, 파일 이름에 3F/4F/5F 또는 3rdFloor…)
  층별로 나뉜 파일은 쉼표로 묶는다. 파일 이름에 층이 없으면 뒤에 @3rdFloor 처럼 붙인다.
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = json.load(open(ROOT / "review" / "survey_v2.json"))
DUPS = set(json.load(open(ROOT / "review" / "dups.json")))
FLOOR = {"3rdFloor": "3F", "4thFloor": "4F", "5thFloor": "5F"}
FAIL = "인식 실패"
NO_TEXT = "글자 없음"
# 전화번호(821-5112)는 빼고, 410호처럼 '호'가 붙은 것은 포함
ROOM_RE = re.compile(r"(?<![\d-])([1-9]\d{2}(?:-\d{1,2})?)(?!\d|-\d)")
SIGN_WORDS = {"비상구표시": re.compile(r"EXIT|비상구", re.I), "화장실표시": re.compile(r"TOILET|화장실|RESTROOM", re.I)}


def image_id(rec, path, floor=None):
    """팀원 형식(image만)일 때 층은 기록 → 인자(@3rdFloor) → 파일 이름 순으로 찾는다."""
    floor = rec.get("floor") or floor or next((f for f in FLOOR if f in path.name), None)
    if floor is None:
        floor = {"3F": "3rdFloor", "4F": "4thFloor", "5F": "5thFloor"}[re.search(r"_([345]F)[_.]", path.name).group(1)]
    return f"{FLOOR[floor]}_{int(Path(rec['image']).stem):03d}", f"{floor}/{rec['image']}"


def load(paths):
    out = {}
    for spec in paths:
        path, _, floor = spec.partition("@")
        p = Path(path)
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("image_id"):
                iid, key = rec["image_id"], f"{rec['floor']}/{rec['image']}"
            else:
                iid, key = image_id(rec, p, floor or None)
            if key in DUPS or iid not in REFERENCE:
                continue
            out[iid] = rec
    return out


def text_of(rec):
    if not rec.get("success"):
        return None
    t = rec.get("signs")
    if t is None and rec.get("result"):
        t = rec["result"].get("signs")
    return t


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def repeats(text):
    parts = [x.strip() for x in text.split(",") if x.strip()]
    return bool(parts) and Counter(parts).most_common(1)[0][1] >= 3


def evaluate(recs):
    ids = sorted(REFERENCE)
    s = Counter()
    lat, lat_ok, out_tok = [], [], []
    examples = {"fp": Counter()}
    sign_cnt = {k: Counter() for k in SIGN_WORDS}
    for i in ids:
        rec = recs.get(i)
        if rec is None:
            s["missing"] += 1
            text = None
        else:
            text = text_of(rec)
            lat.append(rec.get("latency_seconds") or 0)
            if text is not None:
                lat_ok.append(rec["latency_seconds"])
                if rec.get("eval_count"):
                    out_tok.append(rec["eval_count"])
        if text is None:
            s["fail"] += 1
            text = ""
        elif text.strip() == NO_TEXT:
            s["no_text"] += 1
        else:
            s["read"] += 1
            if repeats(text):
                s["repeat"] += 1
        found = set(ROOM_RE.findall(text))
        if "410" in found and not i.startswith("4F"):
            s["copy410"] += 1
        if "410호" in text and "EXIT" in text.upper():
            s["copy_pattern"] += 1
        found -= set(REFERENCE[i].get("ignore_rooms", []))
        truth = {x["room"] for x in REFERENCE[i]["rooms"]}
        near = {x["room"] for x in REFERENCE[i]["rooms"] if x["near"]}
        s["tp"] += len(found & truth)
        s["fp"] += len(found - truth)
        s["fn"] += len(truth - found)
        s["near_tp"] += len(found & near)
        s["near_n"] += len(near)
        s["rooms_out"] += len(found)
        s["floor_ok"] += sum(1 for x in found if x[0] == i[0])
        for x in found - truth:
            examples["fp"][x] += 1
        for name, rx in SIGN_WORDS.items():
            t = name in REFERENCE[i]["landmarks"]
            g = bool(rx.search(text))
            sign_cnt[name]["tp"] += t and g
            sign_cnt[name]["fp"] += g and not t
            sign_cnt[name]["fn"] += t and not g
    return s, lat, lat_ok, out_tok, examples, sign_cnt


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    rows = []
    for name, files in specs:
        recs = load(files.split(","))
        rows.append((name, evaluate(recs)))

    print(f"사진 {len(REFERENCE)}장 (기준 v2 호실 {sum(len(v['rooms']) for v in REFERENCE.values())}개, "
          f"가까운 호실 {sum(x['near'] for v in REFERENCE.values() for x in v['rooms'])}개)\n")
    head = ["모델", "읽음", "글자없음", "실패", "반복", "시간중앙(성공)", "시간평균(전체)", "출력토큰중앙",
            "호실 P/R/F1", "가까운호실", "층일치", "410베끼기", "EXIT P/R", "TOILET P/R"]
    print(" | ".join(head))
    for name, (s, lat, lat_ok, out_tok, ex, sc) in rows:
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        e = prf(sc["비상구표시"]["tp"], sc["비상구표시"]["fp"], sc["비상구표시"]["fn"])
        t = prf(sc["화장실표시"]["tp"], sc["화장실표시"]["fp"], sc["화장실표시"]["fn"])
        print(" | ".join(map(str, [
            name, s["read"], s["no_text"], s["fail"], s["repeat"],
            f"{statistics.median(lat_ok):.1f}s" if lat_ok else "-",
            f"{statistics.mean(lat):.1f}s" if lat else "-",
            f"{statistics.median(out_tok):.0f}" if out_tok else "-",
            f"{p*100:.0f}/{r*100:.0f}/{f*100:.0f} ({s['tp']}/{s['rooms_out']})",
            f"{s['near_tp']}/{s['near_n']}",
            f"{s['floor_ok']}/{s['rooms_out']}",
            s["copy410"],
            f"{e[0]*100:.0f}/{e[1]*100:.0f}", f"{t[0]*100:.0f}/{t[1]*100:.0f}"])))
    print()
    for name, (s, *_rest) in rows:
        ex = _rest[3]
        print(f"{name} 틀린 번호 많이 나온 것: {ex['fp'].most_common(8)}")


if __name__ == "__main__":
    main()
