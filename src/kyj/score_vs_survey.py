"""fields 결과를 사진 조사(survey_*.json)와 사진 단위로 맞춰 본다.

사진 조사는 정답(GT)이 아니다. 축소 사진을 사람 대신 훑어서 적은 참고 기록이라
여기서 나온 P/R/F1은 "참고 일치율"로만 쓴다.

사용법: python3 score_vs_survey.py survey_dir 이름=결과.jsonl [이름=결과.jsonl ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare_fields  # noqa: E402

LANDMARKS = [
    "출입문", "게시판", "복도끝", "사물함", "액자", "CCTV", "유리문", "소파",
    "화장실표시", "기둥", "노란바닥띠", "소화전", "엘리베이터", "화장실입구",
    "미닫이문", "안내판", "셔터", "계단", "비상구표시", "소화기",
]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_survey(folder):
    survey = {}
    for path in sorted(Path(folder).glob("survey_*.json")):
        survey.update(json.load(open(path, encoding="utf-8")))
    return survey


def score(survey, records):
    # 실패한 사진은 빈 답으로 친다 (못 읽은 것과 같음)
    answers = {
        r["image_id"]: (r["result"] if r["success"] else {"rooms": [], "floors": [], "landmarks": [], "directions": []})
        for r in records
    }
    ids = [i for i in survey if i in answers]

    per_landmark = defaultdict(lambda: [0, 0, 0])
    total = [0, 0, 0]
    for i in ids:
        truth = set(survey[i]["landmarks"])
        guess = set(answers[i]["landmarks"])
        for name in LANDMARKS:
            t, g = name in truth, name in guess
            cell = per_landmark[name]
            cell[0] += t and g
            cell[1] += g and not t
            cell[2] += t and not g
        total[0] += len(truth & guess)
        total[1] += len(guess - truth)
        total[2] += len(truth - guess)

    rooms = [0, 0, 0]
    for i in ids:
        truth = set(survey[i]["rooms"])
        guess = {room.replace("호", "").strip() for room in answers[i]["rooms"]}
        rooms[0] += len(truth & guess)
        rooms[1] += len(guess - truth)
        rooms[2] += len(truth - guess)

    floors = [0, 0, 0]
    for i in ids:
        truth = set(survey[i]["floor_text"])
        guess = set(answers[i]["floors"])
        floors[0] += len(truth & guess)
        floors[1] += len(guess - truth)
        floors[2] += len(truth - guess)

    # 화살표: 사진에 화살표가 있는지(사진 단위)만 본다
    arrow = [0, 0, 0, 0]  # 둘 다 있음, 모델만, 조사만, 방향까지 맞음(±1시간)
    for i in ids:
        truth = survey[i]["arrows"]
        guess = [d for d in answers[i]["directions"] if d["type"] == "SIGN_ARROW"]
        if truth and guess:
            arrow[0] += 1
            values = {d["value"] for d in guess}
            if any(near(t["value"], v) for t in truth for v in values):
                arrow[3] += 1
        elif guess:
            arrow[1] += 1
        elif truth:
            arrow[2] += 1

    empty = [i for i in ids if not survey[i]["landmarks"]]
    return {
        "ids": len(ids),
        "landmarks": prf(*total),
        "per_landmark": {name: prf(*per_landmark[name]) + (per_landmark[name][0] + per_landmark[name][2],) for name in LANDMARKS},
        "rooms": prf(*rooms) + (rooms[0], rooms[0] + rooms[2], rooms[0] + rooms[1]),
        "floors": prf(*floors) + (floors[0], floors[0] + floors[2], floors[0] + floors[1]),
        "arrow": arrow,
        "arrow_truth": sum(1 for i in ids if survey[i]["arrows"]),
        "empty": len(empty),
    }


def near(a, b):
    hours = {f"{h}시": h % 12 for h in range(1, 13)}
    if a not in hours or b not in hours:
        return False
    gap = abs(hours[a] - hours[b]) % 12
    return min(gap, 12 - gap) <= 1


def main(folder, pairs):
    survey = load_survey(folder)
    skip = compare_fields.dup_ids()
    results = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        results[name] = score(survey, compare_fields.load(path, skip))
    names = list(results)

    def pct(x):
        return f"{x * 100:.0f}"

    print(f"사진 {results[names[0]]['ids']}장 기준\n")
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| 랜드마크 P / R / F1 | " + " | ".join("{} / {} / **{}**".format(*map(pct, results[n]["landmarks"])) for n in names) + " |")
    print("| rooms P / R / F1 | " + " | ".join("{} / {} / **{}**".format(*map(pct, results[n]["rooms"][:3])) for n in names) + " |")
    print("| rooms 맞힌 번호 / 조사 번호 / 낸 번호 | " + " | ".join("{} / {} / {}".format(*results[n]["rooms"][3:]) for n in names) + " |")
    print("| floors 맞힌 / 조사 / 낸 | " + " | ".join("{} / {} / {}".format(*results[n]["floors"][3:]) for n in names) + " |")
    a0 = results[names[0]]["arrow_truth"]
    print(f"| 화살표 있는 사진 찾음 (조사 {a0}장) | " + " | ".join(f"{results[n]['arrow'][0]}" for n in names) + " |")
    print("| └ 방향까지 맞음(±1시간) | " + " | ".join(f"{results[n]['arrow'][3]}" for n in names) + " |")
    print("| 화살표 없는데 화살표라고 함 (사진) | " + " | ".join(f"{results[n]['arrow'][1]}" for n in names) + " |")

    print("\n| 랜드마크 | 조사 사진 수 | " + " | ".join(f"{n} P/R/F1" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for landmark in LANDMARKS:
        support = results[names[0]]["per_landmark"][landmark][3]
        cells = []
        for n in names:
            p, r, f, _ = results[n]["per_landmark"][landmark]
            cells.append(f"{pct(p)}/{pct(r)}/**{pct(f)}**")
        print(f"| {landmark} | {support} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
