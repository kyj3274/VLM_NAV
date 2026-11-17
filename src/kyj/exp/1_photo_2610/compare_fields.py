"""fields 결과 여러 개를 같은 사진(중복 제외 176장) 기준으로 비교한다. 정답 없이 보는 지표만.

사용법: python3 compare_fields.py 이름=결과.jsonl [이름=결과.jsonl ...]
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DUPS = Path(__file__).resolve().parents[4] / "review" / "dups.json"
FLOOR_IDS = {"3rdFloor": "3F", "4thFloor": "4F", "5thFloor": "5F"}

POSITION_TARGETS = {"엘리베이터", "계단", "화장실표시", "화장실입구", "비상구표시", "유리문", "복도끝"}
POSITION_VALUES = {"11시", "12시", "1시", "판단 불가"}

# 10/17 사진 176장을 사람 대신 직접 훑어서 센 대략적인 수 (정답 아님)
SURVEY = {
    "출입문": 135, "게시판": 51, "복도끝": 43, "사물함": 37, "액자": 29,
    "CCTV": 27, "유리문": 26, "소파": 24, "화장실표시": 22, "기둥": 18,
    "노란바닥띠": 18, "소화전": 17, "엘리베이터": 16, "화장실입구": 14,
    "미닫이문": 11, "안내판": 8, "셔터": 8, "계단": 5, "정수기": 2,
    "비상구표시": 61, "소화기": 45, "점자블록": 0, "자판기": 0,
}


def dup_ids():
    ids = set()
    for path in json.load(open(DUPS)):
        floor, name = path.split("/")
        ids.add(f"{FLOOR_IDS[floor]}_{int(Path(name).stem):03d}")
    return ids


def load(path, skip):
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in records if r["image_id"] not in skip]


def counting_run(values):
    # "101", "102", "103" ... 처럼 1씩 늘어나는 번호가 4개 이상
    numbers = [int(v) for v in values if v.isdigit()]
    return any(
        numbers[i + 3] - numbers[i] == 3 and len(set(numbers[i:i + 4])) == 4
        for i in range(len(numbers) - 3)
    )


def summarize(records):
    ok = [r for r in records if r["success"]]
    res = [r["result"] for r in ok]
    rooms = [(r["image_id"], room) for r in ok for room in r["result"]["rooms"]]
    directions = [(r, d) for r in ok for d in r["result"]["directions"]]

    def share(n, total):
        return f"{n} ({n / total * 100:.0f}%)" if total else "0"

    positions = [d for _, d in directions if d["type"] == "RELATIVE_POSITION"]
    position_bad = [
        d for d in positions
        if d["value"] not in POSITION_VALUES
        or (d["target"] not in POSITION_TARGETS and not d["target"][:1].isdigit())
    ]

    return {
        "images": len(records),
        "success": share(len(ok), len(records)),
        "truncated": sum(1 for r in records if not r["success"] and r.get("eval_count") == 512),
        "timeout": sum(1 for r in records if not r["success"] and "Timeout" in (r["error"] or "")),
        "latency": f"{statistics.median(r['latency_seconds'] for r in ok):.1f}",
        "rooms_images": sum(1 for x in res if x["rooms"]),
        "rooms_values": len(rooms),
        "rooms_floor_match": share(sum(1 for i, room in rooms if room[:1] == i[0]), len(rooms)),
        "rooms_non_digit": sum(1 for _, room in rooms if not room[:1].isdigit()),
        "rooms_999": sum(1 for _, room in rooms if "999" in room),
        "rooms_9xx": sum(1 for _, room in rooms if room[:1] == "9"),
        "rooms_counting": sum(1 for x in res if counting_run(x["rooms"])),
        "floors_images": sum(1 for x in res if x["floors"]),
        "spaces_images": sum(1 for x in res if x["spaces"]),
        "landmarks_median": statistics.median(len(set(x["landmarks"])) for x in res),
        "landmarks_10plus": sum(1 for x in res if len(set(x["landmarks"])) >= 10),
        "landmark_counts": Counter(v for x in res for v in set(x["landmarks"])),
        "landmark_by_floor": {
            name: Counter(r["image_id"][:2] for r in ok if name in r["result"]["landmarks"])
            for name in SURVEY
        },
        "dir_images": sum(1 for x in res if x["directions"]),
        "dir_items": len(directions),
        "dir_arrow": sum(1 for _, d in directions if d["type"] == "SIGN_ARROW"),
        "dir_position": len(positions),
        "dir_12": share(sum(1 for _, d in directions if d["value"] == "12시"), len(directions)),
        "dir_position_bad": len(position_bad),
        "repeated": sum(1 for x in res if any(len(x[f]) != len(set(x[f])) for f in ["rooms", "spaces", "landmarks"])),
    }


def main(pairs):
    skip = dup_ids()
    runs = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        runs[name] = summarize(load(path, skip))

    names = list(runs)
    rows = [
        ("사진 수", "images"), ("성공", "success"), ("답 잘림(512토큰)", "truncated"),
        ("시간 초과", "timeout"), ("응답 시간 중앙값(초)", "latency"),
        ("rooms 있는 사진", "rooms_images"), ("rooms 값 개수", "rooms_values"),
        ("rooms 층 일치", "rooms_floor_match"), ("rooms에 숫자 아닌 값", "rooms_non_digit"),
        ("rooms `999` 베끼기", "rooms_999"), ("rooms 9로 시작", "rooms_9xx"),
        ("rooms 번호 줄줄이 세기(사진)", "rooms_counting"),
        ("floors 있는 사진", "floors_images"), ("spaces 있는 사진", "spaces_images"),
        ("랜드마크 개수(사진당 중앙값)", "landmarks_median"), ("랜드마크 10개 이상 사진", "landmarks_10plus"),
        ("directions 있는 사진", "dir_images"), ("directions 항목", "dir_items"),
        ("└ SIGN_ARROW", "dir_arrow"), ("└ RELATIVE_POSITION", "dir_position"),
        ("└ 값이 12시", "dir_12"), ("RELATIVE_POSITION 규칙 위반", "dir_position_bad"),
        ("같은 값 반복(사진)", "repeated"),
    ]
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, key in rows:
        print(f"| {label} | " + " | ".join(str(runs[n][key]) for n in names) + " |")

    print("\n| 랜드마크 | 사진 조사 | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for landmark, survey in sorted(SURVEY.items(), key=lambda item: -item[1]):
        print(f"| {landmark} | {survey} | " + " | ".join(str(runs[n]["landmark_counts"][landmark]) for n in names) + " |")

    print("\n[층별: 3층에만 있는 미닫이문·셔터, 4층에만 있는 사물함, 5층에만 있는 노란바닥띠·소파]")
    for landmark in ["미닫이문", "셔터", "사물함", "노란바닥띠", "소파", "기둥"]:
        print(landmark, {n: dict(runs[n]["landmark_by_floor"][landmark]) for n in names})


if __name__ == "__main__":
    main(sys.argv[1:])
