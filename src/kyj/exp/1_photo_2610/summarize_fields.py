"""fields 결과(jsonl) 요약: 정답 없이 답 분포와 이상한 답만 본다.

사용법: python3 summarize_fields.py 결과.jsonl
"""

import json
import statistics
import sys
from collections import Counter

FIELDS = ["rooms", "floors", "spaces", "landmarks"]
POSITION_VALUES = {"11시", "12시", "1시", "판단 불가"}


def main(path):
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ok = [r for r in records if r["success"]]
    failed = [r for r in records if not r["success"]]

    latencies = [r["latency_seconds"] for r in ok]
    print(f"전체 {len(records)} / 성공 {len(ok)} / 실패 {len(failed)}")
    if latencies:
        print(f"응답 시간: 중앙값 {statistics.median(latencies):.1f}초, 최대 {max(latencies):.1f}초")
    for r in failed:
        print(f"  실패 {r['image_id']}: {r['error'][:100]}")

    tokens = [r["eval_count"] for r in ok if r.get("eval_count")]
    if tokens:
        print(f"출력 토큰: 중앙값 {statistics.median(tokens):.0f}, 최대 {max(tokens)} (512면 잘린 것)")

    print("\n[필드별: 값이 있는 사진 수 / 많이 나온 값]")
    for field in FIELDS:
        values = Counter(v for r in ok for v in set(r["result"][field]))
        filled = sum(1 for r in ok if r["result"][field])
        print(f"- {field}: {filled}장")
        print("  " + ", ".join(f"{v}({n})" for v, n in values.most_common(15)))

    print("\n[층별 rooms 앞자리 (다른 층 번호가 나오면 지어냈을 가능성)]")
    for floor in ["3F", "4F", "5F"]:
        firsts = Counter(
            room[0]
            for r in ok if r["image_id"].startswith(floor)
            for room in r["result"]["rooms"] if room
        )
        print(f"- {floor}: {dict(firsts)}")

    print("\n[예시 베끼기: 999]")
    copied = [r["image_id"] for r in ok if any("999" in v for v in r["result"]["rooms"])]
    print(f"- {len(copied)}장 {copied[:20]}")

    print("\n[directions]")
    directions = [(r["image_id"], d) for r in ok for d in r["result"]["directions"]]
    print(f"- 방향이 있는 사진 {sum(1 for r in ok if r['result']['directions'])}장, 항목 {len(directions)}개")
    print("  type:", dict(Counter(d["type"] for _, d in directions)))
    print("  value:", dict(Counter(d["value"] for _, d in directions)))

    bad_position = [
        (image, d) for image, d in directions
        if d["type"] == "RELATIVE_POSITION" and d["value"] not in POSITION_VALUES
    ]
    print(f"  RELATIVE_POSITION인데 11시/12시/1시가 아닌 것: {len(bad_position)}개")

    orphan = []
    for r in ok:
        named = {v for field in ["rooms", "spaces", "landmarks"] for v in r["result"][field]}
        orphan += [(r["image_id"], d["target"]) for d in r["result"]["directions"] if d["target"] not in named]
    print(f"  target이 다른 칸에 없는 것: {len(orphan)}개 {orphan[:10]}")

    print("\n[같은 값 반복]")
    repeated = [
        r["image_id"] for r in ok
        if any(len(r["result"][f]) != len(set(r["result"][f])) for f in FIELDS)
    ]
    print(f"- {len(repeated)}장 {repeated[:20]}")


if __name__ == "__main__":
    main(sys.argv[1])
