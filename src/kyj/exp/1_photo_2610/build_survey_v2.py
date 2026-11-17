"""채점 기준 v2 만들기: 사진 조사(v1) + 호실 후보 잘라낸 사진 확인 결과(ref_verdicts.json).

- 확인 결과 문패(yes)인 후보 → 호실 정답에 추가 (읽은 숫자는 확인자가 고친 값)
- 조사(v1)에 있던 번호인데 그 번호의 후보가 모두 no → 정답에서 제거
- unsure → 정답에도 오답에도 넣지 않음 (ignore_rooms, 채점에서 제외)
- near: 그 번호 글자 높이가 사진 높이의 1.5% 이상인 후보가 하나라도 있으면 true
결과: review/survey_v2.json, review/human_check.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NEAR = 1.5
v1 = {}
for path in sorted((ROOT / "review" / "survey").glob("survey_*.json")):
    v1.update(json.load(open(path, encoding="utf-8")))
index = {c["crop"]: c for c in json.load(open(ROOT / "review" / "ref_crops" / "index.json", encoding="utf-8"))}
verdicts = {v["crop"]: v for v in json.load(open(ROOT / "review" / "ref_verdicts.json", encoding="utf-8"))}

out, check = {}, []
for image_id, item in v1.items():
    crops = [c for c in index.values() if c["image_id"] == image_id]
    yes, no, unsure, height = set(), set(), set(), {}
    for c in crops:
        v = verdicts.get(c["crop"], {"is_plate": "unsure", "text": c["text"]})
        number = (v.get("text") or c["text"]).replace("호", "").strip()
        {"yes": yes, "no": no}.get(v["is_plate"], unsure).add(number if v["is_plate"] == "yes" else c["text"])
        if v["is_plate"] == "yes":
            height[number] = max(height.get(number, 0), c["height_pct"])
        if v["is_plate"] == "unsure" or (v["is_plate"] == "yes" and number != c["text"]):
            check.append(f"| {image_id} | {c['crop']} | OCR `{c['text']}` | 확인 `{v['is_plate']}` `{number}` | {v.get('note', '')} |")
    rooms = set(item["rooms"]) | yes
    rooms -= {r for r in item["rooms"] if r in no and r not in yes}
    ignore = (unsure - rooms)
    out[image_id] = {
        "rooms": [{"room": r, "near": height.get(r, 0) >= NEAR} for r in sorted(rooms)],
        "ignore_rooms": sorted(ignore),
        "landmarks": item["landmarks"], "arrows": item["arrows"], "floor_text": item["floor_text"],
    }
json.dump(out, open(ROOT / "review" / "survey_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
total = sum(len(v["rooms"]) for v in out.values()); near = sum(1 for v in out.values() for r in v["rooms"] if r["near"])
open(ROOT / "review" / "human_check.md", "w", encoding="utf-8").write(
    "# 직접 확인할 호실 후보\n\n잘라낸 사진: `review/ref_crops/`\n\n| 사진 | 잘라낸 파일 | OCR | 확인 결과 | 메모 |\n|---|---|---|---|---|\n" + "\n".join(check) + "\n")
print(f"호실 정답 {total}개 (가까움 {near}), v1 {sum(len(v['rooms']) for v in v1.values())}개, 확인 필요 {len(check)}건")
