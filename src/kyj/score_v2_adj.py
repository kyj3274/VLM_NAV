"""score_v2 + 보정 정답(outputs/kyj/exp_a_extra_rooms.json)으로 채점. 사용법은 score_v2.py와 같음.
기준 v2가 1단계(m2048) OCR 후보에서만 만들어져 다른 OCR 설정이 새로 맞게 읽은 호실을 틀림으로 세는 문제를 보정."""
import json, sys
from pathlib import Path
import score_v2

EXTRA = score_v2.ROOT / "outputs" / "kyj" / "exp_a_extra_rooms.json"
_load = json.load


def patched(f, *a, **k):
    data = _load(f, *a, **k)
    if getattr(f, "name", "").endswith("survey_v2.json"):
        extra = _load(open(EXTRA, encoding="utf-8"))
        for i, rooms in extra["add"].items():
            have = {x["room"] for x in data[i]["rooms"]}
            data[i]["rooms"] += [r for r in rooms if r["room"] not in have]
        for i, rooms in extra["ignore"].items():
            data[i]["ignore_rooms"] = list(set(data[i].get("ignore_rooms", [])) | set(rooms))
    return data


score_v2.json.load = patched
if __name__ == "__main__":
    score_v2.main()
