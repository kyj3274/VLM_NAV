"""실험 B: 호실 후보 주변 '문패 판' 특징을 원본 사진 픽셀로 계산 (CPU, GPU 불필요).

exp_a_<설정>.jsonl 의 texts_all 에서 호실 모양 후보(전부)를 뽑아 원본 crop으로
room_rules.plate_features 를 계산하고 outputs/kyj/exp_bc_plate.json 에 저장한다.
사용법 (서버): cd /workspace/capstone/src/kyj && .venv/bin/python exp_bc_plate.py m2048 s2048 m3072
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

import room_rules

DATA = Path("/workspace/capstone/data/cnuCS")
OUT = Path("/workspace/capstone/outputs/kyj")
FLOORS = {"3F": "3rdFloor", "4F": "4thFloor", "5F": "5thFloor"}


def main():
    configs = sys.argv[1:] or ["m2048"]
    records = {c: [json.loads(l) for l in open(OUT / f"exp_a_{c}.jsonl", encoding="utf-8")] for c in configs}
    ids = [r["image_id"] for r in records[configs[0]]]
    result = {c: {} for c in configs}
    times = []
    for n, i in enumerate(ids, 1):
        floor, num = i.split("_")
        image = Image.open(DATA / FLOORS[floor] / f"{int(num)}.jpeg").convert("RGB")
        for c in configs:
            rec = next(r for r in records[c] if r["image_id"] == i)
            cands = room_rules.candidates(rec["texts_all"], rec["ocr_input_size"])
            t0 = time.perf_counter()
            result[c][i] = {x["room"]: room_rules.plate_features(image, x["box"]) for x in cands}
            if cands:
                times.append((time.perf_counter() - t0) / len(cands))
        if n % 20 == 0:
            print(n, len(ids), flush=True)
    result["_seconds_per_candidate"] = sum(times) / max(len(times), 1)
    json.dump(result, open(OUT / "exp_bc_plate.json", "w"), ensure_ascii=False)
    print("done", result["_seconds_per_candidate"])


if __name__ == "__main__":
    main()
