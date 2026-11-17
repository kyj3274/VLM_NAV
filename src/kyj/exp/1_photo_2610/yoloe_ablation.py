"""YOLOE 설정 비교: 입력 크기(640/1280) x 물체 이름 수(26/7). 문·화장실은 합친 이름으로 채점.

사용법: .venv/bin/python src/kyj/yoloe_ablation.py 3F_006,4F_027,...
결과: outputs/kyj/yoloe_ablation_{설정}.jsonl, 화면에 채점표
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_vision import IMAGES, MODELS, OUT, PROMPTS, FLOOR_IDS, clock  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]

# 합친 이름: 문 = 출입문/유리문/미닫이문, 화장실 = 화장실표시/화장실입구
MERGE = {"출입문": "문", "유리문": "문", "미닫이문": "문", "화장실표시": "화장실", "화장실입구": "화장실"}

# 2번: 이름 7개. 영어 표현은 기존 것 그대로 (이름 수 효과만 보기 위해)
PROMPTS_7 = {
    "문": PROMPTS["출입문"],
    "엘리베이터": PROMPTS["엘리베이터"],
    "계단": PROMPTS["계단"],
    "화장실": PROMPTS["화장실표시"],
    "비상구표시": PROMPTS["비상구표시"],
    "사람": PROMPTS["사람"],
    "상자": PROMPTS["상자"],
}

CONFIGS = {
    "기존_26개_640": (PROMPTS, 640),
    "1번_26개_1280": (PROMPTS, 1280),
    "2번_7개_640": (PROMPTS_7, 640),
    "1+2번_7개_1280": (PROMPTS_7, 1280),
}

SCORED = ["문", "엘리베이터", "계단", "화장실", "비상구표시"]
THRESHOLDS = [0.25, 0.15]


def merged(name):
    return MERGE.get(name, name)


def image_path(image_id):
    short, number = image_id.split("_")
    floor = {v: k for k, v in FLOOR_IDS.items()}[short]
    return IMAGES / floor / f"{int(number)}.jpeg"


def run(ids):
    from ultralytics import YOLOE

    os.chdir(MODELS)
    for config, (prompts, size) in CONFIGS.items():
        model = YOLOE("yoloe-11l-seg.pt")
        names = list(prompts)
        texts = [prompts[n] for n in names]
        model.set_classes(texts, model.get_text_pe(texts))
        model.predict(str(image_path(ids[0])), imgsz=size, device="mps", verbose=False)  # 예열

        with open(OUT / f"yoloe_ablation_{config}.jsonl", "w", encoding="utf-8") as out:
            for image_id in ids:
                started = time.perf_counter()
                result = model.predict(str(image_path(image_id)), imgsz=size, conf=0.05, device="mps", verbose=False)[0]
                latency = time.perf_counter() - started
                width = result.orig_shape[1]
                detections = [
                    {"name": merged(names[int(c)]), "conf": round(s, 3), "clock": clock((b[0] + b[2]) / 2 / width)}
                    for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                ]
                out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")


def score(ids):
    survey = {}
    for path in (ROOT / "review" / "survey").glob("*.json"):
        survey.update(json.load(open(path, encoding="utf-8")))
    truth = {i: {merged(n) for n in survey[i]["landmarks"]} for i in ids}

    for threshold in THRESHOLDS:
        print(f"\n### 기준 점수 {threshold} (사진 {len(ids)}장, 사진 단위로 있다/없다)\n")
        print("| 물체 (조사 사진 수) | " + " | ".join(CONFIGS) + " |")
        print("|---|" + "---|" * len(CONFIGS))
        rows = {}
        for config in CONFIGS:
            records = {json.loads(l)["image_id"]: json.loads(l) for l in open(OUT / f"yoloe_ablation_{config}.jsonl", encoding="utf-8")}
            found = {i: {d["name"] for d in records[i]["detections"] if d["conf"] >= threshold} for i in ids}
            for name in SCORED:
                tp = sum(1 for i in ids if name in truth[i] and name in found[i])
                fp = sum(1 for i in ids if name not in truth[i] and name in found[i])
                rows.setdefault(name, []).append(f"맞음 {tp} / 오탐 {fp}")
            total_tp = sum(len(truth[i] & found[i] & set(SCORED)) for i in ids)
            total_fp = sum(len((found[i] - truth[i]) & set(SCORED)) for i in ids)
            total_fn = sum(len((truth[i] - found[i]) & set(SCORED)) for i in ids)
            p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
            r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
            f = 2 * p * r / (p + r) if p + r else 0
            rows.setdefault("합계 P / R / F1", []).append(f"{p * 100:.0f} / {r * 100:.0f} / **{f * 100:.0f}**")
            people = sum(1 for i in ids if "사람" in found[i])
            rows.setdefault("(참고) 사람 잡힌 사진", []).append(str(people))
            median = sorted(rec["latency_seconds"] for rec in records.values())[len(records) // 2]
            rows.setdefault("시간 (중앙값)", []).append(f"{median * 1000:.0f}ms")
        for name, cells in rows.items():
            count = sum(1 for i in ids if name in truth[i]) if name in SCORED else ""
            label = f"{name} ({count})" if count != "" else name
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sample = sys.argv[1].split(",")
    run(sample)
    score(sample)
