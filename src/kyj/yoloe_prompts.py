"""YOLOE 3번 실험: 영어 표현 여러 개 (평균 vs 따로) x 모델 (11l / 26l / 26x). 입력 1280, 기준 0.15.

사용법: .venv/bin/python src/kyj/yoloe_prompts.py 3F_006,4F_027,...
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_vision import MODELS, OUT, clock  # noqa: E402
from yoloe_ablation import ROOT, SCORED, image_path, merged  # noqa: E402

# 사진을 보고 고르지 않고, 물체를 일반적으로 묘사하는 표현만 넣음
PHRASES = {
    "문": ["door", "closed door", "office door", "glass door", "sliding door"],
    "엘리베이터": ["elevator", "elevator doors", "stainless steel elevator door", "lift", "elevator with call button panel"],
    "계단": ["staircase", "stairs", "stairwell"],
    "화장실": ["restroom sign", "toilet sign", "restroom entrance", "bathroom entrance"],
    "비상구표시": ["green emergency exit sign", "exit sign", "running man exit sign"],
    "사람": ["person"],
    "상자": ["cardboard box"],
}

SIZE = 1280
THRESHOLD = 0.15

CONFIGS = [
    ("3A_평균_11l", "yoloe-11l-seg.pt", "mean"),
    ("3B_따로_11l", "yoloe-11l-seg.pt", "separate"),
    ("3B_따로_26l", "yoloe-26l-seg.pt", "separate"),
    ("3B_따로_26x", "yoloe-26x-seg.pt", "separate"),
]


def build(model, mode):
    if mode == "separate":
        texts, labels = [], []
        for name, phrases in PHRASES.items():
            texts += phrases
            labels += [name] * len(phrases)
        model.set_classes(texts, model.get_text_pe(texts))
        return labels

    import torch

    embeddings = []
    for phrases in PHRASES.values():
        pe = model.get_text_pe(phrases)  # (1, n, d)
        mean = pe.mean(dim=1, keepdim=True)
        embeddings.append(mean / mean.norm(dim=-1, keepdim=True))
    labels = list(PHRASES)
    model.set_classes(labels, torch.cat(embeddings, dim=1))
    return labels


def run(ids):
    from ultralytics import YOLOE

    os.chdir(MODELS)
    for config, weights, mode in CONFIGS:
        model = YOLOE(weights)
        labels = build(model, mode)
        model.predict(str(image_path(ids[0])), imgsz=SIZE, device="mps", verbose=False)
        with open(OUT / f"yoloe_prompts_{config}.jsonl", "w", encoding="utf-8") as out:
            for image_id in ids:
                started = time.perf_counter()
                result = model.predict(str(image_path(image_id)), imgsz=SIZE, conf=0.05, device="mps", verbose=False)[0]
                latency = time.perf_counter() - started
                width = result.orig_shape[1]
                detections = [
                    {"name": labels[int(c)], "conf": round(s, 3), "clock": clock((b[0] + b[2]) / 2 / width)}
                    for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                ]
                out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")
        print("done", config)


def score(ids, configs):
    survey = {}
    for path in (ROOT / "review" / "survey").glob("*.json"):
        survey.update(json.load(open(path, encoding="utf-8")))
    truth = {i: {merged(n) for n in survey[i]["landmarks"]} for i in ids}

    print(f"\n### 입력 {SIZE}, 기준 점수 {THRESHOLD} (사진 {len(ids)}장)\n")
    print("| 물체 (조사 사진 수) | " + " | ".join(configs) + " |")
    print("|---|" + "---|" * len(configs))
    rows = {}
    for config in configs:
        records = {json.loads(l)["image_id"]: json.loads(l) for l in open(OUT / config, encoding="utf-8")}
        found = {i: {merged(d["name"]) for d in records[i]["detections"] if d["conf"] >= THRESHOLD} for i in ids}
        for name in SCORED:
            tp = sum(1 for i in ids if name in truth[i] and name in found[i])
            fp = sum(1 for i in ids if name not in truth[i] and name in found[i])
            rows.setdefault(name, []).append(f"맞음 {tp} / 오탐 {fp}")
        tp = sum(len(truth[i] & found[i] & set(SCORED)) for i in ids)
        fp = sum(len((found[i] - truth[i]) & set(SCORED)) for i in ids)
        fn = sum(len((truth[i] - found[i]) & set(SCORED)) for i in ids)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        rows.setdefault("합계 P / R / F1", []).append(f"{p * 100:.0f} / {r * 100:.0f} / **{f * 100:.0f}**")
        median = sorted(rec["latency_seconds"] for rec in records.values())[len(records) // 2]
        rows.setdefault("시간 (중앙값)", []).append(f"{median * 1000:.0f}ms")
    for name, cells in rows.items():
        label = f"{name} ({sum(1 for i in ids if name in truth[i])})" if name in SCORED else name
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sample = sys.argv[1].split(",")
    run(sample)
    score(sample, ["yoloe_ablation_1+2번_7개_1280.jsonl"] + [f"yoloe_prompts_{c}.jsonl" for c, _, _ in CONFIGS])
