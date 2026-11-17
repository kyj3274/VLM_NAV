"""정확도 위주 열린 어휘 탐지 모델 비교 (Mac, transformers). YOLOE 3A와 같은 표현·채점.

사용법: .venv/bin/python src/kyj/accurate_detectors.py 3F_006,4F_027,...
결과: outputs/kyj/det_{모델}.jsonl, 화면에 채점표
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from local_vision import OUT, clock  # noqa: E402
from yoloe_ablation import image_path  # noqa: E402
import yoloe_prompts  # noqa: E402

PHRASES = yoloe_prompts.PHRASES
PHRASE_TO_NAME = {p: n for n, ps in PHRASES.items() for p in ps}
DEVICE = "mps"

MODELS = [
    ("gdino_tiny", "grounding", "IDEA-Research/grounding-dino-tiny"),
    ("gdino_base", "grounding", "IDEA-Research/grounding-dino-base"),
    ("mm_gdino_tiny", "grounding", "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"),
    ("llmdet_tiny", "grounding", "iSEE-Laboratory/llmdet_tiny"),
    ("llmdet_base", "grounding", "iSEE-Laboratory/llmdet_base"),
    ("owlv2_base", "owl", "google/owlv2-base-patch16-ensemble"),
]


def to_name(label):
    label = label.strip().lower()
    if label in PHRASE_TO_NAME:
        return PHRASE_TO_NAME[label]
    # Grounding DINO는 표현 일부만 돌려줄 때가 있음 → 가장 많이 겹치는 표현으로
    best = max(PHRASE_TO_NAME, key=lambda p: len(set(p.split()) & set(label.split())))
    return PHRASE_TO_NAME[best] if set(best.split()) & set(label.split()) else None


def run_grounding(repo, ids, out):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(repo).to(DEVICE).eval()
    phrases = [p for ps in PHRASES.values() for p in ps]
    text = [phrases]
    for k, image_id in enumerate([ids[0]] + ids):  # 첫 장은 예열
        image = Image.open(image_path(image_id)).convert("RGB")
        started = time.perf_counter()
        inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[image.size[::-1]])[0]
        latency = time.perf_counter() - started
        if k == 0:
            continue
        labels = result.get("text_labels", result.get("labels"))
        detections = []
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
            name = to_name(label if isinstance(label, str) else phrases[int(label)])
            if name:
                detections.append({"name": name, "label": label, "conf": round(score, 3), "clock": clock((box[0] + box[2]) / 2 / image.size[0])})
        out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")


def run_owl(repo, ids, out):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(repo)
    model = Owlv2ForObjectDetection.from_pretrained(repo).to(DEVICE).eval()
    phrases = [p for ps in PHRASES.values() for p in ps]
    for k, image_id in enumerate([ids[0]] + ids):
        image = Image.open(image_path(image_id)).convert("RGB")
        started = time.perf_counter()
        inputs = processor(text=[phrases], images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        size = max(image.size)  # OWLv2는 정사각형으로 채워서 처리
        result = processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=torch.tensor([[size, size]]))[0]
        latency = time.perf_counter() - started
        if k == 0:
            continue
        detections = [
            {"name": PHRASE_TO_NAME[phrases[int(l)]], "label": phrases[int(l)], "conf": round(s, 3), "clock": clock((b[0] + b[2]) / 2 / image.size[0])}
            for b, s, l in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist())
        ]
        out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")


def main(ids):
    done = []
    for name, kind, repo in MODELS:
        path = OUT / f"det_{name}.jsonl"
        try:
            with open(path, "w", encoding="utf-8") as out:
                (run_grounding if kind == "grounding" else run_owl)(repo, ids, out)
            done.append(path.name)
            print("done", name)
        except Exception as error:  # 받을 수 없는 모델은 건너뜀
            print("skip", name, type(error).__name__, str(error)[:150])
            path.unlink(missing_ok=True)
    return done


if __name__ == "__main__":
    sample = sys.argv[1].split(",")
    files = main(sample)
    for threshold in (0.3, 0.2, 0.1):
        yoloe_prompts.THRESHOLD = threshold
        yoloe_prompts.score(sample, ["yoloe_prompts_3A_평균_11l.jsonl"] + files)
