"""YOLOE 4번 실험: 엘리베이터만 예시 사진(visual prompt)으로. 나머지 물체는 3A(영어 표현 평균) 그대로.

예시 사진은 조정용 10장에 없는 4F_035, 5F_034 (엘리베이터 상자는 1000x750 기준으로 직접 확인).
비교: 엘리베이터 = 글(3A) / 사진만 / 사진+글, 각각 겹친 상자 정리(agnostic NMS) 켬·끔.

사용법: .venv/bin/python src/kyj/yoloe_visual.py 3F_006,4F_027,...
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from local_vision import MODELS, OUT, clock  # noqa: E402
from yoloe_ablation import image_path  # noqa: E402
import yoloe_prompts  # noqa: E402

EXAMPLES = [
    ("4F_035", [0, 250, 375, 750]),
    ("5F_034", [380, 110, 715, 750]),
]
SIZE = 1280


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def text_embeddings(model):
    rows = []
    for phrases in yoloe_prompts.PHRASES.values():
        rows.append(normalize(model.get_text_pe(phrases).mean(dim=1, keepdim=True)))
    return torch.cat(rows, dim=1)  # (1, 클래스 수, d)


def elevator_visual(model):
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    vectors = []
    for image_id, box in EXAMPLES:
        predictor = YOLOEVPSegPredictor(overrides={"task": "segment", "mode": "predict", "imgsz": SIZE, "device": "mps", "verbose": False, "save": False})
        predictor.set_prompts({"bboxes": np.array([box], dtype=float), "cls": np.array([0])})
        predictor.setup_model(model=model.model, verbose=False)
        vectors.append(normalize(predictor.get_vpe(str(image_path(image_id))).float()))
    return normalize(torch.cat(vectors, dim=1).mean(dim=1, keepdim=True))  # (1, 1, d)


def main(ids):
    from ultralytics import YOLOE

    os.chdir(MODELS)
    model = YOLOE("yoloe-11l-seg.pt")
    names = list(yoloe_prompts.PHRASES)
    elevator = names.index("엘리베이터")
    text = text_embeddings(model)
    visual = elevator_visual(model).to(text.device, text.dtype)
    print("엘리베이터 글-사진 특징 유사도:", round(float((text[0, elevator] * visual[0, 0]).sum()), 3))

    variants = {
        "글": text[0, elevator],
        "사진만": visual[0, 0],
        "사진+글": normalize(text[0, elevator] + visual[0, 0]),
    }
    configs = []
    for label, vector in variants.items():
        embeddings = text.clone()
        embeddings[0, elevator] = vector
        for agnostic in (True, False):
            config = f"4번_엘리베이터{label}_{'겹침정리' if agnostic else '물체별'}"
            model.set_classes(names, embeddings)
            model.predict(str(image_path(ids[0])), imgsz=SIZE, device="mps", verbose=False)
            with open(OUT / f"yoloe_visual_{config}.jsonl", "w", encoding="utf-8") as out:
                for image_id in ids:
                    started = time.perf_counter()
                    result = model.predict(str(image_path(image_id)), imgsz=SIZE, conf=0.05, device="mps",
                                           agnostic_nms=agnostic, verbose=False)[0]
                    latency = time.perf_counter() - started
                    width = result.orig_shape[1]
                    detections = [
                        {"name": names[int(c)], "conf": round(s, 3), "clock": clock((b[0] + b[2]) / 2 / width)}
                        for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
                    ]
                    out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")
            configs.append(f"yoloe_visual_{config}.jsonl")

    for threshold in (0.15, 0.1):
        yoloe_prompts.THRESHOLD = threshold
        yoloe_prompts.score(ids, configs)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
