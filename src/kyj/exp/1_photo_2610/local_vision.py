"""Mac에서 YOLOE(물체 탐지)와 PaddleOCR(글자 인식)을 사진 176장(review/img)에 돌린다.

결과: outputs/kyj/local_yoloe.jsonl, outputs/kyj/local_ocr_{server,mobile}.jsonl
사용법: .venv/bin/python src/kyj/local_vision.py [yoloe|ocr-server|ocr-mobile ...]
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
IMAGES = ROOT / "review" / "img"
OUT = ROOT / "outputs" / "kyj"
MODELS = ROOT / "local_models"
FLOOR_IDS = {"3rdFloor": "3F", "4thFloor": "4F", "5thFloor": "5F"}

# 랜드마크 23개 → YOLOE에 줄 영어 문구 (사진을 보고 고치지 않은 첫 버전)
PROMPTS = {
    "엘리베이터": "elevator",
    "계단": "staircase",
    "화장실표시": "restroom sign",
    "화장실입구": "restroom entrance",
    "비상구표시": "green emergency exit sign",
    "안내판": "evacuation map board",
    "소화전": "fire hose cabinet",
    "소화기": "fire extinguisher",
    "점자블록": "tactile paving",
    "정수기": "water dispenser",
    "자판기": "vending machine",
    "출입문": "door",
    "유리문": "glass door",
    "복도끝": "end of corridor",
    "기둥": "pillar",
    "게시판": "bulletin board",
    "사물함": "locker",
    "액자": "picture frame",
    "CCTV": "security camera",
    "소파": "sofa",
    "노란바닥띠": "yellow floor stripe",
    "미닫이문": "sliding door",
    "셔터": "rolling shutter",
    # 랜드마크는 아니지만 길안내에 쓸 수 있는 것
    "문패": "room number sign",
    "사람": "person",
    "상자": "cardboard box",
}

ROOM_PATTERN = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")


ONLY_IDS = None  # --ids 3F_006,4F_027 로 일부만
SUFFIX = ""


def images():
    for floor, short in FLOOR_IDS.items():
        for path in sorted((IMAGES / floor).glob("*.jpeg"), key=lambda p: int(p.stem)):
            image_id = f"{short}_{int(path.stem):03d}"
            if ONLY_IDS is None or image_id in ONLY_IDS:
                yield image_id, path


def clock(center_x):
    # 화면 3등분: 왼쪽 11시, 가운데 12시, 오른쪽 1시 (docs/05 규칙)
    return "11시" if center_x < 1 / 3 else "12시" if center_x < 2 / 3 else "1시"


def run_yoloe():
    from ultralytics import YOLOE

    os.chdir(MODELS)
    model = YOLOE("yoloe-11l-seg.pt")
    names = list(PROMPTS)
    texts = [PROMPTS[n] for n in names]
    model.set_classes(texts, model.get_text_pe(texts))

    with open(OUT / f"local_yoloe{SUFFIX}.jsonl", "w", encoding="utf-8") as out:
        for image_id, path in images():
            started = time.perf_counter()
            result = model.predict(str(path), conf=0.05, device="mps", verbose=False)[0]
            latency = time.perf_counter() - started
            width, height = result.orig_shape[1], result.orig_shape[0]
            detections = []
            for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2 / width
                detections.append({
                    "name": names[int(cls)],
                    "conf": round(score, 3),
                    "box": [round(x1 / width, 3), round(y1 / height, 3), round(x2 / width, 3), round(y2 / height, 3)],
                    "clock": clock(center_x),
                })
            out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "detections": detections}, ensure_ascii=False) + "\n")
    print("yoloe done")


def run_ocr(variant):
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="korean",
        text_detection_model_name=f"PP-OCRv5_{variant}_det",
        # 위치 모델을 지정하면 인식 모델이 기본(중국어)으로 바뀌므로 한국어 모델을 명시
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    with open(OUT / f"local_ocr_{variant}{SUFFIX}.jsonl", "w", encoding="utf-8") as out:
        for image_id, path in images():
            started = time.perf_counter()
            result = ocr.predict(str(path))[0]
            latency = time.perf_counter() - started
            height, width = result["doc_preprocessor_res"]["output_img"].shape[:2] if result.get("doc_preprocessor_res") else (None, None)
            texts = []
            for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"].tolist()):
                if not text.strip():
                    continue
                texts.append({"text": text, "score": round(float(score), 3), "box": box})
            rooms = sorted({t["text"].replace("호", "").strip() for t in texts
                            if t["score"] >= 0.5 and ROOM_PATTERN.match(t["text"].replace("호", "").strip())})
            out.write(json.dumps({"image_id": image_id, "latency_seconds": latency, "size": [width, height],
                                  "rooms": rooms, "texts": texts}, ensure_ascii=False) + "\n")
    print(f"ocr {variant} done")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--ids" in args:
        i = args.index("--ids")
        ONLY_IDS = set(args[i + 1].split(","))
        SUFFIX = "_sample"
        args = args[:i] + args[i + 2:]
    for job in args or ["yoloe", "ocr-server", "ocr-mobile"]:
        if job == "yoloe":
            run_yoloe()
        else:
            run_ocr(job.split("-")[1])
