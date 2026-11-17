"""YOLOE + PaddleOCR (+ 필요할 때만 VLM) 파이프라인. 결과는 fields 형식(jsonl).

1단계: YOLOE(물체 위치) + OCR(글자) + 합치기 규칙 → fields
2단계: 1단계 결과 중 확인이 필요한 사진만 VLM(qwen3-vl)에 사진 + 단서를 주고 짧게 확인

사용법 (서버, kyj 전용 환경):
  CUDA_VISIBLE_DEVICES=3 src/kyj/.venv/bin/python src/kyj/pipeline.py --stage 1
  src/kyj/.venv/bin/python src/kyj/pipeline.py --stage 2 --ollama-url http://127.0.0.1:11436
"""

import argparse
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import run_task

KYJ = Path(__file__).resolve().parent
MODEL_DIR = KYJ / "models"
OUTPUT_DIR = run_task.OUTPUT_DIR

YOLOE_WEIGHTS = "yoloe-11l-seg.pt"
YOLOE_SIZE = 1280
YOLOE_THRESHOLD = 0.15
OCR_MAX_SIDE = 2048
OCR_THRESHOLD = 0.5

# 3A: 물체마다 영어 표현 여러 개의 평균 (Mac 10장 조정 결과)
PHRASES = {
    "문": ["door", "closed door", "office door", "glass door", "sliding door"],
    "엘리베이터": ["elevator", "elevator doors", "stainless steel elevator door", "lift", "elevator with call button panel"],
    "계단": ["staircase", "stairs", "stairwell"],
    "화장실": ["restroom sign", "toilet sign", "restroom entrance", "bathroom entrance"],
    "비상구표시": ["green emergency exit sign", "exit sign", "running man exit sign"],
    "사람": ["person"],
    "상자": ["cardboard box"],
}
LANDMARK_NAMES = ["문", "엘리베이터", "계단", "화장실", "비상구표시"]
OBSTACLE_NAMES = ["사람", "상자"]

ROOM = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")
FLOOR_DIGIT = re.compile(r"^[1-9]$")
FLOOR_TEXT = re.compile(r"^(B?[1-9])\s*(F|층)$", re.IGNORECASE)
HANGUL = re.compile(r"[가-힣]")
TOILET = re.compile(r"(TOILET|화장실|RESTROOM)", re.IGNORECASE)


def clock(center_x):
    return "11시" if center_x < 1 / 3 else "12시" if center_x < 2 / 3 else "1시"


# ---------------------------------------------------------------
# 1단계
# ---------------------------------------------------------------

def load_yoloe():
    import torch
    from ultralytics import YOLOE

    MODEL_DIR.mkdir(exist_ok=True)
    os.chdir(MODEL_DIR)
    model = YOLOE(YOLOE_WEIGHTS)
    rows = []
    for phrases in PHRASES.values():
        pe = model.get_text_pe(phrases).mean(dim=1, keepdim=True)
        rows.append(pe / pe.norm(dim=-1, keepdim=True))
    model.set_classes(list(PHRASES), torch.cat(rows, dim=1))
    return model


def load_ocr():
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="korean",
        text_detection_model_name="PP-OCRv5_mobile_det",
        # 위치 모델을 지정하면 인식 모델이 기본(중국어)으로 바뀌므로 한국어 모델을 명시
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0",
    )


def detect(model, image):
    result = model.predict(image, imgsz=YOLOE_SIZE, conf=0.05, device=0, verbose=False)[0]
    names = list(PHRASES)
    width, height = image.size
    return [
        {"name": names[int(c)], "conf": round(s, 3),
         "box": [round(b[0] / width, 3), round(b[1] / height, 3), round(b[2] / width, 3), round(b[3] / height, 3)]}
        for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]


def read_text(ocr, image):
    small = image.copy()
    small.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
    array = np.array(small)[:, :, ::-1]  # RGB → BGR
    result = ocr.predict(array)[0]
    width, height = small.size
    texts = []
    for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"].tolist()):
        text = text.strip()
        if text and score >= OCR_THRESHOLD:
            texts.append({"text": text, "score": round(float(score), 3),
                          "box": [round(box[0] / width, 3), round(box[1] / height, 3), round(box[2] / width, 3), round(box[3] / height, 3)]})
    return texts


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def is_floor_digit_over(text, box):
    # 숫자 한 자리가 큰 문(문·엘리베이터 상자)의 위쪽 가까이, 가로로 겹치는 곳에 있음
    tx, ty = center(text["box"])
    above = text["box"][3] <= box[1] + 0.05 and box[1] - text["box"][3] < 0.25
    return box[0] - 0.05 <= tx <= box[2] + 0.05 and above


def combine(detections, texts):
    found = [d for d in detections if d["conf"] >= YOLOE_THRESHOLD]
    rooms, floors, spaces, landmarks, directions = [], [], [], set(), []
    elevator_candidate = False

    room_boxes = []
    for t in texts:
        value = t["text"].replace("호", "").strip()
        if ROOM.match(value) and value not in rooms:
            rooms.append(value)
            room_boxes.append((value, t["box"]))
        match = FLOOR_TEXT.match(t["text"].replace(" ", ""))
        if match and f"{match.group(1).upper()}F" not in floors:
            floors.append(f"{match.group(1).upper()}F")

    # 공간 이름: 호실 번호 글자 바로 아래나 오른쪽에 붙은 한글 줄
    for value, rbox in room_boxes:
        for t in texts:
            if t["box"] is rbox or not HANGUL.search(t["text"]) or ROOM.match(t["text"]):
                continue
            dx = abs(t["box"][0] - rbox[0])
            dy = t["box"][1] - rbox[3]
            right = abs(t["box"][1] - rbox[1]) < 0.03 and 0 <= t["box"][0] - rbox[2] < 0.1
            if (dx < 0.08 and -0.01 <= dy < 0.06) or right:
                name = t["text"].replace(" ", "")
                if name not in spaces:
                    spaces.append(name)
    for t in texts:
        if TOILET.search(t["text"]) and "TOILET" not in spaces and "화장실" not in spaces:
            spaces.append("TOILET" if t["text"].isascii() else "화장실")

    # 엘리베이터: 큰 문 + 그 위 숫자 한 자리 → 확정. 탐지 모델만 엘리베이터라고 하면 후보로 남김
    digits = [t for t in texts if FLOOR_DIGIT.match(t["text"])]
    for d in found:
        if d["name"] in ("문", "엘리베이터"):
            over = [t for t in digits if is_floor_digit_over(t, d["box"])]
            height = d["box"][3] - d["box"][1]
            if over and height > 0.3:
                landmarks.add("엘리베이터")
                floor = f"{over[0]['text']}F"
                if floor not in floors:
                    floors.append(floor)
                directions.append({"target": "엘리베이터", "type": "RELATIVE_POSITION", "value": clock(center(d["box"])[0])})
            elif d["name"] == "엘리베이터":
                elevator_candidate = True
                landmarks.add("문")
            else:
                landmarks.add("문")
        elif d["name"] in LANDMARK_NAMES:
            landmarks.add(d["name"])
            if d["name"] in ("계단", "화장실", "비상구표시"):
                item = {"target": d["name"], "type": "RELATIVE_POSITION", "value": clock(center(d["box"])[0])}
                if item not in directions:
                    directions.append(item)
    if any(TOILET.search(t["text"]) for t in texts):
        landmarks.add("화장실")

    for value, rbox in room_boxes:
        directions.append({"target": value, "type": "RELATIVE_POSITION", "value": clock(center(rbox)[0])})

    obstacles = sorted({d["name"] for d in found if d["name"] in OBSTACLE_NAMES})
    result = {"rooms": rooms, "floors": floors, "spaces": spaces,
              "landmarks": sorted(landmarks), "directions": directions, "obstacles": obstacles}

    # 2단계에서 VLM이 확인할 이유
    reasons = []
    if elevator_candidate and "엘리베이터" not in landmarks:
        reasons.append("엘리베이터 후보")
    if "비상구표시" in landmarks:
        reasons.append("화살표 가능")
    if not rooms and len(texts) >= 3:
        reasons.append("글자는 있는데 호실 없음")
    return result, reasons


def stage1(images):
    model = load_yoloe()
    ocr = load_ocr()
    path = OUTPUT_DIR / "pipeline_stage1.jsonl"
    with open(path, "w", encoding="utf-8") as out:
        for index, image_path in enumerate(images, start=1):
            image = Image.open(image_path).convert("RGB")
            t0 = time.perf_counter()
            detections = detect(model, image)
            t1 = time.perf_counter()
            texts = read_text(ocr, image)
            t2 = time.perf_counter()
            result, reasons = combine(detections, texts)
            record = {
                "image_id": run_task.image_id(image_path), "task": "fields", "method": "yoloe+ocr",
                "result": result, "vlm_reasons": reasons, "success": True,
                "latency_seconds": t2 - t0, "yoloe_seconds": t1 - t0, "ocr_seconds": t2 - t1,
                "detections": detections, "texts": texts,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(images)}] {record['image_id']} | {json.dumps(result, ensure_ascii=False)[:110]} | {t2 - t0:.2f}초")
    print(f"결과 파일: {path}")


# ---------------------------------------------------------------
# 2단계: 필요한 사진만 VLM 확인
# ---------------------------------------------------------------

VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "elevator": {"type": "boolean"},
        "arrows": {
            "type": "array", "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "value": {"type": "string", "enum": run_task.CLOCK_VALUES},
                },
                "required": ["target", "value"],
            },
        },
        "rooms": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
    "required": ["elevator", "arrows", "rooms"],
}


def vlm_prompt(result, record):
    texts = ", ".join(t["text"] for t in record["texts"]) or "없음"
    objects = ", ".join(sorted({d["name"] for d in record["detections"] if d["conf"] >= YOLOE_THRESHOLD})) or "없음"
    return (
        "시각장애인 실내 길안내를 위해 사진을 확인합니다.\n"
        f"글자 인식 결과: {texts}\n"
        f"물체 탐지 결과: {objects}\n\n"
        "아래 세 가지만 JSON으로 답하세요.\n"
        "1. elevator: 사진에 엘리베이터(호출 버튼이나 층 표시가 있는 금속 문)가 확실히 보이면 true, 아니면 false\n"
        "2. arrows: 표지판이나 유도등에 화살표 모양(→ ← ↑)이 그려져 있으면 그 방향을 시계 방향으로. "
        "12시는 정면, 3시는 오른쪽, 9시는 왼쪽. target은 화살표가 안내하는 대상(모르면 '알 수 없음'). "
        "화살표 모양이 없으면 빈 배열. 문패에는 화살표가 없습니다.\n"
        "3. rooms: 글자 인식 결과에 없지만 사진에서 확실히 읽히는 호실 번호(숫자 3자리). 없으면 빈 배열\n"
        "추측하지 마세요."
    )


def stage2(ollama_url, model_name):
    source = OUTPUT_DIR / "pipeline_stage1.jsonl"
    records = [json.loads(line) for line in open(source, encoding="utf-8")]
    path = OUTPUT_DIR / "pipeline_stage2.jsonl"
    paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
    with open(path, "w", encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            result = json.loads(json.dumps(record["result"]))
            vlm = None
            vlm_seconds = 0.0
            if record["vlm_reasons"]:
                started = time.perf_counter()
                image = Image.open(paths[record["image_id"]]).convert("RGB")
                image.thumbnail((1024, 1024))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                payload = {
                    "model": model_name, "format": VLM_SCHEMA, "stream": False,
                    "options": {**run_task.OPTIONS, "num_predict": 128},
                    "messages": [{"role": "user", "content": vlm_prompt(result, record),
                                  "images": [base64.b64encode(buffer.getvalue()).decode()]}],
                }
                try:
                    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=(10, 60))
                    response.raise_for_status()
                    vlm = json.loads(response.json()["message"]["content"])
                except Exception as error:
                    vlm = {"error": f"{type(error).__name__}: {error}"}
                vlm_seconds = time.perf_counter() - started

                if "error" not in vlm:
                    if "엘리베이터 후보" in record["vlm_reasons"] and vlm.get("elevator"):
                        result["landmarks"] = sorted(set(result["landmarks"]) | {"엘리베이터"})
                    for arrow in vlm.get("arrows", []):
                        result["directions"].append({"target": arrow["target"], "type": "SIGN_ARROW", "value": arrow["value"]})
                    for room in vlm.get("rooms", []):
                        room = room.replace("호", "").strip()
                        if ROOM.match(room) and room not in result["rooms"]:
                            result["rooms"].append(room)

            out.write(json.dumps({
                "image_id": record["image_id"], "task": "fields", "method": "yoloe+ocr+vlm",
                "result": result, "success": True, "vlm_reasons": record["vlm_reasons"], "vlm": vlm,
                "latency_seconds": record["latency_seconds"] + vlm_seconds, "vlm_seconds": vlm_seconds,
            }, ensure_ascii=False) + "\n")
            note = f"VLM {vlm_seconds:.1f}초 {record['vlm_reasons']}" if record["vlm_reasons"] else "VLM 안 부름"
            print(f"[{index}/{len(records)}] {record['image_id']} | {note}")
    print(f"결과 파일: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--ids", default=None, help="일부 사진만 (쉼표 구분)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11436")
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    args = parser.parse_args()

    if args.stage == 1:
        images = run_task.drop_duplicates(run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"]))
        if args.ids:
            wanted = set(args.ids.split(","))
            images = [p for p in images if run_task.image_id(p) in wanted]
        stage1(images)
    else:
        stage2(args.ollama_url.rstrip("/"), args.model)


if __name__ == "__main__":
    main()
