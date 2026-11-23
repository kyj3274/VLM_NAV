"""1단계 인식: YOLOE(물체 위치) + PaddleOCR(글자) + 합치기 규칙 → fields(rooms, floors, spaces, landmarks, directions).
길안내 서비스(navigator.py, vlm_client.py)가 쓴다. 예전 사진 1장 파이프라인(pipeline.py, docs/16 1절)의 1단계 부분과 같다.
"""

import os
import re
from pathlib import Path

import numpy as np

BASE = next((d for d in Path(__file__).resolve().parents if (d / "models").is_dir()), Path(__file__).resolve().parents[2])  # models/가 있는 위 폴더 (서버 src/kyj, Mac 저장소 맨 위)
# YOLOE 가중치 폴더: 서버는 <위 폴더>/models, Mac은 저장소의 local_models (둘 다 없으면 <위 폴더>/models에 내려받음)
_LOCAL = [d / "local_models" for d in Path(__file__).resolve().parents if (d / "local_models").is_dir()][:1]
MODEL_DIR = next((d for d in [BASE / "models"] + _LOCAL if (d / "yoloe-11l-seg.pt").is_file()), BASE / "models")

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



def _torch_device():
    """서버는 CUDA 0, Mac은 mps."""
    import torch
    return 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _paddle_device():
    try:
        import paddle
        return "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"

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
        device=_paddle_device(),
    )


def detect(model, image):
    result = model.predict(image, imgsz=YOLOE_SIZE, conf=0.05, device=_torch_device(), verbose=False)[0]
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
