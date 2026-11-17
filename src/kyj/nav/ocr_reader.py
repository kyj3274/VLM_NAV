"""PaddleOCR 불러오기와 읽기 (한국어 인식 모델 + 글자 위치 모델). 길안내 서비스가 쓴다.
예전 OCR 설정 실험(exp_a_ocr.py, docs/16 1절)에서 고른 설정(mobile 위치 모델, 긴 변 2048)을 그대로 쓴다.
"""

import os

import numpy as np

REC = "korean_PP-OCRv5_mobile_rec"
DET = {"m": "PP-OCRv5_mobile_det", "s": "PP-OCRv5_server_det"}

PREDICT_ONLY = {"text_det_max_side_limit"}  # 생성자에서 안 받고 predict()에서만 받는 인자


def load_ocr(det, extra):
    extra = {k: v for k, v in extra.items() if k not in PREDICT_ONLY}
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="korean",
        text_detection_model_name=DET[det],
        text_recognition_model_name=REC,  # 위치 모델 지정 시 인식이 중국어로 바뀌므로 명시
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0",
        **extra,
    )


def read_all(ocr, image, max_side, predict_kw=None):
    small = image.copy()
    if max_side:
        small.thumbnail((max_side, max_side))
    array = np.array(small)[:, :, ::-1]
    if predict_kw:  # PaddleOCR 래퍼가 max_side_limit을 안 넘겨주므로 내부 paddlex 파이프라인을 직접 호출
        result = list(ocr.paddlex_pipeline.predict(array, **predict_kw))[0]
    else:
        result = ocr.predict(array)[0]
    width, height = small.size
    texts = []
    for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"].tolist()):
        text = text.strip()
        if text:
            texts.append({"text": text, "score": round(float(score), 3),
                          "box": [round(box[0] / width, 4), round(box[1] / height, 4),
                                  round(box[2] / width, 4), round(box[3] / height, 4)]})
    return texts, list(small.size)
