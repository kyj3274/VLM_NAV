"""실험 A: OCR 입력 설정 비교 (해상도 / 글자 위치 모델 / det 파라미터).

- YOLOE는 pipeline.py 현재 설정 그대로 한 번만 돌려 exp_a_yoloe.jsonl 에 저장하고 재사용.
- 설정마다 176장 OCR 원자료(점수 0 이상 전부: text, score, box)를 저장하고,
  pipeline.combine(점수 0.5 이상 글자)을 그대로 적용한 결과를 같은 줄에 넣는다.
- latency_seconds = YOLOE 시간 + OCR 시간 (사진 읽기 제외: 기존 pipeline 1단계와 같은 기준).
  사진 읽기(디코딩) 시간은 load_seconds 로 따로 기록.

사용법 (서버):
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python exp_a_ocr.py --configs m1024,m2048,s2048
결과: outputs/kyj/exp_a_<설정>.jsonl, outputs/kyj/exp_a_yoloe.jsonl
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
from PIL import Image

import pipeline
import run_task

OUT = run_task.OUTPUT_DIR
REC = "korean_PP-OCRv5_mobile_rec"
DET = {"m": "PP-OCRv5_mobile_det", "s": "PP-OCRv5_server_det"}

# 이름 → (위치 모델, 입력 긴 변(None=원본), PaddleOCR 추가 인자)
# PaddleOCR 기본: limit_type=min, limit_side_len=736, max_side_limit=4000
#  → 입력 짧은 변이 736 이상이면 입력 크기 그대로 위치 찾기(긴 변 4000 넘으면 4000으로 줄임)
CONFIGS = {
    "m1024": ("m", 1024, {}),
    "m2048": ("m", 2048, {}),
    "m2560": ("m", 2560, {}),
    "m3072": ("m", 3072, {}),
    "morig": ("m", None, {"text_det_max_side_limit": 6000}),
    "s1024": ("s", 1024, {}),
    "s2048": ("s", 2048, {}),
    "s2560": ("s", 2560, {}),
    "s3072": ("s", 3072, {}),
    "sorig": ("s", None, {"text_det_max_side_limit": 6000}),
    # 인식용 글자 조각은 원본에서 자르고, 위치 찾기만 긴 변 2048로 (속도/정확도 절충)
    "morig_det2048": ("m", None, {"text_det_limit_type": "max", "text_det_limit_side_len": 2048}),
    "sorig_det2048": ("s", None, {"text_det_limit_type": "max", "text_det_limit_side_len": 2048}),
    # det 파라미터 (기본 thresh 0.3, box_thresh 0.6)
    "m2048_box05": ("m", 2048, {"text_det_box_thresh": 0.5}),
    "m2048_th02": ("m", 2048, {"text_det_thresh": 0.2, "text_det_box_thresh": 0.5}),
}


def images():
    return run_task.drop_duplicates(run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"]))


def run_yoloe(paths):
    path = OUT / "exp_a_yoloe.jsonl"
    if path.exists() and sum(1 for _ in open(path)) == len(paths):
        print(f"YOLOE 재사용: {path}")
        return {r["image_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
    cwd = os.getcwd()
    model = pipeline.load_yoloe()
    os.chdir(cwd)
    warm = Image.open(paths[0]).convert("RGB")
    for _ in range(2):
        pipeline.detect(model, warm)
    rows = {}
    with open(path, "w", encoding="utf-8") as out:
        for p in paths:
            image = Image.open(p).convert("RGB")
            t0 = time.perf_counter()
            detections = pipeline.detect(model, image)
            sec = time.perf_counter() - t0
            row = {"image_id": run_task.image_id(p), "yoloe_seconds": sec, "detections": detections}
            rows[row["image_id"]] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"YOLOE 완료: {path} 중앙값 {statistics.median(r['yoloe_seconds'] for r in rows.values()):.3f}초")
    del model
    import torch
    torch.cuda.empty_cache()
    return rows


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


def run_config(name, paths, yolo):
    det, max_side, extra = CONFIGS[name]
    ocr = load_ocr(det, extra)
    pkw = {k: v for k, v in extra.items() if k in PREDICT_ONLY}
    warm = Image.open(paths[-1]).convert("RGB")  # 5층 큰 사진으로 예열
    for _ in range(2):
        read_all(ocr, warm, max_side, pkw)
    path = OUT / f"exp_a_{name}.jsonl"
    lat = []
    with open(path, "w", encoding="utf-8") as out:
        for index, p in enumerate(paths, start=1):
            iid = run_task.image_id(p)
            tl = time.perf_counter()
            image = Image.open(p).convert("RGB")
            t0 = time.perf_counter()
            texts_all, size = read_all(ocr, image, max_side, pkw)
            t1 = time.perf_counter()
            texts = [t for t in texts_all if t["score"] >= pipeline.OCR_THRESHOLD]
            detections = yolo[iid]["detections"]
            result, reasons = pipeline.combine(detections, texts)
            latency = yolo[iid]["yoloe_seconds"] + (t1 - t0)
            lat.append(latency)
            out.write(json.dumps({
                "image_id": iid, "task": "fields", "method": f"yoloe+ocr:{name}", "config": name,
                "ocr_config": {"det": DET[det], "rec": REC, "max_side": max_side, **extra},
                "result": result, "vlm_reasons": reasons, "success": True, "vlm_called": False,
                "latency_seconds": latency, "yoloe_seconds": yolo[iid]["yoloe_seconds"],
                "ocr_seconds": t1 - t0, "load_seconds": t0 - tl, "ocr_input_size": size,
                "detections": detections, "texts": texts, "texts_all": texts_all,
            }, ensure_ascii=False) + "\n")
            if index % 20 == 0:
                print(f"  [{name} {index}/{len(paths)}] 중앙값 {statistics.median(lat):.2f}초", flush=True)
    print(f"{name}: {path} 1단계 중앙값 {statistics.median(lat):.3f}초 평균 {statistics.mean(lat):.3f}초", flush=True)
    del ocr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()
    paths = images()
    print(f"사진 {len(paths)}장", flush=True)
    yolo = run_yoloe(paths)
    for name in args.configs.split(","):
        started = time.time()
        try:
            run_config(name, paths, yolo)
        except Exception as error:  # 메모리 부족 등은 기록하고 다음 설정으로
            print(f"{name}: 실패 {type(error).__name__}: {error}", flush=True)
        print(f"{name}: 총 {time.time() - started:.0f}초", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
