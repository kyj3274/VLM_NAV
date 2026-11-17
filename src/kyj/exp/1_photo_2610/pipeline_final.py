"""최종 파이프라인 (실험 A~E에서 고른 설정): 사진 1장 → YOLOE + OCR + 호실 규칙 → 필요할 때만 VLM 고르기.

설정 (docs/07_2026-10-20_파이프라인_정확도_개선.md)
  1단계  YOLOE yoloe-11l-seg 1280 (pipeline.py 그대로) + PaddleOCR mobile_det / korean mobile_rec, 입력 긴 변 2048 (실험 A m2048)
        호실 규칙 room_rules.BEST = 첫 자리 다수결 (실험 B·C)
  2단계  qwen3-vl:4b-instruct (Ollama), 한국어 질문, 답만 (실험 D d3h, 실험 E)
        R 애매 호실 후보 / E 엘리베이터 후보(conf ≥ 0.15, 1단계 미확정) / S 비상구표시(conf ≥ 0.15) 화살표
        호실만 물으면 조각만, E·S가 있으면 전체 사진(1024) + 조각

실험 코드(exp_a_ocr, exp_bc_rules, exp_d_vlm)의 함수를 그대로 불러 써서 실험 결과와 같은 동작을 보장한다.

시간 (사진 1장, 서버에서 잼)
  latency_seconds = YOLOE + OCR + 규칙 + VLM(줄이기·자르기·인코딩·요청)   ← 실험 A~E와 같은 기준 (파일 디코딩 제외)
  latency_with_decode_seconds = 위 + 원본 JPEG 디코딩
  첫 사진 전에 YOLOE·OCR·VLM을 예열하고 그 시간은 뺀다.

사용 (서버):
  cd /workspace/capstone/src/kyj
  ./gpu_lock.sh 3 ./gpu_lock.sh 7 env CUDA_VISIBLE_DEVICES=3 .venv/bin/python pipeline_final.py --name final_all
  → /workspace/capstone/outputs/kyj/exp_<name>.jsonl
"""

import argparse
from pathlib import Path
import json
import os
import statistics
import time

import requests
from PIL import Image

import exp_a_ocr
import exp_d_vlm
import pipeline
import room_rules
import run_task

OUT = run_task.OUTPUT_DIR
CFG = "m2048"
OCR_MAX_SIDE = 2048
RULES = room_rules.BEST          # {"majority": True}
VARIANT = "d3h"
EL_TH = 0.15
EX_TH = 0.15


def load_models():
    cwd = os.getcwd()
    yoloe = pipeline.load_yoloe()   # models/ 로 chdir 함
    os.chdir(cwd)
    ocr = exp_a_ocr.load_ocr("m", {})
    return yoloe, ocr


def ask_vlm(url, model, options, img, q):
    schema, prompt = exp_d_vlm.schema_and_prompt(q, VARIANT)
    payload = {"model": model, "format": schema, "stream": False, "keep_alive": "30m", "options": options,
               "messages": [{"role": "user", "content": prompt, "images": exp_d_vlm.images_for(img, q, VARIANT)}]}
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=(10, 60))
        resp.raise_for_status()
        body = resp.json()
        ans = json.loads(body["message"]["content"])
        ans["_eval_count"] = body.get("eval_count")
        ans["_prompt_eval_count"] = body.get("prompt_eval_count")
    except Exception as error:  # noqa: BLE001  오류면 1단계 결과 유지
        ans = {"error": f"{type(error).__name__}: {error}"}
    return ans


def process(yoloe, ocr, img, url, model, options):
    """디코딩된 사진 1장 처리. 반환: 기록용 dict (시간 포함)."""
    t0 = time.perf_counter()
    detections = pipeline.detect(yoloe, img)
    t1 = time.perf_counter()
    texts_all, size = exp_a_ocr.read_all(ocr, img, OCR_MAX_SIDE)
    t2 = time.perf_counter()
    texts = [t for t in texts_all if t["score"] >= pipeline.OCR_THRESHOLD]
    combined, _ = pipeline.combine(detections, texts)
    rec = {"result": combined, "texts_all": texts_all, "ocr_input_size": size, "detections": detections}
    kept, base, q = exp_d_vlm.plan(rec, RULES, EL_TH, EX_TH)
    t3 = time.perf_counter()
    called = bool(q["rooms"] or q["elev"] or q["signs"])
    ans = ask_vlm(url, model, options, img, q) if called else None
    result, notes = exp_d_vlm.apply(base, kept, q, ans, texts_all)
    t4 = time.perf_counter()
    return {"result": result, "base_result": base, "vlm_called": called, "questions": q, "vlm": ans, "room_notes": notes,
            "latency_seconds": t4 - t0, "yoloe_seconds": t1 - t0, "ocr_seconds": t2 - t1, "rules_seconds": t3 - t2,
            "stage1_seconds": t3 - t0, "vlm_seconds": t4 - t3 if called else 0.0,
            "ocr_input_size": size, "detections": detections, "texts_all": texts_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--num-predict", type=int, default=96)
    ap.add_argument("--out-dir", default=str(OUT), help="결과 폴더 (다른 사람은 자기 outputs 폴더로)")
    args = ap.parse_args()

    paths = run_task.drop_duplicates(run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"]))
    if args.ids_file:
        wanted = {l.strip() for l in open(args.ids_file) if l.strip()}
        paths = [p for p in paths if run_task.image_id(p) in wanted]
    if args.limit:
        paths = paths[:args.limit]
    options = {**run_task.OPTIONS, "num_predict": args.num_predict}
    print(f"사진 {len(paths)}장", flush=True)

    t = time.perf_counter()
    yoloe, ocr = load_models()
    print(f"모델 로드 {time.perf_counter() - t:.1f}초", flush=True)
    # 예열 (시간 제외): 5층 큰 사진으로 YOLOE·OCR, 작은 이미지로 VLM 올리기
    warm = Image.open(paths[-1]).convert("RGB")
    for _ in range(2):
        pipeline.detect(yoloe, warm)
        exp_a_ocr.read_all(ocr, warm, OCR_MAX_SIDE)
    w = Image.new("RGB", (64, 64), (128, 128, 128))
    requests.post(f"{args.url}/api/chat", json={"model": args.model, "stream": False, "keep_alive": "30m", "options": options,
                  "messages": [{"role": "user", "content": "ok", "images": [exp_d_vlm.encode(w, 64)]}]}, timeout=300)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exp_{args.name}.jsonl"
    lat, lat_dec = [], []
    with open(path, "w", encoding="utf-8") as f:
        for n, p in enumerate(paths, 1):
            iid = run_task.image_id(p)
            td = time.perf_counter()
            img = Image.open(p).convert("RGB")
            decode_s = time.perf_counter() - td
            r = process(yoloe, ocr, img, args.url, args.model, options)
            out = {"image_id": iid, "task": "fields", "success": True,
                   "method": f"final: yoloe+ocr:{CFG}+rules(majority)+vlm:{VARIANT}", "model": args.model,
                   **r, "decode_seconds": decode_s, "latency_with_decode_seconds": r["latency_seconds"] + decode_s}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            lat.append(r["latency_seconds"]); lat_dec.append(out["latency_with_decode_seconds"])
            q = r["questions"]
            tag = f"VLM {r['vlm_seconds']:.2f}s R{len(q['rooms'])} E{len(q['elev'])} S{len(q['signs'])}" if r["vlm_called"] else "-"
            print(f"[{n}/{len(paths)}] {iid} {r['latency_seconds']:.2f}s (1단계 {r['stage1_seconds']:.2f}) {tag} "
                  f"rooms={r['result']['rooms']}", flush=True)
    print(f"시간 중앙값 {statistics.median(lat):.2f}s 평균 {statistics.mean(lat):.2f}s 최대 {max(lat):.2f}s | "
          f"디코딩 포함 중앙값 {statistics.median(lat_dec):.2f}s", flush=True)
    print(f"결과 파일: {path}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
