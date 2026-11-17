"""영상 프레임으로 공간 맥락 실험 (실험 A: 사진 1장 맥락, 실험 B: 연속 장면 위치 추적).

프레임: data/video/frames/<순서>_<영상>/f000.jpg … (1초에 1장, 순서 = 파일 이름)
  1_start_졸프_영상3            건물 입구 → 복도 → 엘리베이터 앞 (첫 상황)
  2a_after_elevator_졸프_영상1  엘리베이터에서 내린 뒤 상황 1
  2b_after_elevator_졸프_영상2  엘리베이터에서 내린 뒤 상황 2
  2a·2b는 둘 다 1의 마지막 상태에서 이어서 시작한다 (갈래 2개).

단계
  stage1  YOLOE + OCR + 호실 규칙 (pipeline_final과 같은 설정, VLM 호실 확인은 생략)
  A0      VLM에 사진만 → 맥락 JSON
  A1      VLM에 사진 + 1단계 결과(호실 후보, 글자, 물체) → 맥락 JSON
  B       VLM에 이전 상태 + 사진 + 1단계 결과 → 새 상태 (유지 KEEP / 갱신 UPDATE / 수정 REVISE / 모름 LOST)

사용 (서버):
  cd /workspace/capstone/src/kyj
  ./gpu_lock.sh 3 ./gpu_lock.sh 7 env CUDA_VISIBLE_DEVICES=3 .venv/bin/python video_context.py \
      --frames /workspace/capstone/outputs/kyj/video/frames --out-dir /workspace/capstone/outputs/kyj/video
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests
from PIL import Image

import exp_a_ocr
import exp_d_vlm
import pipeline
import room_rules
import run_task

OCR_MAX_SIDE = 2048
VLM_SIDE = 1024
SCENES = ["CORRIDOR", "LOBBY", "ELEVATOR_HALL", "ENTRANCE", "INTERSECTION", "ROOM_DOOR", "INSIDE_ELEVATOR", "UNKNOWN"]
PATHS = ["FORWARD", "LEFT", "RIGHT", "FORWARD_LEFT", "FORWARD_RIGHT", "LEFT_RIGHT", "ALL", "DEAD_END", "UNKNOWN"]
FLOORS = ["1층", "2층", "3층", "4층", "5층", "모름"]
UPDATES = ["KEEP", "UPDATE", "REVISE", "LOST"]
LEVELS = ["HIGH", "MEDIUM", "LOW"]
SIDE = {"11시": "왼쪽", "12시": "가운데", "1시": "오른쪽"}

# ---------------------------------------------------------------- 1단계

def stage1(yoloe, ocr, img):
    t0 = time.perf_counter()
    detections = pipeline.detect(yoloe, img)
    texts_all, size = exp_a_ocr.read_all(ocr, img, OCR_MAX_SIDE)
    texts = [t for t in texts_all if t["score"] >= pipeline.OCR_THRESHOLD]
    combined, _ = pipeline.combine(detections, texts)
    rec = {"result": combined, "texts_all": texts_all, "ocr_input_size": size, "detections": detections}
    kept, base, _ = exp_d_vlm.plan(rec, room_rules.BEST, 0.15, 0.15)
    return {"result": base, "texts_all": texts_all, "detections": detections, "ocr_input_size": size,
            "stage1_seconds": time.perf_counter() - t0}


def hints(s1):
    """1단계 결과를 VLM에 줄 짧은 글로. 틀릴 수 있다고 알린다."""
    r = s1["result"]
    texts = sorted((t for t in s1["texts_all"] if t["score"] >= 0.5), key=lambda t: -(t["box"][3] - t["box"][1]))
    words = [t["text"] for t in texts[:12]]
    objs = []
    for d in sorted(s1["detections"], key=lambda d: -d["conf"]):
        if d["conf"] < 0.15:
            continue
        side = SIDE[pipeline.clock(pipeline.center(d["box"])[0])]
        item = f"{d['name']}({side})"
        if item not in objs:
            objs.append(item)
    return ("보조 인식 결과 (자동 인식이라 틀릴 수 있음, 사진으로 확인할 것):\n"
            f"- 호실 번호 후보 (가까운 순): {', '.join(r['rooms']) or '없음'}\n"
            f"- 읽힌 글자 (큰 글자 순): {', '.join(words) or '없음'}\n"
            f"- 물체: {', '.join(objs[:8]) or '없음'}")


# ---------------------------------------------------------------- VLM

def room_items(cands):
    return {"type": "string", "enum": cands} if cands else {"type": "string", "maxLength": 8}


def schema_a(cands):
    return {"type": "object", "properties": {
        "visible_rooms": {"type": "array", "items": room_items(cands), "maxItems": 4},
        "scene_type": {"type": "string", "enum": SCENES},
        "navigation_context": {"type": "object", "properties": {
            "location_evidence": {"type": "string", "maxLength": 80},
            "decision_point": {"type": "boolean"},
            "visible_path": {"type": "string", "enum": PATHS},
            "uncertainty": {"type": "string", "maxLength": 80},
        }, "required": ["location_evidence", "decision_point", "visible_path", "uncertainty"]},
    }, "required": ["visible_rooms", "scene_type", "navigation_context"]}


def schema_b(cands):
    return {"type": "object", "properties": {
        "observed_rooms": {"type": "array", "items": room_items(cands), "maxItems": 4},
        "observed_landmarks": {"type": "string", "maxLength": 60},
        "update": {"type": "string", "enum": UPDATES},
        "floor_estimate": {"type": "string", "enum": FLOORS},
        "location_estimate": {"type": "string", "maxLength": 60},
        "confidence": {"type": "string", "enum": LEVELS},
        "reason": {"type": "string", "maxLength": 80},
    }, "required": ["observed_rooms", "observed_landmarks", "update", "floor_estimate",
                    "location_estimate", "confidence", "reason"]}


GUIDE_A = (
    "너는 시각장애인 실내 길안내 보조다. 이 사진은 사용자가 걸으며 휴대폰으로 찍은 영상의 한 장면이다.\n"
    "사진에서 직접 보이는 것(관찰)과 그로부터 판단할 수 있는 공간 상황(추론)을 구분하고, "
    "사진 한 장만으로는 알 수 없는 것은 uncertainty에 적어라.\n"
    "- visible_rooms: 사진에서 실제로 읽을 수 있는 호실 번호만. 없으면 [].\n"
    "- scene_type: CORRIDOR 복도, LOBBY 로비·넓은 홀, ELEVATOR_HALL 엘리베이터 앞, ENTRANCE 건물 출입문, "
    "INTERSECTION 복도 갈림길, ROOM_DOOR 특정 방 문 바로 앞, INSIDE_ELEVATOR 엘리베이터 안, UNKNOWN 판단 불가.\n"
    "- location_evidence: 위치 판단의 근거가 된, 사진에 보이는 것 (짧게).\n"
    "- decision_point: 여기서 방향을 골라야 하는 지점(갈림길, 엘리베이터 앞, 출입문)이면 true.\n"
    "- visible_path: 사진에서 걸어갈 수 있게 보이는 길의 방향. 알 수 없으면 UNKNOWN.\n"
    "- uncertainty: 이 사진만으로 확정할 수 없는 것 (짧게)."
)

GUIDE_B = (
    "너는 시각장애인 실내 길안내 보조다. 사용자가 걸으며 찍은 영상을 1초에 한 장씩 받는다.\n"
    "이전 상태와 지금 장면을 함께 보고 현재 위치 추정을 갱신하라. 이전 답을 그대로 베끼지 말고, "
    "지금 장면의 관찰로 이전 추정을 유지할지 고칠지 판단하라.\n"
    "- observed_rooms, observed_landmarks: 지금 장면에서 실제로 보이는 것만.\n"
    "- update: KEEP 새 근거가 없어 이전 위치 유지 / UPDATE 새 근거로 위치를 갱신 / "
    "REVISE 지금 관찰이 이전 추정과 맞지 않아 고침 / LOST 위치를 알 수 없음.\n"
    "- floor_estimate: 층. 호실 번호의 첫 자리는 보통 층을 뜻한다. 엘리베이터로 이동한 직후 근거가 없으면 모름.\n"
    "- location_estimate: 현재 위치를 짧게 (근처 호실, 엘리베이터 앞, 입구 등).\n"
    "- confidence: 새 근거 없이 시간이 지날수록 낮춘다.\n"
    "- reason: 판단 이유 (짧게)."
)


THINK = None  # 생각 모드 지원 모델(qwen3.5 등)용: None = 모델 기본값, True/False = 켬/끔


def ask(url, model, prompt, schema, img, num_predict):
    payload = {"model": model, "stream": False, "keep_alive": "30m",
               "options": {**run_task.OPTIONS, "num_predict": num_predict},
               "messages": [{"role": "user", "content": prompt}]}
    if img is not None:
        payload["messages"][0]["images"] = [exp_d_vlm.encode(img, VLM_SIDE)]
    if schema:
        payload["format"] = schema
    if THINK is not None:
        payload["think"] = THINK
    t = time.perf_counter()
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=(10, 600))
        resp.raise_for_status()
        body = resp.json()
        ans = json.loads(body["message"]["content"])
        err = None
    except Exception as error:  # noqa: BLE001
        ans, err, body = None, f"{type(error).__name__}: {error}", {}
    out_think = ((body.get("message") or {}).get("thinking") or "")[:2000] if body else ""
    return {"answer": ans, "error": err, "vlm_seconds": time.perf_counter() - t, "thinking": out_think,
            "eval_count": body.get("eval_count"), "prompt_eval_count": body.get("prompt_eval_count")}


# ---------------------------------------------------------------- 실행

def sequences(root):
    dirs = sorted(d for d in Path(root).iterdir() if d.is_dir())
    return [(d.name, sorted(d.glob("f*.jpg"))) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--only", default="A0,A1,B")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = sequences(args.frames)
    print("순서:", [(n, len(f)) for n, f in seqs], flush=True)

    cwd = os.getcwd()
    yoloe = pipeline.load_yoloe()
    os.chdir(cwd)
    ocr = exp_a_ocr.load_ocr("m", {})
    warm = Image.open(seqs[0][1][0]).convert("RGB")
    for _ in range(2):
        stage1(yoloe, ocr, warm)
    ask(args.url, args.model, "ok", None, Image.new("RGB", (64, 64)), 4)

    # 1단계 (모든 프레임)
    s1 = {}
    with open(out / "stage1.jsonl", "w", encoding="utf-8") as f:
        for name, frames in seqs:
            for p in frames:
                r = stage1(yoloe, ocr, Image.open(p).convert("RGB"))
                s1[(name, p.name)] = r
                f.write(json.dumps({"seq": name, "frame": p.name, **r}, ensure_ascii=False) + "\n")
                print(f"[1단계] {name}/{p.name} {r['stage1_seconds']:.2f}s rooms={r['result']['rooms']}", flush=True)

    todo = args.only.split(",")
    # 실험 A: 사진 1장
    for tag in [t for t in ("A0", "A1") if t in todo]:
        lat = []
        with open(out / f"{tag}.jsonl", "w", encoding="utf-8") as f:
            for name, frames in seqs:
                for p in frames:
                    r1 = s1[(name, p.name)]
                    cands = r1["result"]["rooms"] if tag == "A1" else []
                    prompt = GUIDE_A + ("\n\n" + hints(r1) if tag == "A1" else "")
                    r = ask(args.url, args.model, prompt, schema_a(cands), Image.open(p).convert("RGB"), 300)
                    total = r["vlm_seconds"] + (r1["stage1_seconds"] if tag == "A1" else 0)
                    lat.append(total)
                    f.write(json.dumps({"seq": name, "frame": p.name, "exp": tag, **r, "latency_seconds": total},
                                       ensure_ascii=False) + "\n")
                    print(f"[{tag}] {name}/{p.name} {total:.2f}s {json.dumps(r['answer'], ensure_ascii=False)}", flush=True)
        print(f"[{tag}] 시간 중앙값 {statistics.median(lat):.2f}s 최대 {max(lat):.2f}s", flush=True)

    # 실험 B: 연속 장면. 2a·2b는 1의 마지막 상태에서 시작 (엘리베이터 이동 알림과 함께)
    if "B" in todo:
        lat = []
        start = {"floor_estimate": "모름", "location_estimate": "모름 (영상 시작)", "confidence": "LOW"}
        end_of_first = None
        with open(out / "B.jsonl", "w", encoding="utf-8") as f:
            for name, frames in seqs:
                first = name.startswith("1_")
                state = dict(start) if first else dict(end_of_first)
                event = "영상 시작" if first else "사용자가 엘리베이터를 타고 다른 층으로 이동한 뒤 내렸다"
                since = 0
                for k, p in enumerate(frames):
                    r1 = s1[(name, p.name)]
                    prompt = (GUIDE_B + f"\n\n이전 상태: {json.dumps(state, ensure_ascii=False)}\n"
                              f"마지막으로 새 근거를 본 뒤 지난 장면 수: {since}\n"
                              + (f"사건: {event}\n" if k == 0 else "") + "\n" + hints(r1))
                    r = ask(args.url, args.model, prompt, schema_b(r1["result"]["rooms"]),
                            Image.open(p).convert("RGB"), 300)
                    total = r["vlm_seconds"] + r1["stage1_seconds"]
                    lat.append(total)
                    a = r["answer"]
                    if a:
                        since = 0 if a["update"] in ("UPDATE", "REVISE") else since + 1
                        state = {k2: a[k2] for k2 in ("floor_estimate", "location_estimate", "confidence")}
                        state["last_observed"] = ", ".join(a["observed_rooms"]) or a["observed_landmarks"]
                    f.write(json.dumps({"seq": name, "frame": p.name, "exp": "B", "since_evidence": since, **r,
                                        "latency_seconds": total}, ensure_ascii=False) + "\n")
                    print(f"[B] {name}/{p.name} {total:.2f}s {json.dumps(a, ensure_ascii=False)}", flush=True)
                if first:
                    end_of_first = state
        print(f"[B] 시간 중앙값 {statistics.median(lat):.2f}s 최대 {max(lat):.2f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
