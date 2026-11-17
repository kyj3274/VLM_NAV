"""물체 조각 분류 실험: YOLOE가 찾은 물체를 잘라서 VLM(qwen3-vl 고정)이 무엇인지 고르게 한다.

최종 파이프라인(D3h)의 엘리베이터 질문(E)을 물체 분류 질문(O)으로 바꾸고,
호실 질문(R)·화살표 질문(S)은 그대로 같은 호출에 묶는다. 1단계 결과는 exp_final_all.jsonl 재사용.

변형 (--crop):
  tight  : 상자 그대로 자름
  margin : 상자를 가로세로 1.8배로 넓혀 자름 (옆 버튼, 위 숫자 등 주변 단서 포함)
  thumb  : margin + 전체 사진 512px를 맨 앞에 (위치 맥락)

사용법 (서버): ./gpu_lock.sh 7 .venv/bin/python exp_obj_vlm.py --crop margin --name obj_margin
"""

import argparse
from pathlib import Path
import base64
import json
import re
import statistics
import time
from io import BytesIO

import requests
from PIL import Image

import exp_d_vlm
import pipeline
import room_rules
import run_task

MODEL = "qwen3-vl:4b-instruct"
URL = "http://127.0.0.1:11436"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 128}
RULES = {"majority": True}
OBJ_CLASSES = ["문", "엘리베이터", "계단", "화장실", "비상구표시"]
OBJ_TH = 0.15
TOP_K = 5
LABELS = ["문", "엘리베이터", "계단 입구", "화장실", "비상구 표시", "기타", "판단 불가"]
LABEL_TO_LANDMARK = {"문": "문", "엘리베이터": "엘리베이터", "계단 입구": "계단", "화장실": "화장실", "비상구 표시": "비상구표시"}
TOILET = re.compile(r"(TOILET|화장실|RESTROOM)", re.IGNORECASE)


def object_boxes(detections):
    """랜드마크 후보 상자: 기준 점수 이상, 서로 겹치면 점수 높은 것만, 큰 상자(가까운 것) 순 최대 K개."""
    kept = []
    for d in sorted((d for d in detections if d["name"] in OBJ_CLASSES and d["conf"] >= OBJ_TH), key=lambda d: -d["conf"]):
        if all(exp_d_vlm.iou(d["box"], o["box"]) < 0.5 for o in kept):
            kept.append(d)
    kept.sort(key=lambda d: -((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])))
    return kept[:TOP_K], kept[TOP_K:]


def crop_object(img, box, scale):
    W, H = img.size
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    w = max((box[2] - box[0]) * W * scale, 64)
    h = max((box[3] - box[1]) * H * scale, 64)
    # 아주 가는 조각은 Ollama가 오류를 내므로 가로세로 비율을 3:1 이내로 넓히고, 사진 밖이면 안쪽으로 민다
    w, h = min(max(w, h / 3), W), min(max(h, w / 3), H)
    x0 = min(max(0, cx - w / 2), W - w)
    y0 = min(max(0, cy - h / 2), H - h)
    c = img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
    c.thumbnail((448, 448))
    buf = BytesIO()
    c.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


DOOR_MIN_HEIGHT = 0.25
BUTTON_DOORS = 2


def door_boxes(detections):
    """버튼을 확인할 큰 문: YOLOE 문/엘리베이터 상자, 높이 0.25 이상, 큰 것부터 최대 2개."""
    doors = []
    for d in sorted((d for d in detections if d["name"] in ("문", "엘리베이터") and d["conf"] >= OBJ_TH
                     and d["box"][3] - d["box"][1] >= DOOR_MIN_HEIGHT),
                    key=lambda d: -((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))):
        if all(exp_d_vlm.iou(d["box"], o["box"]) < 0.3 for o in doors):
            doors.append(d)
        if len(doors) == BUTTON_DOORS:
            break
    return doors


def crop_door_sides(img, box):
    """문 양옆 벽(버튼이 있을 높이)을 잘라 왼쪽|오른쪽으로 붙인 한 장. 버튼이 크게 보이도록."""
    W, H = img.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    w, h = x1 - x0, y1 - y0
    top, bottom = y0 + 0.2 * h, y0 + 0.95 * h  # 버튼은 꽤 낮게 달려 있음
    side = max(0.7 * w, 0.15 * h)
    strips = []
    for a, b in ((x0 - side, x0 + 0.1 * w), (x1 - 0.1 * w, x1 + side)):
        a, b = max(0, a), min(W, b)
        if b - a < 8:
            a, b = (0, min(W, 16)) if a <= 0 else (max(0, W - 16), W)
        strips.append(img.crop((int(a), int(top), int(b), int(bottom))))
    height = max(s.size[1] for s in strips)
    gap = 12
    sheet = Image.new("RGB", (sum(s.size[0] for s in strips) + gap, height), (255, 255, 255))
    sheet.paste(strips[0], (0, 0))
    sheet.paste(strips[1], (strips[0].size[0] + gap, 0))
    w2, h2 = sheet.size  # 너무 가늘면 Ollama 오류 → 비율 3:1 이내로 흰 여백
    if w2 < h2 / 3:
        pad = Image.new("RGB", (int(h2 / 3) + 1, h2), (255, 255, 255)); pad.paste(sheet, (0, 0)); sheet = pad
    elif h2 < w2 / 3:
        pad = Image.new("RGB", (w2, int(w2 / 3) + 1), (255, 255, 255)); pad.paste(sheet, (0, 0)); sheet = pad
    sheet.thumbnail((640, 640))
    buf = BytesIO()
    sheet.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def build(img, q, objects, mode, doors=()):
    schema, prompt = exp_d_vlm.schema_and_prompt(q, "d3c")  # R(호실 조각), S(화살표)만 들어 있음 (E는 비움)
    lines = [prompt]
    images = []
    names = [f"R{i + 1}" for i in range(len(q["rooms"]))] + [f"O{i + 1}" for i in range(len(objects))] + \
            [f"S{i + 1}" for i in range(len(q["signs"]))] + [f"B{i + 1}" for i in range(len(doors))]
    head = "첫 번째 이미지는 전체 사진이고, 다음 이미지들은 차례대로 " if mode == "thumb" else "이미지들은 차례대로 "
    lines[0] = prompt.split("\n")[0] + "\n" + head + ", ".join(names) + " 부분을 원본 사진에서 확대한 것입니다." + \
        ("\n" + "\n".join(prompt.split("\n")[2:]) if len(prompt.split("\n")) > 2 else "")
    if mode == "thumb":
        images.append(exp_d_vlm.encode(img, 512))
    images += [exp_d_vlm.crop(img, r["box"], "R") for r in q["rooms"]]
    scale = 1.0 if mode == "tight" else 1.8
    images += [crop_object(img, o["box"], scale) for o in objects]
    images += [exp_d_vlm.crop(img, s["box"], "S") for s in q["signs"]]
    images += [crop_door_sides(img, d["box"]) for d in doors]
    if objects:
        lines.append(
            "\n[물체] 물체 탐지기가 찾은 부분입니다. 각 O 조각의 가운데에 있는 것을 하나 고르세요. "
            "엘리베이터=가운데가 갈라지는 금속 문(호출 버튼이나 층 표시가 함께 보이는 경우가 많음), "
            "문=방이나 복도의 일반 출입문, 계단 입구=계단이나 계단실 입구, 화장실=화장실 표지나 입구, "
            "비상구 표시=초록색 피난 유도 표지. 해당하지 않으면 '기타', 확실하지 않으면 '판단 불가'.")
        for i, o in enumerate(objects, 1):
            lines.append(f"- O{i}: {exp_d_vlm.where(o['box'])}")
            schema["properties"][f"O{i}"] = {"type": "string", "enum": LABELS}
            schema["required"].append(f"O{i}")
    if doors:
        lines.append(
            "\n[호출 버튼] 각 B 조각은 큰 문의 양옆 벽을 확대해 왼쪽|오른쪽으로 붙인 것입니다. "
            "벽에 위쪽·아래쪽 화살표가 있는 호출 버튼(엘리베이터를 부르는 버튼)이 보이면 '예', "
            "없으면 '아니요', 확실하지 않으면 '판단 불가'. 전등 스위치, 화재 경보기, 카드 단말기 같은 다른 장치는 '아니요'.")
        for i, d in enumerate(doors, 1):
            lines.append(f"- B{i}: {exp_d_vlm.where(d['box'])}의 문")
            schema["properties"][f"B{i}"] = {"type": "string", "enum": exp_d_vlm.ANSWERS}
            schema["required"].append(f"B{i}")
    return schema, "\n".join(lines), images


def ask(schema, prompt, images):
    payload = {"model": MODEL, "format": schema, "stream": False, "keep_alive": "30m", "options": OPTIONS,
               "messages": [{"role": "user", "content": prompt, "images": images}]}
    try:
        resp = requests.post(f"{URL}/api/chat", json=payload, timeout=(10, 60))
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def apply(base, kept, q, objects, rest, ans, texts_all, doors=(), buttons=False):
    q_no_elev = dict(q, elev=[])
    res, notes = exp_d_vlm.apply(base, kept, q_no_elev, ans, texts_all)
    # 랜드마크를 새로 계산: OCR 근거(층 숫자로 확정된 엘리베이터, 화장실 글자) + 물체 조각 판단 + 묻지 않은 상자
    landmarks = set()
    if "엘리베이터" in base["landmarks"]:
        landmarks.add("엘리베이터")
    if any(TOILET.search(t["text"]) for t in texts_all if t["score"] >= pipeline.OCR_THRESHOLD):
        landmarks.add("화장실")
    directions = [d for d in res["directions"] if d["type"] == "SIGN_ARROW" or room_rules.ROOM.match(d["target"])]
    if base["landmarks"] and "엘리베이터" in base["landmarks"]:
        directions += [d for d in base["directions"] if d["target"] == "엘리베이터"]
    labels = {}
    for i, o in enumerate(objects, 1):
        label = ans.get(f"O{i}") if ans and "error" not in ans else None
        labels[f"O{i}"] = label
        if buttons and label == "엘리베이터":
            name = "문"  # 버튼 모드: 엘리베이터는 버튼(또는 OCR 층 숫자)으로만 판단
        elif label in LABEL_TO_LANDMARK:
            name = LABEL_TO_LANDMARK[label]
        elif label in (None, "판단 불가"):
            name = "문" if o["name"] == "엘리베이터" else o["name"]  # 1단계와 같은 규칙
        else:
            continue  # 기타
        landmarks.add(name)
        if name != "문":
            item = {"target": name, "type": "RELATIVE_POSITION", "value": exp_d_vlm.clock((o["box"][0] + o["box"][2]) / 2)}
            if item not in directions:
                directions.append(item)
    for o in rest:
        landmarks.add("문" if o["name"] == "엘리베이터" else o["name"])
    for i, d in enumerate(doors, 1):
        answer = ans.get(f"B{i}") if ans and "error" not in ans else None
        labels[f"B{i}"] = answer
        if answer == exp_d_vlm.YES:
            landmarks.add("엘리베이터")
            item = {"target": "엘리베이터", "type": "RELATIVE_POSITION", "value": exp_d_vlm.clock((d["box"][0] + d["box"][2]) / 2)}
            if item not in directions:
                directions.append(item)
    res["landmarks"] = sorted(landmarks)
    res["directions"] = directions
    return res, labels


def set_model(model, url):
    global MODEL, URL
    MODEL, URL = model, url.rstrip("/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=["tight", "margin", "thumb"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default=str(run_task.OUTPUT_DIR / "exp_final_all.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--buttons", action="store_true", help="엘리베이터를 문 옆 호출 버튼으로만 판단")
    parser.add_argument("--model", default=MODEL, help=f"Ollama 모델 (기본 {MODEL})")
    parser.add_argument("--url", default=URL, help=f"Ollama 주소 (기본 {URL}, GPU 7)")
    parser.add_argument("--out-dir", default=str(run_task.OUTPUT_DIR), help="결과 폴더 (다른 사람은 자기 outputs 폴더로)")
    args = parser.parse_args()
    set_model(args.model, args.url)

    records = [json.loads(l) for l in open(args.source, encoding="utf-8")][: args.limit]
    paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp_{args.name}.jsonl"

    # 모델 예열 (시간에서 제외)
    requests.post(f"{URL}/api/chat", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                                            "stream": False, "keep_alive": "30m", "options": OPTIONS}, timeout=300)
    times = []
    with open(out_path, "w", encoding="utf-8") as out:
        for n, rec in enumerate(records, 1):
            img = Image.open(paths[rec["image_id"]]).convert("RGB")
            texts = [t for t in rec["texts_all"] if t["score"] >= pipeline.OCR_THRESHOLD]
            combined, _ = pipeline.combine(rec["detections"], texts)
            stage1 = {"result": combined, "texts_all": rec["texts_all"], "ocr_input_size": rec["ocr_input_size"],
                      "detections": rec["detections"]}
            t0 = time.perf_counter()
            kept, base, q = exp_d_vlm.plan(stage1, RULES, 1.1, 0.15)  # 엘리베이터 질문은 끔 (기준 1.1)
            objects, rest = object_boxes(rec["detections"])
            doors = door_boxes(rec["detections"]) if args.buttons and "엘리베이터" not in base["landmarks"] else []
            called = bool(q["rooms"] or q["signs"] or objects or doors)
            ans = None
            if called:
                schema, prompt, images = build(img, q, objects, args.crop, doors)
                ans = ask(schema, prompt, images)
            result, labels = apply(base, kept, q, objects, rest, ans, rec["texts_all"], doors, args.buttons)
            vlm_seconds = time.perf_counter() - t0
            latency = rec["stage1_seconds"] + vlm_seconds
            times.append(latency)
            out.write(json.dumps({
                "image_id": rec["image_id"], "task": "fields", "method": f"obj_{args.crop}", "model": MODEL, "result": result,
                "success": True, "vlm_called": called, "vlm": ans, "object_labels": labels,
                "objects": [{"name": o["name"], "conf": o["conf"], "box": o["box"]} for o in objects],
                "latency_seconds": latency, "stage1_seconds": rec["stage1_seconds"], "vlm_seconds": vlm_seconds,
            }, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(records)}] {rec['image_id']} | 물체 {len(objects)} | {labels} | {latency:.2f}초", flush=True)
    print(f"완료: 중앙값 {statistics.median(times):.2f}초, 결과 {out_path}")


if __name__ == "__main__":
    main()
