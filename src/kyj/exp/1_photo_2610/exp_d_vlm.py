"""실험 D: VLM은 '만들기'가 아니라 '고르기'만. 1단계(규칙) 결과에 VLM 객관식 판단을 반영한다.

1단계 원자료: outputs/kyj/exp_a_<cfg>.jsonl (texts_all, detections, latency_seconds = YOLOE+OCR)
호실 규칙: room_rules.select(cands, params) → kept(출력), ambiguous(애매 후보, VLM 확인)

VLM 호출 조건 (하나라도 해당하면 한 번 호출, 필요한 질문만 묶음)
  R: 애매 호실 후보 (room_rules.ambiguous_reason)
  E: YOLOE '엘리베이터' 상자 conf >= --el-th 이고 1단계가 엘리베이터를 확정하지 못함
  S: YOLOE '비상구표시' 상자 conf >= --ex-th (화살표 가능)

변형
  d1: 전체 사진(1024px) + 후보를 텍스트(번호·위치)로
  d2: d1 + 사진 위에 후보 상자와 번호를 그림(Set-of-Mark)
  d3: d1 + 후보 영역을 원본에서 잘라 추가 이미지로 (전체 사진 + 조각)
  d3c: 조각만 (전체 사진 없음)
  d3h: 호실 질문만 있는 사진은 조각만, 엘리베이터·화살표 질문이 있으면 전체 사진 + 조각

답(JSON schema로 제한): R1.. / E1.. = "예"|"아니요"|"판단 불가", arrows = 시계 방향 enum 배열(maxItems 2)
반영: kept_low 후보 '아니요' → 뺌, maj_big/weak_big 후보 '예' → 넣음, E '예' → 엘리베이터 추가, arrows → SIGN_ARROW.
'판단 불가'/오류 → 1단계 규칙 결과 유지.

사용 (서버):
  ./gpu_lock.sh 7 .venv/bin/python exp_d_vlm.py --variant d1 --ids-file /workspace/capstone/outputs/kyj/exp_d_ids_dev.txt --name d1_dev
  → outputs/kyj/exp_d_<name>.jsonl
"""

import argparse
import base64
import json
import statistics
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

import exp_bc_rules
import room_rules
import run_task

OUT = run_task.OUTPUT_DIR
YES, NO, UNSURE = "예", "아니요", "판단 불가"
ANSWERS = [YES, NO, UNSURE]


def clock(x):
    return "11시" if x < 1 / 3 else "12시" if x < 2 / 3 else "1시"


def where(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "왼쪽" if cx < 1 / 3 else "가운데" if cx < 2 / 3 else "오른쪽"
    v = "위" if cy < 1 / 3 else "중간 높이" if cy < 2 / 3 else "아래"
    return f"사진 {h} {v} (가로 {cx * 100:.0f}%, 세로 {cy * 100:.0f}% 위치)"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def top_boxes(dets, name, th, k=2):
    out = []
    for d in sorted((d for d in dets if d["name"] == name and d["conf"] >= th), key=lambda d: -d["conf"]):
        if all(iou(d["box"], o["box"]) < 0.5 for o in out):
            out.append(d)
        if len(out) == k:
            break
    return out


def plan(rec, params, el_th, ex_th, diag_rooms=False):
    """1단계 결과에서 이 사진에 물어볼 질문을 정한다.
    diag_rooms: 진단용. 점수 0.3 이상 호실 후보 전부(최대 4개)만 묻는다 (VLM의 호실 확인 정확도 측정용, 결과 반영 안 함)."""
    cands = room_rules.candidates(rec["texts_all"], rec["ocr_input_size"])
    kept, amb = room_rules.select(cands, params)
    base = exp_bc_rules.rebuild(rec, kept)
    q = {"rooms": [], "elev": [], "signs": []}
    if diag_rooms:
        for c in sorted((c for c in cands if c["score"] >= 0.3), key=lambda c: -c["h"])[:4]:
            q["rooms"].append({"room": c["room"], "box": c["box"], "reason": "diag", "score": c["score"], "cx": c["cx"]})
        return kept, base, q
    for c in sorted(amb, key=lambda c: -c["h"])[:3]:
        q["rooms"].append({"room": c["room"], "box": c["box"], "reason": c["ambiguous"], "score": c["score"], "cx": c["cx"]})
    if "엘리베이터" not in base["landmarks"]:
        q["elev"] = [{"box": d["box"], "conf": d["conf"]} for d in top_boxes(rec["detections"], "엘리베이터", el_th)]
    q["signs"] = [{"box": d["box"], "conf": d["conf"]} for d in top_boxes(rec["detections"], "비상구표시", ex_th)]
    return kept, base, q


def crops_only(q, variant):
    """d3c: 항상 조각만. d3h: 호실 질문만 있으면 조각만, 엘리베이터·화살표 질문이 있으면 전체 사진 + 조각."""
    return variant == "d3c" or (variant == "d3h" and not q["elev"] and not q["signs"])


def schema_and_prompt(q, variant):
    props, req, lines = {}, [], []
    mark = variant == "d2"
    crops = variant in ("d3", "d3c", "d3h")
    lines.append("시각장애인 실내 길안내를 위해 사진 속 후보를 확인합니다. 아래 질문에만 JSON으로 답하세요. "
                 "확실하지 않으면 추측하지 말고 '판단 불가'로 답하세요.")
    if crops:
        names = [f"R{i + 1}" for i in range(len(q["rooms"]))] + [f"E{i + 1}" for i in range(len(q["elev"]))] + \
                [f"S{i + 1}" for i in range(len(q["signs"]))]
        if crops_only(q, variant):
            lines.append("이미지들은 차례대로 " + ", ".join(names) + " 후보 부분을 원본 사진에서 확대한 것입니다.")
        else:
            lines.append("첫 번째 이미지는 전체 사진이고, 다음 이미지들은 차례대로 " + ", ".join(names) + " 후보 부분을 원본에서 확대한 것입니다.")
    if q["rooms"]:
        lines.append("\n[호실 번호] 글자 인식기가 아래 숫자를 읽었습니다. 각 후보가 문패(방 이름표)에 적힌 호실 번호이고 "
                     "사진 속 숫자가 후보와 정확히 같으면 '예', 문패가 아니거나 숫자가 다르면 '아니요'.")
        for i, r in enumerate(q["rooms"], 1):
            extra = " (사진에 R%d 상자로 표시)" % i if mark else ""
            lines.append(f"- R{i}: \"{r['room']}\" — {where(r['box'])}{extra}")
            props[f"R{i}"] = {"type": "string", "enum": ANSWERS}
            req.append(f"R{i}")
    if q["elev"]:
        lines.append("\n[엘리베이터] 물체 탐지기가 아래 위치를 엘리베이터 후보로 찾았습니다. 그 위치에 엘리베이터 문"
                     "(가운데가 갈라지는 금속 문, 옆에 호출 버튼)이 있으면 '예', 일반 문·벽·다른 물체면 '아니요'.")
        for i, e in enumerate(q["elev"], 1):
            extra = " (사진에 E%d 상자로 표시)" % i if mark else ""
            lines.append(f"- E{i}: {where(e['box'])}{extra}")
            props[f"E{i}"] = {"type": "string", "enum": ANSWERS}
            req.append(f"E{i}")
    if q["signs"]:
        extra = " (S 상자로 표시한 표지 포함)" if mark else ""
        lines.append(f"\n[화살표] 사진 속 표지판이나 유도등{extra}에 화살표 모양이 그려져 있으면, 화살표가 가리키는 방향을 "
                     "시계 방향으로 답하세요 (12시=앞/위쪽, 3시=오른쪽, 9시=왼쪽). 화살표 모양이 없으면 빈 배열.")
        props["arrows"] = {"type": "array", "maxItems": 2, "items": {"type": "string", "enum": run_task.CLOCK_VALUES}}
        req.append("arrows")
    return {"type": "object", "properties": props, "required": req}, "\n".join(lines)


def encode(img, side):
    img = img.copy()
    img.thumbnail((side, side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def draw_marks(img, q):
    small = img.copy()
    small.thumbnail((1024, 1024))
    W, H = small.size
    d = ImageDraw.Draw(small)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    items = [(f"R{i}", r["box"], (255, 0, 0), 1.5) for i, r in enumerate(q["rooms"], 1)] + \
            [(f"E{i}", e["box"], (0, 120, 255), 0) for i, e in enumerate(q["elev"], 1)] + \
            [(f"S{i}", s["box"], (255, 160, 0), 0.2) for i, s in enumerate(q["signs"], 1)]
    for label, b, color, pad in items:
        x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        ph = (y1 - y0) * pad
        x0, y0, x1, y1 = max(0, x0 - ph), max(0, y0 - ph), min(W - 1, x1 + ph), min(H - 1, y1 + ph)
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tx, ty = x0, max(0, y0 - 26)
        tw = d.textlength(label, font=font)
        d.rectangle([tx, ty, tx + tw + 6, ty + 25], fill=color)
        d.text((tx + 3, ty + 1), label, fill=(255, 255, 255), font=font)
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def crop(img, box, kind):
    W, H = img.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    w, h = x1 - x0, y1 - y0
    if kind == "R":   # 글자 상자 주변 문패 전체가 보이게
        px, py = max(w * 1.2, h * 3), h * 3
        side = 384
    elif kind == "E":
        px, py = w * 0.15, h * 0.1
        side = 512
    else:
        px, py = w * 0.5, h * 0.5
        side = 384
    c = img.crop((int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))))
    s = side / max(c.size)
    c = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.LANCZOS)
    buf = BytesIO()
    c.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def images_for(img, q, variant):
    if variant == "d2":
        return [draw_marks(img, q)]
    out = [] if crops_only(q, variant) else [encode(img, 1024)]
    if variant in ("d3", "d3c", "d3h"):
        out += [crop(img, r["box"], "R") for r in q["rooms"]]
        out += [crop(img, e["box"], "E") for e in q["elev"]]
        out += [crop(img, s["box"], "S") for s in q["signs"]]
    return out


def apply(base, kept, q, ans, texts_all):
    """VLM 답을 1단계 결과에 반영. 반환: 새 result, 판단 기록."""
    res = json.loads(json.dumps(base))
    notes = {}
    if not ans or "error" in ans:
        return res, notes
    rooms = list(kept)
    changed = False
    for i, r in enumerate(q["rooms"], 1):
        a = ans.get(f"R{i}", UNSURE)
        notes[r["room"]] = a
        if r["reason"] == "kept_low" and a == NO:
            rooms = [c for c in rooms if c["room"] != r["room"]]
            changed = True
        elif r["reason"] in ("maj_big", "weak_big") and a == YES and all(c["room"] != r["room"] for c in rooms):
            rooms.append({"room": r["room"], "box": r["box"], "h": r["box"][3] - r["box"][1], "cx": r["cx"]})
            changed = True
    if changed:
        rooms.sort(key=lambda c: -c["h"])
        rebuilt = exp_bc_rules.rebuild({"result": base, "texts_all": texts_all}, rooms)
        res["rooms"], res["spaces"] = rebuilt["rooms"], rebuilt["spaces"]
        res["directions"] = [d for d in res["directions"] if not room_rules.ROOM.match(d["target"])] + \
            [d for d in rebuilt["directions"] if room_rules.ROOM.match(d["target"])]
    for i, e in enumerate(q["elev"], 1):
        if ans.get(f"E{i}") == YES:
            if "엘리베이터" not in res["landmarks"]:
                res["landmarks"] = sorted(set(res["landmarks"]) | {"엘리베이터"})
                res["directions"].append({"target": "엘리베이터", "type": "RELATIVE_POSITION",
                                          "value": clock((e["box"][0] + e["box"][2]) / 2)})
    for v in (ans.get("arrows") or [])[:2] if q["signs"] else []:
        item = {"target": "알 수 없음", "type": "SIGN_ARROW", "value": v}
        if item not in res["directions"]:
            res["directions"].append(item)
    return res, notes



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["d1", "d2", "d3", "d3c", "d3h"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="m2048")
    ap.add_argument("--rules", default='{"majority": true}')
    ap.add_argument("--el-th", type=float, default=0.15)
    ap.add_argument("--ex-th", type=float, default=0.15)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--num-predict", type=int, default=96)
    ap.add_argument("--diag-rooms", action="store_true")
    args = ap.parse_args()
    params = json.loads(args.rules)

    recs = [json.loads(l) for l in open(OUT / f"exp_a_{args.cfg}.jsonl", encoding="utf-8")]
    if args.ids_file:
        wanted = [l.strip() for l in open(args.ids_file) if l.strip()]
        recs = [r for r in recs if r["image_id"] in set(wanted)]
    paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
    options = {**run_task.OPTIONS, "num_predict": args.num_predict}

    # 모델 올리기(시간 제외)
    w = Image.new("RGB", (64, 64), (128, 128, 128))
    requests.post(f"{args.url}/api/chat", json={"model": args.model, "stream": False, "keep_alive": "30m", "options": options,
                  "messages": [{"role": "user", "content": "ok", "images": [encode(w, 64)]}]}, timeout=300)

    path = OUT / f"exp_d_{args.name}.jsonl"
    vlm_times = []
    with open(path, "w", encoding="utf-8") as f:
        for n, rec in enumerate(recs, 1):
            kept, base, q = plan(rec, params, args.el_th, args.ex_th, args.diag_rooms)
            called = bool(q["rooms"] or q["elev"] or q["signs"])
            ans, vlm_s, decode_s, prompt = None, 0.0, 0.0, None
            if called:
                t0 = time.perf_counter()
                img = Image.open(paths[rec["image_id"]]).convert("RGB")
                decode_s = time.perf_counter() - t0  # 1단계와 같이 디코딩은 시간에서 뺌 (기록만)
                t1 = time.perf_counter()
                schema, prompt = schema_and_prompt(q, args.variant)
                payload = {"model": args.model, "format": schema, "stream": False, "keep_alive": "30m", "options": options,
                           "messages": [{"role": "user", "content": prompt, "images": images_for(img, q, args.variant)}]}
                try:
                    resp = requests.post(f"{args.url}/api/chat", json=payload, timeout=(10, 60))
                    resp.raise_for_status()
                    body = resp.json()
                    ans = json.loads(body["message"]["content"])
                    ans["_eval_count"] = body.get("eval_count")
                    ans["_prompt_eval_count"] = body.get("prompt_eval_count")
                except Exception as error:  # noqa: BLE001
                    ans = {"error": f"{type(error).__name__}: {error}"}
                vlm_s = time.perf_counter() - t1
                vlm_times.append(vlm_s)
            result, notes = apply(base, kept, q, ans, rec["texts_all"])
            out = {"image_id": rec["image_id"], "task": "fields", "method": f"yoloe+ocr:{args.cfg}+rules+vlm:{args.variant}",
                   "variant": args.variant, "model": args.model, "result": result, "base_result": base, "success": True,
                   "vlm_called": called, "questions": q, "vlm": ans, "room_notes": notes,
                   "latency_seconds": rec["latency_seconds"] + vlm_s, "stage1_seconds": rec["latency_seconds"],
                   "vlm_seconds": vlm_s, "decode_seconds": decode_s}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            tag = f"VLM {vlm_s:.2f}s R{len(q['rooms'])} E{len(q['elev'])} S{len(q['signs'])} {json.dumps(ans, ensure_ascii=False)}" if called else "-"
            print(f"[{n}/{len(recs)}] {rec['image_id']} {tag}", flush=True)
    if vlm_times:
        print(f"호출 {len(vlm_times)}/{len(recs)}, 호출당 중앙 {statistics.median(vlm_times):.2f}s, 최대 {max(vlm_times):.2f}s")
    print(f"결과 파일: {path}")


if __name__ == "__main__":
    main()
