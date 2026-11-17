"""실험 E: 모델 x 프롬프트 언어 x 근거 필드. 실험 D의 D3h(전체 사진 + 조각, 호실만이면 조각만)를 기준으로 프롬프트만 바꾼다.

  --lang ko|en      : 프롬프트 언어. 답 구조는 같고 enum 값만 그 언어로 (반영 전에 한국어 값으로 되돌림)
  --reason none|short: short면 질문마다 {"evidence": 짧은 근거(maxLength), "answer": enum} — 근거를 답 앞에 둔다
  --model           : qwen3-vl:4b-instruct | qwen2.5vl:7b

plan / images_for / apply 는 exp_d_vlm 그대로 재사용 (호출 조건·조각 크기·반영 규칙 동일).
결과: outputs/kyj/exp_e_<name>.jsonl. "vlm"은 한국어 값으로 정규화한 답(R1.., E1.., arrows), "vlm_raw"는 원래 답, "evidence"는 근거.

사용 (서버):
  ./gpu_lock.sh 7 .venv/bin/python exp_e_vlm.py --lang en --reason short --model qwen3-vl:4b-instruct \
      --ids-file /workspace/capstone/outputs/kyj/exp_d_ids_dev.txt --name q3_en_r_dev [--diag-rooms]
"""

import argparse
import json
import statistics
import time

import requests
from PIL import Image

import exp_d_vlm as D
import run_task

OUT = run_task.OUTPUT_DIR

ANS = {"ko": [D.YES, D.NO, D.UNSURE], "en": ["yes", "no", "unsure"]}
TO_KO = {"yes": D.YES, "no": D.NO, "unsure": D.UNSURE}
CLOCK_EN = [f"{h} o'clock" for h in [12] + list(range(1, 12))]
CLOCK_TO_KO = dict(zip(CLOCK_EN, run_task.CLOCK_VALUES))
EV_LEN = 30  # 근거 최대 글자 수


def where_en(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "left" if cx < 1 / 3 else "center" if cx < 2 / 3 else "right"
    v = "top" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "bottom"
    return f"{v} {h} of the photo (x {cx * 100:.0f}%, y {cy * 100:.0f}%)"


T = {
    "ko": {
        "intro": "시각장애인 실내 길안내를 위해 사진 속 후보를 확인합니다. 아래 질문에만 JSON으로 답하세요. "
                 "확실하지 않으면 추측하지 말고 '판단 불가'로 답하세요.",
        "crops_only": "이미지들은 차례대로 {names} 후보 부분을 원본 사진에서 확대한 것입니다.",
        "full_crops": "첫 번째 이미지는 전체 사진이고, 다음 이미지들은 차례대로 {names} 후보 부분을 원본에서 확대한 것입니다.",
        "rooms": "\n[호실 번호] 글자 인식기가 아래 숫자를 읽었습니다. 각 후보가 문패(방 이름표)에 적힌 호실 번호이고 "
                 "사진 속 숫자가 후보와 정확히 같으면 '예', 문패가 아니거나 숫자가 다르면 '아니요'.",
        "room_item": "- R{i}: \"{room}\" — {where}",
        "elev": "\n[엘리베이터] 물체 탐지기가 아래 위치를 엘리베이터 후보로 찾았습니다. 그 위치에 엘리베이터 문"
                "(가운데가 갈라지는 금속 문, 옆에 호출 버튼)이 있으면 '예', 일반 문·벽·다른 물체면 '아니요'.",
        "elev_item": "- E{i}: {where}",
        "signs": "\n[화살표] 사진 속 표지판이나 유도등에 화살표 모양이 그려져 있으면, 화살표가 가리키는 방향을 "
                 "시계 방향으로 답하세요 (12시=앞/위쪽, 3시=오른쪽, 9시=왼쪽). 화살표 모양이 없으면 빈 배열.",
        "reason": "\n각 evidence에는 답을 고르기 전에 확대 이미지에서 실제로 보이는 것만 문장 없이 단어로 적으세요 "
                  "(호실: 문패에 보이는 글자 그대로, 엘리베이터: 그 위치의 물체 이름, 화살표: 화살표 방향). 그다음 answer를 고르세요.",
        "where": D.where,
    },
    "en": {
        "intro": "You are checking candidates in a photo for indoor navigation of a blind person. Answer only the questions "
                 "below, in JSON. If you are not sure, do not guess; answer 'unsure'.",
        "crops_only": "The images are, in order, zoomed crops of the candidates {names} from the original photo.",
        "full_crops": "The first image is the whole photo; the next images are, in order, zoomed crops of the candidates "
                      "{names} from the original photo.",
        "rooms": "\n[Room numbers] A text recognizer read the numbers below. Answer 'yes' if the candidate is a room number "
                 "written on a door plate (room name sign) and the number in the photo is exactly the same as the candidate; "
                 "answer 'no' if it is not on a door plate or the number is different.",
        "room_item": "- R{i}: \"{room}\" — {where}",
        "elev": "\n[Elevator] An object detector found the locations below as elevator candidates. Answer 'yes' if there is "
                "an elevator door there (a metal door that splits in the middle, with call buttons beside it); answer 'no' "
                "if it is an ordinary door, a wall, or another object.",
        "elev_item": "- E{i}: {where}",
        "signs": "\n[Arrows] If a sign or exit light in the photo has an arrow drawn on it, give the direction the arrow "
                 "points as a clock direction (12 o'clock = forward/up, 3 o'clock = right, 9 o'clock = left). "
                 "If there is no arrow shape, give an empty array.",
        "reason": "\nIn each 'evidence', before choosing, write only what is actually visible in the zoomed image, as keywords, "
                  "not a sentence (room: the characters on the plate exactly as shown; elevator: name of the object at that "
                  "spot; arrow: the arrow direction). Then choose 'answer'.",
        "where": where_en,
    },
}


def schema_and_prompt(q, variant, lang, reason):
    t = T[lang]
    answers = ANS[lang]
    clocks = run_task.CLOCK_VALUES if lang == "ko" else CLOCK_EN
    props, req, lines = {}, [], [t["intro"]]
    names = [f"R{i + 1}" for i in range(len(q["rooms"]))] + [f"E{i + 1}" for i in range(len(q["elev"]))] + \
            [f"S{i + 1}" for i in range(len(q["signs"]))]
    key = "crops_only" if D.crops_only(q, variant) else "full_crops"
    lines.append(t[key].format(names=", ".join(names)))

    def field(enum_schema):
        if reason == "none":
            return enum_schema
        return {"type": "object", "properties": {"evidence": {"type": "string", "maxLength": EV_LEN}, "answer": enum_schema},
                "required": ["evidence", "answer"]}

    if q["rooms"]:
        lines.append(t["rooms"])
        for i, r in enumerate(q["rooms"], 1):
            lines.append(t["room_item"].format(i=i, room=r["room"], where=t["where"](r["box"])))
            props[f"R{i}"] = field({"type": "string", "enum": answers})
            req.append(f"R{i}")
    if q["elev"]:
        lines.append(t["elev"])
        for i, e in enumerate(q["elev"], 1):
            lines.append(t["elev_item"].format(i=i, where=t["where"](e["box"])))
            props[f"E{i}"] = field({"type": "string", "enum": answers})
            req.append(f"E{i}")
    if q["signs"]:
        lines.append(t["signs"])
        arr = {"type": "array", "maxItems": 2, "items": {"type": "string", "enum": clocks}}
        if reason == "none":
            props["arrows"] = arr
            req.append("arrows")
        else:
            props["arrow_evidence"] = {"type": "string", "maxLength": EV_LEN}
            props["arrows"] = arr
            req += ["arrow_evidence", "arrows"]
    if reason != "none":
        lines.append(t["reason"])
    return {"type": "object", "properties": props, "required": req}, "\n".join(lines)


def normalize(raw):
    """언어·근거 형식을 실험 D 형식(한국어 값, 평평한 R1/E1/arrows)으로 되돌린다."""
    if raw is None or "error" in raw:
        return raw, {}
    ans, ev = {}, {}
    for k, v in raw.items():
        if k.startswith("_"):
            ans[k] = v
            continue
        if k == "arrow_evidence":
            ev[k] = v
            continue
        if k == "arrows":
            ans[k] = [CLOCK_TO_KO.get(x, x) for x in (v or [])]
            continue
        if isinstance(v, dict):
            ev[k] = v.get("evidence")
            v = v.get("answer", D.UNSURE)
        ans[k] = TO_KO.get(v, v)
    return ans, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="d3h", choices=["d3", "d3c", "d3h"])
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    ap.add_argument("--reason", choices=["none", "short"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="m2048")
    ap.add_argument("--rules", default='{"majority": true}')
    ap.add_argument("--el-th", type=float, default=0.15)
    ap.add_argument("--ex-th", type=float, default=0.15)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--num-predict", type=int, default=None, help="기본: 근거 없음 96, 근거 있음 320")
    ap.add_argument("--diag-rooms", action="store_true")
    args = ap.parse_args()
    params = json.loads(args.rules)
    num_predict = args.num_predict or (96 if args.reason == "none" else 320)

    recs = [json.loads(l) for l in open(OUT / f"exp_a_{args.cfg}.jsonl", encoding="utf-8")]
    if args.ids_file:
        wanted = {l.strip() for l in open(args.ids_file) if l.strip()}
        recs = [r for r in recs if r["image_id"] in wanted]
    paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
    options = {**run_task.OPTIONS, "num_predict": num_predict}

    # 모델 올리기(시간 제외)
    w = Image.new("RGB", (64, 64), (128, 128, 128))
    requests.post(f"{args.url}/api/chat", json={"model": args.model, "stream": False, "keep_alive": "30m", "options": options,
                  "messages": [{"role": "user", "content": "ok", "images": [D.encode(w, 64)]}]}, timeout=300)

    path = OUT / f"exp_e_{args.name}.jsonl"
    vlm_times = []
    with open(path, "w", encoding="utf-8") as f:
        for n, rec in enumerate(recs, 1):
            kept, base, q = D.plan(rec, params, args.el_th, args.ex_th, args.diag_rooms)
            called = bool(q["rooms"] or q["elev"] or q["signs"])
            raw, ans, ev, vlm_s, decode_s, prompt = None, None, {}, 0.0, 0.0, None
            if called:
                t0 = time.perf_counter()
                img = Image.open(paths[rec["image_id"]]).convert("RGB")
                decode_s = time.perf_counter() - t0  # 디코딩은 시간에서 뺌 (기록만)
                t1 = time.perf_counter()
                schema, prompt = schema_and_prompt(q, args.variant, args.lang, args.reason)
                payload = {"model": args.model, "format": schema, "stream": False, "keep_alive": "30m", "options": options,
                           "messages": [{"role": "user", "content": prompt, "images": D.images_for(img, q, args.variant)}]}
                try:
                    resp = requests.post(f"{args.url}/api/chat", json=payload, timeout=(10, 60))
                    resp.raise_for_status()
                    body = resp.json()
                    raw = json.loads(body["message"]["content"])
                    raw["_eval_count"] = body.get("eval_count")
                    raw["_prompt_eval_count"] = body.get("prompt_eval_count")
                    raw["_done_reason"] = body.get("done_reason")
                except Exception as error:  # noqa: BLE001
                    raw = {"error": f"{type(error).__name__}: {error}"}
                vlm_s = time.perf_counter() - t1
                vlm_times.append(vlm_s)
                ans, ev = normalize(raw)
            result, notes = D.apply(base, kept, q, ans, rec["texts_all"])
            out = {"image_id": rec["image_id"], "task": "fields",
                   "method": f"yoloe+ocr:{args.cfg}+rules+vlm:{args.variant}:{args.lang}:{args.reason}",
                   "variant": args.variant, "lang": args.lang, "reason": args.reason, "model": args.model,
                   "result": result, "base_result": base, "success": True, "vlm_called": called, "questions": q,
                   "vlm": ans, "vlm_raw": raw, "evidence": ev, "room_notes": notes,
                   "latency_seconds": rec["latency_seconds"] + vlm_s, "stage1_seconds": rec["latency_seconds"],
                   "vlm_seconds": vlm_s, "decode_seconds": decode_s}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            tag = f"VLM {vlm_s:.2f}s R{len(q['rooms'])} E{len(q['elev'])} S{len(q['signs'])} {json.dumps(raw, ensure_ascii=False)}" if called else "-"
            print(f"[{n}/{len(recs)}] {rec['image_id']} {tag}", flush=True)
    if vlm_times:
        print(f"호출 {len(vlm_times)}/{len(recs)}, 호출당 중앙 {statistics.median(vlm_times):.2f}s, 최대 {max(vlm_times):.2f}s")
    print(f"결과 파일: {path}")


if __name__ == "__main__":
    main()
