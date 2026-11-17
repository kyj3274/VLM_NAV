"""팀원 4층 비교표(4F_4종_비교.xlsx)와 같은 방식으로 영상 1초 1장마다 YOLOE·OCR·Qwen 결과를 뽑는다 (서버).
- YOLOE 팀 목록: 팀원 표에 나온 13종 (엘리베이터 문, 의자, 책상, 문, 소화기, 게시판, 자판기, 호실 문패, 포스터, 사람, 쓰레기통, 정수기, 표지판)
- YOLOE 우리 7종: 파이프라인 그대로 (pipeline.PHRASES, 점수 0.15)
- OCR: 점수 0.5 이상 글자 전부
- Qwen3-VL 4B: 사진만 보고 물체(위치)와 읽은 글자를 JSON으로
Claude 기준과 색칠은 Mac의 team_compare_xlsx.py가 한다.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python team_compare.py --frames <폴더> --glob "i*.jpg" --step 8 --out <json>
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import exp_a_ocr
import pipeline
import video_context as vc

TEAM = {  # 팀원 YOLOE 열에 나온 이름 → 영어 표현
    "엘리베이터 문": ["elevator door"], "의자": ["chair"], "책상": ["desk"], "문": ["door"],
    "소화기": ["fire extinguisher"], "게시판": ["bulletin board"], "자판기": ["vending machine"],
    "호실 문패": ["door plate", "room number sign"], "포스터": ["poster"], "사람": ["person"],
    "쓰레기통": ["trash can"], "정수기": ["water dispenser"], "표지판": ["sign"],
}
TEAM_TH = 0.25
QWEN_PROMPT = ("이 사진은 시각장애인 실내 길안내용으로 걸으면서 찍은 건물 복도 사진이다. "
               "사진에 보이는 물체를 빠짐없이 적고, 각 물체가 화면의 왼쪽·가운데·오른쪽 중 어디 있는지 적어라. "
               "같은 물체가 여러 개면 따로 적되 최대 20개까지만 적어라. 그리고 사진에서 읽을 수 있는 글자(호실 번호, 실 이름, 안내문)를 "
               "보이는 그대로 적어라. 읽을 수 없으면 지어내지 마라.")
QWEN_SCHEMA = {"type": "object", "properties": {
    "objects": {"type": "array", "maxItems": 20, "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "position": {"type": "string", "enum": ["왼쪽", "가운데", "오른쪽"]}},
        "required": ["name", "position"]}},
    "texts": {"type": "array", "maxItems": 15, "items": {"type": "string"}}}, "required": ["objects", "texts"]}
# 처음엔 목록 제한 없이 600토큰으로 물었더니 같은 물체를 끝없이 나열하다 잘려 6장 중 5장 JSON이 깨짐 → 개수 제한 + 1500토큰


def side(box):
    cx = (box[0] + box[2]) / 2
    return "왼쪽" if cx < 1 / 3 else "가운데" if cx < 2 / 3 else "오른쪽"


def set_classes(model, phrases):
    rows = []
    for ph in phrases.values():
        pe = model.get_text_pe(ph).mean(dim=1, keepdim=True)
        rows.append(pe / pe.norm(dim=-1, keepdim=True))
    model.set_classes(list(phrases), torch.cat(rows, dim=1))


def detect(model, names, img, th):
    r = model.predict(img, imgsz=pipeline.YOLOE_SIZE, conf=th, device=0, verbose=False)[0]
    return [{"name": names[int(c)], "conf": round(s, 2),
             "position": side([b[0] / img.width, 0, b[2] / img.width, 0])}
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--glob", default="i*.jpg")
    ap.add_argument("--step", type=int, default=8, help="몇 장마다 하나 (8fps 폴더면 8 = 1초에 1장, 초의 첫 장)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    a = ap.parse_args()
    frames = sorted(Path(a.frames).glob(a.glob))[::a.step]
    yoloe = pipeline.load_yoloe()
    ocr = exp_a_ocr.load_ocr("m", {})
    rows = []
    for sec, p in enumerate(frames):
        img = Image.open(p).convert("RGB")
        set_classes(yoloe, TEAM)
        team = detect(yoloe, list(TEAM), img, TEAM_TH)
        set_classes(yoloe, pipeline.PHRASES)
        ours = detect(yoloe, list(pipeline.PHRASES), img, pipeline.YOLOE_THRESHOLD)
        texts = [t["text"] for t in exp_a_ocr.read_all(ocr, img, pipeline.OCR_MAX_SIDE)[0]
                 if t["score"] >= pipeline.OCR_THRESHOLD]
        q = vc.ask(a.url, a.model, QWEN_PROMPT, QWEN_SCHEMA, img, 1500)
        rows.append({"sec": sec, "frame": p.name, "yoloe_team": team, "yoloe_ours": ours, "ocr": texts,
                     "qwen": q["answer"], "qwen_error": q["error"], "qwen_seconds": round(q["vlm_seconds"], 1)})
        print(f"{sec:2d}s 팀 {[d['name'] for d in team]} | 우리 {[d['name'] for d in ours]} | OCR {texts} | "
              f"Qwen {q['vlm_seconds']:.0f}s {q['error'] or len((q['answer'] or {}).get('objects', []))}", flush=True)
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
