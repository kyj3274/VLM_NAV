"""웹 세션(또는 프레임 폴더)의 1초 대표 사진마다 OCR이 읽은 글자 전부와 YOLOE 물체를 뽑는다. 무엇을 읽는지 보는 용도.
대표 사진은 log.jsonl의 key_frame을 쓰고, 없으면 폴더의 사진을 전부 본다. 결과는 <세션>/ocr_dump.jsonl.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python web_ocr_dump.py <세션 폴더>
"""

import argparse
import json
from pathlib import Path

from PIL import Image

import exp_a_ocr
import pipeline
import video_context as vc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--min-obj", type=float, default=0.25, help="출력에 보일 YOLOE 물체 최소 점수")
    a = ap.parse_args()
    d = Path(a.session)
    log = d / "log.jsonl"
    if log.is_file():
        keys = [json.loads(x)["key_frame"] for x in open(log, encoding="utf-8") if '"key_frame"' in x]
        frames = [d / k for k in keys]
    else:
        frames = sorted(d.glob("*.jpg"))
    yoloe = pipeline.load_yoloe()
    ocr = exp_a_ocr.load_ocr("m", {})
    with open(d / "ocr_dump.jsonl", "w", encoding="utf-8") as f:
        for p in frames:
            img = Image.open(p).convert("RGB")
            r = vc.stage1(yoloe, ocr, img)
            rec = {"frame": p.name, "size": list(img.size), "ocr_input_size": r["ocr_input_size"],
                   "ocr": [{"text": t["text"], "score": round(t["score"], 2), "box": t["box"]} for t in r["texts_all"]],
                   "objects": [{"name": o["name"], "conf": round(o["conf"], 2), "box": o["box"]} for o in r["detections"]],
                   "rooms": r["result"]["rooms"], "floors": r["result"]["floors"], "landmarks": r["result"]["landmarks"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            words = " · ".join(f"{t['text']}({t['score']})" for t in rec["ocr"]) or "-"
            objs = ", ".join(f"{o['name']} {o['conf']}" for o in rec["objects"] if o["conf"] >= a.min_obj) or "-"
            print(f"{p.name} {img.size[0]}x{img.size[1]} | 호실 {rec['rooms']} | 글자 {words} | 물체 {objs}", flush=True)


if __name__ == "__main__":
    main()
