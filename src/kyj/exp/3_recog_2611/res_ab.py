"""같은 사진을 원래 크기(1080)와 720으로 줄인 크기로 각각 OCR을 돌려 해상도 효과만 비교한다.
웹 세션 폴더의 사진(s{구간}_{초}_{k}.jpg)을 모두 쓰고, 초마다 호실 번호를 무엇을 읽었는지와 글자 높이(px)를 비교한다.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python res_ab.py <세션 폴더> [--small 1280]
결과: <세션>/res_ab.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nav"))  # 서비스 모듈 (src/kyj/nav)

import ocr_reader as exp_a_ocr
import perception as pipeline


def rooms(texts, height):
    return [(t["text"], round(t["score"], 2), round((t["box"][3] - t["box"][1]) * height))
            for t in texts if pipeline.ROOM.match(t["text"]) and t["score"] >= pipeline.OCR_THRESHOLD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--small", type=int, default=1280, help="줄인 사진의 긴 변 (웹의 보통 화질 = 1280)")
    a = ap.parse_args()
    d = Path(a.session)
    ocr = exp_a_ocr.load_ocr("m", {})
    by_sec = defaultdict(list)
    with open(d / "res_ab.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(d.glob("s*_*_*.jpg")):
            big = Image.open(p).convert("RGB")
            small = big.copy()
            small.thumbnail((a.small, a.small), Image.LANCZOS)
            sharp = float(cv2.Laplacian(cv2.cvtColor(np.array(big), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
            rb = rooms(exp_a_ocr.read_all(ocr, big, pipeline.OCR_MAX_SIDE)[0], big.height)
            rs = rooms(exp_a_ocr.read_all(ocr, small, pipeline.OCR_MAX_SIDE)[0], small.height)
            rec = {"frame": p.name, "size": list(big.size), "small": list(small.size), "sharpness": round(sharp),
                   "rooms_big": rb, "rooms_small": rs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            by_sec[p.name.rsplit("_", 1)[0]].append(rec)
            print(f"{p.name} 선명도 {sharp:5.0f} | 1080 {rb or '-'} | 720 {rs or '-'}", flush=True)
    nb = sum(any(r["rooms_big"] for r in v) for v in by_sec.values())
    ns = sum(any(r["rooms_small"] for r in v) for v in by_sec.values())
    fb = sum(bool(r["rooms_big"]) for v in by_sec.values() for r in v)
    fs = sum(bool(r["rooms_small"]) for v in by_sec.values() for r in v)
    n = sum(len(v) for v in by_sec.values())
    print(f"\n호실을 하나라도 읽은 초: 1080 {nb}/{len(by_sec)}, 720 {ns}/{len(by_sec)} | 읽은 사진: 1080 {fb}/{n}, 720 {fs}/{n}")


if __name__ == "__main__":
    main()
