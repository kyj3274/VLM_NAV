"""휴대폰이 보내는 사진을 줄여도 문패를 읽는지 본다 (서버). 웹 세션 폴더의 사진(s{구간}_{초}_{k}.jpg)을 그대로 쓴다.
- 장수: 1초 4장을 모두 읽을 때 vs 가장 선명한 1장만 읽을 때, 호실 번호를 읽은 초 수
- 압축: 가장 선명한 1장을 JPEG 품질 70/55/40으로 다시 저장했을 때 크기와 읽은 호실
  (원본은 브라우저 품질 0.85. 다시 저장한 것이라 실제 휴대폰 압축과 조금 다를 수 있음)

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python send_ab.py <세션 폴더> [<세션 폴더> …]
결과: 화면 요약, <첫 세션>/send_ab.jsonl
"""

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import exp_a_ocr
import pipeline

QUALS = (70, 55, 40)


def rooms(ocr, img):
    texts = exp_a_ocr.read_all(ocr, img, pipeline.OCR_MAX_SIDE)[0]
    return sorted({t["text"] for t in texts if pipeline.ROOM.match(t["text"]) and t["score"] >= pipeline.OCR_THRESHOLD})


def sharp(img):
    return float(cv2.Laplacian(cv2.cvtColor(np.array(img.reduce(2)), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    ocr = exp_a_ocr.load_ocr("m", {})
    out = open(Path(sys.argv[1]) / "send_ab.jsonl", "w", encoding="utf-8")
    n = 0
    all_any = key_any = same = 0
    lost, gained = [], []
    q_any = {q: 0 for q in QUALS}
    q_same = {q: 0 for q in QUALS}
    size = defaultdict(list)
    for d in map(Path, sys.argv[1:]):
        by_sec = defaultdict(list)
        for p in sorted(d.glob("s*_*_*.jpg")):
            by_sec[p.name.rsplit("_", 1)[0]].append(p)
        for sec, paths in sorted(by_sec.items()):
            imgs = [Image.open(p).convert("RGB") for p in paths]
            per = [rooms(ocr, im) for im in imgs]
            k = max(range(len(imgs)), key=lambda i: sharp(imgs[i]))
            r_all, r_key = sorted(set().union(*per)), per[k]
            rec = {"session": d.name, "sec": sec, "frames": len(paths), "all": r_all, "key": r_key, "q": {}}
            size["원본 1장"].append(paths[k].stat().st_size)
            size["원본 4장"].append(sum(p.stat().st_size for p in paths))
            for q in QUALS:
                buf = io.BytesIO()
                imgs[k].save(buf, "JPEG", quality=q)
                size[f"품질 {q} 1장"].append(buf.tell())
                rq = rooms(ocr, Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
                rec["q"][q] = rq
                q_any[q] += bool(rq)
                q_same[q] += rq == r_key
            n += 1
            all_any += bool(r_all)
            key_any += bool(r_key)
            same += r_all == r_key
            if set(r_all) - set(r_key):
                lost.append((d.name[-10:], sec, sorted(set(r_all) - set(r_key))))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{d.name[-10:]} {sec} 4장 {r_all or '-'} | 1장 {r_key or '-'} | " + " ".join(f"q{q} {rec['q'][q] or '-'}" for q in QUALS), flush=True)
    print(f"\n전체 {n}초")
    print(f"호실을 읽은 초: 4장 모두 {all_any}, 선명한 1장만 {key_any} (결과가 완전히 같은 초 {same}/{n})")
    print(f"1장만 보면 놓치는 번호가 있는 초 {len(lost)}: {lost[:12]}")
    for q in QUALS:
        print(f"품질 {q}: 호실 읽은 초 {q_any[q]} (원본 1장 {key_any}), 원본 1장과 결과가 같은 초 {q_same[q]}/{n}")
    print("\n| 보내는 것 | 1초치 크기 (평균) |\n|---|---|")
    for k, v in size.items():
        print(f"| {k} | {sum(v) / len(v) / 1024:.0f} KB |")


if __name__ == "__main__":
    main()
