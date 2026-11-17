"""영상 1초마다 YOLOE가 무엇을 찾는지 상자를 그려서 본다 (Mac에서 실행, MPS).
1초에 8장을 뽑아 가장 선명한 장(추적기 --key sharp와 같은 방식)에 파이프라인과 같은 7종·같은 영어 표현으로 YOLOE를 돌린다.
0.15 이상(파이프라인이 쓰는 값)은 실선, 0.05~0.15는 점선. 프레임 정답표(frame_gt.json)의 others와 나란히 저장한다.
--pf: 표현을 주지 않고 프롬프트 없는 모델(yoloe-26l-seg-pf, 내장 이름 4585개)이 스스로 이름을 붙이게 한다.

  .venv/bin/python src/kyj/yoloe_scene_view.py data/video/raw/졸프_영상2.mp4 --seq 2b_after_elevator_졸프_영상2 --out <폴더>
결과: <폴더>/sNN.jpg(상자 그린 사진), <폴더>/scenes.json
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pipeline import PHRASES, YOLOE_SIZE, YOLOE_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"문": (46, 125, 220), "엘리베이터": (220, 60, 50), "계단": (240, 160, 0), "화장실": (150, 70, 200),
          "비상구표시": (30, 170, 80), "사람": (230, 80, 160), "상자": (120, 90, 50)}
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def load(weights, pf=False):
    from ultralytics import YOLOE

    os.chdir(Path(weights).parent)  # MobileCLIP 글자 모델을 같은 폴더에서 찾게
    model = YOLOE(Path(weights).name)
    if pf:
        return model
    rows = []
    for phrases in PHRASES.values():
        pe = model.get_text_pe(phrases).mean(dim=1, keepdim=True)
        rows.append(pe / pe.norm(dim=-1, keepdim=True))
    model.set_classes(list(PHRASES), torch.cat(rows, dim=1))
    return model


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def draw(img, dets):
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, max(18, img.width // 32))
    for o in sorted(dets, key=lambda o: o["conf"]):
        x0, y0, x1, y1 = [v * s for v, s in zip(o["box"], (img.width, img.height) * 2)]
        c = COLORS.get(o["name"]) or tuple(int(v) for v in np.random.default_rng(sum(map(ord, o["name"]))).integers(40, 220, 3))
        if o["conf"] >= YOLOE_THRESHOLD:
            d.rectangle([x0, y0, x1, y1], outline=c, width=5)
            label = f"{o['name']} {o['conf']:.2f}"
            tw = d.textlength(label, font=font)
            d.rectangle([x0, y0, x0 + tw + 10, y0 + font.size + 8], fill=c)
            d.text((x0 + 5, y0 + 2), label, fill="white", font=font)
        else:  # 약한 후보: 점선
            for a, b in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))):
                n = max(1, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 14))
                for i in range(0, n, 2):
                    p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                    q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                    d.line([p, q], fill=c, width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seq", help="frame_gt.json의 seq (정답표와 나란히 보기)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(ROOT / "local_models" / "yoloe-11l-seg.pt"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--pf", action="store_true", help="표현 없이 프롬프트 없는 모델로")
    a = ap.parse_args()
    video, out = Path(a.video).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gt = {}
    if a.seq:
        for f in json.load(open(ROOT / "data/video/frame_gt.json", encoding="utf-8"))["frames"]:
            if f["seq"] == a.seq:
                gt[int(f["frame"][1:4])] = f["정답"]

    with tempfile.TemporaryDirectory() as tmp:  # ffmpeg가 회전 정보를 반영해 세로로 뽑는다
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={a.fps}", "-q:v", "2",
                        f"{tmp}/i%05d.jpg"], check=True)
        frames = sorted(Path(tmp).glob("i*.jpg"))
        weights = str(ROOT / "local_models" / "yoloe-26l-seg-pf.pt") if a.pf and a.weights == ap.get_default("weights") else a.weights
        model = load(weights, a.pf)
        names = model.names if a.pf else list(PHRASES)
        scenes = []
        for sec in range(len(frames) // a.fps):
            group = [Image.open(p).convert("RGB") for p in frames[sec * a.fps:(sec + 1) * a.fps]]
            sharp = [sharpness(g) for g in group]
            k = int(np.argmax(sharp))
            img = group[k]
            r = model.predict(img, imgsz=YOLOE_SIZE, conf=0.05, device="mps", verbose=False)[0]
            dets = [{"name": names[int(c)], "conf": round(s, 3),
                     "box": [round(b[0] / img.width, 3), round(b[1] / img.height, 3),
                             round(b[2] / img.width, 3), round(b[3] / img.height, 3)]}
                    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            shown = draw(img, dets)
            shown.thumbnail((720, 720))
            shown.save(out / f"s{sec:02d}.jpg", quality=82)
            g = gt.get(sec, {})
            scenes.append({"sec": sec, "frame": k, "sharpness": round(sharp[k]), "size": list(img.size),
                           "detections": dets, "gt_scene": g.get("장면", ""), "gt_others": g.get("others", []),
                           "gt_rooms": g.get("호실_표지판", [])})
            strong = ", ".join(f"{o['name']} {o['conf']:.2f}" for o in dets if o["conf"] >= YOLOE_THRESHOLD) or "-"
            print(f"{sec:2d}s 선명도 {sharp[k]:5.0f} | YOLOE {strong} | 정답 {g.get('장면', '')} {g.get('others', [])}",
                  flush=True)
    json.dump({"video": video.name, "seq": a.seq, "threshold": YOLOE_THRESHOLD, "scenes": scenes},
              open(out / "scenes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
