"""사진 한 장 전체가 어떤 장소인지 Places365(ResNet18)로 분류하고 프레임 정답표의 장면과 비교한다 (Mac에서 실행).
1초에 8장 중 가장 선명한 장(추적기 --key sharp와 같은 방식). 장소 이름 365개를 길안내용 묶음(복도/엘리베이터/로비·입구/계단)으로 합친다.

  .venv/bin/python src/kyj/scene_places.py --out <폴더>
결과: <폴더>/scenes_places.json, 화면에 초마다 상위 3개와 채점
모델: src/kyj/models/places365/ (서버) 또는 local_models/places365/ (Mac)의 resnet18_places365.pth.tar (places2.csail.mit.edu 공개 모델)
추적기는 load()와 classify()를 가져다 쓴다 (엘리베이터 문 장면 판단).
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = next((d for d in (Path(__file__).resolve().parent / "models" / "places365", ROOT / "local_models" / "places365")
              if (d / "resnet18_places365.pth.tar").is_file()), None)
VIDEOS = [("졸프_영상3", "1_start_졸프_영상3", "1층"), ("졸프_영상1", "2a_after_elevator_졸프_영상1", "4층"),
          ("졸프_영상2", "2b_after_elevator_졸프_영상2", "3층")]
GROUPS = {"엘리베이터": ["elevator/door", "elevator_lobby", "elevator_shaft"], "복도": ["corridor"],
          "로비·입구": ["lobby", "entrance_hall"], "계단": ["staircase"]}


def load():
    """(모델, 장소 이름 목록). 모델 파일이 없으면 None."""
    if MODEL is None:
        return None
    names = [line.split()[0][3:] for line in open(MODEL / "categories_places365.txt")]
    net = torchvision.models.resnet18(num_classes=365)
    ck = torch.load(MODEL / "resnet18_places365.pth.tar", map_location="cpu", weights_only=True)
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state_dict"].items()})
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return net.eval().to(device), names


PREP = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def classify(model, img):
    """사진 한 장의 장소 확률. (상위 5개 [(이름, 확률)], 묶음별 확률 {복도: ..., 엘리베이터: ...}, elevator/door 확률)"""
    net, names = model
    with torch.no_grad():
        p = torch.softmax(net(PREP(img)[None].to(next(net.parameters()).device)), 1)[0].cpu().numpy()
    top = [(names[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:5]]
    groups = {k: float(sum(p[names.index(n)] for n in v)) for k, v in GROUPS.items()}
    return top, groups, float(p[names.index("elevator/door")])


def sharpness(path):
    return float(cv2.Laplacian(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())


def gt_group(scene):
    if "흐림" in scene and "엘리베이터" not in scene:
        return "흐림"
    if "하차 직후" in scene:
        return "엘리베이터 안·하차"
    if "엘리베이터" in scene:
        return "엘리베이터"
    if "복도" in scene:
        return "복도"
    if "입구" in scene:
        return "로비·입구"
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load()
    gt = json.load(open(ROOT / "data/video/frame_gt.json", encoding="utf-8"))["frames"]
    rows, times = [], []
    for vid, seq, floor in VIDEOS:
        g = {int(f["frame"][1:4]): f["정답"] for f in gt if f["seq"] == seq}
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / f"data/video/raw/{vid}.mp4"), "-vf", "fps=8",
                            "-q:v", "2", f"{tmp}/i%05d.jpg"], check=True)
            fr = sorted(Path(tmp).glob("i*.jpg"))
            for sec in range(len(fr) // 8):
                img = Image.open(max(fr[sec * 8:(sec + 1) * 8], key=sharpness)).convert("RGB")
                t0 = time.perf_counter()
                top, grp, _ = classify(model, img)
                times.append(time.perf_counter() - t0)
                best = max(grp, key=grp.get)
                scene = g.get(sec, {}).get("장면", "")
                rows.append({"floor": floor, "sec": sec, "top": top, "groups": grp, "pred": best if grp[best] >= 0.2 else "기타",
                             "gt_scene": scene, "gt_group": gt_group(scene),
                             "gt_elevator": "엘리베이터" in g.get(sec, {}).get("others", [])})
                r = rows[-1]
                print(f"{floor} {sec:2d}s | {', '.join(f'{n} {v:.2f}' for n, v in top[:3]):60s} | 묶음 {r['pred']:6s} | 정답 {scene}")
    json.dump(rows, open(out / "scenes_places.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n사진 1장 처리 {np.median(times[3:]) * 1000:.0f}ms (Mac MPS 중앙값)")
    pos = [r for r in rows if r["gt_elevator"]]
    neg = [r for r in rows if not r["gt_elevator"]]
    for th in (0.2, 0.3, 0.5):
        tp = sum(r["groups"]["엘리베이터"] >= th for r in pos)
        fp = sum(r["groups"]["엘리베이터"] >= th for r in neg)
        print(f"엘리베이터 묶음 확률 {th} 이상: 진짜 {tp}/{len(pos)}초, 헛것 {fp}/{len(neg)}초")
    hall = [r for r in rows if r["gt_group"] == "복도"]
    print(f"정답 '복도' {len(hall)}초 중 복도로 분류 {sum(r['pred'] == '복도' for r in hall)}초")


if __name__ == "__main__":
    main()
