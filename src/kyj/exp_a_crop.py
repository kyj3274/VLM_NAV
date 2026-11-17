"""실험 A 보조: (image_id, 상대좌표 상자) 주변을 원본에서 잘라 확인용 jpg로 저장.
사용법: python exp_a_crop.py 이름 3F_007:0.36,0.01,0.38,0.03 ...  → outputs/kyj/exp_a_crops/이름_*.jpg"""
import sys
from PIL import Image
import run_task
paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
out = run_task.OUTPUT_DIR / "exp_a_crops"
out.mkdir(exist_ok=True)
for arg in sys.argv[2:]:
    iid, box = arg.split(":")
    x0, y0, x1, y1 = map(float, box.split(","))
    im = Image.open(paths[iid]).convert("RGB")
    w, h = im.size
    pad = max(x1 - x0, y1 - y0) * 3 + 0.03
    c = im.crop((int(max(0, x0 - pad) * w), int(max(0, y0 - pad) * h), int(min(1, x1 + pad) * w), int(min(1, y1 + pad) * h)))
    c.thumbnail((600, 600))
    c.save(out / f"{sys.argv[1]}_{iid}_{x0:.2f}_{y0:.2f}.jpg", quality=90)
