"""채점 기준 v2용: 1단계 OCR의 호실 후보(3자리 숫자)를 원본에서 주변까지 잘라 저장."""
import json, re, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import run_task
ROOM = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")
out_dir = run_task.OUTPUT_DIR / "ref_crops"; out_dir.mkdir(exist_ok=True)
paths = {run_task.image_id(p): p for p in run_task.find_images(["3rdFloor", "4thFloor", "5thFloor"])}
index = []
for line in open(run_task.OUTPUT_DIR / "pipeline_stage1.jsonl", encoding="utf-8"):
    r = json.loads(line)
    image = None
    for k, t in enumerate(r["texts"]):
        value = t["text"].replace("호", "").strip()
        if not ROOM.match(value):
            continue
        image = image or Image.open(paths[r["image_id"]]).convert("RGB")
        W, H = image.size
        x1, y1, x2, y2 = t["box"]; w, h = x2 - x1, y2 - y1
        box = (max(0, (x1 - 2.5 * w) * W), max(0, (y1 - 3 * h) * H), min(W, (x2 + 2.5 * w) * W), min(H, (y2 + 5 * h) * H))
        crop = image.crop(tuple(int(v) for v in box)); crop.thumbnail((480, 480))
        name = f"{r['image_id']}_{k:02d}_{value}.jpg"; crop.save(out_dir / name, quality=90)
        index.append({"crop": name, "image_id": r["image_id"], "text": value, "height_pct": round(h * 100, 2), "center_y": round((y1 + y2) / 2, 3)})
json.dump(index, open(out_dir / "index.json", "w"), ensure_ascii=False, indent=1)
print(len(index), "crops")
