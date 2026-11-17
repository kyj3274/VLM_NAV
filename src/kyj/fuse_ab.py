"""OCR이 읽은 호실 번호가 '문패에 적힌 것'인지 YOLOE로 확인할 수 있는지 잰다 (서버).
11/16 웹 시연 4개에서 대표 사진이 읽은 번호(점수 0.5 이상)마다
  ① 문 기준: 글자가 YOLOE 문 상자 옆·위쪽에 있는가 (좁게 = 거리 추정에 쓰는 조건, 넓게 = 문 상자를 사방으로 넓힌 안)
  ② 조각 분류: 글자 주변을 잘라 키운 조각에서 YOLOE가 문패/포스터/안내도/표지/간판 중 무엇을 찾는가
정답(진짜 문패 / 문패인데 번호 오독 / 문패 아님)은 사진 조각을 보고 사람이 판정한 것.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python fuse_ab.py <결과 폴더>
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import pipeline

S = Path("/workspace/capstone/outputs/kyj/web_sessions")
ROOM = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")
REAL = {"20261116_075501_314": {"307", "309", "313", "315", "316", "317", "319"}, "20261116_092851_410": {"410"},
        "20261116_093803_410": set(), "20261116_094718_410": {"409", "410"}}
MISREAD = {"20261116_075501_314": {"100", "207", "209", "300", "399", "504"}}  # 진짜 문패를 잘못 읽은 것 (문패는 맞음)
CLASSES = {"문패": ["door plate", "room number sign", "room name plate on wall"],
           "포스터": ["poster", "notice paper", "flyer"],
           "안내도": ["floor plan", "building directory map", "map board"],
           "표지": ["warning sign", "safety sign sticker"],
           "간판": ["banner", "signboard"]}


def set_classes(model, phrases):
    rows = []
    for ph in phrases.values():
        pe = model.get_text_pe(ph).mean(dim=1, keepdim=True)
        rows.append(pe / pe.norm(dim=-1, keepdim=True))
    model.set_classes(list(phrases), torch.cat(rows, dim=1))


def near_door(objs, cx, cy, wide):
    for d in objs:
        if d["name"] not in ("문", "엘리베이터") or d["conf"] < 0.15:
            continue
        x0, y0, x1, y1 = d["box"]
        if wide:
            w, h = x1 - x0, y1 - y0
            if x0 - 0.6 * w - 0.05 <= cx <= x1 + 0.6 * w + 0.05 and y0 - 0.25 * h - 0.05 <= cy <= y1:
                return True
        elif x0 - 0.12 <= cx <= x1 + 0.12 and y0 - 0.15 <= cy <= (y0 + y1) / 2:
            return True
    return False


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    yoloe = pipeline.load_yoloe()
    set_classes(yoloe, CLASSES)
    names = list(CLASSES)
    rows, tiles = [], []
    for name in REAL:
        t = -1
        for line in open(S / name / "log.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if "event" in r:
                continue
            t += 1
            cands = [o for o in r["raw"]["ocr"] if ROOM.match(o["text"].replace("호", "")) and o["score"] >= 0.5]
            if not cands:
                continue
            img = Image.open(S / name / r["key_frame"]).convert("RGB")
            W, H = img.size
            for o in cands:
                room = o["text"].replace("호", "")
                kind = "진짜 문패" if room in REAL[name] else "문패 오독" if room in MISREAD.get(name, ()) else "문패 아님"
                x0, y0, x1, y1 = o["box"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                px, py = (x1 - x0) * 2.5 + 0.04, (y1 - y0) * 4 + 0.04
                box = (int(max(0, x0 - px) * W), int(max(0, y0 - py) * H), int(min(1, x1 + px) * W), int(min(1, y1 + py) * H))
                crop = img.crop(box)
                k = max(1, round(640 / max(crop.size)))
                big = crop.resize((crop.width * k, crop.height * k), Image.LANCZOS)
                res = yoloe.predict(big, imgsz=640, conf=0.05, device=0, verbose=False)[0]
                tx, ty = (cx * W - box[0]) * k, (cy * H - box[1]) * k  # 조각 안에서 글자 중심
                dets = []
                for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    inside = b[0] <= tx <= b[2] and b[1] <= ty <= b[3]
                    dets.append({"name": names[int(c)], "conf": round(s, 2), "inside": inside,
                                 "frac": round((b[2] - b[0]) * (b[3] - b[1]) / (big.width * big.height), 2)})
                on = sorted([d for d in dets if d["inside"]], key=lambda d: -d["conf"])
                pred = on[0]["name"] if on else "없음"
                row = {"session": name, "t": t, "room": room, "score": round(o["score"], 2), "kind": kind,
                       "door_narrow": near_door(r["raw"]["objects"], cx, cy, False), "door_wide": near_door(r["raw"]["objects"], cx, cy, True),
                       "pred": pred, "pred_conf": on[0]["conf"] if on else 0, "on_text": on[:4]}
                rows.append(row)
                print(f"{name[9:15]} t{t:2d} {room} {kind} | 문 옆 좁게 {row['door_narrow']} 넓게 {row['door_wide']} | 조각 {pred} "
                      f"{row['pred_conf']} {[(d['name'], d['conf']) for d in on[:3]]}", flush=True)
                tile = Image.new("RGB", (260, 250), "white")
                c2 = crop.copy(); c2.thumbnail((250, 190)); tile.paste(c2, (5, 55))
                d = ImageDraw.Draw(tile)
                d.text((5, 3), f"{name[9:15]} t{t} {room} [{'O' if kind == '진짜 문패' else 'M' if kind == '문패 오독' else 'X'}]", fill="black")
                d.text((5, 18), f"door n={int(row['door_narrow'])} w={int(row['door_wide'])}", fill="black")
                d.text((5, 33), f"crop: {['plate','poster','map','sign','banner','none'][(names + ['없음']).index(pred)]} {row['pred_conf']}", fill="blue")
                tiles.append(tile)
    json.dump(rows, open(out / "fuse_ab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = 8
    sheet = Image.new("RGB", (260 * cols, 250 * ((len(tiles) + cols - 1) // cols)), "white")
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (260 * (i % cols), 250 * (i // cols)))
    sheet.save(out / "sheet.jpg", quality=80)
    print("\n| 정답 | 읽힌 횟수 | ① 문 옆 (좁게) | ① 문 옆 (넓게) | ② 조각을 문패로 봄 | ② 다른 것으로 봄 | ② 아무것도 못 찾음 |\n|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for kind in ("진짜 문패", "문패 오독", "문패 아님"):
        v = by[kind]
        other = Counter(r["pred"] for r in v if r["pred"] not in ("문패", "없음"))
        print(f"| {kind} | {len(v)} | {sum(r['door_narrow'] for r in v)} | {sum(r['door_wide'] for r in v)} | "
              f"{sum(r['pred'] == '문패' for r in v)} | {dict(other)} | {sum(r['pred'] == '없음' for r in v)} |")


if __name__ == "__main__":
    main()
