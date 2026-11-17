"""11/16 시연 기록에서 걸음마다 OCR이 읽은 호실 후보를 모으고 번호별 조각 한 장을 만든다 (서버, 다수결 규칙 시뮬레이션용).
결과는 서버 outputs/kyj/tmp/vote/ (cands.json, sheet.jpg). Mac 사본은 data/video/results/vote_cands.json.
  .venv/bin/python vote_collect.py <세션 이름> [...]
"""
import json, re, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOM = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")
S = Path("/workspace/capstone/outputs/kyj/web_sessions")
out = Path("/workspace/capstone/outputs/kyj/tmp/vote"); out.mkdir(parents=True, exist_ok=True)
rows, best = [], {}
for name in sys.argv[1:]:
    d = S / name
    t = -1
    for line in open(d / "log.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "event" in r:
            continue
        t += 1
        cands = [{"room": o["text"].replace("호", ""), "score": o["score"], "h": round(o["box"][3] - o["box"][1], 4), "box": o["box"]}
                 for o in r["raw"]["ocr"] if ROOM.match(o["text"].replace("호", ""))]
        rows.append({"session": name, "t": t, "seg": r["segment"], "sec": r["sec"], "key": r["key_frame"], "wall": r.get("wall"),
                     "cands": cands, "confirmed": r["rooms"], "sent": r.get("sent_frames", len(r["used_frames"]))})
        for c in cands:
            k = (name, c["room"])
            if k not in best or c["h"] > best[k][0]["h"]:
                best[k] = (c, d / r["key_frame"], t)
json.dump(rows, open(out / "cands.json", "w", encoding="utf-8"), ensure_ascii=False)
tiles = []
for (name, room), (c, path, t) in sorted(best.items()):
    im = Image.open(path).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = c["box"]; pw, ph = (x1 - x0) * 2.5 + 0.03, (y1 - y0) * 3 + 0.03
    crop = im.crop((int(max(0, x0 - pw) * W), int(max(0, y0 - ph) * H), int(min(1, x1 + pw) * W), int(min(1, y1 + ph) * H)))
    crop.thumbnail((300, 190)); tile = Image.new("RGB", (310, 225), "white"); tile.paste(crop, (5, 30))
    ImageDraw.Draw(tile).text((5, 5), f"{name[9:15]} {room} t{t} s{c['score']:.2f}", fill="black")
    tiles.append(tile)
cols = 6
sheet = Image.new("RGB", (310 * cols, 225 * ((len(tiles) + cols - 1) // cols)), "white")
for i, t in enumerate(tiles):
    sheet.paste(t, (310 * (i % cols), 225 * (i // cols)))
sheet.save(out / "sheet.jpg", quality=85)
print(len(rows), "초,", len(best), "번호")
for (name, room), (c, _, t) in sorted(best.items()):
    n = sum(any(x["room"] == room for x in r["cands"]) for r in rows if r["session"] == name)
    print(name[9:15], room, "읽힌 초", n, "최고 조각 점수", c["score"], "높이", c["h"])
