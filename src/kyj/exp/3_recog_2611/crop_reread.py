"""'문패 한글만 읽힘' 장면에서 문패를 잘라 키워 다시 읽으면 호실 번호가 나오는지 본다 (서버). 지금은 이때 VLM에 조각을 물음.
웹 세션 log.jsonl에서 호실이 확정되지 않았고 한글 글자(점수 0.5 이상, 높이 0.012 이상)가 있는 초의 대표 사진을 쓴다.
한글 상자 주변(문패 크기만큼)을 잘라 2배·3배로 키워 OCR을 다시 돌린다.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python crop_reread.py <결과 폴더> <세션 이름> [...]
결과: <결과 폴더>/crop_reread.json, sheet.jpg (조각과 읽은 번호)
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nav"))  # 서비스 모듈 (src/kyj/nav)

import ocr_reader as exp_a_ocr
import perception as pipeline

HANGUL = re.compile(r"[가-힣]")
S = Path("/workspace/capstone/outputs/kyj/web_sessions")
SCALES = (2, 3)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ocr = exp_a_ocr.load_ocr("m", {})
    rows, tiles = [], []
    for name in sys.argv[2:]:
        t = -1
        for line in open(S / name / "log.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if "event" in r:
                continue
            t += 1
            if r["rooms"]:
                continue
            han = [o for o in r["raw"]["ocr"] if o["score"] >= 0.5 and HANGUL.search(o["text"]) and o["box"][3] - o["box"][1] >= 0.012]
            if not han:
                continue
            img = Image.open(S / name / r["key_frame"]).convert("RGB")
            W, H = img.size
            for o in sorted(han, key=lambda o: -(o["box"][3] - o["box"][1]))[:2]:  # 큰 글자 2개까지
                x0, y0, x1, y1 = o["box"]
                px, py = (x1 - x0) * 0.6 + 0.03, (y1 - y0) * 3 + 0.02  # 번호는 보통 실 이름 위·아래 줄에 있음
                crop = img.crop((int(max(0, x0 - px) * W), int(max(0, y0 - py) * H), int(min(1, x1 + px) * W), int(min(1, y1 + py) * H)))
                got = {}
                for k in SCALES:
                    big = crop.resize((crop.width * k, crop.height * k), Image.LANCZOS)
                    texts = exp_a_ocr.read_all(ocr, big, pipeline.OCR_MAX_SIDE)[0]
                    got[k] = [(x["text"], round(x["score"], 2)) for x in texts if pipeline.ROOM.match(x["text"].replace("호", ""))]
                rows.append({"session": name, "t": t, "sec": r["sec"], "key": r["key_frame"], "hangul": o["text"], "h": round(y1 - y0, 4),
                             "crop": list(crop.size), "rooms": got, "all3": [(x["text"], round(x["score"], 2)) for x in texts][:8]})
                print(name[9:15], t, o["text"], crop.size, got, flush=True)
                tile = Image.new("RGB", (330, 250), "white")
                c = crop.copy(); c.thumbnail((320, 200)); tile.paste(c, (5, 45))
                d = ImageDraw.Draw(tile)
                d.text((5, 5), f"{name[9:15]} t{t} {crop.size[0]}x{crop.size[1]}", fill="black")
                d.text((5, 22), f"x2 {got[2]}  x3 {got[3]}", fill="red" if got[2] or got[3] else "gray")
                tiles.append(tile)
    json.dump(rows, open(out / "crop_reread.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = 5
    sheet = Image.new("RGB", (330 * cols, 250 * max(1, (len(tiles) + cols - 1) // cols)), "white")
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (330 * (i % cols), 250 * (i // cols)))
    sheet.save(out / "sheet.jpg", quality=85)
    secs = {(x["session"], x["t"]) for x in rows}
    hit = {(x["session"], x["t"]) for x in rows if x["rooms"][2] or x["rooms"][3]}
    print(f"\n한글만 읽힌 초 {len(secs)}, 잘라 읽어 번호가 나온 초 {len(hit)} (조각 {len(rows)}개)")


if __name__ == "__main__":
    main()
