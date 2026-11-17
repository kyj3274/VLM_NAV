"""영상 프레임별 정답표 초안 만들기.

1단계(YOLOE+OCR) 결과로 칸을 미리 채운다. 이 값은 정답이 아니라 사람이 고칠 초안이다.
항목 이름은 cjy 평가 기준(evaluation_prompt.txt)의 층표시_표지판, 호실_표지판, 화살표, others 21종에 맞춘다.
영상 칸(층, 진행)은 경로 순서를 보고 사람이 채운다.

사용: python3 src/kyj/video_frame_gt.py            → data/video/frame_gt_draft.json
      python3 src/kyj/video_frame_gt.py --sheets   → 검토용 6장 묶음 사진 (scratchpad 등 --sheet-dir)
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "video"
LANDMARKS = [
    "엘리베이터", "에스컬레이터", "계단", "출입문", "셔터", "복도끝",
    "안내판", "입간판", "비상구표시", "화장실표시", "바닥유도표시",
    "화장실입구", "안내데스크", "정수기", "자판기", "소파",
    "소화전", "소화기", "CCTV", "기둥", "사물함",
]
DET_TO_LM = {"엘리베이터": "엘리베이터", "계단": "계단", "문": "출입문", "비상구표시": "비상구표시", "화장실": "화장실표시"}
NEAR_H = 0.012  # 가까운 문패 = 글자 높이가 화면 높이의 1.2% 이상 (video_tracker.py와 같음)


def side(cx):
    return "왼쪽" if cx < 0.4 else "오른쪽" if cx > 0.6 else "가운데"


def draft_row(r):
    rooms = {}
    for t in r["texts_all"]:
        m = re.fullmatch(r"\D{0,2}([1-5]\d\d)\D{0,3}", t["text"].strip())
        if not m or t["score"] < 0.5:
            continue
        b = t["box"]
        h = b[3] - b[1]
        num = m.group(1)
        if num not in rooms or h > rooms[num]["h"]:
            rooms[num] = {"번호": num, "쪽": side((b[0] + b[2]) / 2), "가까움": h >= NEAR_H,
                          "h": round(h, 4), "ocr_score": round(t["score"], 2)}
    floors = [f for f in r["result"].get("floors", [])]
    lms = sorted({DET_TO_LM[d["name"]] for d in r["detections"] if d["name"] in DET_TO_LM and d["conf"] >= 0.4})
    return {
        "seq": r["seq"], "frame": r["frame"],
        "층표시_표지판": floors,
        "호실_표지판": sorted(rooms.values(), key=lambda x: -x["h"]),
        "화살표": [],
        "others": lms,
        "장면": None, "층": None, "확인": False, "메모": "",
    }


def make_sheets(rows, out_dir):
    from PIL import Image, ImageDraw, ImageFont
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    W, H = 360, 640
    for seq in dict.fromkeys(r["seq"] for r in rows):
        sr = [r for r in rows if r["seq"] == seq]
        for k in range(0, len(sr), 6):
            sheet = Image.new("RGB", (W * 3, (H + 60) * 2), "white")
            d = ImageDraw.Draw(sheet)
            for j, r in enumerate(sr[k:k + 6]):
                im = Image.open(ROOT / "frames" / seq / r["frame"]).convert("RGB").resize((W, H))
                x, y = (j % 3) * W, (j // 3) * (H + 60)
                sheet.paste(im, (x, y))
                rooms = " ".join(f'{p["번호"]}{"*" if p["가까움"] else ""}' for p in r["호실_표지판"])
                d.text((x + 4, y + H + 2), f'{r["frame"]}  {rooms}', fill="black", font=font)
                d.text((x + 4, y + H + 30), ",".join(r["others"]), fill="blue", font=font)
            sheet.save(out_dir / f"{seq[:2]}_{k // 6:02d}.jpg", quality=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "results_stage1.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "frame_gt_draft.json"))
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--sheet-dir", default=str(ROOT / "gt_sheets"))
    a = ap.parse_args()
    rows = [draft_row(json.loads(l)) for l in open(a.stage1, encoding="utf-8")]
    if a.sheets:
        make_sheets(rows, Path(a.sheet_dir))
        return
    doc = {"_설명": "프레임별 정답표 초안. 칸 이름은 cjy 평가 기준과 같음. 초안 값은 1단계 결과이며 '확인'이 true인 행만 정답.",
           "landmarks_21": LANDMARKS, "frames": rows}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "frames →", a.out)


if __name__ == "__main__":
    main()
