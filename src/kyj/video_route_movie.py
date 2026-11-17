"""길안내 결과(guide_*.jsonl)를 영상 위에 자막으로 입혀 경로 전체 시연 영상을 만든다 (Mac, ffmpeg 필요).

화면: 위 = 지금 상태(층, 읽은 호실, VLM 호출), 아래 = 안내 문장(새 문장은 4초 동안 표시).
사용: python3 src/kyj/video_route_movie.py data/video/results_guide_314_1_2b_lean.jsonl data/video/route_314_3층.mp4
"""

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2] / "data" / "video" / "frames_8fps"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FPS = 8
HOLD = 4  # 안내 문장을 몇 초 보여 줄지


def main():
    src, out = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    big, small = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 24)
    tmp = Path(tempfile.mkdtemp())
    n, said, said_at, vlm_total, rooms_seen = 0, "", -99, 0, []
    for i, r in enumerate(recs):
        if r["say"]:
            said, said_at = " ".join(r["say"]), i
        vlm_total += r["vlm_called"]
        for x in r["rooms"]:
            if x not in rooms_seen:
                rooms_seen.append(x)
        files = sorted((ROOT / r["seq"]).glob("i*.jpg"))[r["sec"] * FPS:(r["sec"] + 1) * FPS]
        st = r["state"]
        floor = f"{st['floor']}층" if st.get("floor") else "모름"
        seg = "입구 → 엘리베이터" if r["seq"].startswith("1_") else "엘리베이터에서 내린 뒤"
        head = [f"목적지 {r['route_target']}호 | {seg} | {i + 1}초",
                f"지금 층: {floor}   읽은 호실: {', '.join(rooms_seen[-4:]) or '-'}   VLM 호출 누적 {vlm_total}번"
                + ("   ← 지금 VLM 확인 중" if r["vlm_called"] else "")]
        for f in files:
            im = Image.open(f).convert("RGB")
            d = ImageDraw.Draw(im, "RGBA")
            W, H = im.size
            d.rectangle((0, 0, W, 92), fill=(0, 0, 0, 170))
            d.text((16, 12), head[0], font=small, fill="white")
            d.text((16, 50), head[1], font=small, fill=(255, 210, 120) if r["vlm_called"] else (200, 220, 255))
            if said and i - said_at < HOLD:
                lines = textwrap.wrap(said, 22)
                h = 22 + 46 * len(lines)
                d.rectangle((0, H - h - 20, W, H), fill=(10, 60, 160, 210))
                for k, line in enumerate(lines):
                    d.text((20, H - h - 6 + 46 * k), line, font=big, fill="white")
            im.save(tmp / f"{n:05d}.jpg", quality=90)
            n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=720:-2", out], check=True)
    print(n, "frames →", out)


if __name__ == "__main__":
    main()
