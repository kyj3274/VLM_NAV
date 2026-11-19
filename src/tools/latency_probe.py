"""지연 원인 확인 (2026-11-19 사용자 지시 2). 휴대폰 시험 인식 2.9초·재확인 3.5~4초가 영상 시험(1.3초·1.0초)보다 긴 이유.
같은 맥·같은 게이트웨이에서 조건 하나씩 바꿔 부른다 (크레딧 씀, 한도 --credit-limit).

  A 휴대폰 사진 원본(지금처럼 품질 90 재인코딩)       × 3, 차례로
  B 같은 휴대폰 사진을 품질 75로                      × 3, 차례로   → 보내는 용량 차이
  C 영상 사진(같은 720×1280)                          × 3, 차례로   → 장면·용량 차이
  D 휴대폰 사진의 문패 재확인 조각(--crop 사진, --box 상자) × 3, 차례로
  E 휴대폰 사진 3장을 동시에 (+ 재확인 1개 동시)       × 1          → 동시 호출(겹쳐 보내기) 영향
캐시는 쓰지 않는다(매번 실제 호출). 결과: outputs/probe/latency_<시각>.json

  .venv/bin/python src/tools/latency_probe.py --session outputs/web_sessions/<세션 폴더> --phone 33,35,38 \
      --crop 36 --box 34,25,44,29 --dry-run            # 호출 없이 크기·횟수만
  (같은 인자에 --credit-limit 4 를 주고 --dry-run 을 빼면 실제 호출)
"""
import argparse
import io
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))
from PIL import Image  # noqa: E402

import claude_perceiver as cp  # noqa: E402
from claude_client import DryRunClient, GatewayClient, Ledger, ROOT, jpeg_block  # noqa: E402

VIDEO = [ROOT / "data/video/frames_8fps/2a_after_elevator_졸프_영상1" / f"i{i:04d}.jpg" for i in (100, 140, 180)]


def block(path, quality):
    b = io.BytesIO()
    Image.open(path).convert("RGB").save(b, "JPEG", quality=quality)
    img = Image.open(io.BytesIO(b.getvalue()))
    return jpeg_block(img, quality=quality), len(b.getvalue())


def see(c, path, quality=90):
    blk, n = block(path, quality)
    r = c.call(cp.SYSTEM, [blk, {"type": "text", "text": "보이는 것을 적어라."}], cp.SEE_SCHEMA, max_tokens=500, kind="see")
    return {"kind": "see", "file": path.name, "q": quality, "bytes": n, "sec": r.seconds, "attempts": r.attempts,
            "retry_wait": r.retry_wait, "in": r.input_tokens, "out": r.output_tokens, "cr": round(r.credit, 4), "err": r.error}


def recheck(c, crop_frame, box):
    p = cp.Perceiver(c)
    t0 = time.monotonic()
    rc = p.recheck(str(crop_frame), {"n": "", "name": "", "box": box})
    log = rc["call"]
    return {"kind": "recheck", "read": rc["n"], "sec": log.get("seconds", round(time.monotonic() - t0, 3)),
            "attempts": log.get("attempts"), "in": log.get("input_tokens"), "out": log.get("output_tokens"),
            "cr": round(log.get("credit") or 0, 4), "err": log.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="웹 세션 폴더 (frames/ 아래 사진)")
    ap.add_argument("--phone", default="33,35,38", help="인식에 쓸 사진 번호 3개 (쉼표)")
    ap.add_argument("--crop", type=int, required=True, help="재확인 조각을 자를 사진 번호")
    ap.add_argument("--box", required=True, help="문패 상자 왼,위,오,아래 (0~100)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--credit-limit", type=float, default=4.0)
    a = ap.parse_args()
    fr = Path(a.session) / "frames"
    PHONE = [fr / f"{int(i):05d}.jpg" for i in a.phone.split(",")]
    crop_frame, box = fr / f"{a.crop:05d}.jpg", [int(v) for v in a.box.split(",")]
    if a.dry_run:
        for f in PHONE + VIDEO:
            print(f.name, "q90", block(f, 90)[1], "q75", block(f, 75)[1])
        print("호출: A3 + B3 + C3 + D3 + E4 = 16번 (인식 13, 재확인 3+1)")
        return
    c = GatewayClient(timeout=30.0, ledger=Ledger(a.credit_limit))
    rows = []
    for name, fn in [("A 휴대폰 q90", lambda i: see(c, PHONE[i], 90)), ("B 휴대폰 q75", lambda i: see(c, PHONE[i], 75)),
                     ("C 영상 q90", lambda i: see(c, VIDEO[i], 90)), ("D 재확인", lambda i: recheck(c, crop_frame, box))]:
        for i in range(3):
            r = {"cond": name, **fn(i)}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(4) as ex:   # E: 동시에
        futs = [ex.submit(see, c, f, 90) for f in PHONE] + [ex.submit(recheck, c, crop_frame, box)]
        for f in futs:
            r = {"cond": "E 동시 4개", **f.result()}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    summ = {}
    for r in rows:
        summ.setdefault(r["cond"], []).append(r["sec"])
    for k, v in summ.items():
        print(f"{k}: 중앙값 {statistics.median(v):.2f}s ({', '.join(f'{x:.2f}' for x in v)})")
    print(f"크레딧 {c.ledger.total:.3f} ({c.ledger.calls}번)")
    out = ROOT / f"outputs/probe/latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "credit": c.ledger.total}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
