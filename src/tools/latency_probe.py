"""지연 원인 실험 (2026-11-23 사용자 지시 A1. 2026-11-19 판의 조건 A~E는 git 기록에 있음).
같은 맥·같은 게이트웨이에서 조건 하나씩 바꿔 인식 호출을 한다 (캐시 안 씀, 크레딧 씀, 한도 --credit-limit).

  용량  (동시 0, 차례로) 휴대폰 사진 3장 × 3가지
        orig  휴대폰이 올린 파일 그대로 (재인코딩 없음)
        q70   JPEG 품질 70으로 다시 저장 (휴대폰 품질 0.7에 해당)
        q80   JPEG 품질 80 = 영상 사진 수준(약 90~100KB)
  동시  q80 사진으로, 함께 도는 다른 호출 수 1 (2개 동시) × 2번, 2 (3개 동시) × 1번
해상도는 모두 원본 720×1280. 각 호출의 읽은 문패도 남겨 품질별 판독을 비교한다.
결과: outputs/probe/latency_<시각>.json

  .venv/bin/python src/tools/latency_probe.py --session outputs/web_sessions/<세션> --frames 69,80,81 --dry-run
"""
import argparse
import base64
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
from claude_client import GatewayClient, Ledger, ROOT  # noqa: E402

PER_CALL = 0.19  # 크레딧 추정 (휴대폰 사진 인식 1번, 지난 기록 중앙값 근처)


def encode(path, q):
    raw = Path(path).read_bytes()
    if q != "orig":
        b = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(b, "JPEG", quality=int(q))
        raw = b.getvalue()
    return raw


def see(c, path, q):
    raw = encode(path, q)
    blk = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(raw).decode()}}
    system = cp.SYSTEM + cp.ED_RULE
    schema = {**cp.SEE_SCHEMA, "properties": {**cp.SEE_SCHEMA["properties"], "ed": cp._S},
              "required": [*cp.SEE_SCHEMA["required"], "ed"]}
    t0 = time.monotonic()
    r = c.call(system, [blk, {"type": "text", "text": "보이는 것을 적어라."}], schema, max_tokens=500, kind="see")
    wall = time.monotonic() - t0
    plates = [(p.get("n"), p.get("m")) for p in (r.data or {}).get("plates", [])]
    return {"file": Path(path).name, "q": q, "kb": round(len(raw) / 1000), "sec": round(r.seconds or wall, 3),
            "attempts": r.attempts, "in": r.input_tokens, "out": r.output_tokens, "cr": round(r.credit, 4),
            "plates": plates, "err": r.error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames", default="69,80,81", help="사진 번호 3개 (쉼표)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--credit-limit", type=float, default=4.0)
    a = ap.parse_args()
    fr = [Path(a.session) / "frames" / f"{int(i):05d}.jpg" for i in a.frames.split(",")]
    plan = [("size", q, 0, [f]) for q in ("orig", "q70", "q80") for f in fr] + \
           [("conc", "80", 1, fr[:2]), ("conc", "80", 1, fr[1:3]), ("conc", "80", 2, fr)]
    n = sum(len(x[3]) for x in plan)
    print(f"호출 {n}번, 예상 크레딧 약 {n * PER_CALL:.1f}")
    for f in fr:
        print(f.name, {q: f"{len(encode(f, q)) // 1000}KB" for q in ("orig", "70", "80")})
    if a.dry_run:
        return
    c = GatewayClient(timeout=30.0, ledger=Ledger(a.credit_limit))
    rows = []
    for kind, q, others, files in plan:
        q = q if q in ("orig",) else q.lstrip("q")
        if others == 0:
            r = see(c, files[0], q)
            rows.append({"cond": f"용량 {'orig' if q == 'orig' else 'q' + q}", "others": 0, **r})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        else:
            with ThreadPoolExecutor(len(files)) as ex:
                for r in [f.result() for f in [ex.submit(see, c, f, q) for f in files]]:
                    rows.append({"cond": f"동시 {others}", "others": others, **r})
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summ = {}
    for r in rows:
        summ.setdefault(r["cond"], []).append(r)
    print("\n| 조건 | 용량 중앙값 | 호출 시간 중앙값 | 각 호출 | 읽은 문패 |\n|---|---|---|---|---|")
    for k, v in summ.items():
        print(f"| {k} | {statistics.median(x['kb'] for x in v)}KB | {statistics.median(x['sec'] for x in v):.2f}s | "
              f"{', '.join(f'{x['sec']:.2f}' for x in v)} | {'; '.join(str(x['plates']) for x in v)} |")
    print(f"크레딧 {c.ledger.total:.3f} ({c.ledger.calls}번)")
    out = ROOT / f"outputs/probe/latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "credit": c.ledger.total}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
