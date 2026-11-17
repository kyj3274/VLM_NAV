"""게이트웨이 확인용 작은 호출.
  schema : 스키마마다 max_tokens=1 요청 (400 'grammar too large' 등이 나는지만 봄)
  measure: 프레임 몇 장을 원본·긴 변 768로 인식 + 문패 재확인 1회 → 시간·토큰·크레딧

  .venv/bin/python src/tools/claude_probe.py schema
  .venv/bin/python src/tools/claude_probe.py measure
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))
import claude_perceiver as cp  # noqa: E402
from claude_client import GatewayClient, ROOT  # noqa: E402

F = ROOT / "data/video/frames_8fps"
FRAMES = [F / "2a_after_elevator_졸프_영상1/i0028.jpg", F / "2b_after_elevator_졸프_영상2/i0222.jpg"]
OUT = ROOT / "outputs/probe"


def schema(c):
    for name, sch in [("see", cp.SEE_SCHEMA), ("see+dest", cp.SEE_DEST_SCHEMA), ("recheck", cp.RECHECK_SCHEMA)]:
        r = c.call("schema check", [{"type": "text", "text": "ok"}], sch, max_tokens=1, kind="schema")
        print(f"{name:9s} stop={r.stop_reason} err={r.error} in={r.input_tokens} out={r.output_tokens} "
              f"{r.seconds:.2f}s {r.credit:.4f}cr", flush=True)


def measure(c):
    rows = []
    for side in (None, 768):
        p = cp.Perceiver(c, send_side=side)
        for f in FRAMES:
            s = p.see(f)
            log, obs = s["call"], s["obs"]
            rows.append({"side": side or 1280, "frame": f"{f.parent.name[:2]}/{f.name}", **{k: log[k] for k in (
                "seconds", "input_tokens", "output_tokens", "cache_read", "cache_write", "credit", "error")}, "obs": obs})
            print(f"[{side or 1280}] {f.parent.name[:2]}/{f.name} {log['seconds']:.2f}s in={log['input_tokens']} "
                  f"cr={log['cache_read']} cw={log['cache_write']} out={log['output_tokens']} {log['credit']:.4f}cr | "
                  f"{json.dumps(obs, ensure_ascii=False)}", flush=True)
            if side is None and obs and obs["plates"]:
                rc = p.recheck(f, obs["plates"][0])
                if rc:
                    lg = rc["call"]
                    rows.append({"side": "recheck", "frame": f.name, "seconds": lg["seconds"], "credit": lg["credit"],
                                 "input_tokens": lg["input_tokens"], "output_tokens": lg["output_tokens"], "n": rc["n"],
                                 "match": rc["match"]})
                    print(f"   재확인 {obs['plates'][0]['n']} → {rc['n']} match={rc['match']} {lg['seconds']:.2f}s "
                          f"in={lg['input_tokens']} out={lg['output_tokens']} {lg['credit']:.4f}cr", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measure.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for side in (1280, 768):
        s = [r["seconds"] for r in rows if r["side"] == side]
        print(f"긴 변 {side}: 시간 {', '.join(f'{x:.2f}' for x in s)} (중앙값 {statistics.median(s):.2f}s)")


if __name__ == "__main__":
    c = GatewayClient()
    before = c.credits_used()
    {"schema": schema, "measure": measure}[sys.argv[1]](c)
    after = c.credits_used()
    print(f"토큰 계산 합계 {c.ledger.total:.4f}cr, 잔액 차이 {after - before if before is not None and after is not None else '?'}")
