"""겹쳐 보내기 시험: 영상을 실제 시간으로 재생하며 휴대폰처럼 보내고, 실제 갱신 간격을 잰다.

- 보내기: interval초(기본 1.5)마다 그동안 찍힌 장(1초 8장) 중 가장 선명한 장 1장을 Claude 인식에 보냄.
  이미 처리 중인 요청이 max_inflight개(기본 3)면 이번 것은 보내지 않음(busy).
- 받기: 응답은 다른 스레드에서 오고, 적용 스레드 하나가 찍은 시각 순서로 Navigator.step(t=찍은 시각)에 넣음.
  이미 더 나중에 찍은 결과를 적용했으면 버림(stale). 확대 재확인은 적용 스레드에서 부름.
- 기록: 찍은 시각, 보낸 시각, 응답 시각, 적용 시각, 버림 이유. 갱신 간격 = 적용 시각끼리의 차이,
  지연 = 적용 시각 − 찍은 시각(사진이 찍힌 뒤 안내 상태가 바뀌기까지).

  .venv/bin/python src/nav/live_sim.py --route "410=1_start_졸프_영상3,2a_after_elevator_졸프_영상1" --skip-first
"""
from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import claude_perceiver as cp
from claude_client import CreditLimit, FakeClient, GatewayClient, Ledger, ROOT
from navigator import Navigator, Runner

FPS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data/video/frames_8fps"))
    ap.add_argument("--route", required=True, help="목적지=구간1,구간2")
    ap.add_argument("--skip-first", action="store_true", help="첫 구간은 건너뛰고 엘리베이터에서 내린 상태로 시작")
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--max-inflight", type=int, default=3)
    ap.add_argument("--credit-limit", type=float, default=8.0)
    ap.add_argument("--name", default="live")
    ap.add_argument("--fake", type=float, default=None, help="API 없이 이 초만큼 기다리는 가짜 응답 (흐름 확인용)")
    ap.add_argument("--seconds", type=float, default=None, help="구간마다 앞에서부터 이 초만")
    a = ap.parse_args()
    target, seqs = a.route.split("=")
    seqs = seqs.split(",")
    if a.fake is not None:
        def fake(kind, content, n):
            time.sleep(a.fake)
            return {"plates": [], "fs": "", "place": "corridor", "door": "none", "el": [], "arrows": []}
        client = FakeClient(fake)
        client.credits_used = lambda: None
    else:
        client = GatewayClient(ledger=Ledger(a.credit_limit))
    before = client.credits_used()
    per = cp.Perceiver(client)
    runner = Runner(per)
    nav = Navigator(runner, target)
    if a.skip_first:  # 첫 구간을 지난 것으로 (엘리베이터에서 내린 상태)
        nav.new_segment(seqs[0])
        nav.seg_n = 1
        seqs = seqs[1:]
    log, applied_wall = [], []
    lock = threading.Lock()
    q: queue.PriorityQueue = queue.PriorityQueue()
    state = {"inflight": 0, "last_applied": -1.0, "stop": False}

    def worker(job):
        try:
            job["sent_w"] = time.monotonic()
            r = per.see(job["frame"])
            job["resp_w"] = time.monotonic()
            job["o"] = {"key": job["frame"], "keys": [job["frame"]], "pick_seconds": job["pick"], **r}
        except CreditLimit as e:
            job["resp_w"], job["o"] = time.monotonic(), None
            job["error"] = str(e)
        finally:
            with lock:
                state["inflight"] -= 1
            q.put((job["t_abs"], id(job), job))

    def applier():
        while not (state["stop"] and q.empty()):
            try:
                _, _, job = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if job["o"] is None:
                job["result"] = "error"
            elif job["t_abs"] <= state["last_applied"]:
                job["result"] = "stale"  # 더 나중에 찍은 결과를 이미 적용함
            else:
                if job["seg"] != nav.seq:
                    nav.clock = job["seg_start"]
                    nav.new_segment(job["seg"])
                rec = nav.step(t=job["t_abs"], observed=job["o"])
                job["applied_w"] = time.monotonic()
                job["result"] = "applied"
                job["say"], job["action"], job["rooms"] = rec["say"], rec["action"], rec["rooms"]
                state["last_applied"] = job["t_abs"]
                applied_wall.append(job["applied_w"])
            log.append(job)

    th = threading.Thread(target=applier, daemon=True)
    th.start()
    pool = ThreadPoolExecutor(max_workers=a.max_inflight)
    seg_offset = 0.0
    w0 = time.monotonic()
    for seq in seqs:
        files = sorted((Path(a.frames) / seq).glob("i*.jpg"))
        dur = len(files) / FPS if a.seconds is None else min(a.seconds, len(files) / FPS)
        t = 0.0
        while t <= dur:
            wait = w0 + seg_offset + t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            lo, hi = max(0, int((t - a.interval) * FPS) + 1), min(len(files), int(t * FPS) + 1)
            job = {"seg": seq, "t": round(t, 2), "t_abs": round(seg_offset + t, 2), "seg_start": seg_offset,
                   "cap_w": time.monotonic()}
            with lock:
                busy = state["inflight"] >= a.max_inflight
                if not busy:
                    state["inflight"] += 1
            if busy:
                job["result"] = "busy"
                log.append(job)
            else:
                tb = time.perf_counter()
                job["frame"] = str(Runner.sharpest(files[lo:hi]))
                job["pick"] = round(time.perf_counter() - tb, 3)
                pool.submit(worker, job)
            t += a.interval
        seg_offset += dur
    pool.shutdown(wait=True)
    state["stop"] = True
    th.join()
    after = client.credits_used()

    ok = [j for j in log if j.get("result") == "applied"]
    gaps = [b - a_ for a_, b in zip(applied_wall, applied_wall[1:])]
    lat = [j["applied_w"] - j["cap_w"] for j in ok]
    call = [j["resp_w"] - j["sent_w"] for j in log if "resp_w" in j and "sent_w" in j]
    summary = {
        "interval_s": a.interval, "max_inflight": a.max_inflight, "sent": sum(1 for j in log if "frame" in j),
        "busy": sum(1 for j in log if j.get("result") == "busy"), "stale": sum(1 for j in log if j.get("result") == "stale"),
        "applied": len(ok), "errors": sum(1 for j in log if j.get("result") == "error"),
        "update_interval_median": round(statistics.median(gaps), 2), "update_interval_mean": round(statistics.mean(gaps), 2),
        "update_interval_max": round(max(gaps), 2),
        "lag_median": round(statistics.median(lat), 2), "lag_max": round(max(lat), 2),
        "call_median": round(statistics.median(call), 2), "call_max": round(max(call), 2),
        "credit": round(client.ledger.total, 3), "balance_diff": round(after - before, 3) if before and after else None,
    }
    out = ROOT / f"outputs/live/live_{target}_{a.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for j in sorted(log, key=lambda j: j["t_abs"]):
            j = {k: v for k, v in j.items() if k != "o"}
            for k in ("cap_w", "sent_w", "resp_w", "applied_w"):
                if k in j:
                    j[k] = round(j[k] - w0, 3)
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    with open(out.with_name(out.stem + "_nav.jsonl"), "w", encoding="utf-8") as f:  # 적용한 걸음의 Navigator 기록(재확인·확정 근거)
        for r in nav.records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(nav.mem.dump(), open(out.with_name(out.stem + "_memory.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
