"""웹 세션 기록을 지금 규칙(navigator)에 다시 넣어 본다 (Claude 호출 없음, 크레딧 0).

- 인식: 기록(log.jsonl)에 저장된 그 프레임의 obs를 그대로 쓴다.
- 확대 재확인: 기록에 같은 프레임·같은 문패의 재확인 결과가 있으면 그것을, 없으면 '못 읽음'(None)으로 둔다
  (새 규칙이 기록에 없는 재확인을 부르면 missing_recheck로 센다. 그 시간은 기록의 재확인 시간 중앙값으로 잡는다).
- 시간: 서버 동작을 그대로 흉내 낸다. 사진마다 (찍은 시각 + 올리는 시간 + 인식 시간)에 인식이 끝나고,
  상태에 넣는 단계(확대 재확인 포함)는 한 번에 하나씩(잠금). 더 나중에 보낸 결과를 이미 넣었으면 버림(stale).
  올리는 시간·인식 시간은 기록값(lag_s − server_s, see_s), 재확인 시간은 기록의 (latency_seconds − see_s).
- 문 앞 빠른 확인(confirm_now): 응답을 받은 뒤 휴대폰이 곧바로 찍어 보내는 것을 흉내 낸다.
  응답을 받은 뒤 0.4·0.8초에 2장(QUICK_GAPS) + 올리기(기록 중앙값) + 짧은 확인 호출(--quick-call초, 실측 전이라 기본은 이 세션 인식 시간 중앙값).
  사진 내용은 --quick-photo: next(방아쇠 프레임 바로 다음 기록 프레임) / nearest(찍었을 시각에 가장 가까운 기록 프레임).
  짧은 확인은 가장 가까운 문패 번호만 읽으므로 그 프레임 obs의 문패 목록(가까운 순 2개)을 쓴다.

  .venv/bin/python src/eval/replay_web_session.py outputs/web_sessions/<세션 폴더> --name after
"""
from __future__ import annotations

import argparse
import heapq
import json
import statistics
import sys
from pathlib import Path

ROOT = next(d for d in Path(__file__).resolve().parents if (d / "src" / "nav").is_dir())
sys.path.insert(0, str(ROOT / "src" / "nav"))

import claude_perceiver as cp  # noqa: E402
import navigator as nv  # noqa: E402
from claude_client import GatewayClient, Ledger, ReplayClient  # noqa: E402

SHARP_MIN = 320  # 휴대폰 app.js의 sharpMin과 같은 값 (일찍 보내기 기준)
QUICK_GAPS = (0.4, 0.8)  # 빠른 확인 사진 2장을 찍는 때 (응답을 받은 뒤 초, 2026-11-23 사용자 지시 B3)


class PerceiveRunner:
    """--reperceive: 인식·재확인을 새 프롬프트(판 3)로 다시 (캐시 outputs/cache/claude_v3.jsonl). 시간은 그 호출의 실제 초."""

    def __init__(self, live, limit):
        client = GatewayClient(timeout=30.0, ledger=Ledger(limit)) if live else ReplayClient()
        self.client = client
        self.p = cp.Perceiver(client, cache_path=ROOT / "outputs/cache/claude_v3.jsonl", version=3)
        self.asked = []

    def see(self, frame):
        return self.p.see(str(frame))

    def recheck(self, frame, plate):
        rc = self.p.recheck(frame, plate)
        self.asked.append({"frame": Path(frame).name, "asked": plate["n"] or plate["name"], "missing": False,
                           "sec": (rc or {}).get("call", {}).get("seconds") or 0.0})
        return rc


class LogRunner:
    """기록의 재확인 결과를 돌려주는 Runner 대신. (frame, 물어본 문패) → 읽은 번호"""

    def __init__(self, rechecks, recheck_default_s):
        self.rechecks, self.default_s = rechecks, recheck_default_s
        self.asked = []

    def recheck(self, frame, plate):
        asked = plate["n"] or plate["name"]
        hit = self.rechecks.get((Path(frame).name, asked))
        if hit is None:
            self.asked.append({"frame": Path(frame).name, "asked": asked, "missing": True})
            n, sec = None, self.default_s
        else:
            n, sec = hit["read"], hit["sec"]
            self.asked.append({"frame": Path(frame).name, "asked": asked, "missing": False})
        return {"n": n, "name": "", "match": (n == plate["n"]) if plate["n"] and n else None, "cached": True,
                "call": {"seconds": sec, "credit": 0.0}}


def load(d):
    lines = [json.loads(x) for x in (Path(d) / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    start = next(x for x in lines if x.get("event") == "start")
    steps = [x for x in lines if "event" not in x]
    stale = [x for x in lines if x.get("event") == "stale"]
    return start, steps, stale


def sharpness(path):
    """휴대폰 app.js sharpness()와 같은 계산: 가로 160px 흑백의 4방향 라플라시안 분산."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w = 160
    a = np.asarray(im.resize((w, round(im.height * w / im.width)), Image.BILINEAR), dtype=np.float64)
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def run(d, quick_photo="nearest", quick_call=None, verbose=False, recheck_outside=True, reperceive=None, limit=30.0,
        quick_on=True, early_send=False):
    start, steps, stale = load(d)
    d = Path(d)
    # 기록에 남은 프레임: 반영된 것 + 버려진 것(stale, obs가 없어 다시 넣을 수 없음 → 같은 시각에 버린 것으로 둠)
    rechecks, rc_secs = {}, []
    for r in steps:
        if r.get("recheck"):
            sec = max(0.0, r["latency_seconds"] - r["see_s"])
            rechecks[(r["frame"], r["recheck"]["asked"])] = {"read": r["recheck"]["read"], "sec": sec}
            rc_secs.append(sec)
    rc_default = statistics.median(rc_secs) if rc_secs else 1.5
    ups = [r["lag_s"] - r["server_s"] for r in steps if r.get("lag_s") is not None]
    up_med = statistics.median(ups)
    see_med = statistics.median(r["see_s"] for r in steps)
    qcall = quick_call if quick_call is not None else see_med
    runner = LogRunner(rechecks, rc_default) if not reperceive else PerceiveRunner(reperceive == "live", limit)
    if reperceive:  # 기록의 obs 대신 새 프롬프트 인식 (시간은 새 호출 시간)
        new = []
        for r in steps:
            o = runner.see(d / "frames" / r["frame"])
            if o["obs"] is None:
                raise SystemExit(f"인식 실패 {r['frame']}: {o['call'].get('error')}")
            new.append({**r, "obs": o["obs"], "see_s": o["call"].get("seconds") or r["see_s"]})
        steps = new
    nav = nv.Navigator(runner, start["target"])
    nav.announce = False
    nav.quick_max = nav.quick_max if quick_on else 0
    nav.new_segment("live_0")
    nav.clock = 0.0

    # 사건 큐: (인식 끝난 시각, 받은 순서 seq, 종류, 자료)
    heap, seq = [], 0
    for r in steps:
        recv = r["t"] + (r["lag_s"] - r["server_s"])
        cl = r.get("client") or {}
        if early_send and cl.get("pick_end") is not None and sharpness(d / "frames" / r["frame"]) >= SHARP_MIN:
            # 일찍 보내기 (2026-11-23, 추정): 선명도 기준 이상인 사진은 구간 끝(pick_end)까지 기다리지 않고 보냄
            recv -= max(0.0, cl["pick_end"] - r["t"])
            r = {**r, "early": True}
        seq_r = r["web_seq"]
        heapq.heappush(heap, (recv + r["see_s"], seq_r, "frame", r))
    for s in stale:  # 원래 버려진 프레임: 인식 결과가 기록에 없음 → 다시 넣지 않음
        pass
    lock_free, last_applied, out, quick_n = 0.0, 0, [], 0
    by_t = sorted(steps, key=lambda r: r["t"])
    while heap:
        done, sq, kind, r = heapq.heappop(heap)
        if sq <= last_applied:
            out.append({"kind": kind, "seq": sq, "stale": True, "t": r["t"]})
            continue
        begin = max(done, lock_free)
        b_rec = len(runner.asked)
        frame = d / "frames" / r["frame"]
        o = {"key": str(frame), "keys": [str(frame)], "pick_seconds": 0.0, "obs": r["obs"], "cached": True,
             "call": {"seconds": r["see_s"], "credit": 0.0}}
        before = b_rec
        t_cap = max(r["t"], float(nav.clock) + 0.01) if nav.records else r["t"]
        rec = nav.step(t=round(t_cap, 2), observed=o, quick=(kind == "quick")) if kind == "quick" else \
            nav.step(t=round(t_cap, 2), observed=o, heading=r.get("heading"))
        asked = runner.asked[before:]
        dur = sum(a["sec"] if "sec" in a else (rechecks.get((a["frame"], a["asked"])) or {"sec": rc_default})["sec"]
                  for a in asked)
        if recheck_outside:  # 2026-11-19부터 서버 동작: 확대 재확인을 잠금 밖(사진마다 따로)에서 부름 → 그 사진만 늦어짐
            begin = max(done + dur, lock_free)
            lock_free = begin
        else:
            lock_free = begin + dur
        last_applied = sq
        row = {"kind": kind, "seq": sq, "t": r["t"], "frame": r["frame"], "resp": round(lock_free, 2),
               "say": rec["say"], "action": rec["action"], "arrived": rec["state"]["arrived"],
               "floor": rec["state"]["floor"], "recheck": [a["asked"] + ("(기록 없음)" if a["missing"] else "") for a in asked],
               "confirm_now": bool(rec.get("confirm_now")), "logged_resp": round(r["t"] + r["lag_s"], 2) if kind == "frame" else None}
        out.append(row)
        if rec.get("confirm_now") and hasattr(nv.Navigator, "quick_obs") and quick_on:
            # 빠른 확인 (2026-11-23): 응답이 휴대폰에 닿은 뒤 QUICK_GAPS초에 2장. 사진 내용은 그 시각에 가장 가까운 기록 사진
            # (방아쇠 사진·서로 같은 사진 제외). 기록 사진 간격이 약 1.5초라 '근처' 사진이다.
            used = {r["frame"]}
            for k, gap in enumerate(QUICK_GAPS):
                cap = lock_free + gap
                cand = [x for x in by_t if x["t"] > r["t"] and x["frame"] not in used]
                if quick_photo == "stop":  # 사용자가 "잠시 멈춰 주세요"를 듣고 멈춤: 응답이 닿기 전 마지막 기록 사진의 장면을 새로 찍은 것으로
                    cand = [max([x for x in by_t if x["t"] <= lock_free] or [r], key=lambda x: x["t"])]
                if not cand:
                    break
                aim = r["t"] + gap if quick_photo == "trigger" else cap  # trigger: 처음 본 사진을 찍은 뒤 0.4·0.8초 근처 사진 (상한 추정)
                src = min(cand, key=lambda x: abs(x["t"] - aim)) if quick_photo in ("nearest", "trigger") else cand[0]
                if quick_photo != "stop":
                    used.add(src["frame"])
                quick_n += 1
                q = dict(src)
                q["obs"] = nv.Navigator.quick_obs(src["obs"]["plates"][:2], rec["obs"])
                q["t"], q["frame"] = round(cap, 2), src["frame"]
                q["see_s"], q["latency_seconds"] = qcall, qcall
                q["photo_from"] = src["frame"]
                recv = cap + up_med
                # 받은 순서: 이 시각 전에 받은 프레임보다 뒤, 뒤에 받은 프레임보다 앞
                seq = max([x["web_seq"] for x in steps if x["t"] + (x["lag_s"] - x["server_s"]) <= recv] + [0]) + 0.1 * (k + 1)
                heapq.heappush(heap, (recv + qcall, seq, "quick", q))
    if reperceive:
        print(f"새 인식 크레딧 {runner.client.ledger.total:.3f} ({runner.client.ledger.calls}번)")
    return {"start": start, "rows": out, "steps": steps, "missing_recheck": sum(a["missing"] for a in runner.asked),
            "rechecks": len(runner.asked), "up_med": up_med, "see_med": see_med, "quick_call": qcall,
            "events": nav.mem.events, "nav": nav}


def target_wait(res, target, steps):
    """목적지 문패가 2m 안에서 처음 보인 사진의 찍은 시각 → 첫 안내 / 도착 안내가 휴대폰에 닿은 시각."""
    rows = [x for x in res["rows"] if not x.get("stale")]
    steps = {r["frame"]: r for r in steps}
    first = None
    for x in sorted(rows, key=lambda x: x["t"]):
        obs = steps[x["frame"]]["obs"] if x["kind"] == "frame" else None
        if obs is None:
            continue
        names = [p["n"] for p in obs["plates"]] + [r["recheck"]["read"] for r in [steps[x["frame"]]] if r.get("recheck")]
        near = [p for p in obs["plates"] + obs["unread"] if p.get("m") is not None and p["m"] <= nv.ARRIVE_M]
        if target in names and near:
            first = x
            break
    arr = next((x for x in sorted(rows, key=lambda x: x["resp"]) if x["arrived"]), None)
    spoke = next((x for x in sorted(rows, key=lambda x: x["resp"]) if first and x["resp"] >= first["t"] and x["say"]
                  and x["t"] >= first["t"]), None)
    return {"first_t": first["t"] if first else None, "first_frame": first["frame"] if first else None,
            "first_say_at": spoke["resp"] if spoke else None, "first_say": spoke["say"] if spoke else None,
            "arrived_at": arr["resp"] if arr else None, "arrived_by": (arr["kind"], arr["frame"]) if arr else None,
            "wait": round(arr["resp"] - first["t"], 2) if arr and first else None}


def silence(res):
    """휴대폰에 안내가 닿은 시각들 사이의 가장 긴 빈 구간 (첫 안내 ~ 도착 또는 끝)."""
    rows = sorted([x for x in res["rows"] if not x.get("stale")], key=lambda x: x["resp"])
    said = [x["resp"] for x in rows if x["say"]]
    end = next((x["resp"] for x in rows if x["arrived"]), rows[-1]["resp"])
    said = [t for t in said if t <= end] + [end]
    gaps = [(round(b - a, 2), a, b) for a, b in zip(said, said[1:])]
    return max(gaps) if gaps else None


def first_elevator(res):
    rows = sorted([x for x in res["rows"] if not x.get("stale")], key=lambda x: x["resp"])
    hit = next((x for x in rows if any("엘리베이터가" in s for s in x["say"])), None)
    return {"frame": hit["frame"], "t": hit["t"], "resp": hit["resp"], "say": hit["say"]} if hit else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--name", default="now")
    ap.add_argument("--quick-photo", default="nearest", choices=["next", "nearest", "trigger", "stop"])
    ap.add_argument("--no-quick", action="store_true", help="빠른 확인 없이")
    ap.add_argument("--early-send", action="store_true", help="선명한 사진은 구간 끝을 기다리지 않고 보낸 것으로 (추정)")
    ap.add_argument("--quick-call", type=float, default=None)
    ap.add_argument("--reperceive", choices=["replay", "live"], default=None,
                    help="인식을 새 프롬프트(판 3)로: replay=캐시만(크레딧 0), live=없는 것은 실제 호출(크레딧 씀)")
    ap.add_argument("--credit-limit", type=float, default=30.0)
    ap.add_argument("--recheck-inside", action="store_true", help="이전 서버 동작(확대 재확인을 잠금 안에서)으로 계산")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "eval"))
    a = ap.parse_args()
    res = run(a.session, a.quick_photo, a.quick_call, recheck_outside=not a.recheck_inside, reperceive=a.reperceive,
              limit=a.credit_limit, quick_on=not a.no_quick, early_send=a.early_send)
    tw = target_wait(res, res["start"]["target"], res["steps"])
    print(f"올리기 중앙값 {res['up_med']:.2f}s, 인식 중앙값 {res['see_med']:.2f}s, 짧은 확인 {res['quick_call']:.2f}s(추정), "
          f"재확인 {res['rechecks']}번 (기록 없음 {res['missing_recheck']})")
    for x in sorted(res["rows"], key=lambda x: (x.get("resp") or 0, x["seq"])):
        if x.get("stale"):
            print(f"  [버림] {x['kind']} seq {x['seq']} t={x['t']}")
            continue
        if x["t"] < 55 and not x["say"]:
            continue
        lr = f" (기록 {x['logged_resp']})" if x["logged_resp"] is not None else ""
        print(f"  {x['kind']:5} {x['frame']} 찍음 {x['t']:6.2f} → 응답 {x['resp']:6.2f}{lr} 층 {x['floor']} {x['action']}"
              f" 재확인 {x['recheck']}" + (" ★빠른확인요청" if x["confirm_now"] else "") + f" | {' / '.join(x['say'])}")
    print("대기:", json.dumps(tw, ensure_ascii=False))
    sil, fe = silence(res), first_elevator(res)
    print("무안내 최장:", sil, " 엘리베이터 첫 안내:", json.dumps(fe, ensure_ascii=False))
    tw.update(silence=sil, first_elevator=fe)
    out = Path(a.out_dir) / f"web_replay_{Path(a.session).name}_{a.name}.json"
    out.write_text(json.dumps({"wait": tw, "rows": res["rows"], "events": res["events"],
                               "missing_recheck": res["missing_recheck"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
