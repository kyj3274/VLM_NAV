"""실시간 길안내 웹 (맥북에서 실행, FastAPI). 휴대폰(또는 맥 브라우저)이 카메라·영상 파일 화면을 1.5초마다 보내면
Claude 인식(claude_perceiver) + 규칙(navigator.Navigator)이 안내 문장을 돌려준다. 브라우저가 소리로 읽는다.

겹쳐 보내기: 브라우저는 응답을 기다리지 않고 1.5초마다 보낸다. 서버는 한 세션에 동시에 3개까지 처리하고
(넘으면 busy), 인식이 끝난 결과를 보낸 순서(seq)대로만 상태에 넣는다. 더 나중에 보낸 결과를 이미 넣었으면
그 결과는 버린다(stale). 확대 재확인은 상태에 넣는 단계에서 부른다 (live_sim.py와 같은 방식).

  NAV_ACCESS_CODE=코드 .venv/bin/python src/nav/web_server.py            # 실제 Claude (크레딧 씀)
  NAV_ACCESS_CODE=코드 .venv/bin/python src/nav/web_server.py --fake     # 가짜 응답 (크레딧 0)
  → 맥 브라우저 http://localhost:8765 , 관찰자 화면 /monitor , 다시보기 기록 /records

- 접속 코드(NAV_ACCESS_CODE)가 맞지 않는 요청은 Claude를 부르지 않는다. API 키는 서버에만 있다.
- 크레딧 한도: 세션 60, 하루 300 (넘으면 음성으로 알리고 멈춤). 받은 사진·기록은 outputs/web_sessions/<세션>/.
- 서버는 127.0.0.1에만 열린다. 휴대폰 접속(HTTPS 터널)은 cloudflared가 필요하다(설치는 사용자 확인 후).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import claude_perceiver as cp  # noqa: E402
from claude_client import ROOT, CreditLimit, GatewayClient, Ledger  # noqa: E402
from dest_parse import parse_destination, parse_here_floor  # noqa: E402
from navigator import Navigator, Runner  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
RETRY = "다시 말씀해 주세요."


@dataclass
class Settings:
    sessions_dir: Path = ROOT / "outputs" / "web_sessions"
    max_inflight: int = 3              # 한 세션에 동시에 처리하는 사진 수
    credit_per_session: float = 60.0
    credit_per_day: float = 300.0
    model: str = "claude-haiku-5-5"
    timeout: float = 15.0
    fake: bool = False
    fake_delay: float = 1.2
    access_code: str = ""
    quick_confirm: bool = False        # 문 앞 빠른 확인(/confirm). 2026-11-19 사용자 지시로 기본 끔


class DailyCredit:
    """하루 누적 크레딧(서버 전체). sessions_dir/daily_credit.json에 날짜별로 저장."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def today(self):
        return float(self._load().get(dt.date.today().isoformat(), 0.0))

    def add(self, credit):
        if not credit:
            return
        with self.lock:
            d = self._load()
            k = dt.date.today().isoformat()
            d[k] = round(float(d.get(k, 0.0)) + credit, 4)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d, indent=1), encoding="utf-8")


@dataclass
class WebSession:
    id: str
    dir: Path
    nav: Navigator
    client: object
    target: str
    info: dict
    started: float = field(default_factory=time.monotonic)
    slock: threading.Lock = field(default_factory=threading.Lock)   # 숫자 세기
    alock: threading.Lock = field(default_factory=threading.Lock)   # 상태에 넣기 (보낸 순서대로)
    seq: int = 0
    inflight: int = 0
    last_applied: int = 0
    counts: dict = field(default_factory=lambda: {"sent": 0, "busy": 0, "stale": 0, "applied": 0, "errors": 0})
    applied_at: list = field(default_factory=list)
    last: dict | None = None
    last_frame: Path | None = None
    last_reply: dict | None = None
    credit_counted: float = 0.0
    stopped: bool = False
    stop_reason: str = ""

    @property
    def credit(self):
        return round(self.client.ledger.total, 4)

    def log(self, obj):
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


class DestBody(BaseModel):
    destination: str
    source: str = "keyboard"            # keyboard | speech_recognition | whisper
    transcript_ms: float | None = None
    camera: dict = {}
    device: dict = {}
    mode: str = "camera"                # camera | file
    files: list[str] = []


class EventBody(BaseModel):
    type: str                           # exited | end | stop | repeat | pause | resume
    detail: dict = {}


def wall():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def detail_of(ws: WebSession) -> dict:
    """'자세히'·관찰자 화면: Claude가 본 것과 규칙 상태."""
    rec = ws.last or {}
    obs = rec.get("obs") or {}
    st = ws.nav.st
    gaps = [b - a for a, b in zip(ws.applied_at, ws.applied_at[1:])]
    plates = [{"n": p["n"], "name": p.get("name", ""), "side": cp.SIDE_KO.get(p["side"], p["side"]), "m": p["m"],
               "on": p.get("on")} for p in obs.get("plates", [])]
    plates += [{"n": "", "name": p.get("name", ""), "side": cp.SIDE_KO.get(p["side"], p["side"]), "m": p["m"],
                "on": p.get("on")} for p in obs.get("unread", [])]
    el = obs.get("elev") or {}
    return {
        "seq": rec.get("web_seq"), "t": rec.get("t"),
        "place": cp.PLACE_KO.get(obs.get("place"), obs.get("place") or "-"),
        "plates": plates, "floor_sign": obs.get("floor_sign", ""),
        "elevator": ({"why": {"button": "호출 버튼", "display": "층 표시창"}.get(el.get("why"), el.get("why")),
                      "side": cp.SIDE_KO.get(el.get("side"), el.get("side")), "m": el.get("m")} if el.get("seen") else None),
        "arrows": obs.get("arrows", []), "door": obs.get("door_ahead", "none"),
        "recheck": rec.get("recheck"), "confirmed": ws.nav.mem.confirmed(),
        "floor": st.get("floor"), "wrong_floor": st.get("wrong_floor"), "action": st.get("act"),
        "arrived": st.get("arrived"), "target_pending": bool(rec.get("target_pending")),
        "events": [f"{e['t'] if isinstance(e['t'], int) else round(e['t'], 1)}초 {e['text']}" for e in ws.nav.mem.events[-10:]],
        "counts": dict(ws.counts), "inflight": ws.inflight,
        "update_interval_median": round(statistics.median(gaps), 2) if gaps else None,
        "latency_s": rec.get("latency_seconds"), "lag_s": rec.get("lag_s"),
        "credit": ws.credit, "segment": ws.nav.si,
    }


def create_app(cfg: Settings | None = None, client_factory=None, transcriber=None) -> FastAPI:
    """client_factory(target, limit) → Claude 클라이언트 (테스트는 가짜를 넣음)."""
    cfg = cfg or Settings()
    code = cfg.access_code or os.environ.get("NAV_ACCESS_CODE", "")
    if not code:
        raise RuntimeError("접속 코드가 없습니다. 환경변수 NAV_ACCESS_CODE를 정해 주세요.")
    if client_factory is None:
        if cfg.fake:
            from fake_scene import fake_client
            client_factory = lambda target, limit: fake_client(target, cfg.fake_delay, limit=limit)  # noqa: E731
        else:
            client_factory = lambda target, limit: GatewayClient(cfg.model, timeout=cfg.timeout,  # noqa: E731
                                                                 ledger=Ledger(limit))
    sessions: dict[str, WebSession] = {}
    daily = DailyCredit(cfg.sessions_dir / "daily_credit.json")
    stt = {"t": transcriber}
    app = FastAPI(title="vlm_nav_claude web")
    app.state.sessions, app.state.cfg, app.state.daily = sessions, cfg, daily

    def check_code(x_access_code: str | None = Header(None), code_q: str | None = Query(None, alias="code")):
        if not secrets.compare_digest((x_access_code or code_q or "").encode(), code.encode()):
            raise HTTPException(status_code=401, detail="접속 코드가 맞지 않습니다.")

    def get_ws(sid: str) -> WebSession:
        ws = sessions.get(sid)
        if not ws:
            raise HTTPException(status_code=404, detail="세션이 없습니다.")
        return ws

    def sync_credit(ws: WebSession):
        d = ws.client.ledger.total - ws.credit_counted
        if d > 0:
            ws.credit_counted += d
            daily.add(d)

    def limit_reason(ws: WebSession):
        if ws.client.ledger.total >= cfg.credit_per_session:
            return f"세션 크레딧 한도({cfg.credit_per_session:g})"
        if daily.today() >= cfg.credit_per_day:
            return f"하루 크레딧 한도({cfg.credit_per_day:g})"
        return None

    def stop(ws: WebSession, reason: str, kind: str = "limit"):
        if not ws.stopped:
            ws.stopped, ws.stop_reason = True, reason
            ws.log({"event": kind, "reason": reason, "wall": wall(), "t": ws.nav.clock})
        speech = "크레딧 한도에 도달해 안내를 멈춥니다." if kind == "limit" else "안내를 멈췄습니다."
        return {"status": "limit" if kind == "limit" else "stopped", "say": [speech], "reason": reason}

    # ------------------------------------------------------------ 목적지
    def start(dest: dict, body: DestBody, info: dict) -> dict:
        sid = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
        d = cfg.sessions_dir / sid
        (d / "frames").mkdir(parents=True, exist_ok=True)
        client = client_factory(dest["target"], cfg.credit_per_session)
        nav = Navigator(Runner(cp.Perceiver(client)), dest["target"], start_floor=dest.get("here"))
        nav.announce = False   # "410호로 안내하겠습니다"를 이미 말함
        nav.quick_max = nav.quick_max if cfg.quick_confirm else 0
        nav.new_segment("live_0")
        nav.clock = 0.0
        ws = WebSession(sid, d, nav, client, dest["target"],
                        {"camera": body.camera, "device": body.device, "mode": body.mode, "files": body.files})
        sessions[sid] = ws
        ws.log({"event": "start", "wall": wall(), "t": 0.0, "target": dest["target"], "destination": dest,
                "model": getattr(client, "model", ""), "fake": cfg.fake, **info, **ws.info})
        here = f" 지금 {dest['here']}층에서 출발합니다." if dest.get("here") else ""
        return {"status": "ok", "session_id": sid, "target": dest["target"], "name": dest["name"],
                "say": [f"{dest['name']}로 안내하겠습니다.{here}"], "floor": dest.get("here")}

    def change(ws: WebSession, dest: dict, info: dict) -> dict:
        """안내 중 목적지 변경. 같이 말한 지금 층("4층에서 105호")이 있으면 그 층도 반영한다."""
        with ws.alock:
            ws.nav.set_target(dest["target"])
            nxt = ws.nav.set_floor(dest["here"]) if dest.get("here") else []
            ws.target = dest["target"]
            ws.stopped, ws.stop_reason = False, ""
            floor, action = ws.nav.st["floor"], ws.nav.st["act"]
        ws.log({"event": "destination", "wall": wall(), "t": ws.nav.clock, "target": dest["target"],
                "here": dest.get("here"), "next": nxt, "action": action, **info})
        return {"status": "ok", "session_id": ws.id, "target": dest["target"], "name": dest["name"],
                "say": [f"{dest['name']}로 안내하겠습니다."] + nxt, "floor": floor, "action": action}

    def here_floor(ws: WebSession, floor: str, info: dict) -> dict:
        """안내 중 "여기 4층이야": 지금 층으로 바로 반영 (2026-11-23 사용자 지시 2). 문패와 같은 근거."""
        with ws.alock:
            say = ws.nav.set_floor(floor)
            action = ws.nav.st["act"]
        ws.log({"event": "here_floor", "wall": wall(), "t": ws.nav.clock, "floor": floor, "say": say, **info})
        return {"status": "ok", "kind": "floor", "session_id": ws.id, "target": ws.target, "name": f"{ws.target}호",
                "say": say, "floor": floor, "action": action}

    def change_or_floor(ws: WebSession, text: str, info: dict) -> dict:
        dest = parse_destination(text)
        if dest:
            return change(ws, dest, info)
        floor = parse_here_floor(text, alone=True)
        return here_floor(ws, floor, info) if floor else retry(info, ws)

    def retry(info: dict, ws: WebSession | None = None) -> dict:
        rec = {"event": "destination_retry", "wall": wall(), **info}
        if ws:
            ws.log(rec)
        else:
            cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
            with open(cfg.sessions_dir / "_destination_retries.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"status": "retry", "say": [RETRY], "heard": info.get("heard", "")}

    @app.post("/api/session", dependencies=[Depends(check_code)])
    def new_session(body: DestBody):
        if daily.today() >= cfg.credit_per_day:
            return {"status": "limit", "say": ["오늘 크레딧 한도에 도달했습니다."], "reason": "하루 크레딧 한도"}
        info = {"source": body.source, "heard": body.destination, "transcript_ms": body.transcript_ms}
        dest = parse_destination(body.destination)
        return start(dest, body, info) if dest else retry(info)

    @app.post("/api/session/{sid}/destination", dependencies=[Depends(check_code)])
    def change_destination(sid: str, body: DestBody):
        ws = get_ws(sid)
        info = {"source": body.source, "heard": body.destination, "transcript_ms": body.transcript_ms}
        return change_or_floor(ws, body.destination, info)

    def transcribe(audio: UploadFile, where: Path) -> tuple[str, dict]:
        data = audio.file.read()
        ext = {"audio/mp4": ".m4a", "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav",
               "audio/aac": ".aac"}.get((audio.content_type or "").split(";")[0], ".bin")
        where.mkdir(parents=True, exist_ok=True)
        p = where / f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}{ext}"
        p.write_bytes(data)
        if stt["t"] is None:
            from stt import Transcriber
            stt["t"] = Transcriber("small")
        tr = stt["t"].transcribe(p)
        return tr.text, {"source": "whisper", "heard": tr.text, "stt": tr.to_dict(), "audio": p.name,
                         "reject": tr.reject}

    @app.post("/api/voice", dependencies=[Depends(check_code)])
    def voice_new(audio: UploadFile = File(...), meta: str = Form("{}")):
        """녹음 → 맥 로컬 Whisper → 목적지 (브라우저 음성 인식이 없을 때). 받아쓰기는 크레딧을 쓰지 않는다."""
        if daily.today() >= cfg.credit_per_day:
            return {"status": "limit", "say": ["오늘 크레딧 한도에 도달했습니다."], "reason": "하루 크레딧 한도"}
        try:
            m = json.loads(meta or "{}")
        except ValueError:
            m = {}
        text, info = transcribe(audio, cfg.sessions_dir / "_voice")
        dest = None if info["reject"] else parse_destination(text)
        if not dest:
            return retry(info)
        body = DestBody(destination=text, source="whisper", camera=m.get("camera", {}), device=m.get("device", {}),
                        mode=m.get("mode", "camera"), files=m.get("files", []))
        return {**start(dest, body, info), "heard": text}

    @app.post("/api/session/{sid}/voice", dependencies=[Depends(check_code)])
    def voice_change(sid: str, audio: UploadFile = File(...)):
        ws = get_ws(sid)
        text, info = transcribe(audio, ws.dir / "voice")
        return {**(retry(info, ws) if info["reject"] else change_or_floor(ws, text, info)), "heard": text}

    # ------------------------------------------------------------ 사진 (겹쳐 보내기)
    @app.post("/api/session/{sid}/frame", dependencies=[Depends(check_code)])
    def frame(sid: str, image: UploadFile = File(...), t: float | None = Form(None), heading: float | None = Form(None),
              steps: int | None = Form(None), client: str = Form("{}")):
        return process(get_ws(sid), image, t, heading, steps, client, quick=False)

    @app.post("/api/session/{sid}/confirm", dependencies=[Depends(check_code)])
    def confirm(sid: str, image: UploadFile = File(...), t: float | None = Form(None), heading: float | None = Form(None),
                steps: int | None = Form(None), client: str = Form("{}")):
        """문 앞 빠른 확인 (2026-11-19 사용자 지시 4). 응답에 confirm이 오면 휴대폰이 곧바로 찍어 보낸다.
        가장 가까운 문패 번호만 읽는 짧은 호출 → 같은 순서 규칙(seq)으로 상태에 넣는다. 동시 한도에 하나 더 허용."""
        return process(get_ws(sid), image, t, heading, steps, client, quick=True)

    def process(ws: WebSession, image, t, heading, steps, client, quick):
        recv = time.monotonic()
        if ws.stopped:
            return {"status": "stopped", "say": [], "reason": ws.stop_reason}
        reason = limit_reason(ws)
        if reason:
            return stop(ws, reason)
        with ws.slock:
            if ws.inflight >= cfg.max_inflight + (1 if quick else 0):
                ws.counts["busy"] += 1
                return {"status": "busy", "say": []}
            ws.inflight += 1
            ws.seq += 1
            seq = ws.seq
            ws.counts["sent"] += 1
        try:
            path = ws.dir / "frames" / f"{seq:05d}{'_q' if quick else ''}.jpg"
            path.write_bytes(image.file.read())
            ws.last_frame = path
            t_cap = float(t) if t is not None else round(recv - ws.started, 2)
            t0 = time.monotonic()
            try:
                if quick:
                    q = ws.nav.runner.p.quick(str(path))
                    base = (ws.last or {}).get("obs")
                    r = {"obs": Navigator.quick_obs(q["plates"], base) if q["plates"] is not None else None,
                         "call": q["call"], "cached": q["cached"]}
                else:
                    r = ws.nav.runner.p.see(str(path))
            except CreditLimit as e:
                sync_credit(ws)
                return stop(ws, str(e))
            see_s = time.monotonic() - t0
            o = {"key": str(path), "keys": [str(path)], "pick_seconds": 0.0, **r}
            # 확대 재확인은 상태 잠금 밖에서 (2026-11-19 사용자 지시 1). 고르는 것만 잠금 안에서 짧게, 부르는 것은 밖에서.
            # 앞 사진의 결과가 아직 안 들어간 상태로 고르므로, 그 사이 확정된 번호를 한 번 더 볼 수는 있다(호출 1번 더, 판단은 같음).
            rechecked, rc_s = None, 0.0
            if not quick and r.get("obs") is not None:
                with ws.alock:
                    cand, why = ws.nav.recheck_pick(r["obs"])
                if cand:
                    t1 = time.monotonic()
                    try:
                        rc = ws.nav.runner.recheck(cand.get("frame") or str(path), cand)
                    except CreditLimit as e:
                        sync_credit(ws)
                        return stop(ws, str(e))
                    rc_s = time.monotonic() - t1
                    rechecked = (cand, why, rc)
                else:
                    rechecked = (None, None, None)
            ready = time.monotonic()
            with ws.alock:
                if seq <= ws.last_applied or ws.stopped:  # 더 나중에 보낸 결과를 이미 넣었음
                    ws.counts["stale"] += 1
                    sync_credit(ws)
                    ws.log({"event": "stale", "web_seq": seq, "t": t_cap, "after": ws.last_applied, "credit": r["call"].get("credit"),
                            "quick": quick})
                    return {"status": "stale", "seq": seq, "say": []}
                t_cap = max(t_cap, float(ws.nav.clock) + 0.01) if ws.nav.records else t_cap
                try:
                    rec = ws.nav.step(t=round(t_cap, 2), observed=o, quick=quick, rechecked=rechecked)
                except CreditLimit as e:  # 확대 재확인에서 한도
                    sync_credit(ws)
                    return stop(ws, str(e))
                ws.last_applied = seq
                now = time.monotonic()
                ws.applied_at.append(now)
                ws.counts["applied"] += 1
                ws.counts["errors"] += bool(rec["error"])
                try:
                    cl = json.loads(client or "{}")
                except ValueError:
                    cl = {}
                rec.update(web_seq=seq, quick=quick, wall=wall(), heading=heading, steps=steps, client=cl, frame=path.name,
                           see_s=round(see_s, 3), server_s=round(now - recv, 3), recheck_s=round(rc_s, 3),
                           lock_wait_s=round(now - ready, 3), recv_s=round(recv - ws.started, 3),
                           img_bytes=path.stat().st_size,
                           lag_s=round(now - ws.started - t_cap, 3) if t is not None else None)
                ws.last = rec
                ws.log(rec)
                (ws.dir / "memory.json").write_text(json.dumps(ws.nav.mem.dump(), ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
                sync_credit(ws)
                st = rec["state"]
                reply = {"status": "ok", "seq": seq, "ask_floor": bool(rec.get("ask_floor")),
                         "say": rec["say"], "action": rec["action"], "floor": st.get("floor"),
                         "arrived": st.get("arrived"), "rooms": rec["rooms"], "error": rec["error"],
                         "server_s": rec["server_s"], "see_s": rec["see_s"], "credit": ws.credit,
                         "target": ws.target, "detail": detail_of(ws),
                         "confirm": bool(rec.get("confirm_now")), "quick": quick}
                ws.last_reply = reply
                return reply
        finally:
            with ws.slock:
                ws.inflight -= 1

    @app.post("/api/session/{sid}/event", dependencies=[Depends(check_code)])
    def event(sid: str, body: EventBody):
        ws = get_ws(sid)
        if body.type not in ("exited", "end", "stop", "repeat", "pause", "resume"):
            raise HTTPException(status_code=400, detail="알 수 없는 기록")
        say = []
        with ws.alock:
            if body.type == "exited":  # 엘리베이터에서 내림 (영상 파일 모드는 두 번째 파일부터 자동)
                if not ws.nav.records:  # 사진을 보내기 전에 눌러도 층이 바뀐 것으로 (새 구간)
                    ws.nav.seg_n = 1
                ws.nav.new_segment(f"live_{ws.nav.si + 1}")
                carry = ws.nav.st["floor"]
                say = ["엘리베이터에서 내렸습니다. " + (f"표시창으로는 {carry}층입니다. 문패로 확인하겠습니다."
                                                    if carry else "지금 층을 확인하겠습니다.")]
            elif body.type in ("end", "stop"):
                ws.stopped, ws.stop_reason = True, {"end": "사용자가 끝냄", "stop": "관찰자 비상 정지"}[body.type]
                say = ["안내를 마칩니다."] if body.type == "end" else ["안내를 멈췄습니다."]
        ws.log({"event": body.type, "wall": wall(), "t": ws.nav.clock, "segment": ws.nav.si, **body.detail})
        return {"status": "ok", "say": say, "segment": ws.nav.si}

    # ------------------------------------------------------------ 관찰자·기록
    @app.get("/api/monitor", dependencies=[Depends(check_code)])
    def monitor():
        out = [{"session_id": sid, "target": ws.target, "stopped": ws.stopped, "credit": ws.credit,
                "applied": ws.counts["applied"], "arrived": ws.nav.st["arrived"]}
               for sid, ws in sorted(sessions.items(), key=lambda kv: -kv[1].started)]
        return {"sessions": out, "daily_credit": round(daily.today(), 3), "fake": cfg.fake,
                "limits": {"per_session": cfg.credit_per_session, "per_day": cfg.credit_per_day,
                           "max_inflight": cfg.max_inflight}}

    @app.get("/api/session/{sid}", dependencies=[Depends(check_code)])
    def get_session(sid: str):
        ws = get_ws(sid)
        last = ws.last or {}
        return {"session_id": sid, "target": ws.target, "stopped": ws.stopped, "stop_reason": ws.stop_reason,
                "say": last.get("say", []), "last_say": next((r["say"] for r in reversed(ws.nav.records) if r["say"]), []),
                "detail": detail_of(ws), "info": ws.info, "frame_seq": last.get("web_seq")}

    @app.get("/api/session/{sid}/latest-frame", dependencies=[Depends(check_code)])
    def latest_frame(sid: str):
        ws = get_ws(sid)
        p = ws.dir / "frames" / last_name(ws)
        if not p.is_file():
            raise HTTPException(status_code=404, detail="사진 없음")
        return FileResponse(p, media_type="image/jpeg")

    def last_name(ws):
        return (ws.last or {}).get("frame") or (ws.last_frame.name if ws.last_frame else "none")

    @app.get("/api/records", dependencies=[Depends(check_code)])
    def records():
        out = []
        for d in sorted(cfg.sessions_dir.glob("2*"), reverse=True)[:50]:
            log = d / "log.jsonl"
            if not log.is_file():
                continue
            lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
            st = next((x for x in lines if x.get("event") == "start"), {})
            steps = [x for x in lines if "event" not in x]
            out.append({"session_id": d.name, "target": st.get("target"), "mode": st.get("mode"), "steps": len(steps),
                        "says": sum(len(x["say"]) for x in steps), "arrived": any(x["state"].get("arrived") for x in steps),
                        "credit": round(sum(x.get("credit") or 0 for x in steps), 2), "fake": st.get("fake")})
        return {"records": out}

    @app.get("/api/records/{sid}/replay", dependencies=[Depends(check_code)])
    def replay(sid: str):
        import web_replay
        d = cfg.sessions_dir / sid
        if not (d / "log.jsonl").is_file() or "/" in sid or ".." in sid:
            raise HTTPException(status_code=404, detail="기록 없음")
        return HTMLResponse(web_replay.build(d))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "fake": cfg.fake}

    for page in ("monitor", "records"):
        app.add_api_route(f"/{page}", (lambda p=page: FileResponse(WEB / f"{p}.html", media_type="text/html")),
                          methods=["GET"])
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fake", action="store_true", help="가짜 응답 (Claude를 부르지 않음, 크레딧 0)")
    ap.add_argument("--fake-delay", type=float, default=1.2, help="가짜 응답이 기다리는 초")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--credit-session", type=float, default=60.0)
    ap.add_argument("--credit-day", type=float, default=300.0)
    ap.add_argument("--max-inflight", type=int, default=3)
    ap.add_argument("--quick-confirm", action="store_true", help="문 앞 빠른 확인 켜기 (기본 끔)")
    ap.add_argument("--save-dir", default=str(ROOT / "outputs" / "web_sessions"))
    a = ap.parse_args()
    cfg = Settings(sessions_dir=Path(a.save_dir), max_inflight=a.max_inflight, credit_per_session=a.credit_session,
                   credit_per_day=a.credit_day, model=a.model, fake=a.fake, fake_delay=a.fake_delay,
                   quick_confirm=a.quick_confirm)
    app = create_app(cfg)
    print(f"READY http://{a.host}:{a.port}  ({'가짜 응답' if a.fake else a.model}, 세션 {a.credit_session:g} / "
          f"하루 {a.credit_day:g} 크레딧, 동시 {a.max_inflight}, 빠른 확인 {'켬' if a.quick_confirm else '끔'})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
