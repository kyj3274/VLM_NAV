"""실시간 길안내 웹 (시연용). 브라우저가 카메라(또는 영상 파일) 화면을 1초마다 보내면
video_tracker.Navigator(YOLOE+OCR+규칙, VLM은 사건 때만)가 그 초의 안내 문장을 돌려준다. 브라우저가 소리로 읽는다.

서버는 127.0.0.1에만 열린다 (학교 서버를 밖에 열지 않음). Mac에서 SSH로 포트를 이어 붙여 쓴다:
  서버: cd /workspace/capstone/src/kyj && env CUDA_VISIBLE_DEVICES=3 .venv/bin/python nav_web.py --port 8765
  Mac : ssh -S /tmp/kyj-cnu.sock -O forward -L 8765:127.0.0.1:8765 cnu   → 브라우저 http://localhost:8765
VLM은 GPU 7의 Ollama(11436, qwen3-vl:4b-instruct). 받은 사진은 outputs/kyj/web_sessions/<세션>/ 에 남긴다.
"""

import argparse
import base64
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import video_tracker as vt

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
S = {"runner": None, "nav": None, "dir": None, "args": None, "password": None}


def start(body):
    target = str(body.get("target", "")).strip().replace("호", "")
    if not (target.isdigit() and len(target) == 3):
        return {"error": "호실 번호 3자리를 넣어 주세요 (예: 314)"}
    sid = time.strftime("%Y%m%d_%H%M%S") + f"_{target}"
    S["dir"] = Path(S["args"].save_dir) / sid
    S["dir"].mkdir(parents=True, exist_ok=True)
    S["nav"] = vt.Navigator(S["runner"], target, body.get("mode") or S["args"].mode)
    S["nav"].new_segment("live_0")
    note({"event": "start", "target": target, "mode": S["nav"].mode, "key": S["args"].key,
          "quality": body.get("quality"), "camera": body.get("camera"), "pick": body.get("pick")})
    return {"session": sid, "target": target}


def note(ev):
    """버튼 기록(시작·일시 정지·다시 시작·엘리베이터·끝내기)을 log.jsonl에 같이 남긴다. web_replay.py가 읽는다."""
    if S["dir"] is None:
        return
    nav = S["nav"]
    ev = {**ev, "wall": time.strftime("%Y-%m-%d %H:%M:%S"), "segment": nav.si if nav else 0, "sec": nav.sec if nav else 0}
    with open(S["dir"] / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def event(body):
    kind = body.get("kind")
    if kind not in ("pause", "resume", "end"):
        return {"error": "알 수 없는 기록"}
    note({"event": kind})
    return {"ok": True}


def segment(_body):
    nav = S["nav"]
    if nav is None:
        return {"error": "먼저 시작하세요"}
    nav.new_segment(f"live_{nav.si + 1}")
    note({"event": "segment"})
    return {"segment": nav.si}


def step(body):
    nav = S["nav"]
    if nav is None:
        return {"error": "먼저 시작하세요"}
    paths = []
    raw = body.get("frames_bin") or [base64.b64decode(b.split(",", 1)[-1]) for b in body.get("frames", [])]
    for k, b in enumerate(raw[:vt.DENSE]):
        p = S["dir"] / f"s{nav.si}_{nav.sec:04d}_{k}.jpg"
        p.write_bytes(b)
        paths.append(p)
    if not paths:
        return {"error": "사진이 없습니다"}
    t0 = time.perf_counter()
    rec = nav.step(paths)
    S["runner"].cache.clear()  # 웹은 같은 사진을 다시 보지 않으므로 메모리만 차지
    with open(S["dir"] / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**rec, "wall": time.strftime("%Y-%m-%d %H:%M:%S"), "segment": nav.si,
                            "sent_frames": len(paths), "client": body.get("client")}, ensure_ascii=False) + "\n")
    (S["dir"] / "memory.json").write_text(json.dumps(nav.mem.dump(), ensure_ascii=False, indent=1), encoding="utf-8")
    st = rec["state"]
    # 다음 초에 4장을 보내 달라고 할지: 1장만 받았는데 애매했거나, 목적지 근처일 때 (4장을 본 직후에는 다시 1장으로)
    more = bool(rec["ambiguous"]) and (len(paths) == 1 or "목적지 근처" in rec["ambiguous"])
    return {"say": rec["say"], "rooms": rec["rooms"], "floor": st.get("floor"), "arrived": st.get("arrived"),
            "wrong_floor": st.get("wrong_floor"), "scene": st.get("scene"), "ambiguous": rec["ambiguous"],
            "vlm": [{"why": v.get("why"), "answer": v.get("answer")} for v in rec["vlm"]],
            "advice": rec.get("advice"), "frames_used": len(rec["used_frames"]), "server_seconds": round(time.perf_counter() - t0, 2),
            "segment": nav.si, "sec": rec["sec"],
            "size": rec["raw"]["size"], "ocr": [[t["text"], t["score"]] for t in rec["raw"]["ocr"]],
            "objects": [[d["name"], d["conf"]] for d in rec["raw"]["objects"]],
            "place": rec["raw"].get("place", []), "more": more}


ROUTES = {"/api/login": lambda _b: {"ok": True}, "/api/start": start, "/api/segment": segment, "/api/step": step, "/api/event": event}


class H(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if fn is None:
            return self._send(404, b"not found", "text/plain")
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if (self.headers.get("Content-Type") or "").startswith("application/octet-stream"):  # 사진을 그대로(글자로 안 바꾸고) 받음
            cuts, at = [], 0
            for n in [int(x) for x in self.headers.get("X-Frames", "").split(",") if x]:
                cuts.append(data[at:at + n])
                at += n
            try:
                client = json.loads(self.headers.get("X-Client") or "{}")
            except ValueError:
                client = {}
            body = {"frames_bin": cuts, "client": client}
        else:
            body = json.loads(data or b"{}")
        if S["password"] and not hmac.compare_digest(self.headers.get("X-Pass", "").encode(), S["password"].encode()):
            return self._send(401, json.dumps({"error": "비밀번호가 틀렸습니다"}, ensure_ascii=False).encode(),
                              "application/json; charset=utf-8")
        with LOCK:  # GPU 작업은 한 번에 하나
            try:
                res = fn(body)
            except Exception as e:  # noqa: BLE001
                res = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(res, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", default="situ", choices=["rule", "c", "lean", "situ"])
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--key", default="sharp", choices=["first", "sharp"],
                    help="1초 대표 장면: sharp = 받은 사진 중 가장 선명한 장 (docs/12), first = 첫 장")
    ap.add_argument("--save-dir", default="/workspace/capstone/outputs/kyj/web_sessions")
    ap.add_argument("--password-file", default="/workspace/capstone/outputs/kyj/nav_web_password",
                    help="이 파일이 있으면 그 내용을 비밀번호로 요구 (명령줄에 비밀번호를 쓰지 않으려고 파일로)")
    a = ap.parse_args()
    S["args"] = a
    pw = Path(a.password_file)
    S["password"] = pw.read_text(encoding="utf-8").strip() if pw.is_file() else None
    S["runner"] = vt.Runner(a)
    from PIL import Image
    warm = Path(a.save_dir) / "_warm.jpg"
    warm.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (720, 1280), "white").save(warm)
    S["runner"].warm(warm)
    print(f"READY http://127.0.0.1:{a.port} mode={a.mode} password={'on' if S['password'] else 'off'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
