"""Gemini API를 Ollama /api/chat 형식으로 중계하는 작은 서버 (Mac에서 실행).

키는 Mac의 ~/.config/gemini_key 에만 둔다 (학교 서버는 팀원 모두 root라 서버에 두지 않음).
서버 스크립트는 SSH 역방향 터널로 이 중계 서버를 부른다:
  Mac:  python3 src/kyj/gemini_proxy.py --model gemini-3.5-flash          (127.0.0.1:11438)
        ssh -S /tmp/kyj-cnu.sock -O forward -R 11438:127.0.0.1:11438 cnu
  서버: video_reason.py --url http://127.0.0.1:11438 --model gemini-3.5-flash

형식 강제: Ollama format(JSON schema) → Gemini responseSchema (enum 유지, 지원 안 하는 키는 뺌)
무료 등급 한도(분당 요청 수)에 걸리면(429) 기다렸다가 다시 보낸다.
모델 목록: python3 src/kyj/gemini_proxy.py --list
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

API = "https://generativelanguage.googleapis.com/v1beta"
KEY = None
LOCK = Lock()
MIN_GAP = 0.0
THINK = None
_last = [0.0]


def key():
    return (Path.home() / ".config" / "gemini_key").read_text().strip()


def call(method, path, body=None, timeout=300):
    req = urllib.request.Request(f"{API}/{path}", method=method, headers={"Content-Type": "application/json",
                                                                         "x-goog-api-key": KEY},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def to_gemini_schema(s):
    out = {}
    t = s.get("type")
    if isinstance(t, list):  # ["array", "null"] 같은 것
        nn = [x for x in t if x != "null"]
        t = nn[0] if nn else "string"
        out["nullable"] = True
    if t:
        out["type"] = t.upper()
    if "enum" in s:
        out["enum"] = [str(v) for v in s["enum"]]
        out["type"] = "STRING"
    if "properties" in s:
        out["properties"] = {k: to_gemini_schema(v) for k, v in s["properties"].items()}
        out["required"] = s.get("required", list(s["properties"]))
        out["propertyOrdering"] = list(s["properties"])
    if "items" in s:
        out["items"] = to_gemini_schema(s["items"])
    for k in ("minItems", "maxItems", "description"):
        if k in s:
            out[k] = s[k]
    return out


def chat(req, model_default):
    msg = req["messages"][-1]
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in msg.get("images", [])]
    parts.append({"text": msg["content"]})
    opts = req.get("options") or {}
    gen = {"temperature": opts.get("temperature", 0)}
    if THINK:
        gen["thinkingConfig"] = {"thinkingLevel": THINK}
    gen["mediaResolution"] = "MEDIA_RESOLUTION_MEDIUM"  # 기본(높음)은 무료 등급에서 사진 1장 20~35초까지 걸림, 중간은 문패도 읽음
    if req.get("format"):
        gen["responseMimeType"] = "application/json"
        if isinstance(req["format"], dict):
            gen["responseSchema"] = to_gemini_schema(req["format"])
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
    model = req.get("model") or model_default
    t0 = time.perf_counter()
    for attempt in range(8):
        with LOCK:  # 무료 등급: 요청 사이 간격 유지
            wait = MIN_GAP - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            res = call("POST", f"models/{model}:generateContent", body)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 7:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
    cand = (res.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
    use = res.get("usageMetadata", {})
    return {"model": model, "message": {"role": "assistant", "content": text, "thinking": ""},
            "done": True, "finish_reason": cand.get("finishReason"),
            "prompt_eval_count": use.get("promptTokenCount", 0), "eval_count": use.get("candidatesTokenCount", 0),
            "total_duration": int((time.perf_counter() - t0) * 1e9)}


def main():
    global KEY, MIN_GAP, THINK
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11438)
    ap.add_argument("--model", default="gemini-3.5-flash", help="요청에 model이 없을 때")
    ap.add_argument("--rpm", type=float, default=8, help="분당 최대 요청 수 (무료 등급 한도보다 낮게)")
    ap.add_argument("--think", default="low", help="생각 수준 (low/medium/high, 빈 값이면 모델 기본)")
    ap.add_argument("--list", action="store_true", help="쓸 수 있는 모델 목록만 출력")
    a = ap.parse_args()
    KEY = key()
    if a.list:
        for m in call("GET", "models?pageSize=200")["models"]:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/", 1)[1], "|", m.get("displayName", ""))
        return
    MIN_GAP = 60.0 / a.rpm
    THINK = a.think or None

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                res, code = chat(body, a.model), 200
            except Exception as e:  # noqa: BLE001
                res, code = {"error": f"{type(e).__name__}: {e}"}, 500
            data = json.dumps(res, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"READY 127.0.0.1:{a.port} model={a.model} rpm={a.rpm}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
