"""Qwen3.5-4B를 transformers로 GPU에서 돌리고, Ollama /api/chat과 같은 형식으로 답하는 작은 서버.
(서버 GPU 드라이버가 오래돼 GPU용 Ollama 0.24.0이 qwen3.5를 못 열기 때문)

형식 강제(format)
  한 칸짜리 객관식 {"a": enum}  → 선택지마다 로그 확률을 계산해 가장 높은 것 (Ollama 문법 강제와 같은 효과)
  그 밖의 schema              → 형식 설명을 붙여 생성한 뒤 JSON을 꺼내 허용 값인지 검사 (어기면 오류)
생각 모드: 요청의 "think": true 이면 chat template의 enable_thinking=True

사용 (서버, 별도 환경 .venv_q35):
  CUDA_VISIBLE_DEVICES=3 HF_HOME=/workspace/capstone/src/kyj/models/hf \
    .venv_q35/bin/python q35_server.py --port 11437
"""

import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-4B"
LOCK = None
M = P = None


def load():
    global M, P
    P = AutoProcessor.from_pretrained(MODEL_ID)
    M = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda:0").eval()


def inputs_for(text, images, think):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    return P.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                 return_tensors="pt", enable_thinking=think).to(M.device)


@torch.no_grad()
def choose(inputs, key, options):
    """'{"key": "' 뒤에 올 선택지 중 로그 확률이 가장 높은 것."""
    prefix = P.tokenizer(f'{{"{key}": "', add_special_tokens=False, return_tensors="pt").input_ids.to(M.device)
    base = {k: v for k, v in inputs.items()}
    ids = torch.cat([base["input_ids"], prefix], dim=1)
    scores = {}
    for opt in options:
        tail = P.tokenizer(f'{opt}"}}', add_special_tokens=False, return_tensors="pt").input_ids.to(M.device)
        full = torch.cat([ids, tail], dim=1)
        kw = {}
        for k, v in base.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == base["input_ids"].shape[1]:
                # 토큰별 텐서(mm_token_type_ids 등)는 덧붙인 글자 토큰만큼 0(글자)으로 늘린다
                v = torch.cat([v, torch.zeros(1, full.shape[1] - v.shape[1], dtype=v.dtype, device=v.device)], dim=1)
            kw[k] = v
        out = M(input_ids=full, attention_mask=torch.ones_like(full), **kw)
        logp = torch.log_softmax(out.logits[0, ids.shape[1] - 1:-1].float(), dim=-1)
        scores[opt] = float(logp.gather(1, tail[0].unsqueeze(1)).sum())
    return max(scores, key=scores.get), scores


def describe(schema):
    lines = []
    for k, v in schema["properties"].items():
        if v.get("type") == "array":
            it = v["items"]
            allowed = it.get("enum")
            lines.append(f'"{k}": 배열' + (f" (원소 허용 값: {allowed})" if allowed else ""))
        elif "enum" in v:
            lines.append(f'"{k}": {v["enum"]} 중 하나')
        else:
            lines.append(f'"{k}": {v.get("type")}')
    return "\n\n다음 키를 모두 가진 JSON 객체 하나로만 답하라. 다른 글은 쓰지 마라.\n" + "\n".join(lines)


def valid(obj, schema):
    for k, v in schema["properties"].items():
        if k not in obj:
            return False
        if "enum" in v and obj[k] not in v["enum"]:
            return False
        if v.get("type") == "array" and "enum" in v.get("items", {}):
            if not isinstance(obj[k], list) or any(x not in v["items"]["enum"] for x in obj[k]):
                return False
    return True


@torch.no_grad()
def chat(req):
    msg = req["messages"][-1]
    images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in msg.get("images", [])]
    think = bool(req.get("think"))
    schema = req.get("format")
    num_predict = (req.get("options") or {}).get("num_predict", 256)
    t0 = time.perf_counter()
    props = (schema or {}).get("properties", {})
    if schema and len(props) == 1 and "enum" in next(iter(props.values())) and not think:
        key, prop = next(iter(props.items()))
        inputs = inputs_for(msg["content"], images, think)
        best, scores = choose(inputs, key, prop["enum"])
        return {"message": {"role": "assistant", "content": json.dumps({key: best}, ensure_ascii=False), "thinking": ""},
                "eval_count": 0, "prompt_eval_count": int(inputs["input_ids"].shape[1]), "scores": scores,
                "total_duration": int((time.perf_counter() - t0) * 1e9)}
    text = msg["content"] + (describe(schema) if schema else "")
    inputs = inputs_for(text, images, think)
    out = M.generate(**inputs, max_new_tokens=num_predict, do_sample=False)
    gen = P.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    thinking, answer = "", gen
    if "</think>" in gen:
        thinking, answer = gen.split("</think>", 1)
    content = answer.strip()
    if schema:
        m = re.search(r"\{.*\}", content, re.S)
        obj = None
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
        content = json.dumps(obj, ensure_ascii=False) if obj is not None and valid(obj, schema) else ""
    return {"message": {"role": "assistant", "content": content, "thinking": thinking.strip(), "raw": gen[-2000:]},
            "eval_count": int(out.shape[1] - inputs["input_ids"].shape[1]),
            "prompt_eval_count": int(inputs["input_ids"].shape[1]),
            "total_duration": int((time.perf_counter() - t0) * 1e9)}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            try:
                res, code = chat(body), 200
            except Exception as e:  # noqa: BLE001
                res, code = {"error": f"{type(e).__name__}: {e}"}, 500
        data = json.dumps(res, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11437)
    a = ap.parse_args()
    LOCK = threading.Lock()
    load()
    print("READY", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
