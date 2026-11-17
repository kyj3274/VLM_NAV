"""Claude 호출 래퍼 (충남대 게이트웨이 BAZE 경유). claude_nav/client.py를 줄여 옮긴 것.

"시스템 프롬프트 + 내용(텍스트·이미지) + 스키마 → JSON, 시간, 토큰, 크레딧"만 주고받는다.
- 키는 .env의 BAZE_API_KEY로만 읽는다. 코드·로그·결과에 남기지 않는다.
- 구조화 출력: output_config.format = json_schema (2026-11-17 claude_nav에서 게이트웨이 동작 확인).
- prompt caching: 시스템 블록에 cache_control (베타 헤더 prompt-caching-2024-07-31).
- 시간 초과(기본 15초)는 재시도하지 않는다. 429·5xx·연결 끊김만 최대 2회 재시도.
- 크레딧: 이 계정은 응답에 billing이 없어 토큰 × 단가 × 1000으로 계산한다(claude_nav에서 잔액 차이와 정확히 일치 확인).
- 작업 한도(job_limit)를 넘으면 CreditLimit을 던진다.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway/claude"
CREDITS_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway/credits"
CACHE_BETA = "prompt-caching-2024-07-31"
# 100만 토큰당 달러: 입력, 출력, 캐시 읽기, 캐시 쓰기. 크레딧 = 달러 × 1000
PRICE = {"claude-haiku-5-5": (0.1, 0.5, 0.01, 0.125), "claude-sonnet-5-5": (2.0, 10.0, 0.2, 2.5)}
THINKING = {"claude-haiku-5-5": {"type": "disabled"}}
ROOT = next(d for d in Path(__file__).resolve().parents if (d / "data" / "video").is_dir())


class CreditLimit(RuntimeError):
    pass


@dataclass
class Result:
    data: dict | None = None
    raw_text: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    credit: float = 0.0
    seconds: float = 0.0
    error: str | None = None
    timed_out: bool = False
    model: str = ""

    @property
    def ok(self):
        return self.error is None and self.data is not None

    def log(self):
        d = asdict(self)
        d.pop("data")
        return d


def load_key(env="BAZE_API_KEY"):
    if not os.environ.get(env):
        p = ROOT / ".env"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == env:
                    os.environ[env] = v.strip().strip('"').strip("'")
    key = os.environ.get(env, "")
    if not key:
        raise RuntimeError(f"{env}가 .env에 없습니다")
    return key


def jpeg_block(img, quality=90):
    """PIL 이미지 → base64 JPEG content block."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def credit_of(model, r: Result):
    p = PRICE.get(model)
    if not p:
        return 0.0
    usd = r.input_tokens * p[0] + r.output_tokens * p[1] + r.cache_read * p[2] + r.cache_write * p[3]
    return round(usd / 1e6 * 1000, 5)


class Ledger:
    """이번 작업 누적 크레딧. limit을 넘으면 더 부르지 않는다."""

    def __init__(self, limit=None):
        self.limit, self.total, self.calls = limit, 0.0, 0

    def check(self):
        if self.limit is not None and self.total >= self.limit:
            raise CreditLimit(f"작업 누적 {self.total:.2f} / 한도 {self.limit}")

    def add(self, c):
        self.total += c or 0.0
        self.calls += 1


class GatewayClient:
    def __init__(self, model="claude-haiku-5-5", timeout=15.0, caching=True, ledger=None, retries=2):
        import anthropic
        self.anthropic, self.model, self.caching, self.retries = anthropic, model, caching, retries
        self.timeout = timeout
        self.ledger = ledger or Ledger()
        self._key = load_key()
        self.sdk = anthropic.Anthropic(api_key=self._key, base_url=BASE_URL, max_retries=0, timeout=timeout,
                                       default_headers={"anthropic-beta": CACHE_BETA} if caching else None)

    def credits_used(self):
        """이번 달 사용 크레딧(게이트웨이 잔액 조회). 실행 시작·끝에만 쓴다."""
        import httpx
        try:
            j = httpx.get(CREDITS_URL, headers={"Authorization": f"Bearer {self._key}"}, timeout=10).json()
            parts = [j.get(k, {}).get("used") for k in ("monthly_allocated", "purchased", "org_granted")]
            return float(sum(x for x in parts if x is not None))
        except Exception:  # noqa: BLE001
            return None

    def call(self, system, content, schema, max_tokens=400, kind="see") -> Result:
        self.ledger.check()
        sys_block = {"type": "text", "text": system}
        if self.caching:
            sys_block["cache_control"] = {"type": "ephemeral"}
        p = {"model": self.model, "max_tokens": max_tokens, "system": [sys_block],
             "messages": [{"role": "user", "content": content}],
             "output_config": {"format": {"type": "json_schema", "schema": schema}}}
        if self.model in THINKING:
            p["thinking"] = THINKING[self.model]
        res = Result(model=self.model)
        a = self.anthropic
        t0 = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                msg = self.sdk.messages.create(**p)
                break
            except Exception as e:  # noqa: BLE001
                res.seconds = round(time.monotonic() - t0, 3)
                if isinstance(e, a.APITimeoutError):
                    res.timed_out, res.error = True, f"시간 초과({self.timeout:g}초)"
                    return res
                retry = isinstance(e, (a.RateLimitError, a.APIConnectionError, a.InternalServerError)) or (
                    isinstance(e, a.APIStatusError) and e.status_code >= 500)
                if retry and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                status = getattr(e, "status_code", None)
                res.error = f"{type(e).__name__}{f' {status}' if status else ''}: {str(e)[:300]}"
                return res
        res.seconds = round(time.monotonic() - t0, 3)
        u = msg.usage
        res.input_tokens, res.output_tokens = u.input_tokens or 0, u.output_tokens or 0
        res.cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        res.cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        res.credit = credit_of(self.model, res)
        self.ledger.add(res.credit)
        res.stop_reason = msg.stop_reason
        res.raw_text = "".join(b.text for b in msg.content if b.type == "text")
        if msg.stop_reason == "max_tokens":
            res.error = "잘림(max_tokens)"
        elif msg.stop_reason == "refusal":
            res.error = "거절"
        else:
            try:
                res.data = json.loads(res.raw_text)
            except json.JSONDecodeError as e:
                res.error = f"JSON 파싱 실패: {e}"
        return res


class FakeClient:
    """테스트용. responder(kind, content, n) → dict."""

    def __init__(self, responder, credit=0.0, seconds=0.0):
        self.responder, self.credit, self.seconds = responder, credit, seconds
        self.model, self.calls, self.ledger = "fake", [], Ledger()

    def call(self, system, content, schema, max_tokens=400, kind="see") -> Result:
        n = sum(1 for c in self.calls if c["kind"] == kind)
        self.calls.append({"kind": kind, "content": content})
        data = self.responder(kind, content, n)
        self.ledger.add(self.credit)
        return Result(data=data, raw_text=json.dumps(data, ensure_ascii=False), stop_reason="end_turn",
                      credit=self.credit, seconds=self.seconds, model="fake")


class DryRunClient:
    def __init__(self, model="claude-haiku-5-5"):
        self.model, self.ledger, self.shown = model, Ledger(), False

    def call(self, system, content, schema, max_tokens=400, kind="see") -> Result:
        if not self.shown:
            print(f"=== 시스템 프롬프트 ({len(system)}자) ===\n{system}\n=== 스키마 ===\n"
                  f"{json.dumps(schema, ensure_ascii=False)}\n===", flush=True)
            self.shown = True
        imgs = [c for c in content if c.get("type") == "image"]
        kb = sum(len(c["source"]["data"]) * 3 // 4 for c in imgs) / 1024
        text = " | ".join(c["text"] for c in content if c.get("type") == "text")
        print(f"[dry-run] {kind} 이미지 {len(imgs)}장 {kb:.0f}KB, 텍스트: {text[:200]}", flush=True)
        return Result(error="dry_run", model=self.model)


class ReplayClient:
    """캐시 재생 전용: 부르면 실패로 돌려준다(크레딧 0). 캐시에 없는 프레임은 오류로 남는다."""

    def __init__(self, model="claude-haiku-5-5"):
        self.model, self.ledger = model, Ledger()

    def call(self, system, content, schema, max_tokens=400, kind="see") -> Result:
        return Result(error="replay: 캐시에 없음", model=self.model)
