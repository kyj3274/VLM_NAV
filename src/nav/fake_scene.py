"""가짜 응답 모드(--fake): Claude를 부르지 않고 정해 둔 장면을 차례로 돌려준다 (크레딧 0).
웹 화면·겹쳐 보내기·소리·관찰자 화면을 맥 브라우저에서 확인하려고 쓴다. 사진 내용과는 상관없다.

장면(인식 호출 순서): 건물 유리문 2번 → 로비 2번 → 엘리베이터(정면 2.5m) 3번 → 엘리베이터 안 3번
→ 목적지 층 표시 → 목적지-2, -1 문패 → 목적지 문패 6m, 3m, 1.8m, 1.5m … (목적지가 호실 번호일 때)
"""
from __future__ import annotations

import random
import threading
import time

from claude_client import FakeClient, Ledger


def _plate(n, m, box_x=45, on="door"):
    return {"n": n, "name": "", "m": m, "on": on, "box": [box_x, 30, box_x + 8, 35]}


def _scene(i, target):
    base = {"plates": [], "fs": "", "place": "corridor", "door": "none", "el": [], "arrows": []}
    t = int(target) if target.isdigit() else 410
    tf = str(t)[0]
    seq = (
        [{**base, "place": "entrance", "door": "glass"}] * 2
        + [{**base, "place": "lobby"}] * 2
        + [{**base, "place": "elev_hall", "el": [{"why": "button", "side": "C", "m": 2.5, "btn": "R"}]}] * 3
        + [{**base, "place": "in_elev"}] * 3
        + [{**base, "fs": f"{tf}F", "place": "elev_hall"}]
        + [{**base, "plates": [_plate(str(t - 2), 7, 70)]}, {**base, "plates": [_plate(str(t - 1), 5, 70)]}]
        + [{**base, "plates": [_plate(str(t), m, 20)]} for m in (6, 3)]
        + [{**base, "plates": [_plate(str(t), m, 45)]} for m in (1.8, 1.5)]
    )
    return seq[min(i, len(seq) - 1)]


class FakeScene:
    """FakeClient에 넣을 응답기. delay초(±30%)를 실제로 기다려 겹쳐 보내기가 보이게 한다."""

    def __init__(self, target, delay=1.2, credit=0.18):
        self.target, self.delay, self.credit = target, delay, credit
        self.i, self.last, self.lock = 0, None, threading.Lock()

    def __call__(self, kind, content, n):
        if self.delay:
            time.sleep(self.delay * random.uniform(0.7, 1.3))
        with self.lock:
            if kind == "quick":  # 문 앞 빠른 확인: 마지막 장면의 문패를 그대로
                return {"plates": [{k: p[k] for k in ("n", "m", "on", "box")} for p in (self.last or {}).get("plates") or []][:2]}
            if kind == "recheck":  # 마지막 장면에서 목적지에 가장 가까운 번호를 그대로 읽은 것으로
                ps = (self.last or {}).get("plates") or []
                t = int(self.target) if self.target.isdigit() else 0
                p = min(ps, key=lambda p: abs(int(p["n"]) - t), default=None) if ps else None
                return {"n": p["n"] if p else "", "name": ""}
            out = _scene(self.i, self.target)
            self.i += 1
            self.last = out
            return out


def fake_client(target, delay=1.2, credit=0.18, limit=None):
    c = FakeClient(FakeScene(target, delay, credit), credit=credit, seconds=delay)
    c.ledger = Ledger(limit)
    _call = c.call

    def call(system, content, schema, max_tokens=400, kind="see"):
        c.ledger.check()  # 진짜 클라이언트처럼 한도를 먼저 확인
        return _call(system, content, schema, max_tokens, kind)

    c.call = call
    c.credits_used = lambda: None
    return c
