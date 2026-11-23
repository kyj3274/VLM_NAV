"""목적지 음성 받아쓰기: 맥북 로컬 faster-whisper(small, 한국어). 외부 API를 쓰지 않는다. (claude_nav/stt.py에서 가져옴)

- 시스템 ffmpeg가 없어 PyAV로 디코딩하는 faster-whisper를 쓴다(mlx-whisper는 ffmpeg CLI 필요).
- 모델은 프로젝트 안 models/whisper/에 둔다.
- 장치: MPS(애플 GPU, 실험적, 약 1.3초) → 실패하면 CPU int8(약 3.2초). 2026-11-17 합성음 4문장 측정.
- 초기 프롬프트: 사용자 지정 단어(호, 층, 강의실, 화장실, 엘리베이터)를 문장 안에 넣는다. 단어만 나열하면
  "화장실 어디야"가 "화장실, 엘리베이터"로 오염되었고(프롬프트 복사), 무음에서는 프롬프트 자체를 그대로 냈다.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from claude_client import ROOT as PROJECT_ROOT

STT_PROMPT = "실내 길안내 목적지를 말합니다. 예를 들면 호, 층, 강의실, 화장실, 엘리베이터 같은 말을 씁니다."
PROMPT_WORDS = ("호", "층", "강의실", "화장실", "엘리베이터")
# 무음·잡음에서 Whisper가 자주 지어내는 문장
HALLUCINATIONS = ("시청해 주셔서 감사합니다", "구독", "좋아요", "자막", "MBC", "KBS", "SBS", "감사합니다.",
                  "실내 길안내 목적지를 말합니다", "같은 말을 씁니다")


@dataclass
class Transcript:
    text: str = ""
    duration_s: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    elapsed_s: float = 0.0
    no_speech_prob: float = 0.0
    device: str = ""
    reject: str | None = None  # 다시 말해 달라고 할 이유

    def to_dict(self) -> dict:
        return asdict(self)


def reject_reason(text: str, duration_s: float, rms: float, no_speech: float,
                  min_duration_s: float = 0.5, min_rms: float = 0.005) -> str | None:
    if duration_s < min_duration_s:
        return f"녹음이 짧음({duration_s:.2f}초)"
    if rms < min_rms:
        return f"소리가 작음(RMS {rms:.4f})"
    t = (text or "").strip()
    if not t:
        return "받아쓴 글자 없음"
    if not re.search(r"[가-힣0-9]", t):
        return f"한글·숫자 없음: {t!r}"
    if any(h in t for h in HALLUCINATIONS) and len(t) < 30:
        return f"무음 환각 의심: {t!r}"
    words = [w for w in re.split(r"[\s,.]+", t) if w]
    if words and all(w in PROMPT_WORDS for w in words) and len(words) >= 3:
        return f"프롬프트 복사 의심: {t!r}"
    if no_speech > 0.8:
        return f"말소리 없음 확률 {no_speech:.2f}"
    return None


class Transcriber:
    def __init__(self, model_size: str = "small", root: Path | None = None, device: str = "auto",
                 min_duration_s: float = 0.5, min_rms: float = 0.005):
        self.model_size = model_size
        self.root = Path(root or PROJECT_ROOT / "models" / "whisper")
        self.device_pref = device
        self.min_duration_s = min_duration_s
        self.min_rms = min_rms
        self._model = None
        self.device = ""
        self._lock = threading.Lock()  # 모델 하나를 여러 요청이 동시에 쓰지 않게

    def _load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        tries = [("mps", "float16"), ("cpu", "int8")] if self.device_pref == "auto" else \
            [(self.device_pref, "float16" if self.device_pref == "mps" else "int8")]
        last = None
        for dev, ct in tries:
            try:
                self._model = WhisperModel(self.model_size, device=dev, compute_type=ct, download_root=str(self.root))
                self.device = dev
                return
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"Whisper 모델을 불러오지 못했습니다: {last}")

    def transcribe(self, path: str | Path) -> Transcript:
        import numpy as np
        from faster_whisper import decode_audio
        t0 = time.monotonic()
        audio = decode_audio(str(path))
        dur = len(audio) / 16000.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        tr = Transcript(duration_s=round(dur, 2), rms=round(rms, 5), peak=round(peak, 4))
        pre = reject_reason("가", dur, rms, 0.0, self.min_duration_s, self.min_rms)  # 길이·음량만 먼저
        if pre:
            tr.reject = pre
            tr.elapsed_s = round(time.monotonic() - t0, 3)
            return tr
        with self._lock:
            self._load()
            segs, _ = self._model.transcribe(audio, language="ko", initial_prompt=STT_PROMPT, beam_size=1)
            segs = list(segs)
        tr.text = "".join(s.text for s in segs).strip()
        tr.no_speech_prob = round(max((s.no_speech_prob for s in segs), default=1.0), 3)
        tr.device = self.device
        tr.elapsed_s = round(time.monotonic() - t0, 3)
        tr.reject = reject_reason(tr.text, dur, rms, tr.no_speech_prob, self.min_duration_s, self.min_rms)
        return tr


class FakeTranscriber:
    """테스트용: 정해 둔 글자를 돌려준다(오디오는 길이 검사만 흉내)."""

    def __init__(self, text: str = "410호로 가고 싶어", duration_s: float = 1.5, rms: float = 0.05):
        self.text, self.duration_s, self.rms = text, duration_s, rms
        self.calls = 0

    def transcribe(self, path) -> Transcript:
        self.calls += 1
        tr = Transcript(text=self.text, duration_s=self.duration_s, rms=self.rms, elapsed_s=0.01, device="fake")
        tr.reject = reject_reason(self.text, self.duration_s, self.rms, 0.0)
        return tr
