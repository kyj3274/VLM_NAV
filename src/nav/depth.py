"""깊이 추정 (Depth Anything V2 Metric Indoor, small). 사진 한 장에서 화면 각 점까지 거리(m)를 낸다.

길안내에서 쓰는 것
- 복도 방향: 눈높이 띠에서 가장 먼 열. 문패가 그 왼쪽이면 왼쪽 벽 (휴대폰을 옆으로 돌려도 흔들리지 않음)
- 문패까지 거리: 글자 상자 주변 깊이의 가운데값
- 정면 거리: 화면 가운데의 깊이 (앞이 막혔는지)

모델 위치: <nav의 위 폴더>/models/depth_anything_v2_metric_indoor_small (서버), 없으면 Hugging Face 캐시
11/17 시험: 3층 문패 21번의 벽을 모두 맞힘 (화면 위치 기준 14번). Mac M4 장당 0.13초.
"""

from pathlib import Path

import numpy as np

HF_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
LOCAL = Path(__file__).resolve().parents[1] / "models" / "depth_anything_v2_metric_indoor_small"


def load():
    """깊이 모델. 불러올 수 없으면 None (길안내는 깊이 없이 예전처럼 동작)."""
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        return None
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return pipeline("depth-estimation", model=str(LOCAL) if LOCAL.is_dir() else HF_ID, device=dev)
    except Exception:  # noqa: BLE001  모델 파일이 없고 내려받을 수도 없을 때
        return None


def measure(model, img):
    """{"far_x": 복도 방향(0~1), "far_m": 그 거리, "ahead_m": 정면 거리}, 깊이 지도"""
    dep = model(img)["predicted_depth"].squeeze().float().cpu().numpy()
    h, w = dep.shape
    col = np.convolve(np.median(dep[int(0.30 * h):int(0.60 * h)], axis=0), np.ones(9) / 9, mode="same")
    ahead = dep[int(0.45 * h):int(0.55 * h), int(0.45 * w):int(0.55 * w)]
    return {"far_x": round(int(np.argmax(col)) / w, 3), "far_m": round(float(col.max()), 1),
            "ahead_m": round(float(np.median(ahead)), 1)}, dep


def box_distance(dep, box):
    """글자 상자(0~1 좌표) 주변 깊이의 가운데값 (m)."""
    h, w = dep.shape
    x0, y0, x1, y1 = box
    pad = (x1 - x0) * 0.5
    patch = dep[int(max(0, y0 - 0.005) * h):int(min(1, y1 + 0.005) * h) + 1,
                int(max(0, x0 - pad) * w):int(min(1, x1 + pad) * w) + 1]
    return round(float(np.median(patch)), 1)
