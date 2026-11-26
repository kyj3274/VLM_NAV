"""깊이 추정 (Depth Anything V2 Metric Indoor, small). 사진 한 장에서 화면 각 점까지 거리(m)를 낸다.

길안내에서 쓰는 것
- 복도 방향: 눈높이 띠에서 가장 먼 열. 문패가 그 왼쪽이면 왼쪽 벽 (휴대폰을 옆으로 돌려도 흔들리지 않음)
- 문패까지 거리: 글자 상자 주변 깊이의 가운데값
- 정면 거리: 화면 가운데의 깊이 (앞이 막혔는지)

모델 위치: models/가 있는 위 폴더의 models/depth_anything_v2_metric_indoor_small (서버), 없으면 Hugging Face 캐시
Claude 버전(src/nav)과 로컬 버전(src/nav/local)이 같이 쓴다.
11/17 시험: 3층 문패 21번의 벽을 모두 맞힘 (화면 위치 기준 14번). Mac M4 장당 0.13초.
"""

import sys
from pathlib import Path

import numpy as np

COLS = 7          # 화면을 가로로 이만큼 나눠 칸마다 벽·막힌 곳까지 거리 (머릿속 지도에서 본 곳을 그 거리까지만)
SIDE_GAP = 0.08   # 문패가 복도 방향(가장 먼 열)에서 화면 너비의 이만큼 안이면 복도 끝 벽일 수 있어 벽 쪽을 정하지 않음
CORRIDOR_M = 3.0  # 가장 먼 곳이 이보다 가까우면 복도가 안 보이는 사진(벽·문을 마주봄)으로 보고 벽 쪽을 정하지 않음
HF_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
LOCAL = next((d for d in Path(__file__).resolve().parents if (d / "models").is_dir()), Path(__file__).resolve().parents[2]) / "models" / "depth_anything_v2_metric_indoor_small"


def load():
    """깊이 모델. 불러올 수 없으면 None (길안내는 깊이 없이 예전처럼 동작)."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        print(f"[depth] torch/transformers가 없어 깊이 없이 동작합니다: {e}", file=sys.stderr, flush=True)
        return None
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return pipeline("depth-estimation", model=str(LOCAL) if LOCAL.is_dir() else HF_ID, device=dev)
    except Exception as e:  # noqa: BLE001  모델 파일이 없고 내려받을 수도 없을 때, 버전이 안 맞을 때
        print(f"[depth] 깊이 모델을 불러오지 못해 깊이 없이 동작합니다 (벽 쪽은 화면 위치로): {type(e).__name__}: {str(e)[:200]}",
              file=sys.stderr, flush=True)
        return None


def measure(model, img):
    """{"far_x": 복도 방향(0~1), "far_m": 그 거리, "ahead_m": 정면 거리, "cols": 화면 왼쪽→오른쪽 칸별 거리}, 깊이 지도"""
    dep = model(img)["predicted_depth"].squeeze().float().cpu().numpy()
    h, w = dep.shape
    col = np.convolve(np.median(dep[int(0.30 * h):int(0.60 * h)], axis=0), np.ones(9) / 9, mode="same")
    ahead = dep[int(0.45 * h):int(0.55 * h), int(0.45 * w):int(0.55 * w)]
    return {"far_x": round(int(np.argmax(col)) / w, 3), "far_m": round(float(col.max()), 1),
            "ahead_m": round(float(np.median(ahead)), 1),
            "cols": [round(float(np.median(c)), 1) for c in np.array_split(col, COLS)]}, dep


def box_distance(dep, box):
    """글자 상자(0~1 좌표) 주변 깊이의 가운데값 (m)."""
    h, w = dep.shape
    x0, y0, x1, y1 = box
    pad = (x1 - x0) * 0.5
    patch = dep[int(max(0, y0 - 0.005) * h):int(min(1, y1 + 0.005) * h) + 1,
                int(max(0, x0 - pad) * w):int(min(1, x1 + pad) * w) + 1]
    return round(float(np.median(patch)), 1)


def wall_side(info, cx):
    """문패 글자 가운데 x(0~1) → '왼쪽'/'오른쪽', 정할 수 없으면 '?' (Memory.see가 표를 넣지 않음)."""
    if info["far_m"] < CORRIDOR_M or abs(cx - info["far_x"]) < SIDE_GAP:
        return "?"
    return "왼쪽" if cx < info["far_x"] else "오른쪽"
