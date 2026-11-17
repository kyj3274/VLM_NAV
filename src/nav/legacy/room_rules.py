"""호실 결정 규칙 (실험 B·C). OCR 원자료(texts_all: text/score/box 0~1 상대좌표)에서 호실을 고른다.

건물 특화 정보(파일의 층, 우리 건물 호실 목록 등)는 쓰지 않는다. 한 사진 안의 정보만 쓴다.

사용 (서버 파이프라인에서 import):
    import room_rules
    cands = room_rules.candidates(texts_all, image_size)            # 호실 모양 글자 후보 + 특징
    room_rules.add_plate_features(cands, original_pil_image)         # (선택) 문패 판 특징
    kept, ambiguous = room_rules.select(cands, room_rules.BEST)      # 출력할 호실, VLM이 확인할 애매 후보
"""

import json
import re
from collections import defaultdict

ROOM = re.compile(r"^[1-9]\d{2}(-\d{1,2})?$")
HANGUL = re.compile(r"[가-힣]")

# 기본 파라미터 (실험 B·C dev에서 고른 값은 BEST)
DEFAULT = {
    "min_score": 0.5,        # OCR 점수 기준
    "max_cy": 1.0,           # 글자 중심 세로 위치 상한 (0=위, 1=아래). 1.0이면 끔
    "edge": 0.0,             # 사진 좌우 가장자리 거리(0~1) 이하에 닿은 상자 = 잘린 글자로 보고 제외. 0이면 끔
    "edge_max_score": 1.01,  # 가장자리 제외는 OCR 점수가 이 값 미만일 때만 (확신 높은 글자는 살림)
    "min_aspect": 0.0,       # 상자 가로/세로(픽셀) 하한. 잘린 3자리 숫자 제외용. 0이면 끔
    "min_height": 0.0,       # 글자 높이(사진 높이 대비) 하한
    "majority": False,       # 같은 사진 호실 첫 자리 다수결 (동점이면 가장 큰 글자 쪽)
    "need_pair": False,      # 숫자 아래/옆에 한글 이름이 붙어 있어야 함
    "pair_or_big": 0.0,      # need_pair일 때 글자 높이가 이 값 이상이면 짝 없어도 통과
    "plate_min": None,       # 문패 판 점수 하한 (add_plate_features 필요). None이면 끔
    "top_k": 0,              # 글자 높이 순 상위 K개만 출력. 0이면 전부
}

# 실험 B·C(dev) 선택
BEST = {"majority": True}                          # m2048 + 첫 자리 다수결. 애매 후보(kept_low 등)는 VLM 확인
BEST_NO_VLM = {"majority": True, "min_score": 0.7}  # VLM 없이 1단계만 쓸 때 (정밀도 우선)
BEST_M3072 = {"majority": True, "min_score": 0.7}   # m3072 원자료용 (보정 기준 최고)


def _norm(text):
    return text.replace("호", "").replace(" ", "").strip()


def candidates(texts_all, image_size):
    """호실 모양 글자(1~9로 시작하는 3자리, -숫자 허용) 후보와 특징을 만든다. 같은 번호는 가장 큰 상자 하나만."""
    width, height = image_size
    hangul = [t for t in texts_all if HANGUL.search(t["text"]) and t["score"] >= 0.5]
    best = {}
    for t in texts_all:
        value = _norm(t["text"])
        if not ROOM.match(value):
            continue
        x0, y0, x1, y1 = t["box"]
        h = y1 - y0
        w = x1 - x0
        c = {
            "room": value, "score": float(t["score"]), "box": t["box"],
            "h": h, "cy": (y0 + y1) / 2, "cx": (x0 + x1) / 2,
            "edge_dist": min(x0, 1 - x1),
            "aspect": (w * width) / max(h * height, 1e-6),
            "pair": _has_pair(t["box"], hangul),
        }
        old = best.get(value)
        if old is None or (c["score"] >= 0.5, c["h"]) > (old["score"] >= 0.5, old["h"]):
            best[value] = c
    return list(best.values())


def _has_pair(box, hangul):
    """숫자 상자 바로 아래(글자 높이 3배 이내, 가로 겹침) 또는 바로 오른쪽(같은 줄)에 한글 글자가 있는지."""
    x0, y0, x1, y1 = box
    h = max(y1 - y0, 1e-6)
    w = x1 - x0
    for t in hangul:
        a0, b0, a1, b1 = t["box"]
        overlap_x = min(x1, a1) - max(x0, a0)
        below = overlap_x > -0.5 * w and -0.3 * h <= b0 - y1 <= 3 * h
        right = abs((b0 + b1) / 2 - (y0 + y1) / 2) < h and -0.3 * h <= a0 - x1 <= 2 * h
        if below or right:
            return True
    return False


def plate_features(image, box, pad=0.6):
    """원본 PIL 이미지에서 글자 상자 주변을 잘라 '단색 판' 특징을 계산한다.
    inner = 글자 상자를 글자 높이 pad배만큼 넓힌 영역, ring = 그 바깥 글자 높이 1배 띠.
    반환: bg_std(판 배경 밝기 표준편차, 낮을수록 균일), contrast(글자-배경 밝기 차),
          ring_diff(판 배경과 바깥 띠의 평균 색 차이), plate(0~1 종합 점수)."""
    import numpy as np

    W, H = image.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    h = max(y1 - y0, 2)

    def crop(m):
        return np.asarray(image.crop((int(max(0, x0 - m)), int(max(0, y0 - m)),
                                      int(min(W, x1 + m)), int(min(H, y1 + m)))).convert("RGB"), dtype=np.float32)

    inner = crop(pad * h)
    outer = crop((pad + 1.0) * h)
    if inner.size == 0 or outer.size == 0:
        return {"bg_std": 99.0, "contrast": 0.0, "ring_diff": 0.0, "plate": 0.0}
    gray = inner.mean(axis=2)
    # Otsu로 글자/배경 나누기 (배경 = 더 많은 쪽)
    hist, edges = np.histogram(gray, bins=64, range=(0, 256))
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    best_t, best_v = 128, -1
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * centers)
    for i in range(1, 63):
        w0 = cum[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cum_mean[i] / w0
        m1 = (cum_mean[-1] - cum_mean[i]) / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, centers[i]
    dark = gray < best_t
    bg_mask = ~dark if (~dark).sum() >= dark.sum() else dark
    fg_mask = ~bg_mask
    bg = inner[bg_mask]
    bg_std = float(gray[bg_mask].std()) if bg_mask.any() else 99.0
    contrast = float(abs(gray[bg_mask].mean() - gray[fg_mask].mean())) if fg_mask.any() and bg_mask.any() else 0.0
    # 바깥 띠 = outer에서 inner 위치를 뺀 부분
    oh, ow = outer.shape[:2]
    ih, iw = inner.shape[:2]
    oy = int(min(y0, (pad + 1.0) * h) - min(y0, pad * h))
    ox = int(min(x0, (pad + 1.0) * h) - min(x0, pad * h))
    ring_mask = np.ones((oh, ow), bool)
    ring_mask[oy:oy + ih, ox:ox + iw] = False
    ring = outer[ring_mask]
    ring_diff = float(np.linalg.norm(bg.mean(axis=0) - ring.mean(axis=0))) if len(ring) and len(bg) else 0.0
    # 종합: 배경이 균일하고(bg_std 작음) 글자 대비가 큰 판
    plate = float(np.clip(contrast / 80.0, 0, 1) * np.clip(1 - bg_std / 40.0, 0, 1))
    return {"bg_std": round(bg_std, 1), "contrast": round(contrast, 1), "ring_diff": round(ring_diff, 1), "plate": round(plate, 3)}


def add_plate_features(cands, image):
    for c in cands:
        c.update(plate_features(image, c["box"]))
    return cands


def _basic_ok(c, p):
    if c["score"] < p["min_score"]:
        return False
    if c["cy"] > p["max_cy"]:
        return False
    if p["edge"] and c["edge_dist"] <= p["edge"] and c["score"] < p["edge_max_score"]:
        return False
    if p["min_aspect"] and c["aspect"] < p["min_aspect"]:
        return False
    if c["h"] < p["min_height"]:
        return False
    if p["need_pair"] and not c["pair"] and not (p["pair_or_big"] and c["h"] >= p["pair_or_big"]):
        return False
    if p["plate_min"] is not None and c.get("plate", 1.0) < p["plate_min"]:
        return False
    return True


def majority_digit(cands):
    """첫 자리별 후보 수 다수결. 동점이면 OCR 점수가 가장 높은 후보의 첫 자리, 그다음 큰 글자."""
    groups = defaultdict(list)
    for c in cands:
        groups[c["room"][0]].append(c)
    if not groups:
        return None
    return max(groups, key=lambda d: (len(groups[d]), round(max(c["score"] for c in groups[d]), 2),
                                      max(c["h"] for c in groups[d])))


def select(cands, params=None, ambig=None):
    """규칙을 적용해 (출력할 후보 목록, 애매 후보 목록)을 돌려준다. 출력 목록은 글자 높이 큰 순(가까운 것 먼저)."""
    p = dict(DEFAULT, **(params or {}))
    passed = [c for c in cands if _basic_ok(c, p)]
    digit = majority_digit(passed) if p["majority"] else None
    rejected_majority = [c for c in passed if digit and c["room"][0] != digit]
    if digit:
        passed = [c for c in passed if c["room"][0] == digit]
    passed.sort(key=lambda c: -c["h"])
    kept = passed[: p["top_k"]] if p["top_k"] else passed
    a = dict(AMBIG, **(ambig or {}))
    ambiguous = [c for c in cands if ambiguous_reason(c, kept, rejected_majority, digit, p, a)]
    for c in ambiguous:
        c["ambiguous"] = ambiguous_reason(c, kept, rejected_majority, digit, p, a)
    return kept, ambiguous


# 애매 후보 = 실험 D에서 VLM이 확인할 후보. 규칙이 확신하지 못하는 경우만.
AMBIG = {"score_lo": 0.5, "score_hi": 0.9, "near_h": 0.012, "weak_lo": 0.3}


def ambiguous_reason(c, kept, rejected_majority, digit, p, a=AMBIG):
    """(1) kept_low: 출력하는 후보인데 OCR 점수가 중간(score_lo 이상 score_hi 미만)
    (2) maj_big: 첫 자리 다수결로만 버렸는데 글자가 큰 편(h >= near_h): 가까운 문패를 잘못 버렸을 수 있음
    (3) weak_big: OCR 점수가 기준 미만(weak_lo 이상)이지만 글자가 큰 편이고 첫 자리가 다수와 같거나 다른 후보가 없음
    해당 없으면 빈 문자열."""
    if c in kept and a["score_lo"] <= c["score"] < a["score_hi"]:
        return "kept_low"
    if c in rejected_majority and c["h"] >= a["near_h"]:
        return "maj_big"
    if (a["weak_lo"] <= c["score"] < p["min_score"] and c["h"] >= a["near_h"]
            and (digit is None or c["room"][0] == digit)
            and all(k["room"] != c["room"] for k in kept)):
        return "weak_big"
    return ""


def apply(record, params=None, image_size=None):
    """원자료 한 줄(exp_a_*.jsonl)에 규칙을 적용해 호실 목록을 돌려준다 (Mac 재계산용)."""
    size = image_size or record.get("ocr_input_size") or (1, 1)
    cands = record.get("_cands") or candidates(record["texts_all"], size)
    kept, ambiguous = select(cands, params)
    return [c["room"] for c in kept], ambiguous


# ---------------------------------------------------------------- 1단계 결과에 반영 (exp_bc_rules.rebuild와 같음)

def clock(x):
    return "11시" if x < 1 / 3 else "12시" if x < 2 / 3 else "1시"


def result_rooms(rec, params=None):
    """1단계 기록(result, texts_all, ocr_input_size)에서 호실을 규칙으로 고르고 결과를 다시 만든다. (확정 호실, 결과)"""
    kept, _ = select(candidates(rec["texts_all"], rec["ocr_input_size"]), params or BEST)
    return kept, rebuild(rec, kept)


def rebuild(rec, kept):
    """기존 combine 결과에서 호실 관련 부분만 규칙 결과로 바꾼다."""
    res = json.loads(json.dumps(rec["result"]))
    rooms = [c["room"] for c in kept]
    res["rooms"] = rooms
    res["directions"] = [d for d in res["directions"] if not ROOM.match(d["target"])]
    res["directions"] += [{"target": c["room"], "type": "RELATIVE_POSITION", "value": clock(c["cx"])} for c in kept]
    # 공간 이름: 남은 호실 번호 바로 아래/옆 한글 (combine과 같은 기준)만 남김
    texts = [t for t in rec["texts_all"] if t["score"] >= 0.5]
    spaces = []
    for c in kept:
        rb = c["box"]
        for t in texts:
            if not HANGUL.search(t["text"]):
                continue
            dx = abs(t["box"][0] - rb[0]); dy = t["box"][1] - rb[3]
            right = abs(t["box"][1] - rb[1]) < 0.03 and 0 <= t["box"][0] - rb[2] < 0.1
            if (dx < 0.08 and -0.01 <= dy < 0.06) or right:
                n = t["text"].replace(" ", "")
                if n not in spaces:
                    spaces.append(n)
    spaces += [s for s in res["spaces"] if s in ("TOILET", "화장실") and s not in spaces]
    res["spaces"] = spaces
    return res
