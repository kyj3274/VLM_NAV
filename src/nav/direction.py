"""엘리베이터에서 내린 뒤 목적지 방향 정하기 (2026-11-23 사용자 지시 B7·B8). navigator.Navigator가 쓴다.

근거 세기(높을수록 우선):
  3 강   호실 범위 안내판·층 안내도(rr), 목적지가 든 화살표 표지, 같은 쪽 벽 문패 비교 2개 이상이 같은 결론
  2 보통 같은 쪽 벽 문패 비교 1개
  1 약   이전 층에서 같은 끝자리 번호(310 → 410)를 본 나침반 방향
  0 없음 복도가 길게 보이는 쪽(lw), 그것도 없으면 앞
방향은 사진 기준(L/F/R/B)으로 받고, 나침반 방향(heading, 도)이 있으면 절대 방향으로 바꿔 두었다가 말할 때의
사용자 방향으로 다시 바꾼다. 나침반이 없으면(영상) 사진 기준 방향을 그대로 쓴다.
나침반이 없으면 둘러보는 동안 몸을 돌린 정도를 모르므로, 문패 비교는 같은 사진 안의 문패끼리만 한다.
"""
from __future__ import annotations

import math
import re

REL_DEG = {"L": -90, "F": 0, "C": 0, "R": 90, "B": 180}
WORD = {"F": "정면", "L": "왼쪽", "R": "오른쪽", "B": "뒤쪽"}
GO = {"F": "앞으로 가세요.", "L": "왼쪽으로 가세요.", "R": "오른쪽으로 가세요.", "B": "뒤로 돌아가세요."}
TURN = {"F": "", "L": "왼쪽으로 돌아 ", "R": "오른쪽으로 돌아 ", "B": "뒤로 돌아 "}
ACT = {"F": "직진", "L": "왼쪽", "R": "오른쪽", "B": "뒤로 돌기"}
STRONG, MEDIUM, WEAK, NONE = 3, 2, 1, 0
STRENGTH_KO = {STRONG: "강", MEDIUM: "보통", WEAK: "약", NONE: "없음"}
_RANGE = re.compile(r"(\d{3})\s*[~∼\-–]\s*(\d{3})")


def to_abs(rel, heading):
    """사진 기준 방향(L/F/R/B) + 그 사진의 나침반 방향 → 절대 방향(도). 나침반이 없으면 None."""
    return None if heading is None else (heading + REL_DEG[rel]) % 360


def rel_of(abs_deg, now_heading):
    """절대 방향 → 지금 사용자 기준 L/F/R/B."""
    d = (abs_deg - now_heading + 540) % 360 - 180
    return "F" if abs(d) <= 45 else "R" if 45 < d <= 135 else "L" if -135 <= d < -45 else "B"


def evidence(rel, heading, strength, basis, t):
    return {"rel": rel, "abs": to_abs(rel, heading), "strength": strength, "basis": basis, "t": t}


def now_rel(ev, heading):
    """근거를 지금 사용자 기준 방향으로 (나침반 둘 다 있으면 다시 계산, 아니면 받은 그대로)."""
    if ev["abs"] is not None and heading is not None:
        return rel_of(ev["abs"], heading)
    return ev["rel"]


def board_hit(ranges, target):
    """호실 범위 안내판에서 목적지가 든 범위의 방향 (L/F/R/B), 없으면 None."""
    t = int(target)
    for r in ranges or []:
        if r["lo"] <= t <= r["hi"] and r["to"] in REL_DEG and r["to"] != "C":
            return r
    return None


def arrow_hit(arrows, target):
    """화살표 표지 글자에 목적지 번호나 목적지가 든 범위가 있으면 그 화살표 방향."""
    t = int(target)
    for a in arrows or []:
        lab, to = str(a.get("label") or ""), a.get("to")
        if to not in ("L", "R", "F", "B"):
            continue
        if target in re.findall(r"\d{3}", lab) or any(int(lo) <= t <= int(hi) for lo, hi in _RANGE.findall(lab)):
            return a
    return None


def wall_votes(reads, target):
    """둘러보는 동안 읽은 문패(나침반 방향·좌우·거리)를 같은 벽끼리 묶어, 앞뒤 두 문패 중 더 먼 쪽이 목적지 번호에
    가까우면 그쪽 방향에 한 표. 반환: (사진 기준 방향 'F'/'B' 또는 절대 방향 도, 표 수, 근거 글) 또는 None.
    reads: [{"h": 나침반 방향 또는 None, "side": "L"/"R", "m": 거리, "n": 번호, "photo": 사진 번호}].
    나침반이 없으면 같은 사진 안의 문패끼리만 비교하고 결과는 그 사진 기준 방향(F/B)."""
    t = int(target)
    has_heading = any(r["h"] is not None for r in reads)
    pts = []
    for r in reads:
        if r["side"] not in ("L", "R") or r["m"] is None or not re.fullmatch(r"\d{3}", r["n"]) or r["n"] == target:
            continue
        h = r["h"] if r["h"] is not None else 0.0
        pts.append({**r, "h0": h, "wall": (h + (90 if r["side"] == "R" else -90)) % 360})
    if len(pts) < 2:
        return None
    axis = pts[0]["h0"]
    groups = []
    for p in pts:
        p["pos"] = p["m"] * math.cos(math.radians(p["h0"] - axis))
        g = next((g for g in groups if abs((g[0]["wall"] - p["wall"] + 540) % 360 - 180) <= 60
                  and (has_heading or g[0].get("photo") == p.get("photo"))), None)
        if g is None:
            groups.append([p])
        elif p["n"] not in [x["n"] for x in g]:
            g.append(p)
    votes, why = {"fwd": 0, "back": 0}, []
    for g in groups:
        g = sorted(g, key=lambda x: x["pos"])
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if b["pos"] - a["pos"] < 1.0 or abs(int(a["n"]) - t) == abs(int(b["n"]) - t):
                    continue
                k = "fwd" if abs(int(b["n"]) - t) < abs(int(a["n"]) - t) else "back"
                votes[k] += 1
                why.append(f"{a['n']}→{b['n']}")
    if votes["fwd"] == votes["back"]:
        return None
    k = max(votes, key=votes.get)
    n = votes[k] - votes["back" if k == "fwd" else "fwd"]
    d = (axis + (0 if k == "fwd" else 180)) % 360
    return {"abs": d if has_heading else None, "rel": "F" if k == "fwd" else "B", "votes": n, "pairs": why}


def sentence(target, rel, strength, long_way=False, floor_known=True):
    """floor_known=False: 아직 층을 모름 (근거가 약하면 '층과 방향을 확인하겠습니다')."""
    w = WORD[rel]
    if strength >= STRONG:
        return f"{target}호는 {w}입니다. {GO[rel]}"
    if strength == MEDIUM:
        return f"{target}호는 {w}일 가능성이 높습니다. {TURN[rel]}천천히 가면서 문패로 확인하겠습니다."
    what = "방향을" if floor_known else "층과 방향을"
    if strength == WEAK:
        return f"{target}호는 {w}일 수 있습니다. {TURN[rel]}천천히 가면서 문패로 {what} 확인하겠습니다."
    if long_way:
        where = {"F": "앞쪽", "L": "왼쪽", "R": "오른쪽", "B": "뒤쪽"}[rel]
        return f"복도가 길게 보이는 {where}으로 천천히 가세요. 문패로 {what} 확인하겠습니다."
    return f"앞으로 천천히 가세요. 문패로 {what} 확인하겠습니다."
