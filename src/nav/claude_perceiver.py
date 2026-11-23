"""Claude 인식: 사진 1장 → 문패·층 표시·장소·엘리베이터·화살표·출입문 (YOLOE·PaddleOCR·Places365·Ollama 대신).

- 사진마다 Claude 호출 1번(see). 목적지와 상관없는 사실만 묻는다(호실 목적지일 때). 그래서 같은 프레임 결과를
  경로·목적지가 달라도 다시 쓸 수 있다(캐시). 호실이 아닌 목적지(화장실 등)는 dest 칸을 더해 따로 묻는다.
- 중요한 문패(목적지 근처 번호, 층을 처음 정할 번호, 층 충돌 번호)는 원본 해상도에서 잘라 확대한 조각으로
  한 번 더 읽는다(recheck). 8장 OCR 투표 대신 쓴다.
- 판단·안내 문장은 규칙(navigator)이 만든다. (2단계의 say 칸은 3단계에서 뺌: 출력 토큰 줄이기)
- 2장 방식: 같은 1초의 앞·뒤 절반에서 고른 2장을 한 번의 호출에 넣고, 문패마다 어느 사진(img)에서 읽었는지 받는다.

출력은 짧은 영문 키로 받는다(출력 토큰을 줄여 응답 시간을 줄이려고).
"""
from __future__ import annotations

import base64
import json
import re
import threading
from pathlib import Path

from PIL import Image

from claude_client import jpeg_block

SIDES = ["L", "C", "R"]
PLACES = ["corridor", "lobby", "elev_hall", "entrance", "junction", "room_door", "in_elev", "stairs", "unknown"]
PLACE_KO = {"corridor": "복도", "lobby": "로비", "elev_hall": "엘리베이터 앞", "entrance": "건물 출입문 앞",
            "junction": "복도 갈림길", "room_door": "방 문 앞", "in_elev": "엘리베이터 안", "stairs": "계단 근처"}
SIDE_KO = {"L": "왼쪽", "C": "정면", "R": "오른쪽"}
CLOCK = {"L": 11, "C": 12, "R": 1}


def _obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props), "additionalProperties": False}


_S = {"type": "string"}
_N = {"type": "number"}
# 출력 토큰을 줄이려고 키를 짧게 하고 엘리베이터는 칸 몇 개로 폈다(2026-11-18). 받은 뒤 clean()이 안쪽 형식으로 바꾼다.
#  문패 side는 box 가운데로 정하므로 묻지 않는다.
PLATE = _obj({"n": _S, "name": _S, "m": _N, "on": {"type": "string", "enum": ["door", "wall", "board"]},
              "box": {"type": "array", "items": {"type": "integer"}}})
EL_WHY = ["button", "display", "doors", "hall"]  # 2026-11-19 사용자 지시 8: 근거 종류 넷 (doors·hall은 후보 근거만)
EL_STRONG = ("button", "display")              # 확정에 쓰는 근거
SEE = {"plates": {"type": "array", "items": PLATE}, "fs": _S, "ss": _S,
       "place": {"type": "string", "enum": PLACES},
       "door": {"type": "string", "enum": ["glass", "door", "none"]},
       "el": {"type": "array", "items": _obj({"why": {"type": "string", "enum": EL_WHY},
                                               "side": {"type": "string", "enum": SIDES}, "m": _N,
                                               "btn": {"type": "string", "enum": ["L", "R", "none"]}})},
       "arrows": {"type": "array", "items": _obj({"to": {"type": "string", "enum": ["L", "R", "F", "B"]}, "label": _S})}}
DEST = _obj({"seen": {"type": "boolean"}, "side": {"type": "string", "enum": SIDES + ["none"]}, "m": _N,
             "why": {"type": "string", "enum": ["plate", "pictogram", "text", "none"]}})
SEE_SCHEMA = _obj(SEE)
SEE_DEST_SCHEMA = _obj({**SEE, "dest": DEST})
PLATE2 = _obj({**PLATE["properties"], "img": {"type": "integer", "enum": [1, 2]}})
SEE2_SCHEMA = _obj({**SEE, "plates": {"type": "array", "items": PLATE2}})
SEE2_DEST_SCHEMA = _obj({**SEE, "plates": {"type": "array", "items": PLATE2}, "dest": DEST})
# 판 4 (2026-11-23 사용자 지시 4): 엘리베이터 안 층 표시창 ed 칸을 see()에서 더한다. 판 2·3(캐시 재생)의 스키마·프롬프트는 그대로.
RECHECK_SCHEMA = _obj({"n": _S, "name": _S})
QUICK_SCHEMA = _obj({"plates": {"type": "array", "items": _obj({k: PLATE["properties"][k] for k in ("n", "m", "on", "box")})}})

SYSTEM = """너는 시각장애인 실내 길안내 시스템의 눈이다. 사용자는 가슴 높이에 휴대폰 후면 카메라를 걸고 걷는다. 사진에서 확인되는 사실만 짧게 적는다. 추측하지 않는다.

좌표: 화면 가로를 셋으로 나눠 왼쪽 1/3은 L, 가운데는 C, 오른쪽 1/3은 R. 거리 m은 사용자에서 대상까지 대략의 미터이고, 모르면 -1.

- plates: 방 문 옆이나 위에 붙은 호실 문패. 가까운 것부터 최대 4개. 없으면 빈 목록.
  - n: 호실 번호(숫자와 하이픈만). 가려지거나 흐려 확실하지 않은 자리가 하나라도 있으면 빈 문자열. 전화번호, 게시물·포스터의 숫자, 층 안내판 목록의 숫자는 문패가 아니다.
  - name: 번호를 읽었으면 빈 문자열. 번호를 읽을 수 없을 때만 같은 문패의 방 이름.
  - 번호도 방 이름도 읽을 수 없는 문패는 적지 않는다.
  - m: 문패까지 거리. on: 문 옆이나 문 위면 door, 문과 떨어진 벽이면 wall, 여러 방을 적은 안내판 속이면 board.
  - box: 문패가 차지하는 영역 [왼쪽, 위, 오른쪽, 아래]. 화면 가로와 세로를 각각 100으로 본 정수.
- fs: 지금 층을 나타내는 층 표지(층 번호 표지판, 층 안내판 제목)의 글자가 또렷하게 읽힐 때만 그대로. 아니면 빈 문자열. 엘리베이터 문 위나 옆의 운행 표시(엘리베이터가 지금 있는 층)는 적지 않는다.
- place: 지금 장소. corridor 복도, lobby 로비·넓은 홀, elev_hall 엘리베이터 앞, entrance 건물 출입문, junction 복도 갈림길, room_door 특정 방 문 바로 앞, in_elev 엘리베이터 안, stairs 계단 근처, unknown 판단 불가.
- door: 사용자가 지나가야 할 출입문(건물 출입문, 복도를 막는 문)이 정면 가까이에 있으면 유리문은 glass, 그 밖의 문은 door, 없으면 none. 옆 벽의 방 문은 none.
- ss: 계단 층 표시. 계단참이나 계단 벽에 붙은 층 표시(위층과 아래층이 함께 적힌 표지 포함)의 글자가 또렷할 때만 그대로. 아니면 빈 문자열.
- el: 엘리베이터가 사진에 보이면 가장 가까운 것 하나. 멀리(10m 이상) 작게 보여도 적는다. 아니면 빈 목록.
  why는 가장 강한 근거 하나: 문 옆 호출 버튼이 보이면 button, 층 표시창이 보이면 display, 버튼·표시창은 안 보이지만 금속 양문(가운데가 갈라지는 문) 모양이면 doors, 문은 안 보이지만 엘리베이터 홀 모양이면 hall.
  side, m: 엘리베이터 문(홀이면 그 홀)의 가로 위치와 거리(멀어도 대략의 미터). btn: 호출 버튼이 엘리베이터 문의 왼쪽 벽이면 L, 오른쪽 벽이면 R, 안 보이면 none.
- arrows: 화살표 표지판이 사진에서 모양으로 확인될 때만. to는 화살표가 가리키는 방향(L 왼쪽, R 오른쪽, F 앞, B 뒤), label은 화살표와 함께 적힌 글자(없으면 빈 문자열). 최대 3개."""

TWO_RULE = """

사진 두 장은 같은 1초 안에 앞뒤로 찍은 것이다(1이 먼저, 2가 나중). 두 장을 합쳐 한 번만 적는다.
- 같은 문패가 두 장에 다 보이면 한 번만 적고, 번호가 더 또렷한 사진을 img에 적는다. m, box는 그 사진 기준이다.
- 나머지 칸은 나중 사진(2) 기준으로 적되, 1에만 보인 것도 적는다."""

ED_RULE = """
- ed: 사용자가 엘리베이터 안에 있을 때(place가 in_elev) 엘리베이터 안 층 표시창의 글자가 또렷하면 그대로(예: 숫자 하나나 B와 숫자). 엘리베이터 밖이거나 안 보이면 빈 문자열."""

# 판 5 (2026-11-23 사용자 지시 B7·B8): 호실 범위 안내판 rr, 복도가 길게 보이는 방향 lw.
# in_elev 보충: 휴대폰 기록(11-19 두 번째)에서 밖에서 닫힌 엘리베이터 문을 가까이 본 사진 31장을 in_elev로 읽어
# 자동 하차 감지(B5)가 잘못 켜졌다. 칸 안쪽이 보일 때만 in_elev로 (규칙만으로는 진짜 탑승(11-20 첫 번째)과 구분 안 됨).
V5_RULE = """
- rr: 여러 호실의 번호 범위와 화살표를 함께 적은 안내판이나 층 안내도에서 범위와 방향이 또렷할 때만 [시작 번호, 끝 번호, 방향] 목록. 방향은 화살표가 가리키는 쪽: L 왼쪽, R 오른쪽, F 위쪽 화살표(앞으로), B 뒤로, 화살표가 없거나 아래쪽이면 N. 최대 3개, 없으면 빈 목록.
- lw: 사진에서 복도가 가장 길게 이어져 보이는 방향 L, C, R. 복도가 아니거나 모르면 none.
- place의 in_elev는 엘리베이터 칸 안에 있을 때만(칸 안쪽 벽, 손잡이, 층 버튼 판, 거울 같은 칸 내부가 보일 때). 엘리베이터 밖에서 닫힌 문을 가까이 본 것은 elev_hall."""
V5_PROPS = {"rr": {"type": "array", "items": {"type": "array", "items": _S}},
            "lw": {"type": "string", "enum": ["L", "C", "R", "none"]}}

# 판 6 (2026-11-23 사용자 지시): 엘리베이터 안·밖 구분. 시험 1에서 칸 안에서 닫힌 문을 본 사진 10장 중 9장을 elev_hall로 읽어
# 자동 하차가 잘못 켜짐(안에서 본 닫힌 문과 홀에서 본 닫힌 문이 비슷함). 근거 panel(층 버튼 패널)·cd(닫힌 문만 보임) 추가.
V6_RULE = """
- rr: 여러 호실의 번호 범위와 화살표를 함께 적은 안내판이나 층 안내도에서 범위와 방향이 또렷할 때만 [시작 번호, 끝 번호, 방향] 목록. 방향은 화살표가 가리키는 쪽: L 왼쪽, R 오른쪽, F 위쪽 화살표(앞으로), B 뒤로, 화살표가 없거나 아래쪽이면 N. 최대 3개, 없으면 빈 목록.
- lw: 사진에서 복도가 가장 길게 이어져 보이는 방향 L, C, R. 복도가 아니거나 모르면 none.
- place의 엘리베이터 안·밖 구분:
  - in_elev: 닫힌 엘리베이터 문이 1m 안팎으로 아주 가깝고 문 옆에 층 버튼 패널(숫자 버튼 여러 개)이나 안쪽 층 표시창이 보일 때, 또는 좁은 사방 벽·손잡이·거울 같은 칸 내부가 보일 때. 열린 문 너머로 밖이 보여도 칸 안쪽 벽·손잡이·패널이 함께 보이면 in_elev.
  - elev_hall: 문 옆에 위·아래 호출 버튼만 있거나, 복도·다른 문·문패처럼 엘리베이터 밖의 것이 보일 때.
- el의 why 보충: 위·아래 호출 버튼은 button, 숫자 버튼이 여러 개인 층 버튼 패널은 panel(칸 안). 둘을 섞지 않는다.
- cd: 닫힌 엘리베이터 문이 화면 대부분을 차지하고 안·밖을 가를 단서(층 버튼 패널, 호출 버튼, 손잡이, 거울, 복도, 다른 문, 문패)가 하나도 안 보이면 true, 아니면 false."""
V6_PROPS = {**V5_PROPS, "cd": {"type": "boolean"}}

DEST_RULE = """
- dest: 사용자의 목적지가 사진에 보일 때만 seen true. why는 근거(plate 문패 글자, pictogram 그림 표지, text 그 밖의 글자). 안 보이면 seen false, side none, m -1, why none."""

RECHECK_SYSTEM = """이 사진은 실내 문패 부분을 잘라 확대한 것이다. 문패의 호실 번호를 그대로 적는다(n: 숫자와 하이픈만). 가려지거나 흐려 확실하지 않은 자리가 하나라도 있으면 n은 빈 문자열. name은 같은 문패의 방 이름(없으면 빈 문자열)."""

# 문 앞 빠른 확인 (2026-11-19 사용자 지시 4). 목적지 번호를 알려 주지 않는다(알려 주면 그 번호로 읽으려는 쏠림이 생길 수 있어서).
QUICK_SYSTEM = """사진에서 사용자에게 가장 가까운 방 문패 최대 2개의 호실 번호만 적는다. 없으면 빈 목록.
- n: 호실 번호(숫자와 하이픈만). 가려지거나 흐려 확실하지 않은 자리가 하나라도 있으면 그 문패는 적지 않는다. 전화번호, 게시물 숫자는 문패가 아니다.
- m: 사용자에서 문패까지 대략의 미터(모르면 -1). on: 문 옆이나 문 위면 door, 문과 떨어진 벽이면 wall, 여러 방을 적은 안내판 속이면 board.
- box: 문패 영역 [왼쪽, 위, 오른쪽, 아래]. 화면 가로와 세로를 각각 100으로 본 정수."""

ROOM = re.compile(r"^\d{2,4}(-\d{1,2})?$")


def resize(img, side):
    if side and max(img.size) > side:
        img = img.copy()
        img.thumbnail((side, side), Image.LANCZOS)
    return img


def floor_of_sign(s):
    """'3F', '3층', 'B1' → '3', 'B1'. 모르면 None."""
    s = (s or "").upper().replace(" ", "")
    m = re.search(r"(?:지하|B)(\d)", s)
    if m:
        return "B" + m.group(1)
    m = re.search(r"(\d{1,2})(?:F|층)", s)
    return str(int(m.group(1))) if m else None


def floor_of_display(s):
    """엘리베이터 안 층 표시창 글자 → 층. '4', '4F', 'B1', '지하1' → '4', 'B1'. 모르면 None."""
    s = (s or "").upper().replace(" ", "")
    m = re.fullmatch(r"(?:지하|B)(\d)(?:F|층)?", s)
    if m:
        return "B" + m.group(1)
    m = re.fullmatch(r"(\d{1,2})(?:F|층)?", s)
    return str(int(m.group(1))) if m and int(m.group(1)) > 0 else None


def ranges_of(rr):
    """호실 범위 안내판 [[시작, 끝, 방향], …] → [{"lo", "hi", "to"}] (3자리 번호, 시작 ≤ 끝, 범위 100 미만만, 최대 3개)."""
    out = []
    for x in (rr or [])[:3]:
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            continue
        lo, hi = (re.sub(r"\D", "", str(v)) for v in x[:2])
        to = str(x[2]).upper() if len(x) > 2 else "N"
        if len(lo) == 3 and len(hi) == 3 and int(lo) <= int(hi) and int(hi) - int(lo) < 100:
            out.append({"lo": int(lo), "hi": int(hi), "to": to if to in ("L", "R", "F", "B") else "N"})
    return out


def floors_of_stair_sign(s):
    """계단 층 표시 글자 → 층 목록(낮은 층부터). '4F↗3F' → ['3', '4']. 모르면 []."""
    s = (s or "").upper().replace(" ", "")
    out = ["B" + m for m in re.findall(r"(?:지하|B)(\d)", s)]
    s2 = re.sub(r"(?:지하|B)\d", "", s)
    out += [str(int(m)) for m in re.findall(r"(\d{1,2})(?:F|층)", s2)]
    key = lambda f: -int(f[1:]) if f.startswith("B") else int(f)  # noqa: E731
    return sorted(set(out), key=key)


def clean(data, img_size=None):
    """Claude 답을 정리. 형식이 어긋난 값은 고치거나 버리고, 고친 내용을 fixes에 남긴다."""
    fixes, plates, unread = [], [], []
    for p in (data.get("plates") or [])[:4]:
        n = re.sub(r"[^\d-]", "", p.get("n", ""))
        name = (p.get("name") or "").strip()
        if not ROOM.match(n):
            if n or not name:
                fixes.append(f"문패 번호 버림: {p.get('n')!r}")
                continue
            n = ""  # 이름만 읽힌 문패: 확대 재확인 후보
        box = p.get("box") or []
        if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
            box = None
        else:
            x0, y0, x1, y1 = (max(0, min(100, v)) for v in box)
            box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)] if x1 != x0 and y1 != y0 else None
        m = p.get("m", -1)
        side = p.get("side", "C")  # 2단계 캐시(side 칸이 있던 형식)와 상자 없는 문패용
        if box:  # 화면 가로 위치는 상자 가운데로 다시 정한다 (모델의 side와 상자가 어긋나는 경우가 있음)
            cx = (box[0] + box[2]) / 2
            side2 = "L" if cx < 100 / 3 else "R" if cx > 200 / 3 else "C"
            if "side" in p and side2 != side:
                fixes.append(f"{n or name} 방향 {side}→{side2} (상자 기준)")
            side = side2
        (plates if n else unread).append({"img": p.get("img"), "n": n, "name": name, "side": side,
                       "m": round(float(m), 1) if isinstance(m, (int, float)) and m > 0 else None,
                       "on": p.get("on", "wall"), "box": box})
    seen = set()
    plates = [p for p in plates if not (p["n"] in seen or seen.add(p["n"]))]
    plates.sort(key=lambda p: p["m"] if p["m"] is not None else 99)
    if "elev" not in data:  # 짧은 키 형식(2026-11-18) → 안쪽 형식
        el = (data.get("el") or [{}])[0] if data.get("el") else {}
        data = {**data, "floor_sign": data.get("fs", ""), "door_ahead": data.get("door", "none"),
                "elev": {"seen": bool(el), "why": el.get("why", "none"), "side": el.get("side", "none"),
                         "m": el.get("m", -1), "btn": el.get("btn", "none")}}
    e = data.get("elev") or {}
    panel = e.get("why") == "panel"  # 판 6: 층 버튼 패널 = 칸 안 (타러 갈 엘리베이터 근거가 아님)
    if panel:
        e = {**e, "seen": False}
    elev = {"seen": bool(e.get("seen")) and e.get("why") in EL_STRONG,
            "cand": bool(e.get("seen")) and e.get("why") in EL_WHY, "why": e.get("why", "none"),
            "side": e.get("side", "none"), "m": e.get("m") if isinstance(e.get("m"), (int, float)) and e.get("m", 0) > 0 else None,
            "btn": e.get("btn", "none")}
    if e.get("seen") and not elev["cand"] and not panel:
        fixes.append("엘리베이터 근거 없음 → 안 본 것으로")
    out = {"plates": plates, "unread": unread, "floor_sign": (data.get("floor_sign") or "").strip(),
           "floor": floor_of_sign(data.get("floor_sign")), "place": data.get("place", "unknown"),
           "stair_sign": (data.get("ss") or "").strip(), "stair_floors": floors_of_stair_sign(data.get("ss")),
           "door_ahead": data.get("door_ahead", "none"), "elev": elev,
           "arrows": (data.get("arrows") or [])[:3], "fixes": fixes,
           "elev_floor": floor_of_display(data.get("ed")) if data.get("place") == "in_elev" else None,
           "ranges": ranges_of(data.get("rr")), "long_way": data.get("lw") if data.get("lw") in ("L", "C", "R") else None,
           "panel": panel, "closed_doors": bool(data.get("cd"))}
    if "dest" in data:
        d = data["dest"]
        out["dest"] = {"seen": bool(d.get("seen")), "side": d.get("side", "none"),
                       "m": d.get("m") if isinstance(d.get("m"), (int, float)) and d.get("m", 0) > 0 else None,
                       "why": d.get("why", "none")}
    return out


class Perceiver:
    """client: claude_client의 GatewayClient / FakeClient / DryRunClient.
    send_side: Claude에 보낼 사진 긴 변(None = 원본). cache: 같은 프레임·같은 설정이면 다시 부르지 않음(jsonl)."""

    def __init__(self, client, send_side=None, cache_path=None, dest_text=None, version=6, quality=90, pass_kb=120):
        # version: 인식 프롬프트 판. 6 = 2026-11-23 (엘리베이터 안·밖 기준, panel, cd). 5 = 2026-11-23 (호실 범위 안내판 rr, 긴 복도 방향 lw). 4 = 2026-11-23 (엘리베이터 안 층 표시창 ed). 3 = 2026-11-19 (계단 층 표시 ss,
        # 엘리베이터 근거 4종·먼 거리). 캐시 키에 붙여
        # 이전 프롬프트 결과와 섞이지 않게 한다. 2 = 이전 캐시(claude_v2.jsonl)를 그대로 재생할 때.
        self.client, self.send_side, self.dest_text, self.version = client, send_side, dest_text, version
        # quality: Claude에 보내는 JPEG 품질 (2026-11-23 사용자 지시 A2: 휴대폰과 같은 수준으로, 해상도는 원본 유지).
        # 이미 pass_kb 이하인 JPEG 파일(휴대폰이 품질 0.7로 올린 사진)은 다시 압축하지 않고 그대로 보낸다.
        # 확대 재확인 조각은 이 값과 상관없이 원본에서 잘라 품질 90으로 보낸다.
        self.quality, self.pass_kb = quality, pass_kb
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache, self.lock = {}, threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r

    def _key(self, kind, frame, extra=""):
        q = f"|q{self.quality}" if self.quality != 90 and kind != "recheck" else ""  # 품질 90(이전 캐시)은 키 그대로
        return f"{kind}|{getattr(self.client, 'model', '')}|{self.send_side}|{Path(frame).parent.name}/{Path(frame).name}|{extra}{q}"

    def _block(self, frame, img=None):
        """보낼 사진 블록. 품질 90이면 전처럼 다시 저장, 아니면 작은 JPEG는 그대로, 큰 것은 그 품질로 다시 저장."""
        if self.quality != 90 and img is None and self.send_side is None:
            raw = Path(frame).read_bytes()
            if raw[:2] == b"\xff\xd8" and len(raw) <= self.pass_kb * 1000:
                return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(raw).decode()}}
        return jpeg_block(resize(img or Image.open(frame).convert("RGB"), self.send_side), quality=self.quality)

    def _store(self, key, res):
        rec = {"key": key, "data": res.data, "log": res.log()}
        if res.ok:  # 실패한 호출은 저장하지 않음 (다음 실행에서 다시 부름)
            with self.lock:
                self.cache[key] = rec
                if self.cache_path:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def see(self, frame, img=None):
        """frame: 사진 경로 1개 또는 [앞, 뒤] 2개. 반환: {"obs": 정리한 인식(실패면 None), "call": 호출 기록, "cached": bool}
        2장이면 문패마다 읽은 사진 경로를 "frame"에 붙인다(확대 재확인에 씀)."""
        dest = self.dest_text
        frames = list(frame) if isinstance(frame, (list, tuple)) else [frame]
        two = len(frames) == 2
        key = self._key("see2" if two else "see", frames[-1], (Path(frames[0]).name + "+" if two else "") + (dest or "")
                        + (f"|p{self.version}" if self.version >= 3 else ""))
        if key in self.cache:
            r = self.cache[key]
            return {"obs": self._attach(clean(r["data"]), frames), "call": r["log"], "cached": True}
        v4 = self.version >= 4
        v5 = self.version >= 5
        v6 = self.version >= 6
        system = SYSTEM + (ED_RULE if v4 else "") + (V6_RULE if v6 else V5_RULE if v5 else "") + \
            (DEST_RULE if dest else "") + (TWO_RULE if two else "")
        if two:
            content = []
            for i, f in enumerate(frames, 1):
                content += [{"type": "text", "text": f"사진 {i}"}, self._block(f)]
            schema = SEE2_DEST_SCHEMA if dest else SEE2_SCHEMA
        else:
            content = [self._block(frames[0], img)]
            schema = SEE_DEST_SCHEMA if dest else SEE_SCHEMA
        if v4:
            schema = {**schema, "properties": {**schema["properties"], "ed": _S}, "required": [*schema["required"], "ed"]}
        if v5:
            props = V6_PROPS if v6 else V5_PROPS
            schema = {**schema, "properties": {**schema["properties"], **props}, "required": [*schema["required"], *props]}
        if v6:  # el.why에 panel 추가
            el = schema["properties"]["el"]
            item = {**el["items"], "properties": {**el["items"]["properties"], "why": {"type": "string", "enum": EL_WHY + ["panel"]}}}
            schema = {**schema, "properties": {**schema["properties"], "el": {**el, "items": item}}}
        content.append({"type": "text", "text": f"사용자의 목적지: {dest}" if dest else "보이는 것을 적어라."})
        res = self.client.call(system, content, schema, max_tokens=500, kind="see")
        rec = self._store(key, res)
        return {"obs": self._attach(clean(res.data), frames) if res.ok else None, "call": rec["log"], "cached": False}

    @staticmethod
    def _attach(obs, frames):
        for p in obs["plates"] + obs["unread"]:
            i = p.get("img") if p.get("img") in (1, 2) else len(frames)
            p["frame"] = str(frames[min(i, len(frames)) - 1])
        return obs

    def quick(self, frame):
        """문 앞 빠른 확인: 가장 가까운 문패 번호만 짧게 읽는다. 반환: {"plates"(clean 형식, 실패면 None), "call", "cached"}"""
        key = self._key("quick", frame)
        if key in self.cache:
            r = self.cache[key]
            return {"plates": self._attach(clean(r["data"]), [frame])["plates"], "call": r["log"], "cached": True}
        content = [self._block(frame), {"type": "text", "text": "가장 가까운 문패 번호를 적어라."}]
        res = self.client.call(QUICK_SYSTEM, content, QUICK_SCHEMA, max_tokens=120, kind="quick")
        rec = self._store(key, res)
        plates = self._attach(clean(res.data), [frame])["plates"] if res.ok else None
        return {"plates": plates, "call": rec["log"], "cached": False}

    def recheck(self, frame, plate):
        """문패 조각을 원본 해상도에서 잘라 확대해 다시 읽는다. 반환: {"n", "name", "match", "call", "cached"}"""
        if not plate.get("box"):
            return None
        key = self._key("recheck", frame, ",".join(str(v) for v in plate["box"]))
        if key in self.cache:
            r = self.cache[key]
            data, log, cached = r["data"], r["log"], True
        else:
            img = Image.open(frame).convert("RGB")
            w, h = img.size
            x0, y0, x1, y1 = (v / 100 for v in plate["box"])
            px, py = (x1 - x0) * 1.0 + 0.04, (y1 - y0) * 1.0 + 0.03
            crop = img.crop((int(max(0, x0 - px) * w), int(max(0, y0 - py) * h),
                             int(min(1, x1 + px) * w), int(min(1, y1 + py) * h)))
            if max(crop.size) < 512:  # 작은 조각은 키워서 보냄
                s = 512 / max(crop.size)
                crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
            res = self.client.call(RECHECK_SYSTEM, [jpeg_block(crop)], RECHECK_SCHEMA, max_tokens=60, kind="recheck")
            rec = self._store(key, res)
            data, log, cached = res.data, rec["log"], False
        if not data:
            return {"n": None, "name": "", "match": None, "call": log, "cached": cached}
        n = re.sub(r"[^\d-]", "", data.get("n", ""))
        n = n if ROOM.match(n) else None
        return {"n": n, "name": data.get("name", ""), "match": (n == plate["n"]) if plate["n"] and n else None,
                "call": log, "cached": cached}
