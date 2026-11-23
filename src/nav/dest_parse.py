"""목적지 말 → 호실 번호 (규칙, Claude 호출 없음).

받아쓰기 결과는 여러 모양으로 온다: "410호로 가고 싶어", "410%"(아이폰 받아쓰기가 '호'를 %로 씀),
"사백십호", "사일공호", "4 1 0 호", "삼백 십칠 호". 모두 3자리 호실 번호로 바꾼다.
번호를 못 찾으면 None → 웹이 "다시 말씀해 주세요"라고 말한다.
지금 길안내 규칙(navigator)은 3자리 호실 목적지만 시험했으므로 호실 번호만 받는다.
"""
from __future__ import annotations

import re

DIG = {"영": 0, "공": 0, "빵": 0, "일": 1, "하나": 1, "이": 2, "둘": 2, "삼": 3, "셋": 3, "사": 4, "넷": 4,
       "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
UNIT = {"십": 10, "백": 100, "천": 1000}
_KNUM = re.compile(r"(?:하나|둘|셋|넷|[영공빵일이삼사오육륙칠팔구십백천])+")
_FLOOR_ONLY = re.compile(r"^\s*(\d+|[일이삼사오육칠팔구십]+)\s*층")


def sino(word: str) -> int | None:
    """'사백십' → 410, '삼백십칠' → 317, '사일공' → 410 (자릿수 읽기)."""
    if not word:
        return None
    if any(u in word for u in UNIT):
        total, cur, i = 0, None, 0
        while i < len(word):
            two = word[i:i + 2]
            if two in ("하나", "둘", "셋", "넷"):  # 고유어 숫자는 단위 앞에 거의 안 오지만 받아쓰기 대비
                cur, i = DIG[two], i + 2
                continue
            ch = word[i]
            if ch in DIG:
                cur = DIG[ch]
            elif ch in UNIT:
                total += (cur if cur is not None else 1) * UNIT[ch]
                cur = None
            else:
                return None
            i += 1
        return total + (cur or 0)
    digits, i = "", 0
    while i < len(word):
        two = word[i:i + 2]
        if two in ("하나", "둘", "셋", "넷"):
            digits, i = digits + str(DIG[two]), i + 2
            continue
        if word[i] not in DIG:
            return None
        digits, i = digits + str(DIG[word[i]]), i + 1
    return int(digits) if digits else None


def parse_room(text: str) -> str | None:
    """말한 문장에서 3자리 호실 번호. 없으면 None."""
    t = (text or "").strip()
    if not t:
        return None
    t = t.replace("%", "호").replace("％", "호")
    t = re.sub(r"(?<=\d)[\s,.-]+(?=\d)", "", t)          # "4 1 0" → "410"
    m = re.search(r"(?<!\d)(\d{3})(?!\d)", t)
    if m:
        return m.group(1)
    # 한글 숫자: 붙여 쓴 덩어리부터 (띄어 쓴 "삼백 십칠"도 붙여서 한 번 더)
    for s in (re.sub(r"\s+", "", t), t):
        for run in sorted(_KNUM.findall(s), key=len, reverse=True):
            v = sino(run)
            if v is not None and 100 <= v <= 999:
                return str(v)
    return None


# 지금 층 말하기 (2026-11-23 사용자 지시 2): "4층인데 410호", "4층에서 410호", 안내 중 "여기 4층이야".
# 목적지와 같이 말할 때는 '지금 층'임을 나타내는 말이 붙어야 지금 층으로 본다("4층 410호"는 목적지 층일 수 있어서).
_FLOOR_WORD = r"(지하\s*\d|[bB]\s*\d|\d{1,2}|지하\s*[일이삼]|[일이삼사오육칠팔구십]{1,3})\s*층"
_HERE_AFTER = r"\s*(?:인데|이고|이니까|이니|이야|이에요|예요|입니다|이요|요|에서|에\s*있|에\s*왔)"
_HERE_BEFORE = r"(?:여기|지금|현재)\s*(?:는|은)?\s*"


def _floor_value(w: str) -> str | None:
    w = re.sub(r"\s+", "", w)
    b = re.fullmatch(r"(?:지하|[bB])(\d|[일이삼])", w)
    if b:
        d = b.group(1)
        return "B" + (d if d.isdigit() else str(DIG[d]))
    if w.isdigit():
        return str(int(w)) if int(w) > 0 else None
    v = sino(w)
    return str(v) if v and 0 < v < 100 else None


def parse_here_floor(text: str, alone: bool = False) -> str | None:
    """말한 문장에서 '지금 층'. alone=True(호실 번호 없이 층만 말함)이면 "4층", "사층이요"도 지금 층으로 본다."""
    t = (text or "").strip()
    for pat in (_HERE_BEFORE + _FLOOR_WORD, _FLOOR_WORD + _HERE_AFTER) + ((_FLOOR_WORD,) if alone else ()):
        m = re.search(pat, t)
        if m:
            v = _floor_value(m.group(1))
            if v:
                return v
    return None


def parse_destination(text: str) -> dict | None:
    """{"원문", "target": "410", "name": "410호", "floor": "4", "here": "1" 또는 None} 또는 None.
    here: 같이 말한 지금 층 ("1층인데 410호" → "1")."""
    room = parse_room(text)
    if room is None:
        return None
    return {"원문": (text or "").strip(), "target": room, "name": f"{room}호", "floor": room[0],
            "here": parse_here_floor(text)}
