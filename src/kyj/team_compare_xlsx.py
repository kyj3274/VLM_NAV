"""team_compare.py 결과 + Claude 기준으로 팀원 4층 표(4F_4종_비교.xlsx)와 같은 형식의 색칠한 엑셀을 만든다 (Mac).
색: 초록 Claude에도 있음 / 빨강 굵게 Claude에 없음 / 주황 글자가 비슷하지만 틀림 / 파랑 굵게(Claude 열) 둘 다 놓침 / 회색 없음·실패
같은 종류로 묶는 이름: 문패=표지판, TV=전자칠판, 포스터=게시판 (팀원 규칙) + 입간판=게시판, 엘리베이터 문=엘리베이터, 쓰레기통=수거함.
위치는 비교하지 않고 개수로 센다.

  .venv/bin/python src/kyj/team_compare_xlsx.py results.json claude_ref.json 3F_4종_비교.xlsx --floor 3F
"""

import argparse
import difflib
import json
import re
from collections import Counter

from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill

GREEN, RED, ORANGE, BLUE, GRAY, BLACK = "0000A040", "00E00000", "00E08000", "001F4FD0", "00909090", "00000000"
NAV = {"문", "호실 문패", "엘리베이터", "계단", "비상구 표시", "화장실", "사람", "소화기"}  # 길안내에 필요한 것
SYN = [  # (묶음 이름, 이름에 들어 있으면 이 묶음) 위에서부터 먼저 맞는 것
    ("엘리베이터", ["엘리베이터"]), ("호실 문패", ["문패", "표지판", "명패", "안내판"]), ("창문", ["창문", "유리창"]),
    ("게시물", ["게시물", "종이", "안내문", "공지"]), ("게시판", ["게시판", "포스터", "입간판", "배너", "현수막"]),
    ("수거함", ["쓰레기", "휴지통", "수거함", "분리수거"]), ("전자칠판", ["TV", "전자칠판", "모니터"]),
    ("사람", ["사람", "남성", "남자", "여성", "보행자", "인물"]), ("CCTV", ["CCTV", "cctv", "감시카메라"]),
    ("비상구 표시", ["비상구"]), ("소화기", ["소화기"]), ("우산", ["우산"]), ("계단", ["계단"]), ("화장실", ["화장실"]),
]


def canon(name):
    for c, keys in SYN:
        if any(k in name for k in keys):
            return c
    if name in ("문", "도어") or (name.endswith("문") and len(name) <= 4):  # 출입문, 철문, 방문, 유리문, 도어(Qwen)
        return "문"
    return name


def parse_ref(s):
    """'문×4(왼쪽/왼쪽/오른쪽/오른쪽), 호실 문패(가운데)' → [(이름, [위치...])]"""
    out = []
    for it in [x.strip() for x in s.split(", ") if x.strip()]:
        m = re.match(r"^(.*?)(?:×(\d+))?(?:\((.*)\))?$", it)
        name, n, pos = m.group(1), int(m.group(2) or 1), (m.group(3) or "").split("/")
        out.append((name, (pos + [""] * n)[:n]))
    return out


def group(items):
    """[{name, position}] → [(이름, [위치...])] 순서 유지"""
    g = {}
    for d in items:
        g.setdefault(d["name"], []).append(d.get("position", ""))
    return list(g.items())


def fmt(name, pos):
    p = "/".join(x for x in pos if x)
    return f"{name}{'×' + str(len(pos)) if len(pos) > 1 else ''}{f'({p})' if p else ''}"


def rich(blocks):
    """[(글자, 색, 굵게)] → 셀 값. ', '로 잇는다."""
    if not blocks:
        return CellRichText([TextBlock(InlineFont(color=GRAY), "없음")])
    parts = []
    for i, (t, color, bold) in enumerate(blocks):
        if i:
            parts.append(TextBlock(InlineFont(color=BLACK), ", "))
        parts.append(TextBlock(InlineFont(color=color, b=bold), t))
    return CellRichText(parts)


def judge_objects(sys_groups, ref_count):
    """시스템 물체 → (색칠 블록, 맞은 수 Counter, 틀린 수, 길안내 맞음, 길안내 틀림)"""
    left = Counter(ref_count)
    blocks, hit, wrong, nav_hit, nav_wrong = [], Counter(), 0, 0, 0
    for name, pos in sys_groups:
        c = canon(name)
        k = min(len(pos), left[c])
        left[c] -= k
        hit[c] += k
        if k:
            blocks.append((fmt(name, pos[:k]), GREEN, False))
        if len(pos) > k:
            blocks.append((fmt(name, pos[k:]), RED, True))
            wrong += len(pos) - k
            nav_wrong += (len(pos) - k) if c in NAV else 0
        nav_hit += k if c in NAV else 0
    return blocks, hit, wrong, nav_hit, nav_wrong


def norm(t):
    return re.sub(r"\s+", "", t)


def similar(a, b):
    a, b = norm(a), norm(b)
    if a.isdigit() and b.isdigit() and len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def judge_texts(texts, ref):
    """글자 → (블록, 정확히 맞힌 Claude 글자 번호 집합, 정확, 비슷, 틀림)"""
    used, blocks, ok, near, bad = set(), [], 0, 0, 0
    for t in texts:
        j = next((i for i, r in enumerate(ref) if i not in used and norm(r) == norm(t)), None)
        if j is not None:
            used.add(j); ok += 1; blocks.append((t, GREEN, False))
        elif any(similar(t, r) for r in ref):
            near += 1; blocks.append((t, ORANGE, False))
        else:
            bad += 1; blocks.append((t, RED, True))
    return blocks, used, ok, near, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--floor", default="3F")
    a = ap.parse_args()
    rows = json.load(open(a.results, encoding="utf-8"))
    ref = json.load(open(a.ref, encoding="utf-8"))

    wb = Workbook()
    summary = wb.active
    summary.title = "요약"
    ws = wb.create_sheet("4종 비교")
    head = ["사진", "YOLOE 물체", "OCR 글자", "Qwen3-VL 4B 물체", "Qwen 글자", "Claude 물체 (기준)", "Claude 글자 (기준)",
            "YOLOE 우리 7종 (참고)"]
    for i, (h, w) in enumerate(zip(head, [16, 42, 22, 42, 22, 50, 26, 34]), 1):
        c = ws.cell(1, i, h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
        ws.column_dimensions[c.column_letter].width = w
    ws.freeze_panes = "B2"

    S = {k: Counter() for k in ("yoloe", "qwen", "ours")}
    T = {k: Counter() for k in ("ocr", "qwen")}
    ref_total = ref_nav = ref_text_total = 0
    qwen_fail = 0
    for r in rows:
        rf = ref[str(r["sec"])]
        ref_groups = parse_ref(rf["objects"])
        ref_count = Counter()
        for name, pos in ref_groups:
            ref_count[canon(name)] += len(pos)
        ref_total += sum(ref_count.values())
        ref_nav += sum(v for k, v in ref_count.items() if k in NAV)
        ref_text_total += len(rf["texts"])

        yb, yhit, yw, ynh, ynw = judge_objects(group(r["yoloe_team"]), ref_count)
        ob, ohit, ow, onh, onw = judge_objects(group(r["yoloe_ours"]), ref_count)
        q = r["qwen"]
        if q is None:
            qwen_fail += 1
            qb, qhit, qw, qnh, qnw = [(f"실패 ({(r['qwen_error'] or '')[:30]})", GRAY, True)], Counter(), 0, 0, 0
            qtb, qused, qt = [("-", GRAY, False)], set(), (0, 0, 0)
        else:
            qb, qhit, qw, qnh, qnw = judge_objects(group(q.get("objects", [])), ref_count)
            qtb, qused, *qt = judge_texts(q.get("texts", []), rf["texts"])
        otb, oused, *ot = judge_texts(r["ocr"], rf["texts"])

        for key, hit, w, nh, nw in (("yoloe", yhit, yw, ynh, ynw), ("qwen", qhit, qw, qnh, qnw), ("ours", ohit, ow, onh, onw)):
            S[key]["hit"] += sum(hit.values()); S[key]["wrong"] += w; S[key]["nav_hit"] += nh; S[key]["nav_wrong"] += nw
        if q is not None:
            S["qwen"]["ok_frames"] += 1
        for key, t in (("ocr", ot), ("qwen", qt)):
            T[key]["ok"] += t[0]; T[key]["near"] += t[1]; T[key]["bad"] += t[2]

        # Claude 열: YOLOE(팀 목록)·Qwen 둘 다 못 찾은 만큼 파랑 굵게
        cb, found = [], Counter({k: max(yhit[k], qhit[k]) for k in set(yhit) | set(qhit)})
        for name, pos in ref_groups:
            c = canon(name)
            k = min(len(pos), found[c])
            found[c] -= k
            if k:
                cb.append((fmt(name, pos[:k]), BLACK, False))
            if len(pos) > k:
                cb.append((fmt(name, pos[k:]), BLUE, True))
        ctb = [(t, BLACK if (i in oused or i in qused) else BLUE, not (i in oused or i in qused))
               for i, t in enumerate(rf["texts"])]

        row = [f"{a.floor}_t{r['sec']:04d}.0s.jpg", rich(yb), rich(otb), rich(qb), rich(qtb), rich(cb), rich(ctb), rich(ob)]
        n = ws.max_row + 1
        for i, v in enumerate(row, 1):
            c = ws.cell(n, i, v)
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def pct(x, y):
        return f"{round(100 * x / y)}%" if y else "-"

    ok_q = S["qwen"]["ok_frames"]
    lines = [
        ["색 범례"], ["초록", "Claude 결과에도 있음 (맞은 것으로 봄)"], ["빨강 굵게", "Claude 결과에 없음 (오탐 의심)"],
        ["주황", "글자가 비슷하지만 틀림 (예: 412→512, 강의실→경의실)"], ["파랑 굵게 (Claude 열)", "YOLOE·Qwen 둘 다 놓친 것"],
        ["회색", "없음 / Qwen 실패"], [],
        ["물체", "YOLOE", "Qwen3-VL 4B", "YOLOE 우리 7종 (참고)"],
        ["맞은 항목 (초록)", S["yoloe"]["hit"], S["qwen"]["hit"], S["ours"]["hit"]],
        ["틀린 항목 (빨강)", S["yoloe"]["wrong"], S["qwen"]["wrong"], S["ours"]["wrong"]],
        ["정밀도 (초록 비율)"] + [pct(S[k]["hit"], S[k]["hit"] + S[k]["wrong"]) for k in ("yoloe", "qwen", "ours")],
        [f"Claude 항목 중 찾은 비율 (총 {ref_total}개)"] + [pct(S[k]["hit"], ref_total) for k in ("yoloe", "qwen", "ours")],
        [],
        ["길안내 물체만 (문·문패·엘리베이터·계단·비상구·화장실·사람·소화기)", "YOLOE", "Qwen3-VL 4B", "YOLOE 우리 7종 (참고)"],
        ["맞은 항목", S["yoloe"]["nav_hit"], S["qwen"]["nav_hit"], S["ours"]["nav_hit"]],
        ["틀린 항목", S["yoloe"]["nav_wrong"], S["qwen"]["nav_wrong"], S["ours"]["nav_wrong"]],
        ["정밀도"] + [pct(S[k]["nav_hit"], S[k]["nav_hit"] + S[k]["nav_wrong"]) for k in ("yoloe", "qwen", "ours")],
        [f"Claude 길안내 항목 중 찾은 비율 (총 {ref_nav}개)"] + [pct(S[k]["nav_hit"], ref_nav) for k in ("yoloe", "qwen", "ours")],
        [],
        ["글자", "OCR", "Qwen3-VL 4B"],
        ["정확 (초록)", T["ocr"]["ok"], T["qwen"]["ok"]],
        ["비슷하지만 틀림 (주황)", T["ocr"]["near"], T["qwen"]["near"]],
        ["틀림·지어냄 (빨강)", T["ocr"]["bad"], T["qwen"]["bad"]],
        [f"Claude가 읽은 글자 중 맞힌 비율 (총 {ref_text_total}개)", pct(T["ocr"]["ok"], ref_text_total), pct(T["qwen"]["ok"], ref_text_total)],
        [],
        ["주의", "Claude 결과는 정답표가 아니라 기준 삼은 것. 같은 이름의 물체는 같은 종류로 묶어 비교(문패=표지판, TV=전자칠판, "
                "포스터=게시판, 입간판=게시판, 엘리베이터 문=엘리베이터, 쓰레기통=수거함). 위치는 비교하지 않음. "
                f"Qwen은 {len(rows)}장 중 {qwen_fail}장이 실패"
                + (f"해서, 정밀도·글자 개수는 성공한 {ok_q}장만 셌고 \"찾은 비율\"은 전체 기준(실패=못 찾음)." if qwen_fail else ".")
                + " YOLOE는 팀원 표에 나온 13종 이름(점수 0.25 이상), 우리 7종은 길안내 파이프라인 설정(점수 0.15 이상). "
                  "Qwen은 JSON 형식을 강제하고 물체 20개·글자 15개로 제한해서 물었음(팀원과 질문이 다를 수 있음). "
                  "제한 없이 물었을 때는 같은 물체를 끝없이 나열하다 잘려 JSON이 깨졌음(팀원 표의 실패와 같은 원인으로 보임). "
                  "제한해도 대부분 20칸을 같은 이름 반복(예: 도어 핸들×17, 101~115 지어냄)으로 채워서 빨강이 많음. 사진은 각 초의 첫 장."],
    ]
    for line in lines:
        summary.append(line)
    for cell in ("A1", "A8", "A14", "A20", "A26"):
        summary[cell].font = Font(bold=True)
    for col, w in (("A", 40), ("B", 45), ("C", 18), ("D", 22)):
        summary.column_dimensions[col].width = w
    summary["B26"].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(a.out)
    print(a.out)
    for line in lines[7:25]:
        if line:
            print(" | ".join(str(x) for x in line))


if __name__ == "__main__":
    main()
