"""관측 기록(영상·시연의 걸음별 기록)으로 장면 그래프 초안을 만든다 (Mac, 모델 없음).

점(node): 구간(엘리베이터 홀, 복도 …), 문패, 문, 안내도 후보
선(edge): 구간 → 구간(이어짐), 문패 → 구간(왼쪽/오른쪽 벽), 문패 → 문(붙어 있음), 문패 → 문패(같은 벽의 다음 번호)
그 위에서 벽별 번호 흐름(앞으로 갈수록 커지나/작아지나)과 목적지 방향을 계산한다.

  .venv/bin/python src/eval/scene_graph.py data/video/results/locate/guide_314_1_2b_situ_sharp.jsonl --target 318 --out <폴더>
결과: <폴더>/<이름>.json (그래프), .txt (텍스트 모델용 글), .png (그림, graphviz)
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))  # 서비스 모듈 (src/nav)
import nav_memory  # noqa: E402

ROOM = re.compile(r"^[1-9]\d{2}$")
FLOOR_MARK = re.compile(r"^[1-9]F$|^[1-9]층$")
SIDE_L, SIDE_R = 0.45, 0.55  # 화면 가로 위치로 벽 구분 (초안: 휴대폰을 돌리면 틀림)


def near_door(objs, cx, cy):
    """문패 글자 아래·옆의 문 상자 (navigator.door_distance와 같은 조건)."""
    for d in objs:
        if d["name"] not in ("문", "엘리베이터") or d["conf"] < 0.15:
            continue
        x0, y0, x1, y1 = d["box"]
        if x0 - 0.12 <= cx <= x1 + 0.12 and y0 - 0.15 <= cy <= (y0 + y1) / 2:
            return [round(v, 2) for v in d["box"]]
    return None


def build(records, target):
    mem = nav_memory.Memory(target)
    g = {"target": target, "segments": [], "plates": {}, "doors": [], "maps": [], "edges": []}
    seg_of_seq, seg = {}, None

    def new_seg(t, kind, seq):
        nonlocal seg
        sid = f"S{len(g['segments'])}"
        g["segments"].append({"id": sid, "kind": kind, "from": t, "to": t, "seq": seq})
        if seg is not None:
            g["edges"].append({"from": seg, "to": sid, "rel": "이어짐" if g["segments"][-2]["seq"] == seq else "엘리베이터로 이동"})
        seg = sid

    for t, r in enumerate(records):
        seq = r["seq"][:2]
        if seq not in seg_of_seq:  # 영상 구간이 바뀌면(엘리베이터 이동) 새 구간과 새 기억
            seg_of_seq[seq] = True
            if t:
                mem.new_segment(t)
            new_seg(t, "출발 구간" if not t else "하차 후 구간", seq)
        # 구간은 엘리베이터를 탈 때만 나눈다. 장소(Places365)는 문 앞에서 '엘리베이터'로 자주 튀어서 구간 안의 기록으로만 남김
        pg = r["raw"].get("place_groups") or {}
        b = max(pg, key=pg.get) if pg else None
        g["segments"][-1].setdefault("places", Counter())[b if b and pg[b] >= 0.35 else "모름"] += 1
        g["segments"][-1]["to"] = t
        # 안내도 후보: 층 표시(2F, 3층 …)가 한 장에 2개 이상
        marks = [o["text"] for o in r["raw"]["ocr"] if FLOOR_MARK.match(o["text"]) and o["score"] >= 0.8]
        if len(set(marks)) >= 2:
            g["maps"].append({"t": t, "seg": seg, "marks": sorted(set(marks))})
        # 문패 후보 → 기억(믿음 점수)
        for o in r["raw"]["ocr"]:
            txt = o["text"].replace("호", "")
            if not ROOM.match(txt) or o["score"] < 0.5:
                continue
            x0, y0, x1, y1 = o["box"]
            cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0
            door = near_door(r["raw"]["objects"], cx, cy)
            on_map = len(set(marks)) >= 2
            rec = mem.see(t, txt, o["score"], h, cx, bool(door) and not on_map)
            key = f"{mem.seg}:{txt}"
            p = g["plates"].setdefault(key, {"id": f"P{len(g['plates'])}", "room": txt, "seg": seg, "cx": [], "reads": 0,
                                             "door": None, "on_map": False, "first": t, "last": t, "best": 0})
            p["cx"].append(round(cx, 2)); p["reads"] += 1; p["last"] = t; p["best"] = max(p["best"], o["score"])
            p["door"] = p["door"] or door
            p["on_map"] |= on_map
            p["ok"] = rec["ok"]
    # 벽 쪽: 읽힌 화면 위치의 가운데값
    for p in g["plates"].values():
        m = sorted(p["cx"])[len(p["cx"]) // 2]
        p["side"] = "왼쪽" if m < SIDE_L else "오른쪽" if m > SIDE_R else "정면"
        p["kind"] = "안내도 번호" if p["on_map"] else "문패" if p["ok"] else "미확정"
    # 같은 구간·같은 벽의 확정 문패를 처음 본 순서로 잇고, 번호 흐름을 계산
    walls = {}
    for p in sorted(g["plates"].values(), key=lambda p: p["first"]):
        if p["kind"] == "미확정" and p["best"] >= 0.95:
            p["kind"] = "흐름에만 사용"  # 한 번 읽혔지만 또렷함
        if p["kind"] not in ("문패", "흐름에만 사용"):
            continue
        g["edges"].append({"from": p["id"], "to": p["seg"], "rel": f"{p['side']} 벽"})
        if p["door"]:
            g["edges"].append({"from": p["id"], "to": "문", "rel": "문 옆"})
        walls.setdefault((p["seg"], p["side"]), []).append(p)
    trend = Counter()
    for (sg, side), ps in walls.items():
        for a, b in zip(ps, ps[1:]):
            if a["room"][0] != b["room"][0]:
                continue
            d = int(b["room"]) - int(a["room"])
            g["edges"].append({"from": a["id"], "to": b["id"], "rel": f"다음 ({'+' if d > 0 else ''}{d})"})
            if side != "정면" and d:
                trend[(sg, "감소" if d < 0 else "증가")] += 1
    g["trend"] = {f"{sg}:{k}": v for (sg, k), v in trend.items()}
    g["walls"] = {f"{sg}:{side}": [p["room"] for p in ps] for (sg, side), ps in walls.items()}
    g["conclusion"] = conclude(g, target, walls, trend)
    return g


def conclude(g, target, walls, trend):
    """마지막 구간에서 목적지가 앞인지 뒤인지 (근거 수와 함께)."""
    last = g["segments"][-1]["id"]
    seen = [p for p in g["plates"].values() if p["kind"] == "문패" and p["seg"] == last]
    flow_seen = [p for p in g["plates"].values() if p["kind"] in ("문패", "흐름에만 사용") and p["seg"] == last]
    floors = Counter(p["room"][0] for p in seen)
    if not floors:
        return {"floor": None, "direction": "모름", "why": "확정 문패 없음"}
    fl = floors.most_common(1)[0][0]
    if fl != target[0]:
        return {"floor": fl, "direction": "다른 층", "why": f"확정 문패 {[p['room'] for p in seen]}의 첫 자리 {fl}"}
    if any(p["room"] == target for p in seen):
        return {"floor": fl, "direction": "보임", "why": f"{target}호 문패 확정"}
    dec, inc = trend.get((last, "감소"), 0), trend.get((last, "증가"), 0)
    if dec == inc:  # 근거가 없거나 반반
        return {"floor": fl, "direction": "모름", "why": f"벽별 번호 흐름이 정해지지 않음 (감소 {dec}, 증가 {inc})"}
    flow = "감소" if dec > inc else "증가"
    near = max((p for p in flow_seen if p["side"] != "정면"), key=lambda p: p["last"], default=seen[-1])
    if abs(int(target) - int(near["room"])) <= 2:  # 번호가 바로 옆이면 건물마다 순서가 어긋날 수 있어 방향을 단정하지 않음
        return {"floor": fl, "direction": "근처", "why": f"마지막 문패 {near['room']}가 목적지와 2 이내, 문패를 확인하며 천천히"}
    ahead = (int(target) < int(near["room"])) == (flow == "감소")
    return {"floor": fl, "direction": "앞쪽" if ahead else "뒤쪽", "flow": flow, "support": max(dec, inc), "against": min(dec, inc),
            "why": f"벽별로 앞으로 갈수록 번호 {flow} (근거 {max(dec, inc)}쌍, 반대 {min(dec, inc)}쌍), 마지막 문패 {near['room']}"}


def to_text(g):
    lines = [f"목적지: {g['target']}호", "구간:"]
    for s in g["segments"]:
        pl = ", ".join(f"{k} {v}초" for k, v in Counter(s["places"]).most_common(3))
        lines.append(f"- {s['id']} {s['kind']} ({s['from']}~{s['to']}초, 장소 인식: {pl})")
    for e in g["edges"]:
        if e["rel"] in ("이어짐", "엘리베이터로 이동"):
            lines.append(f"- {e['from']} → {e['to']}: {e['rel']}")
    lines.append("벽별 문패 (본 순서):")
    for k, rooms in g["walls"].items():
        lines.append(f"- {k}: {' → '.join(rooms)}")
    odd = [p for p in g["plates"].values() if p["kind"] != "문패"]
    if odd:
        lines.append("문패로 쓰지 않은 번호: " + ", ".join(f"{p['room']}({p['kind']})" for p in odd))
    if g["maps"]:
        lines.append("안내도 후보: " + ", ".join(f"{m['t']}초 {m['marks']}" for m in g["maps"]))
    c = g["conclusion"]
    lines.append(f"계산 결과: 층 {c.get('floor')}, 목적지 방향 {c['direction']} ({c['why']})")
    return "\n".join(lines)


def to_dot(g):
    out = ["digraph G {", 'graph [rankdir=LR, fontname="Apple SD Gothic Neo", labelloc=t, label="' +
           f"목적지 {g['target']}호 → {g['conclusion']['direction']}" + '"];',
           'node [fontname="Apple SD Gothic Neo", fontsize=11]; edge [fontname="Apple SD Gothic Neo", fontsize=9];']
    for s in g["segments"]:
        pl = ", ".join(f"{k} {v}" for k, v in Counter(s["places"]).most_common(2))
        out.append(f'{s["id"]} [shape=box, style="rounded,filled", fillcolor="#e8eefb", label="{s["id"]} {s["kind"]}\\n{s["from"]}~{s["to"]}초\\n{pl}"];')
    color = {"문패": "#e3f3e7", "미확정": "#f1f1f1", "안내도 번호": "#fbe5e2", "흐름에만 사용": "#fff4d6"}
    for p in g["plates"].values():
        door = "\\n문 옆" if p["door"] else ""
        out.append(f'{p["id"]} [shape=ellipse, style=filled, fillcolor="{color[p["kind"]]}", label="{p["room"]}\\n{p["kind"]} · {p["side"]}{door}\\n{p["reads"]}번 읽음"];')
        if p["kind"] not in ("문패", "흐름에만 사용"):
            out.append(f'{p["id"]} -> {p["seg"]} [style=dotted, color="#999999"];')
    for e in g["edges"]:
        if e["to"] == "문":
            continue
        style = "bold" if e["rel"].startswith("다음") else "solid"
        out.append(f'{e["from"]} -> {e["to"]} [label="{e["rel"]}", style={style}];')
    out.append("}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name")
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(a.records, encoding="utf-8")]
    g = build(recs, a.target)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{Path(a.records).stem}_{a.target}"
    for sg in g["segments"]:
        sg["places"] = dict(sg["places"])
    json.dump(g, open(out / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (out / f"{name}.txt").write_text(to_text(g), encoding="utf-8")
    (out / f"{name}.dot").write_text(to_dot(g), encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=110", str(out / f"{name}.dot"), "-o", str(out / f"{name}.png")], check=True)
    print(to_text(g))


if __name__ == "__main__":
    main()
