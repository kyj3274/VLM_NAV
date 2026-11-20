"""지도 없는 복도에서 '번호 배치 가설'을 배우며 목적지 방향을 확률로 추정하는 실험 (Mac, 모델 없음, 기존 기록만).

지금 길안내(nav_memory.direction)는 확정 문패를 벽별로 놓고 증감을 세어 다수결, 목적지와 ±1이면 '근처'로 정한다.
여기서는 사람이 정한 숫자 대신 확률로 판단한다.

1) 문패 관측: 초마다 OCR이 읽은 세 자리 번호. 믿음 = OCR 점수 × 읽힌 횟수(많이 읽힐수록 1에 가까움).
   지나간 순서 = 글자가 가장 크게 보인 때(가장 가까이 지난 때).
2) 가설 d: 앞으로 갈수록 번호가 커진다(+1) / 작아진다(-1). 처음엔 반반.
   지나간 문패 두 개(a 다음 b)마다 P(b-a의 부호 | d)로 가설 가중치를 갱신한다.
   번호가 건너뛰거나(315 → 307) 맞은편 벽이라 거꾸로 보이는 경우를 '예외' 확률 EPS로 흡수한다.
   (가까이 붙은 번호일수록 같은 벽 이웃일 가능성이 커서 근거가 강하고, 멀리 떨어진 번호는 약하게 셈)
3) 지금 위치: 최근에 지난 문패들의 번호 분포(최근일수록 무게 큼).
4) P(목적지가 앞) = Σ_d P(d) · Σ_i w_i · [ (목적지 - n_i)·d > 0 ]
   목적지가 최근 번호들 사이에 있으면 자연히 0.5 근처가 된다 = '근처' (±1 같은 숫자 없이).

  .venv/bin/python src/eval/layout_belief.py            # 판단 지점 13개 채점 + 3층 경로 초별 확률
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(d for d in Path(__file__).resolve().parents if (d / "data" / "video").is_dir())
ROOM = re.compile(r"^[1-9]\d{2}$")

EPS = 0.2         # 예외(건너뜀·맞은편 벽) 확률: 증감 근거 한 쌍이 틀릴 수 있는 정도
MIN_READ = 0.5    # 이 점수 미만 OCR 글자는 문패 관측으로 안 씀
HALF_LIFE = 3     # 지금 위치 분포: 지난 문패 몇 개 전이면 무게가 반이 되나
SIDE_GAP = 0.1   # 화면 가운데 ±이 안의 문패는 벽을 정하지 않음
SURE = 0.8        # P(앞) 이 이상이면 '앞쪽', 1-SURE 이하면 '뒤쪽'
NEAR_P = 0.35     # 목적지가 최근 번호들 사이에 있고 P(앞)이 0.5±이 안이면 '근처'


_DEPTH = {"model": None, "cache": {}}
FRAMES = ROOT / "data/video/frames_8fps"


def depth_info(r):
    """대표 사진의 깊이 정보 (src/nav/depth.py, 처음 부를 때 모델을 불러옴). 사진이 없으면 None."""
    d = next((x for x in FRAMES.iterdir() if x.name.startswith(r["seq"])), None) if FRAMES.is_dir() else None
    path = d / r["key_frame"] if d else None
    if path is None or not path.is_file():
        return None
    if path not in _DEPTH["cache"]:
        sys.path.insert(0, str(ROOT / "src/nav"))
        import depth
        from PIL import Image
        if _DEPTH["model"] is None:
            _DEPTH["model"] = depth.load()
        _DEPTH["cache"][path] = depth.measure(_DEPTH["model"], Image.open(path).convert("RGB"))[0] if _DEPTH["model"] else None
    return _DEPTH["cache"][path]


def observations(records, use_depth=False):
    """초별 기록 → 문패별 {번호, 믿음, 지나간 때, 벽}. 마지막 구간(엘리베이터에서 내린 뒤)만.
    use_depth: 벽 쪽을 깊이 지도(복도가 뻗는 쪽) 기준으로. 아니면 화면 위치."""
    seg = records[-1]["seq"][:2]
    plates = defaultdict(lambda: {"reads": 0, "score": 0.0, "best_h": 0.0, "pass_t": None, "last": None, "cx": [], "votes": []})
    for t, r in enumerate(records):
        if r["seq"][:2] != seg:
            continue
        info = depth_info(r) if use_depth else None
        for o in r["raw"]["ocr"]:
            n = o["text"].replace("호", "")
            if not ROOM.match(n) or o["score"] < MIN_READ:
                continue
            p = plates[n]
            h = o["box"][3] - o["box"][1]
            p["reads"] += 1
            p["score"] = max(p["score"], o["score"])
            p["last"] = t
            cx = (o["box"][0] + o["box"][2]) / 2
            p["cx"].append(cx)
            if info is not None:
                import depth
                v = depth.wall_side(info, cx)
                if v != "?":
                    p["votes"].append("L" if v == "왼쪽" else "R")
            if h >= p["best_h"]:
                p["best_h"], p["pass_t"] = h, t
    out = []
    for n, p in plates.items():
        trust = p["score"] * (1 - 0.5 ** p["reads"])  # 1번 0.5, 2번 0.75, 3번 0.88 … × 점수
        if use_depth:  # 깊이 표 중 2/3 이상이 같은 쪽일 때만 벽을 정함
            top = max(set(p["votes"]), key=p["votes"].count) if p["votes"] else "?"
            side = top if p["votes"] and p["votes"].count(top) / len(p["votes"]) >= 2 / 3 else "?"
        else:
            cx = sorted(p["cx"])[len(p["cx"]) // 2]
            side = "L" if cx < 0.5 - SIDE_GAP else "R" if cx > 0.5 + SIDE_GAP else "?"
        out.append({"n": n, "trust": round(trust, 3), "pass_t": p["pass_t"], "reads": p["reads"], "side": side})
    return sorted(out, key=lambda p: p["pass_t"])


def belief(plates, target):
    """번호 배치 가설 가중치와 목적지 방향 확률."""
    fl_w = defaultdict(float)
    for p in plates:
        fl_w[p["n"][0]] += p["trust"]
    if not fl_w:
        return {"direction": "모름", "why": "문패 없음"}
    floor = max(fl_w, key=fl_w.get)
    if floor != target[0] and fl_w[floor] >= 1.0:
        return {"direction": "다른 층", "floor": floor, "why": f"층 근거 {dict(fl_w)}"}
    same = [p for p in plates if p["n"][0] == target[0]]
    if any(p["n"] == target and p["trust"] >= 0.7 for p in same):
        return {"direction": "보임", "why": f"{target} 문패"}
    if not same:
        return {"direction": "모름", "why": "같은 층 문패 없음"}
    # 2) 가설 가중치 (로그). 같은 벽에서 연속한 두 문패의 증감 부호 (두 벽은 걷는 방향이 같으므로 같은 d를 공유)
    logw = {+1: 0.0, -1: 0.0}
    pairs = []
    walls = [[p for p in same if p["side"] == sd] for sd in ("L", "R")]
    for a, b in [pr for wl in walls for pr in zip(wl, wl[1:])]:
        diff = int(b["n"]) - int(a["n"])
        if diff == 0:
            continue
        rel = a["trust"] * b["trust"] * (1.0 if abs(diff) <= 3 else 0.5)  # 가까운 번호 = 같은 벽 이웃일 가능성 큼
        for d in (+1, -1):
            like = (1 - EPS) if (diff > 0) == (d > 0) else EPS
            logw[d] += rel * math.log(like)
        pairs.append(f"{a['n']}→{b['n']}")
    m = max(logw.values())
    pd = {d: math.exp(v - m) for d, v in logw.items()}
    z = sum(pd.values())
    pd = {d: v / z for d, v in pd.items()}
    # 3) 지금 위치: 목적지와 번호가 가장 가까운 문패의 벽에서, 최근 문패일수록 무게 (맞은편 벽 번호는 번호 체계가 엇갈려 섞지 않음)
    tnum = int(target)
    key = min(same, key=lambda p: abs(int(p["n"]) - tnum))
    wall = [p for p in same if p["side"] == key["side"]] if key["side"] != "?" else same
    k = len(wall)
    w = [p["trust"] * 0.5 ** ((k - 1 - i) / HALF_LIFE) for i, p in enumerate(wall)]
    zw = sum(w)
    p_ahead = 0.0
    for d, pdv in pd.items():
        s_ = sum(wi * (1.0 if (tnum - int(p["n"])) * d > 0 else 0.5 if tnum == int(p["n"]) else 0.0)
                 for wi, p in zip(w, wall))
        p_ahead += pdv * s_ / zw
    # 근처: 그 벽의 최근 3개 번호 사이에 목적지가 있는가
    recent = [int(p["n"]) for p in wall[-3:]]
    info = {"p_up": round(pd[+1], 2), "p_ahead": round(p_ahead, 2), "pairs": pairs, "walk": [p["n"] for p in same]}
    if min(recent) <= tnum <= max(recent) and abs(p_ahead - 0.5) <= NEAR_P:
        return {"direction": "근처", **info}
    if p_ahead >= SURE:
        return {"direction": "앞쪽", **info}
    if p_ahead <= 1 - SURE:
        return {"direction": "뒤쪽", **info}
    return {"direction": "모름", **info}


ACT = {"앞쪽": "직진", "뒤쪽": "뒤로 돌기", "다른 층": "엘리베이터로 돌아가기", "보임": "목적지 접근", "근처": "목적지 접근", "모름": "직진"}
SRC = {"A": ("guide_314_1_2b_situ_sharp", "314"), "C": ("guide_314_1_2b_situ_sharp", "318"), "B": ("guide_314_2a_situ_sharp", "314")}


def score(recs, cps, use_depth, show=False):
    ok, miss = 0, []
    for cp in cps:
        f, tg = SRC[cp["route"]]
        cut = [r for r in recs[f] if r["sec"] <= cp["sec"]] if cp["route"] == "B" else \
              [r for r in recs[f] if r["seq"].startswith("1_") or r["sec"] <= cp["sec"]]
        b = belief(observations(cut, use_depth), tg)
        hit = ACT[b["direction"]] in cp["action"] or (b["direction"] == "근처" and "직진" in cp["action"])
        ok += hit
        if not hit:
            miss.append(cp["id"])
        if show:
            print(f"{cp['id']:>3} 정답 {cp['action'][0]:12s} → {b['direction']:4s} {'O' if hit else 'X'}  "
                  f"P(앞)={b.get('p_ahead', '-')} P(증가)={b.get('p_up', '-')} {' → '.join(b.get('walk', []))}")
    return ok, miss


def main():
    loc = ROOT / "data/video/results/locate"
    recs = {f: [json.loads(line) for line in open(loc / f"{f}.jsonl", encoding="utf-8")] for f, _ in SRC.values()}
    cps = [c for c in json.load(open(ROOT / "data/video/checkpoints.json", encoding="utf-8"))["checkpoints"] if c["seg"] == 1]
    ok = 0
    print("== 판단 지점 (엘리베이터에서 내린 뒤 13개)")
    for cp in cps:
        f, tg = SRC[cp["route"]]
        cut = [r for r in recs[f] if r["sec"] <= cp["sec"]] if cp["route"] == "B" else \
              [r for r in recs[f] if r["seq"].startswith("1_") or r["sec"] <= cp["sec"]]
        b = belief(observations(cut), tg)
        act = ACT[b["direction"]]
        hit = act in cp["action"] or (b["direction"] == "근처" and "직진" in cp["action"])
        ok += hit
        print(f"{cp['id']:>3} 정답 {cp['action'][0]:12s} → {b['direction']:4s} {'O' if hit else 'X'}  "
              f"P(앞)={b.get('p_ahead', '-')} P(증가)={b.get('p_up', '-')} {' → '.join(b.get('walk', []))}")
    print(f"점수 {ok}/{len(cps)} (지금 길안내 그래프 방식 12/13)\n")
    for tg in ("314", "318"):
        print(f"== 3층 영상, 목적지 {tg}: 초별 확률")
        f = "guide_314_1_2b_situ_sharp"
        prev = None
        seg_recs = [r for r in recs[f] if r["seq"].startswith("2b")]
        for s in range(len(seg_recs)):
            cut = [r for r in recs[f] if r["seq"].startswith("1_")] + seg_recs[:s + 1]
            b = belief(observations(cut), tg)
            line = f"{b['direction']} P(앞)={b.get('p_ahead', '-')} P(증가)={b.get('p_up', '-')} [{' → '.join(b.get('walk', []))}]"
            if line != prev:
                print(f"  {s:>2}초  {line}")
                prev = line
        print()
    # 돌아선 상황 흉내: 3층 영상의 문패 관측을 거꾸로 (지나간 순서를 뒤집음)
    print("== 돌아선 상황 흉내: 3층 문패 순서를 뒤집어 목적지 318")
    obs = observations([r for r in recs["guide_314_1_2b_situ_sharp"] if r["seq"].startswith("2b")])
    last = max(p["pass_t"] for p in obs)
    rev = sorted(({**p, "pass_t": last - p["pass_t"]} for p in obs), key=lambda p: p["pass_t"])
    for i in range(2, len(rev) + 1):
        b = belief(rev[:i], "318")
        print(f"  {' → '.join(b.get('walk', []))}: {b['direction']} P(앞)={b.get('p_ahead')} P(증가)={b.get('p_up')}")


if __name__ == "__main__":
    sys.exit(main())
