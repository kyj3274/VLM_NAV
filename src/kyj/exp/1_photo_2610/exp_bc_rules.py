"""실험 B·C: 호실 결정 규칙을 실험 A 원자료로 재계산해 결과 jsonl을 만든다 (Mac, GPU 불필요).

사용법:
  python3 src/kyj/exp_bc_rules.py diag m2048              # dev 후보 특징 표 (보정 기준 라벨)
  python3 src/kyj/exp_bc_rules.py write <이름> <원자료 설정> '<json 파라미터>'
      → outputs/kyj/exp_bc_<이름>.jsonl  (rooms/spaces/directions 를 규칙 결과로 다시 만든다)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import room_rules  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "kyj"
PLATE = json.load(open(OUT / "exp_bc_plate.json")) if (OUT / "exp_bc_plate.json").exists() else {}
HANGUL = re.compile(r"[가-힣]")


def load(cfg):
    return [json.loads(l) for l in open(OUT / f"exp_a_{cfg}.jsonl", encoding="utf-8")]


def cands_for(rec, cfg):
    cands = room_rules.candidates(rec["texts_all"], rec["ocr_input_size"])
    feats = PLATE.get(cfg, {}).get(rec["image_id"], {})
    for c in cands:
        c.update(feats.get(c["room"], {}))
    return cands


def clock(x):
    return "11시" if x < 1 / 3 else "12시" if x < 2 / 3 else "1시"


def rebuild(rec, kept):
    """기존 combine 결과에서 호실 관련 부분만 규칙 결과로 바꾼다."""
    res = json.loads(json.dumps(rec["result"]))
    rooms = [c["room"] for c in kept]
    res["rooms"] = rooms
    res["directions"] = [d for d in res["directions"] if not room_rules.ROOM.match(d["target"])]
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


def write(name, cfg, params):
    path = OUT / f"exp_bc_{name}.jsonl"
    n_amb = {}
    with open(path, "w", encoding="utf-8") as f:
        for rec in load(cfg):
            cands = cands_for(rec, cfg)
            kept, amb = room_rules.select(cands, params)
            out = {k: rec[k] for k in ("image_id", "success", "vlm_called", "latency_seconds", "yoloe_seconds", "ocr_seconds")}
            out.update({"task": "fields", "method": f"yoloe+ocr:{cfg}+rules", "config": cfg, "rules": params,
                        "result": rebuild(rec, kept),
                        "room_candidates": [{k: c.get(k) for k in ("room", "score", "h", "cy", "pair", "plate")} for c in cands],
                        "ambiguous": [c["room"] for c in amb]})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return path


def diag(cfg):
    ref = json.load(open(ROOT / "review/survey_v2.json")); split = json.load(open(ROOT / "review/split.json"))
    extra = json.load(open(OUT / "exp_a_extra_rooms.json"))
    for i, rs in extra["add"].items():
        ref[i]["rooms"] += rs
    recs = {r["image_id"]: r for r in load(cfg)}
    for i in split["dev"]:
        truth = {x["room"]: x["near"] for x in ref[i]["rooms"]}
        cands = cands_for(recs[i], cfg)
        for c in sorted(cands, key=lambda c: c["cy"]):
            lab = ("NEAR" if truth[c["room"]] else "TP") if c["room"] in truth else "FP"
            if c["room"] in ref[i].get("ignore_rooms", []) or c["room"] in extra["ignore"].get(i, []):
                lab = "IGN"
            print(f"{i} {c['room']:5} {lab:4} s={c['score']:.2f} h={c['h']:.3f} cy={c['cy']:.2f} edge={c['edge_dist']:.3f} asp={c['aspect']:.1f} "
                  f"pair={int(c['pair'])} plate={c.get('plate', -1):.2f} bgstd={c.get('bg_std', -1):.0f} con={c.get('contrast', -1):.0f} ring={c.get('ring_diff', -1):.0f}")
        miss = [k for k in truth if k not in {c["room"] for c in cands}]
        if miss:
            print(i, "MISS", miss, [truth[m] for m in miss])


if __name__ == "__main__":
    if sys.argv[1] == "diag":
        diag(sys.argv[2])
    else:
        print(write(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}))


def grid_score(cfg, params, split="dev"):
    """공식/보정 기준 점수를 바로 계산 (파일 쓰지 않음)."""
    import score_v2
    ref = json.load(open(ROOT / "review/survey_v2.json"))
    ids = json.load(open(ROOT / "review/split.json"))[split]
    adj = json.loads(json.dumps(ref))
    extra = json.load(open(OUT / "exp_a_extra_rooms.json"))
    for i, rs in extra["add"].items():
        have = {x["room"] for x in adj[i]["rooms"]}
        adj[i]["rooms"] += [r for r in rs if r["room"] not in have]
    for i, rs in extra["ignore"].items():
        adj[i]["ignore_rooms"] = list(set(adj[i].get("ignore_rooms", [])) | set(rs))
    records, amb_imgs = [], 0
    for rec in load(cfg):
        kept, amb = room_rules.select(cands_for(rec, cfg), params)
        records.append({"image_id": rec["image_id"], "result": rebuild(rec, kept), "latency_seconds": rec["latency_seconds"]})
        amb_imgs += bool(amb) and rec["image_id"] in ids
    return score_v2.score(ref, records, ids), score_v2.score(adj, records, ids), amb_imgs
