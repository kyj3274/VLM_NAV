"""실험 D 결과 분석 (Mac): 공식/보정 점수 + VLM 호출 비율·호출당 시간·판단 불가 비율 + 호출 조건 부분집합 시뮬레이션.

사용: python3 src/kyj/exp_d_analyze.py outputs/kyj/exp_d_d1_dev.jsonl [...] [--el-th 0.25 --ex-th 0.4 --no-rooms]
부분집합 시뮬레이션: 더 좁은 조건에 해당하지 않는 질문의 답은 버리고(1단계 유지), 그 질문만 있던 사진은 VLM 시간을 뺀다.
(실제로는 질문을 묶어 한 번에 부르므로 근사치)
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "kyj"


def refs():
    ref = json.load(open(ROOT / "review/survey_v2.json"))
    adj = json.loads(json.dumps(ref))
    extra = json.load(open(OUT / "exp_a_extra_rooms.json"))
    for i, rs in extra["add"].items():
        have = {x["room"] for x in adj[i]["rooms"]}
        adj[i]["rooms"] += [r for r in rs if r["room"] not in have]
    for i, rs in extra["ignore"].items():
        adj[i]["ignore_rooms"] = list(set(adj[i].get("ignore_rooms", [])) | set(rs))
    return ref, adj


def subset(path, out, el_th, ex_th, rooms_on, cfg="m2048", params=None):
    """더 좁은 호출 조건으로 다시 반영해 새 jsonl을 쓴다 (exp_d_vlm.plan/apply 재사용)."""
    import types
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    import exp_d_vlm
    raw = {json.loads(l)["image_id"]: json.loads(l) for l in open(OUT / f"exp_a_{cfg}.jsonl", encoding="utf-8")}
    with open(out, "w", encoding="utf-8") as f:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rec = raw[r["image_id"]]
            kept, base, q = exp_d_vlm.plan(rec, params or {"majority": True}, 0.0, 0.0)
            q = r["questions"]
            ans = r["vlm"]
            keep_e = [i for i, e in enumerate(q["elev"]) if e["conf"] >= el_th]
            nq = {"rooms": q["rooms"] if rooms_on else [], "elev": [q["elev"][i] for i in keep_e],
                  "signs": [s for s in q["signs"] if s["conf"] >= ex_th]}
            nans = None
            if ans and "error" not in ans:
                nans = {f"R{i + 1}": ans.get(f"R{i + 1}") for i in range(len(nq["rooms"]))}
                for j, i in enumerate(keep_e):
                    nans[f"E{j + 1}"] = ans.get(f"E{i + 1}")
                nans["arrows"] = ans.get("arrows", [])
            elif ans:
                nans = ans
            called = bool(nq["rooms"] or nq["elev"] or nq["signs"])
            result, notes = exp_d_vlm.apply(base, kept, nq, nans, rec["texts_all"])
            vlm_s = r["vlm_seconds"] if called else 0.0
            r.update({"result": result, "vlm_called": called, "questions": nq, "vlm": nans if called else None,
                      "vlm_seconds": vlm_s, "latency_seconds": r["stage1_seconds"] + vlm_s})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--subset", default=None, help="el_th,ex_th,rooms(0/1) → 파일마다 _sub 결과를 만들어 채점")
    args = ap.parse_args()
    if args.subset:
        el, ex, ro = args.subset.split(",")
        tag = f"_e{el}_x{ex}_r{ro}".replace(".", "")
        args.files = [subset(p, str(p).replace(".jsonl", tag + ".jsonl"), float(el), float(ex), ro == "1") for p in args.files]
    ref, adj = refs()
    ids = json.load(open(ROOT / "review/split.json"))[args.split]
    rows = []
    for path in args.files:
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        recs = [r for r in recs if r["image_id"] in set(ids)]
        a = score_v2.score(ref, recs, ids)
        b = score_v2.score(adj, recs, ids)
        called = [r for r in recs if r["vlm_called"]]
        vt = [r["vlm_seconds"] for r in called]
        qs = {"R": [0, 0], "E": [0, 0]}  # [질문 수, 판단 불가 수]
        errors = 0
        for r in called:
            v = r["vlm"] or {}
            if "error" in v:
                errors += 1
                continue
            for k, val in v.items():
                if k[:1] in qs and k[1:].isdigit():
                    qs[k[0]][0] += 1
                    qs[k[0]][1] += val == "판단 불가"
        arrows_asked = sum(1 for r in called if r["questions"]["signs"])
        arrows_given = sum(1 for r in called if r["questions"]["signs"] and (r["vlm"] or {}).get("arrows"))
        rows.append({
            "name": Path(path).stem, "off": a, "adj": b,
            "calls": len(called), "n": len(recs),
            "vlm_med": round(statistics.median(vt), 2) if vt else 0, "vlm_max": round(max(vt), 2) if vt else 0,
            "R": qs["R"], "E": qs["E"], "errors": errors, "arrows": [arrows_asked, arrows_given],
        })
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    pct = lambda d: f"{d['p'] * 100:.0f}/{d['r'] * 100:.0f}/{d['f1'] * 100:.0f}"
    print("| 결과 | 공식 호실 P/R/F1 | 맞/틀/놓 | 가까운 | 보정 P/R/F1 | 맞/틀/놓 | 가까운(보정) | 랜드마크 F1 | 엘리베이터 P/R/F1 | 화살표 찾음/방향/오탐 | 중앙(초) | 호출 | 호출당 중앙/최대(초) | R 판단불가 | E 판단불가 | 화살표 답함/질문 |")
    print("|---" * 16 + "|")
    for x in rows:
        a, b = x["off"], x["adj"]
        el = a["per_landmark"]["엘리베이터"]
        ar = a["arrows"]
        print(f"| {x['name']} | {pct(a['rooms'])} | {a['rooms']['tp']}/{a['rooms']['fp']}/{a['rooms']['fn']} | "
              f"{a['near_rooms_found']}/{a['near_rooms_total']} | {pct(b['rooms'])} | {b['rooms']['tp']}/{b['rooms']['fp']}/{b['rooms']['fn']} | "
              f"{b['near_rooms_found']}/{b['near_rooms_total']} | {a['landmarks']['f1'] * 100:.0f} | {pct(el)} | "
              f"{ar['found']}/{ar['direction_ok']}/{ar['false_alarm_images']} (정답 {ar['truth_images']}) | {a['latency_median']} | "
              f"{x['calls']}/{x['n']} | {x['vlm_med']}/{x['vlm_max']} | {x['R'][1]}/{x['R'][0]} | {x['E'][1]}/{x['E'][0]} | {x['arrows'][1]}/{x['arrows'][0]} |")


if __name__ == "__main__":
    main()
