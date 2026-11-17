"""팀 형식 JSON을 cjy 평가기(src/cjy/evaluate.py, 고치지 않고 그대로 실행)로 1장씩 채점하고 합산한다.

  # kyj 추적기 내보내기 (video_team_format.py가 만든 index.jsonl)
  python3 video_team_eval.py run  --index OUT/kyj/index.jsonl --out OUT/kyj_eval
  # lsh 결과(video_nav_*.jsonl)의 image 모드 답을 같은 방식으로
  python3 video_team_eval.py lsh  --src /workspace/capstone/outputs/lsh/video_nav_1F-3F-4F_sharp.jsonl --mode image --out OUT/lsh
  python3 video_team_eval.py run  --index OUT/lsh/index.jsonl --out OUT/lsh_eval
  python3 video_team_eval.py sum  OUT/kyj_eval OUT/lsh_eval
평가 VLM은 cjy 기본값(qwen3-vl:4b-instruct, 11436).
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

CJY = Path("/workspace/capstone/src/cjy/evaluate.py")
SEQ = {"1F.mp4": "1", "3F.mp4": "2b", "4F.mp4": "2a"}


def lsh(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(a.src, encoding="utf-8"):
        r = json.loads(line)
        res = ((r.get("modes") or {}).get(a.mode) or {}).get("result")
        if not res:
            continue
        sec = int(r.get("window_start_sec", r["t_sec"]))
        p = out / f"{SEQ[r['video']]}_{sec:02d}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append({"json": str(p), "image": r["frame_path"], "request": r["request"], "sec": sec})
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(len(rows), "→", out)


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    for line in open(a.index, encoding="utf-8"):
        x = json.loads(line)
        dst = out / Path(x["json"]).name
        if dst.exists():
            continue
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, str(CJY), "--image", x["image"], "--request", x["request"],
                            "--vlm-output", x["json"], "--output", str(dst)], capture_output=True, text=True)
        rec = {"name": dst.name, "seconds": round(time.perf_counter() - t0, 1), "ok": p.returncode == 0,
               "error": p.stderr.strip()[-300:] if p.returncode else None}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(rec["name"], rec["ok"], rec["seconds"], rec["error"] or "", flush=True)


def summary(a):
    for d in a.dirs:
        files = sorted(Path(d).glob("*.json"))
        vis = {}
        ctx = {"공간이름_및_판단근거": [], "목적지와의_관계": []}
        act, req = [], []
        for f in files:
            e = json.loads(f.read_text(encoding="utf-8"))
            v = e["시각정보 품질"]
            for k in ("층표시_표지판", "호실_표지판", "화살표"):
                vis.setdefault(k, [0, 0, 0])
                for i, t in enumerate(("TP", "FP", "FN")):
                    vis[k][i] += v[k][t]
            for k, c in v["others"].items():
                vis.setdefault("others", [0, 0, 0])
                for i, t in enumerate(("TP", "FP", "FN")):
                    vis["others"][i] += c[t]
            for k in ctx:
                ctx[k].append(e["상황_맥락 품질"][k]["점수"])
            act.append(e["방향정보 품질"]["다음_행동_방향"]["점수"])
            req.append(e["요청인식 품질"]["인식한_목적지_텍스트"]["점수"])
        logs = [json.loads(l) for l in open(Path(d) / "log.jsonl", encoding="utf-8")] if (Path(d) / "log.jsonl").exists() else []
        print(f"\n## {d}: 채점된 장 {len(files)}, 평가 실패 {sum(not l['ok'] for l in logs)}, "
              f"평가 시간 중앙값 {statistics.median([l['seconds'] for l in logs]) if logs else '-'}초")
        print("| 항목 | TP | FP | FN | P | R |\n|---|---|---|---|---|---|")
        for k, (tp, fp, fn) in vis.items():
            p = f"{tp / (tp + fp):.2f}" if tp + fp else "-"
            r = f"{tp / (tp + fn):.2f}" if tp + fn else "-"
            print(f"| {k} | {tp} | {fp} | {fn} | {p} | {r} |")
        m = lambda xs: f"{statistics.mean(xs):.2f}" if xs else "-"
        print(f"공간이름 {m(ctx['공간이름_및_판단근거'])}/2, 목적지와의_관계 {m(ctx['목적지와의_관계'])}/2, "
              f"다음_행동_방향 {m(act)}/2, 목적지 인식 {m(req)}/2")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lsh"); p.add_argument("--src", required=True); p.add_argument("--mode", default="image"); p.add_argument("--out", required=True)
    p = sp.add_parser("run"); p.add_argument("--index", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("sum"); p.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    {"lsh": lsh, "run": run, "sum": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
