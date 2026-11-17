"""410호 경로(1층 입구 → 엘리베이터 → 4층) 세 방식 비교.

  VLM_NAV     data/video/results/results_guide_410_1_2a_situ_sharp.jsonl (YOLOE·OCR·Places365 + Ollama, 서버 GPU)
  claude_nav  $CLAUDE_NAV_RUN(claude_nav의 nav_1F-3F-4F_full_v6.jsonl) 중 1F·4F (Claude가 매 장 판단까지, 캐싱)
              3F 장면은 이 경로에 없어서 뺀다. 4F 4.9초(맥 화면 잠금으로 87초)는 시간 통계에서 뺀다.
  새 방식     outputs/routes/guide_410_1_2a_<이름>.jsonl (Claude 인식 + VLM_NAV 규칙)
              크레딧은 캐시에서 그 프레임의 인식·재확인 호출 값을 더한다(다른 경로와 같이 쓴 호출 포함).

  .venv/bin/python src/eval/compare_410.py --name claude
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = next(d for d in Path(__file__).resolve().parents if (d / "data" / "video").is_dir())
CLAUDE_NAV = Path(__import__("os").environ.get("CLAUDE_NAV_RUN", "nav_1F-3F-4F_full_v6.jsonl"))


def stats(xs):
    xs = [x for x in xs if x is not None]
    return (statistics.median(xs), max(xs), len(xs)) if xs else (None, None, 0)


def vlm_nav():
    recs = [json.loads(x) for x in open(ROOT / "data/video/results/results_guide_410_1_2a_situ_sharp.jsonl", encoding="utf-8")]
    seen = next((r for r in recs if "410" in r["rooms"]), None)
    arr = next((r for r in recs if r["state"].get("arrived")), None)
    says = [f"{r['seq'][:2]} {r['sec']}s: {s}" for r in recs for s in r["say"] if "410" in s]
    return {"name": "VLM_NAV (OCR·YOLOE·Places365 + qwen3-vl, GPU)", "steps": len(recs),
            "lat": stats([r["latency_seconds"] for r in recs]), "calls": sum(r["vlm_called"] for r in recs),
            "credit": 0.0, "first_seen": f"4층 {seen['sec']}초" if seen else "못 봄",
            "arrived": f"4층 {arr['sec']}초" if arr else "도착 판정 없음", "says": says}


def claude_nav():
    recs = [json.loads(x) for x in open(CLAUDE_NAV, encoding="utf-8")]
    recs = [r for r in recs if r["영상"] in ("1F.mp4", "4F.mp4")]
    lat = [r["처리_시간_s"] for r in recs if not r.get("시간_통계_제외")]
    first = next((r for r in recs if r["영상"] == "4F.mp4" and "410" in json.dumps(r["변환_결과"].get("시각정보", {}).get("호실_표지판", []), ensure_ascii=False)), None)
    arr = next((r for r in recs if r["결정"].get("도착")), None)
    says = [f"{r['영상'][:2]} {r['시각_s']:.1f}s: {r['결정']['안내_문장']}" for r in recs if "410" in r["결정"]["안내_문장"]]
    return {"name": "claude_nav full_v6 (Claude가 판단까지, 1F+4F)", "steps": len(recs), "lat": stats(lat),
            "calls": len(recs), "credit": sum(r["호출"]["credit"] or 0 for r in recs),
            "first_seen": f"4층 {first['시각_s']:.1f}초" if first else "못 봄",
            "arrived": f"{arr['영상'][:2]} {arr['시각_s']:.1f}초" if arr else "도착 판정 없음", "says": says}


def new(name):
    recs = [json.loads(x) for x in open(ROOT / f"outputs/routes/guide_410_1_2a_{name}.jsonl", encoding="utf-8")]
    if all("credit_all" in r for r in recs):  # 캐시에서 온 호출도 포함한 이 경로의 크레딧
        credit = sum(r["credit_all"] for r in recs)
        calls = sum(1 + bool(r["recheck"]) for r in recs if r["obs"] is not None)
    else:  # 2단계 첫 실행 기록: 캐시 파일에서 그 프레임 호출을 더함
        cache = [json.loads(x) for x in open(ROOT / "outputs/cache/claude.jsonl", encoding="utf-8")]
        frames = {f"{r['seq']}/{r['key_frame']}" for r in recs}
        credit = sum(c["log"]["credit"] for c in cache if c["key"].split("|")[3] in frames)
        calls = sum(1 for c in cache if c["key"].split("|")[3] in frames)
    seen = next((r for r in recs if r["seq"].startswith("2a") and "410" in [p["n"] for p in (r["obs"] or {}).get("plates", [])]), None)
    conf = next((r for r in recs if "410" in r["rooms"]), None)
    arr = next((r for r in recs if r["state"].get("arrived")), None)
    says = [f"{r['seq'][:2]} {r['sec']}s: {s}" for r in recs for s in r["say"] if "410" in s and "안내를 시작" not in s]
    return {"name": f"새 방식 {name}", "steps": len(recs),
            "lat": stats([r["latency_seconds"] for r in recs if not r["error"]]), "calls": calls, "credit": credit,
            "first_seen": (f"4층 {seen['sec']}초 (읽음)" if seen else "못 봄") + (f", {conf['sec']}초 확정" if conf else ""),
            "arrived": f"4층 {arr['sec']}초" if arr else "도착 판정 없음", "says": says}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="claude", help="새 방식 실행 이름들(쉼표)")
    a = ap.parse_args()
    rows = [vlm_nav(), claude_nav()] + [new(n) for n in a.name.split(",")]
    print("| 방식 | 초(장) | 판단 시간 중앙값 | 최대 | 모델 호출 | 크레딧 | 410 처음 | 도착 판정 |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        med, mx, _ = r["lat"]
        print(f"| {r['name']} | {r['steps']} | {med:.2f}초 | {mx:.2f}초 | {r['calls']} | {r['credit']:.2f} | "
              f"{r['first_seen']} | {r['arrived']} |")
    for r in rows:
        print(f"\n[{r['name']}] 410이 들어간 안내")
        for s in r["says"][:8]:
            print("  " + s)
    out = ROOT / f"outputs/eval/compare_410_{a.name.replace(',', '+')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
