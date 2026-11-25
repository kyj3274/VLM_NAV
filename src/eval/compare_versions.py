"""같은 캐시로 돌린 여러 판(이름)의 진행 방향 안내 비교 (크레딧 0, 2026-11-24 PR #2 병합 확인용, docs/17 21절).

먼저 판마다 (그 판의 코드 폴더에서) 캐시로 다시 돌린다:
  .venv/bin/python src/nav/navigator.py --replay --name <이름> --route "314=1_start_졸프_영상3,2b_after_elevator_졸프_영상2" \\
      --route "314=1_start_졸프_영상3,2a_after_elevator_졸프_영상1" --route "318=1_start_졸프_영상3,2b_after_elevator_졸프_영상2" \\
      --route "410=1_start_졸프_영상3,2a_after_elevator_졸프_영상1"
  .venv/bin/python src/eval/score_checkpoints.py --name <이름>
  .venv/bin/python src/eval/replay_web_session.py outputs/web_sessions/<세션> --name <이름>
그다음:
  .venv/bin/python src/eval/compare_versions.py pre21 main21 merge21 --session "시험 1=<세션 폴더 이름>" --session …
세션 폴더 이름(outputs/web_sessions/ 아래)은 저장소에 적지 않고 실행할 때 넘긴다.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = next(d for d in Path(__file__).resolve().parents if (d / "src" / "nav").is_dir())
ROUTES = ["314_1_2b", "314_1_2a", "318_1_2b", "410_1_2a"]
DIRW = re.compile(r"앞쪽|뒤쪽|뒤로|방향이 맞|멀어지|지나|근처입니다|비춰|가능성|찾지 못했|직진하세요")


def route(tag, name):
    rows = [json.loads(x) for x in open(ROOT / f"outputs/routes/guide_{tag}_{name}.jsonl", encoding="utf-8")]
    arr = next((f"{r['seq'][:2]} {r['sec']}s" for r in rows if r["state"].get("arrived")), "없음")
    return arr, [(f"{r['seq'][:2]} {r['sec']:>3}s", s) for r in rows for s in r["say"] if DIRW.search(s)]


def session(sid, name):
    d = json.load(open(ROOT / f"outputs/eval/web_replay_{sid}_{name}.json", encoding="utf-8"))
    return d["wait"].get("arrived_at"), [(f"{r['kind'][:5]} {r['t']:6.1f}", s) for r in d["rows"]
                                         for s in (r.get("say") or []) if DIRW.search(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--session", action="append", default=[], help="이름=세션 폴더 이름 (여러 번)")
    a = ap.parse_args()
    print("## 영상 경로")
    for tag in ROUTES:
        print(f"\n### {tag}")
        for n in a.names:
            arr, says = route(tag, n)
            print(f"- {n}: 도착 {arr}")
            for k, s in says:
                print(f"    {k}  {s}")
    print("\n## 휴대폰 기록 재생")
    for lab, sid in (x.split("=", 1) for x in a.session):
        print(f"\n### {lab} {sid}")
        for n in a.names:
            arr, says = session(sid, n)
            print(f"- {n}: 도착 {arr}")
            for k, s in says:
                print(f"    {k}  {s}")


if __name__ == "__main__":
    main()
