"""video_tracker.py 결과(guide_*.jsonl)를 장면별 안내 타임라인 HTML로 만든다 (Mac에서 확인용).

사용: python3 src/kyj/video_demo_html.py data/video/guide_314_1_2b.jsonl data/video/guide_314_1_2a.jsonl
  → data/video/demo_<파일이름>.html  (그림은 data/video/frames_8fps/ 를 상대 경로로 씀)
"""

import html
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "data" / "video"
ROUTE = {"1_": "입구 → 엘리베이터 (영상3)", "2a": "엘리베이터 하차 후 (영상1, 4층)", "2b": "엘리베이터 하차 후 (영상2, 3층)"}

CSS = """
:root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e4e2dc; --say:#0b5cad; --vlm:#b5480b; --dense:#4b7a1e; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --card:#20201e; --line:#34332f; --say:#7db7f2; --vlm:#f0a072; --dense:#a6d36f; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, "Apple SD Gothic Neo", sans-serif; }
main { max-width:980px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 10px; }
.sub { color:var(--muted); margin-bottom:16px; }
table.sum { border-collapse:collapse; margin:8px 0 4px; } table.sum td, table.sum th { border:1px solid var(--line); padding:4px 10px; text-align:left; }
.row { display:grid; grid-template-columns:64px 96px 1fr; gap:12px; padding:8px 0; border-bottom:1px solid var(--line); align-items:start; }
.row.quiet { opacity:.55; } .row.quiet img { width:48px; }
.t { font-variant-numeric:tabular-nums; color:var(--muted); font-size:13px; }
img { width:96px; border-radius:4px; display:block; }
.say { color:var(--say); font-weight:600; font-size:16px; }
.tag { display:inline-block; font-size:12px; border:1px solid var(--line); border-radius:10px; padding:0 7px; margin:2px 4px 0 0; color:var(--muted); }
.tag.vlm { color:var(--vlm); border-color:var(--vlm); } .tag.dense { color:var(--dense); border-color:var(--dense); }
"""


def render(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    target = recs[0]["route_target"]
    seqs = []
    for r in recs:
        if r["seq"] not in seqs:
            seqs.append(r["seq"])
    lat = [r["latency_seconds"] for r in recs]
    vlm = sum(r["vlm_called"] for r in recs)
    dense = sum(len(r["used_frames"]) > 1 for r in recs)
    out = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{target}호 안내 시연</title><style>{CSS}</style></head><body><main>",
           f"<h1>{target}호로 가는 안내 시연</h1>",
           f"<div class='sub'>경로: {' → '.join(ROUTE.get(s[:2], s) for s in seqs)}. 1초마다 1장을 보고, 애매하면 그 1초의 8장을 더 봄. VLM은 사건이 있을 때만.</div>",
           "<table class='sum'><tr><th>영상 길이</th><th>안내 문장</th><th>VLM 호출</th><th>8장을 다시 본 초</th><th>1초 처리 시간 중앙값 / 최대</th></tr>",
           f"<tr><td>{len(recs)}초</td><td>{sum(len(r['say']) for r in recs)}개</td><td>{vlm}번 ({vlm / len(recs) * 100:.0f}%)</td>"
           f"<td>{dense}초</td><td>{statistics.median(lat):.2f}초 / {max(lat):.2f}초</td></tr></table>"]
    for seq in seqs:
        out.append(f"<h2>{html.escape(ROUTE.get(seq[:2], seq))}</h2>")
        for r in [x for x in recs if x["seq"] == seq]:
            img = f"frames_8fps/{seq}/{r['key_frame']}"
            tags = []
            if r["rooms"]:
                tags.append(f"<span class='tag'>읽은 호실 {', '.join(r['rooms'])}</span>")
            if len(r["used_frames"]) > 1:
                tags.append(f"<span class='tag dense'>8장 다시 봄: {', '.join(r['ambiguous'])}</span>")
            for v in r["vlm"]:
                a = v.get("answer") or {}
                brief = ", ".join(f"{k}={a[k]}" for k in ("scene_type", "visible_path", "door_ahead", "elevator_visible",
                                                          "call_button_side", "room") if k in a)
                tags.append(f"<span class='tag vlm'>VLM ({', '.join(v['why'])}, {v.get('vlm_seconds', 0):.1f}초): {html.escape(brief)}</span>")
            if r.get("target_distance_m") is not None or r.get("distance_basis"):
                tags.append(f"<span class='tag'>거리 {r.get('target_distance_m')}m ({html.escape(r.get('distance_basis', ''))})</span>")
            say = "".join(f"<div class='say'>🔊 {html.escape(s)}</div>" for s in r["say"])
            quiet = "" if (r["say"] or r["vlm"]) else " quiet"
            out.append(f"<div class='row{quiet}'><div class='t'>{r['sec']:02d}초<br>{r['latency_seconds']:.2f}s</div>"
                       f"<img src='{img}' alt='' loading='lazy'><div>{say}{''.join(tags)}</div></div>")
    out.append("</main></body></html>")
    dest = ROOT / "results" / f"demo_{Path(path).stem}.html"
    dest.write_text("\n".join(out), encoding="utf-8")
    return dest


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(render(p))
