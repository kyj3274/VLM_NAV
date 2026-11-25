"""시뮬레이터 재생 화면: 실행 몇 개를 HTML 한 파일로 만들어 브라우저로 본다 (외부 파일·인터넷 없이 열림).

  .venv/bin/python src/sim/view.py                         # 배치 5종 × 시드 0~3 → outputs/sim/view.html 열기
  .venv/bin/python src/sim/view.py --seeds 0 1 --lat 2     # 지연 2초
  .venv/bin/python src/sim/view.py --layouts u_shape --seeds 5 --no-open

복도를 위에서 본 그림: 문(번호, 목적지는 강조), 사용자(삼각형)와 카메라 시야, 이번 사진에 읽힌 문패.
지연이 있으면 사진을 찍은 위치를 흐린 삼각형으로 같이 그린다. 오른쪽에 안내 문장·행동·P(앞)·사건.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import world  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "sim" / "view.html"

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>시뮬레이터 재생</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#d9d7d1;--floor:#ecebe6;--wall:#8a877f;
 --door:#b9b5ab;--seen:#2f6fdb;--target:#d9480f;--user:#1d1d1b;--fov:rgba(47,111,219,.12);--ok:#2b8a3e;--bad:#c92a2a;--say:#fff4e6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#20201e;--ink:#ecebe6;--mute:#9c9a93;--line:#34332f;--floor:#2a2a27;
 --wall:#77746c;--door:#55524b;--seen:#6ea0ff;--target:#ff8a4c;--user:#ecebe6;--fov:rgba(110,160,255,.14);--ok:#69db7c;--bad:#ff8787;--say:#3a2a16}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 8px 0 0}select,button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
button{cursor:pointer}.res{font-weight:600}.res.ok{color:var(--ok)}.res.bad{color:var(--bad)}
main{display:grid;grid-template-columns:1fr 360px;gap:16px;padding:16px;align-items:start}@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
svg{width:100%;height:auto;display:block}.ctl{display:flex;gap:8px;align-items:center;margin-top:10px}input[type=range]{flex:1}
.k{color:var(--mute);font-size:12px}.big{font-size:18px;font-weight:600}.say{background:var(--say);border-radius:6px;padding:6px 8px;margin:4px 0}
.log{max-height:260px;overflow:auto;font-size:13px}.log div{padding:2px 0;border-bottom:1px dashed var(--line)}.log .cur{font-weight:600}
.row{display:flex;justify-content:space-between;gap:8px}.legend{font-size:12px;color:var(--mute);margin-top:8px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
</style></head><body>
<header><h1>시뮬레이터 재생</h1><select id="ep"></select><span id="res" class="res"></span><span id="info" class="k"></span></header>
<main>
 <div class="card"><svg id="map"></svg>
  <div class="ctl"><button id="play">▶ 재생</button><button id="prev">◀</button><button id="next">▶</button>
   <input id="slider" type="range" min="0" value="0"><select id="speed"><option value="1">1배</option><option value="2" selected>2배</option><option value="4">4배</option></select></div>
  <div class="legend"><span style="background:var(--target)"></span>목적지 <span style="background:var(--seen)"></span>이번 사진에 읽힌 문패
   <span style="background:var(--fov)"></span>카메라 시야 · 흐린 삼각형 = 사진 찍은 위치(지연)</div>
 </div>
 <div>
  <div class="card"><div class="row"><div><div class="k">시각</div><div class="big" id="t"></div></div>
   <div><div class="k">행동</div><div class="big" id="act"></div></div><div><div class="k">목적지까지</div><div class="big" id="ahead"></div></div></div>
   <div class="k" style="margin-top:8px">이번 안내</div><div id="say"></div>
   <div class="k" style="margin-top:8px">읽힌 문패 · P(앞)</div><div id="seen"></div>
   <div class="k" style="margin-top:8px">판단 근거</div><div id="why" style="font-size:13px"></div>
  </div>
  <div class="card" style="margin-top:12px"><div class="k">안내 기록 (누르면 그 시각으로)</div><div class="log" id="log"></div></div>
 </div>
</main>
<script>
const EPS = __DATA__;
const $ = id => document.getElementById(id);
let ep = 0, i = 0, timer = null;
EPS.forEach((e, k) => { const o = document.createElement("option"); o.value = k;
  o.textContent = `${e.meta.layout} #${e.meta.seed} → ${e.meta.target.n}호 ${e.result.success ? "✓" : "✗"}`; $("ep").appendChild(o); });
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function draw() {
  const e = EPS[ep], m = e.meta, f = e.steps[i], L = m.corridor.length, W = m.width;
  const S = 900 / (L + 2), pad = 40, H = 180, mid = H / 2;
  const X = x => pad + (x + 1) * S - pad + 20, Y = y => mid - y * S;
  let s = `<svg viewBox="0 0 ${900 + 40} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  s += `<rect x="${X(0)}" y="${Y(W/2)}" width="${L*S}" height="${W*S}" fill="var(--floor)"/>`;
  s += `<line x1="${X(0)}" y1="${Y(W/2)}" x2="${X(L)}" y2="${Y(W/2)}" stroke="var(--wall)" stroke-width="3"/>`;
  s += `<line x1="${X(0)}" y1="${Y(-W/2)}" x2="${X(L)}" y2="${Y(-W/2)}" stroke="var(--wall)" stroke-width="3"/>`;
  const fov = (x, h, yaw, op) => { const a0 = (h > 0 ? 0 : 180) + yaw, r = m.range * S, pts = [];
    for (let a = a0 - m.fov/2; a <= a0 + m.fov/2 + .01; a += 5) pts.push(`${X(x) + r*Math.cos(a*Math.PI/180)},${Y(0) - r*Math.sin(a*Math.PI/180)}`);
    return `<polygon points="${X(x)},${Y(0)} ${pts.join(" ")}" fill="var(--fov)" opacity="${op}"/>`; };
  s += `<clipPath id="c"><rect x="${X(0)}" y="${Y(W/2)}" width="${L*S}" height="${W*S}"/></clipPath><g clip-path="url(#c)">${fov(f.cap_x, f.cap_heading, f.cap_yaw, 1)}</g>`;
  for (const d of m.corridor.doors) {
    const up = d.wall === "A", y = up ? W/2 : -W/2, tg = d.n === m.target.n, sn = f.seen.includes(d.n);
    const col = tg ? "var(--target)" : sn ? "var(--seen)" : "var(--door)";
    s += `<rect x="${X(d.x) - 0.45*S}" y="${up ? Y(y) - 5 : Y(y) - 1}" width="${0.9*S}" height="6" fill="${col}"/>`;
    s += `<text x="${X(d.x)}" y="${up ? Y(y) - 10 : Y(y) + 20}" text-anchor="middle" font-size="12" font-weight="${tg||sn?700:400}" fill="${col === "var(--door)" ? "var(--mute)" : col}">${d.n}</text>`;
  }
  const tri = (x, h, op) => { const p = h > 0 ? [[10,0],[-7,-7],[-7,7]] : [[-10,0],[7,-7],[7,7]];
    return `<polygon points="${p.map(([a,b]) => `${X(x)+a},${Y(0)+b}`).join(" ")}" fill="var(--user)" opacity="${op}"/>`; };
  if (m.lat > 0) s += tri(f.cap_x, f.cap_heading, .3);
  s += tri(f.x, f.heading, 1);
  if (f.yaw) { const a = ((f.heading > 0 ? 0 : 180) + f.yaw) * Math.PI / 180;
    s += `<line x1="${X(f.x)}" y1="${Y(0)}" x2="${X(f.x) + 28*Math.cos(a)}" y2="${Y(0) - 28*Math.sin(a)}" stroke="var(--seen)" stroke-width="2"/>`; }
  s += `</svg>`;
  $("map").outerHTML = s.replace("<svg", '<svg id="map"');
  $("t").textContent = f.t.toFixed(1) + "초"; $("act").textContent = f.action;
  $("ahead").textContent = (f.ahead > 0 ? "앞 " : "뒤 ") + Math.abs(f.ahead).toFixed(1) + "m";
  $("say").innerHTML = f.say.length ? f.say.map(x => `<div class="say">🔊 ${x}</div>`).join("") : `<span class="k">(없음)</span>`;
  $("seen").textContent = (f.seen.join(", ") || "없음") + (f.p_ahead != null ? ` · P(앞)=${f.p_ahead}` : "");
  $("why").textContent = [f.direction, ...f.events].filter(Boolean).join(" / ") || "-";
  $("slider").value = i;
  document.querySelectorAll("#log div").forEach(d => d.classList.toggle("cur", +d.dataset.i === lastSay(i)));
}
function lastSay(k) { const st = EPS[ep].steps; for (let j = k; j >= 0; j--) if (st[j].say.length) return j; return -1; }
function load(k) {
  ep = k; i = 0; const e = EPS[ep], r = e.result;
  $("slider").max = e.steps.length - 1;
  $("res").textContent = r.success ? `성공 (${r.end})` : `실패 (${r.end})`; $("res").className = "res " + (r.success ? "ok" : "bad");
  $("info").textContent = `걸은 거리 ${r.walked}m / 최단 ${r.shortest}m · SPL ${r.spl.toFixed(2)} · 틀린 방향 ${r.wrong_dir} · 돌기 ${r.turns}` + (e.meta.lat ? ` · 지연 ${e.meta.lat}초` : "");
  $("log").innerHTML = e.steps.map((f, j) => f.say.length ? `<div data-i="${j}">${f.t.toFixed(1)}초 · ${f.say.join(" / ")}</div>` : "").join("");
  document.querySelectorAll("#log div").forEach(d => d.onclick = () => { i = +d.dataset.i; draw(); });
  draw();
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "▶ 재생"; }
$("play").onclick = () => { if (timer) return stop(); $("play").textContent = "⏸ 멈춤";
  timer = setInterval(() => { if (i >= EPS[ep].steps.length - 1) return stop(); i++; draw(); }, 1500 / +$("speed").value); };
$("speed").onchange = () => { if (timer) { stop(); $("play").click(); } };
$("prev").onclick = () => { i = Math.max(0, i - 1); draw(); };
$("next").onclick = () => { i = Math.min(EPS[ep].steps.length - 1, i + 1); draw(); };
$("slider").oninput = e => { i = +e.target.value; draw(); };
$("ep").onchange = e => { stop(); load(+e.target.value); };
document.onkeydown = e => { if (e.key === "ArrowRight") $("next").click(); if (e.key === "ArrowLeft") $("prev").click(); if (e.key === " ") { e.preventDefault(); $("play").click(); } };
load(0);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=list(world.LAYOUTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--lat", type=float, default=0.0)
    ap.add_argument("--range", type=float, default=world.RANGE)
    ap.add_argument("--err", type=int, default=0, choices=sorted(world.ERR_PRESETS), help="인식 오류 0·1·2")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    world.RANGE = a.range
    world.ERR.update(world.ERR_PRESETS[a.err])
    eps = []
    for lay in a.layouts:
        for s in a.seeds:
            tr = []
            res = bench.episode(lay, s, a.lat, trace=tr)
            eps.append({"meta": tr[0], "steps": tr[1:], "result": res})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(eps, ensure_ascii=False)), encoding="utf-8")
    print(f"{out} ({len(eps)}개 실행, 성공 {sum(e['result']['success'] for e in eps)})")
    if not a.no_open:
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
