"""시뮬레이터 v3 재생 화면: 갈림길이 있는 층과 로비를 위에서 본 그림으로 (HTML 한 파일, 인터넷 없이 열림).

  .venv/bin/python src/sim/view_floor.py                          # 층 6종 × 시드 0~1 × (번호 / 이름 / 이름+표지판 / 영어+표지판)
                                                                  # + 로비 3크기 × 시드 0~1 × (번호 / 영어) × (가운데 장소 없음 / 3개) → outputs/sim/view_floor.html
  .venv/bin/python src/sim/view_floor.py --layouts loop --seeds 9 12

그림: 복도, 문(목적지 강조, 이번 사진에 읽힌 문패 파랑), 사용자(삼각형)와 카메라 시야, 지나온 길, 표지판 있는 갈림길(■).
오른쪽: 안내 문장, 행동, 보인 갈림길, 판단 근거(사건), 안내 기록.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_floor  # noqa: E402
import floor as fw  # noqa: E402
import lobby  # noqa: E402
import world  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "sim" / "view_floor.html"
CASES = [("number", 0.0, "번호"), ("name", 0.0, "이름만"), ("name", 1.0, "이름+표지판"), ("english", 1.0, "영어+표지판")]
LOBBY_CASES = [("number", 0, "로비 번호"), ("english", 0, "로비 영어"), ("number", 3, "로비+가운데 장소"), ("english", 3, "로비+가운데 장소 영어")]

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>층 시뮬레이터 재생</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#d9d7d1;--floor:#e4e2dc;--wall:#8a877f;
 --door:#b9b5ab;--seen:#2f6fdb;--target:#d9480f;--user:#1d1d1b;--fov:rgba(47,111,219,.16);--trail:#9c6ade;--sign:#2b8a3e;--ok:#2b8a3e;--bad:#c92a2a;--say:#fff4e6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#20201e;--ink:#ecebe6;--mute:#9c9a93;--line:#34332f;--floor:#2e2e2b;
 --wall:#77746c;--door:#55524b;--seen:#6ea0ff;--target:#ff8a4c;--user:#ecebe6;--fov:rgba(110,160,255,.18);--trail:#c4a3ff;--sign:#69db7c;--ok:#69db7c;--bad:#ff8787;--say:#3a2a16}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 8px 0 0}select,button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
button{cursor:pointer}.res{font-weight:600}.res.ok{color:var(--ok)}.res.bad{color:var(--bad)}
main{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px;align-items:start}@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
svg{width:100%;height:auto;display:block;max-height:70vh}.ctl{display:flex;gap:8px;align-items:center;margin-top:10px}input[type=range]{flex:1;min-width:0}
.k{color:var(--mute);font-size:12px}.big{font-size:18px;font-weight:600}.say{background:var(--say);border-radius:6px;padding:6px 8px;margin:4px 0}
.log{max-height:300px;overflow:auto;font-size:13px}.log div{padding:2px 0;border-bottom:1px dashed var(--line);cursor:pointer}.log .cur{font-weight:600}
.row{display:flex;justify-content:space-between;gap:8px}.legend{font-size:12px;color:var(--mute);margin-top:8px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
</style></head><body>
<header><h1>층 시뮬레이터 재생</h1><select id="ep"></select><span id="res" class="res"></span><span id="info" class="k"></span></header>
<main>
 <div class="card"><svg id="map"></svg>
  <div class="ctl"><button id="play">▶ 재생</button><button id="prev">◀</button><button id="next">▶</button>
   <input id="slider" type="range" min="0" value="0"><select id="speed"><option value="1">1배</option><option value="3" selected>3배</option><option value="8">8배</option></select></div>
  <div class="legend"><span style="background:var(--target)"></span>목적지 <span style="background:var(--seen)"></span>이번 사진에 읽힌 문패
   <span style="background:var(--trail)"></span>지나온 길 <span style="background:var(--sign)"></span>표지판 있는 갈림길 <span style="background:var(--fov)"></span>카메라 시야</div>
 </div>
 <div>
  <div class="card"><div class="row"><div><div class="k">시각</div><div class="big" id="t"></div></div>
   <div><div class="k">행동</div><div class="big" id="act"></div></div><div><div class="k">걸은 거리</div><div class="big" id="walked"></div></div></div>
   <div class="k" style="margin-top:8px">이번 안내</div><div id="say"></div>
   <div class="k" style="margin-top:8px">보인 것</div><div id="seen"></div>
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
  o.textContent = `${e.case} · ${e.kind} #${e.seed} → ${e.target.n} ${e.result.success ? "✓" : "✗"}`; $("ep").appendChild(o); });
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function draw() {
  const e = EPS[ep], f = e.steps[i], b = e.box, S = 14, P = 30;
  const X = x => P + (x - b[0]) * S, Y = y => P + (b[3] - y) * S;
  const Wd = (b[2] - b[0]) * S + 2 * P, Hd = (b[3] - b[1]) * S + 2 * P;
  let s = `<svg viewBox="0 0 ${Wd} ${Hd}" xmlns="http://www.w3.org/2000/svg">`;
  for (const r of e.rooms) s += `<rect x="${X(r[0])}" y="${Y(r[3])}" width="${(r[2]-r[0])*S}" height="${(r[3]-r[1])*S}" fill="var(--floor)"/>`;
  for (const r of e.blocks || []) s += `<rect x="${X(r[0])}" y="${Y(r[3])}" width="${(r[2]-r[0])*S}" height="${(r[3]-r[1])*S}" fill="var(--wall)" opacity=".55"/>`;
  for (const c of e.cors) s += `<line x1="${X(c.a[0])}" y1="${Y(c.a[1])}" x2="${X(c.b[0])}" y2="${Y(c.b[1])}" stroke="var(--floor)" stroke-width="${e.width*S}" stroke-linecap="square"/>`;
  for (const j of e.signs) s += `<rect x="${X(j[0])-5}" y="${Y(j[1])-5}" width="10" height="10" fill="var(--sign)"/>`;
  const tr = e.steps.slice(0, i + 1).map(g => `${X(g.p[0])},${Y(g.p[1])}`).join(" ");
  s += `<polyline points="${tr}" fill="none" stroke="var(--trail)" stroke-width="2.5" stroke-opacity=".7" stroke-linejoin="round"/>`;
  const cam = (f.body - f.yaw) * Math.PI / 180, r = e.range * S, pts = [];
  for (let a = -30; a <= 30; a += 5) { const q = cam + a * Math.PI / 180; pts.push(`${X(f.p[0]) + r*Math.sin(q)},${Y(f.p[1]) - r*Math.cos(q)}`); }
  s += `<polygon points="${X(f.p[0])},${Y(f.p[1])} ${pts.join(" ")}" fill="var(--fov)"/>`;
  for (const d of e.doors) {
    const tg = d.n === e.target.n, sn = f.seen.includes(d.n) || (tg && f.seen.includes("★"));
    const col = tg ? "var(--target)" : sn ? "var(--seen)" : "var(--door)";
    s += `<circle cx="${X(d.p[0])}" cy="${Y(d.p[1])}" r="${tg ? 5 : 3.5}" fill="${col}"><title>${esc(d.n)}</title></circle>`;
    if (e.mode === "number" || tg || sn)
      s += `<text x="${X(d.l[0])}" y="${Y(d.l[1]) + 4}" text-anchor="${d.a || "middle"}" font-size="${tg||sn ? 11 : 9}" font-weight="${tg||sn ? 700 : 400}" fill="${col === "var(--door)" ? "var(--mute)" : col}">${esc(d.n)}</text>`;
  }
  const u = f.body * Math.PI / 180, tri = [[0, 11], [-7, -7], [7, -7]].map(([dx, fw]) =>
    `${X(f.p[0]) + fw*Math.sin(u) + dx*Math.cos(u)},${Y(f.p[1]) - fw*Math.cos(u) + dx*Math.sin(u)}`).join(" ");
  s += `<polygon points="${tri}" fill="var(--user)"/></svg>`;
  $("map").outerHTML = s.replace("<svg", '<svg id="map"');
  $("t").textContent = f.t.toFixed(1) + "초"; $("act").textContent = f.act || "-"; $("walked").textContent = f.walked.toFixed(0) + "m";
  $("say").innerHTML = f.say.length ? f.say.map(x => `<div class="say">🔊 ${esc(x)}</div>`).join("") : `<span class="k">(없음)</span>`;
  const j = f.junction, jt = j ? ` · 갈림길 앞 ${j.m}m (트인 쪽: ${j.open.map(o => ({ahead:"앞",left:"왼쪽",right:"오른쪽"}[o])).join("·")})` +
    (j.signs.length ? " · 표지판 " + j.signs.map(g => `${({ahead:"앞",left:"왼쪽",right:"오른쪽"}[g.dir])}: ${g.items.slice(0, 4).join(", ")}${g.items.length > 4 ? " …" : ""}`).join(" / ") : "") : "";
  $("seen").textContent = "문패 " + (f.seen.join(", ").replace("★", "목적지") || "없음") + jt;
  $("why").textContent = f.events.join(" / ") || "-";
  $("slider").value = i;
  document.querySelectorAll("#log div").forEach(d => d.classList.toggle("cur", +d.dataset.i === lastSay(i)));
}
function lastSay(k) { const st = EPS[ep].steps; for (let j = k; j >= 0; j--) if (st[j].say.length) return j; return -1; }
function load(k) {
  ep = k; i = 0; const e = EPS[ep], r = e.result;
  $("slider").max = e.steps.length - 1;
  $("res").textContent = r.success ? `성공 (${r.end})` : `실패 (${r.end})`; $("res").className = "res " + (r.success ? "ok" : "bad");
  $("info").textContent = `${r.t}초 · 걸은 거리 ${r.walked}m / 최단 ${r.shortest}m · SPL ${r.spl.toFixed(2)} · 끝에서 돎 ${r.end_turns}`;
  $("log").innerHTML = e.steps.map((f, j) => f.say.length ? `<div data-i="${j}">${f.t.toFixed(1)}초 · ${esc(f.say.join(" / "))}</div>` : "").join("");
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


def pack(tr, res, case):
    """bench_floor 기록 → 그림용 (평면 좌표)."""
    m, steps = tr[0], tr[1:]
    fl, half = m["floor"], world.WIDTH / 2
    cors = [{"a": fw.point(c, 0), "b": fw.point(c, c["length"])} for c in fl["corridors"]]
    doors, signs = [], []
    for c in fl["corridors"]:
        vx, vy = fw.vec(c["ang"])
        for d in c["doors"]:
            s = 1 if d["wall"] == "A" else -1  # 벽 A = +방향 왼쪽 (왼쪽 법선 (-vy, vx))
            px, py = fw.point(c, d["x"])
            doors.append({"n": d["n"], "p": (px - vy * half * s, py + vx * half * s),
                          "l": (px - vy * (half + 1.2) * s, py + vx * (half + 1.2) * s)})
        signs += [fw.point(c, j["x"]) for j in c["junc"] if j["signs"]]
    xs = [p[0] for c in cors for p in (c["a"], c["b"])]
    ys = [p[1] for c in cors for p in (c["a"], c["b"])]
    out, walked, prev = [], 0.0, None
    for f in steps:
        c = fl["corridors"][f["c"]]
        p = fw.point(c, f["x"])
        walked += math.dist(prev, p) if prev else 0.0
        prev = p
        out.append({"t": f["t"], "p": p, "body": (c["ang"] + (0 if f["h"] > 0 else 180)) % 360, "yaw": f["yaw"],
                    "say": f["say"], "act": f["act"], "seen": f["seen"], "junction": f["junction"], "events": f["events"],
                    "walked": walked})
    pad = half + 2
    return {"case": case, "kind": m["kind"], "seed": m["seed"], "target": m["target"], "mode": fl["mode"],
            "width": world.WIDTH, "range": world.RANGE, "box": (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
            "cors": cors, "rooms": [], "doors": doors, "signs": signs, "steps": out, "result": res}


def pack_lobby(tr, res, case):
    """lobby.py 기록 → 그림용."""
    m, lb = tr[0], tr[0]["lobby"]
    out = lambda d: -1 if not d.get("island") else 1  # 벽 글자는 로비 바깥, 섬 글자는 섬 바깥(로비 쪽)
    doors = [{"n": d["n"], "p": d["p"], "l": (d["p"][0] + out(d) * d["n_in"][0] * 0.8, d["p"][1] + out(d) * d["n_in"][1] * 1.2),
              "a": {1: "end", -1: "start"}.get(-out(d) * d["n_in"][0], "middle")} for d in lb["doors"]]
    steps = [{"t": f["t"], "p": f["p"], "body": f["b"], "yaw": f["yaw"], "say": f["say"], "act": f["act"], "seen": f["seen"],
              "junction": f.get("junction"), "events": f["events"], "walked": f["walked"]} for f in tr[1:]]
    return {"case": case, "kind": m["kind"], "seed": m["seed"], "target": m["target"], "mode": lb["mode"],
            "width": world.WIDTH, "range": world.RANGE, "box": (-2.5, -2.5, lb["W"] + 2.5, lb["H"] + 2.5),
            "cors": [], "rooms": [(0, 0, lb["W"], lb["H"])], "blocks": lb.get("blocks", []), "doors": doors, "signs": [], "steps": steps, "result": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=list(fw.FLOORS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    world.RANGE = a.range
    eps = []
    for mode, signs, case in CASES:
        for lay in a.layouts:
            for s in a.seeds:
                tr = []
                res = bench_floor.episode(lay, s, mode, signs, trace=tr)
                eps.append(pack(tr, res, case))
    for mode, isl, case in LOBBY_CASES:
        for W in lobby.SIZES:
            for s in a.seeds:
                tr = []
                res = lobby.episode(W, s, mode, trace=tr, islands=isl)
                eps.append(pack_lobby(tr, res, case))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(eps, ensure_ascii=False)), encoding="utf-8")
    print(f"{out} ({len(eps)}개 실행, 성공 {sum(e['result']['success'] for e in eps)})")
    if not a.no_open:
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
