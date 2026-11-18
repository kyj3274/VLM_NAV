"""웹 길안내 세션 하나(outputs/web_sessions/<세션>/)를 웹 화면과 같은 모양의 다시보기 HTML 한 장으로 만든다.
걸음마다 보낸 사진, 그때 말한 안내, 층·확정 문패·Claude가 본 것·확대 재확인, 버튼 기록을 순서대로 넘겨 본다.
사진은 줄여서 HTML 안에 넣으므로 파일 하나만 옮기면 어디서나 열린다. (VLM_NAV web_replay.py 화면 구조 그대로, 내용만 Claude 인식으로)

  .venv/bin/python src/nav/web_replay.py outputs/web_sessions/<세션> [--out replay.html]
  웹에서는 /records → 기록을 누르면 서버가 이 함수로 바로 만든다.
"""

import argparse
import base64
import html
import io
import json
import statistics
from pathlib import Path

from PIL import Image

EVENT_SAY = {"start": "안내 시작", "pause": "일시 정지", "resume": "다시 시작", "exited": "엘리베이터에서 내림 (층·위치 다시 확인)",
             "end": "안내를 끝냄", "stop": "관찰자 비상 정지", "limit": "크레딧 한도로 멈춤", "destination": "목적지 변경",
             "destination_retry": "목적지를 못 알아들음", "stale": "늦게 온 응답 버림", "repeat": "다시 듣기"}
SIDE = {"L": "왼쪽", "C": "정면", "R": "오른쪽"}
PLACE = {"corridor": "복도", "lobby": "로비", "elev_hall": "엘리베이터 앞", "entrance": "건물 출입문 앞", "junction": "복도 갈림길",
         "room_door": "방 문 앞", "in_elev": "엘리베이터 안", "stairs": "계단 근처", "unknown": "모름"}


def thumb(p, side=560):
    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def load(d):
    """log.jsonl → 화면 한 칸씩(걸음 기록 또는 버튼 기록)."""
    items, imgs = [], {}
    for line in open(d / "log.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "event" in r:
            items.append({"kind": "event", **{k: r.get(k) for k in ("event", "wall", "target", "mode", "heard", "source", "fake",
                                                                    "reason")}, "seq": r.get("web_seq")})
            continue
        k = r.get("frame") or r["key_frame"]
        if k not in imgs and (d / "frames" / k).is_file():
            imgs[k] = thumb(d / "frames" / k)
        st, obs = r["state"], r.get("obs") or {}
        plates = [f"{p['n']}호 {SIDE.get(p['side'], '')} {p['m'] or '?'}m" + (f" ({p['name']})" if p.get("name") else "")
                  for p in obs.get("plates", [])] + [f"번호 못 읽음 '{p['name']}'" for p in obs.get("unread", [])]
        el = obs.get("elev") or {}
        items.append({"kind": "step", "img": k, "seg": int(str(r["seq"]).split("_")[-1] or 0), "sec": r["sec"],
                      "wall": r.get("wall"), "say": r["say"], "rooms": r["rooms"], "floor": st.get("floor"),
                      "wrong_floor": st.get("wrong_floor"), "scene": PLACE.get(obs.get("place"), "-"),
                      "arrived": st.get("arrived"), "action": r["action"], "recheck": r.get("recheck"),
                      "plates": plates, "fs": obs.get("floor_sign", ""),
                      "elev": (f"엘리베이터 {SIDE.get(el.get('side'), '')} {el.get('m') or '?'}m" if el.get("seen") else ""),
                      "pending": r.get("target_pending"), "latency": round(r.get("latency_seconds", 0), 2),
                      "lag": r.get("lag_s"), "heading": r.get("heading"), "steps": r.get("steps"),
                      "target": r["route_target"], "calls": r.get("claude_calls", 0), "credit": r.get("credit", 0)})
    return items, imgs


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>안내 기록 __SID__</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#15171a; --muted:#5d6570; --accent:#1f6feb; --ok:#1a7f37; --warn:#b54708; --line:#dde1e6; --pause:#9a6700; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0f1115; --card:#181b21; --text:#eef0f3; --muted:#9aa3ad; --accent:#4c8dff; --ok:#3fb950; --warn:#f0883e; --line:#2a2f37; --pause:#d29922; } }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:980px; margin:0 auto; padding:16px; display:grid; gap:12px; }
  h1 { font-size:20px; margin:4px 0; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  button, select { font-size:16px; padding:8px 12px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--text); cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  #state { font-size:13px; padding:3px 10px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
  #state[data-s=running] { color:#fff; background:var(--ok); border-color:var(--ok); }
  #state[data-s=paused] { color:#fff; background:var(--pause); border-color:var(--pause); }
  #state[data-s=arrived] { color:var(--ok); border-color:var(--ok); }
  .sum { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; font-size:14px; }
  .sum b { display:block; font-size:18px; font-variant-numeric:tabular-nums; }
  .sum span { color:var(--muted); font-size:12px; }
  .grid { display:grid; grid-template-columns: minmax(0,320px) minmax(0,1fr); gap:12px; }
  @media (max-width:720px) { .grid { grid-template-columns:1fr; } }
  .vbox { position:relative; }
  #img { width:100%; max-height:62vh; background:#000; border-radius:10px; object-fit:contain; display:block; aspect-ratio:9/16; }
  #veil { position:absolute; inset:12px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:22px; font-weight:700; background:rgba(0,0,0,.45); border-radius:10px; }
  #clock { position:absolute; left:20px; top:20px; color:#fff; background:rgba(0,0,0,.55); padding:2px 8px; border-radius:6px; font-size:13px; font-variant-numeric:tabular-nums; }
  #say { font-size:24px; line-height:1.45; min-height:72px; font-weight:600; }
  #say.fresh { color:var(--accent); }
  .stat { display:grid; grid-template-columns:auto 1fr; gap:4px 10px; font-size:14px; color:var(--muted); }
  .stat b { color:var(--text); font-weight:600; }
  #log { max-height:42vh; overflow:auto; font-size:14px; }
  #log div { padding:6px 0; border-bottom:1px solid var(--line); }
  #log .t { color:var(--muted); margin-right:6px; font-variant-numeric:tabular-nums; }
  #log .vlm { color:var(--warn); } #log .ev { color:var(--pause); } .arrived { color:var(--ok); }
  #bar { position:relative; height:26px; background:var(--bg); border:1px solid var(--line); border-radius:6px; cursor:pointer; }
  #bar i { position:absolute; top:3px; bottom:3px; width:3px; border-radius:2px; }
  #bar i.say { background:var(--accent); } #bar i.vlm { background:var(--warn); top:14px; } #bar i.ev { background:var(--pause); width:4px; }
  #pos { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--text); }
  input[type=range] { flex:1; min-width:160px; }
  .muted { color:var(--muted); font-size:13px; }
  details { font-size:14px; } details li { margin:3px 0; }
</style></head>
<body><main>
  <h1>안내 기록 · 목적지 <span id="tgt"></span>호 <span id="state" data-s="running">안내 중</span></h1>
  <div class="card sum" id="sum"></div>
  <div class="card" style="display:grid; gap:8px">
    <div class="row">
      <button id="prev" title="한 칸 뒤로">◀</button>
      <button id="play" class="primary">재생</button>
      <button id="next" title="한 칸 앞으로">▶</button>
      <button id="nextSay">다음 안내 문장으로</button>
      <select id="speed"><option value="1000">1배</option><option value="500" selected>2배</option><option value="250">4배</option></select>
      <label class="row muted"><input id="speak" type="checkbox"> 소리로 읽기</label>
      <input id="seek" type="range" min="0" value="0">
    </div>
    <div id="bar"><span id="pos"></span></div>
    <div class="muted">파란 줄 = 안내 문장, 주황 줄 = 확대 재확인, 갈색 줄 = 버튼 기록 (엘리베이터에서 내림·목적지 변경·끝내기)</div>
  </div>
  <div class="grid">
    <div class="card vbox"><img id="img" alt="그 초의 대표 사진"><span id="clock"></span><div id="veil" hidden>일시 정지</div></div>
    <div class="card" style="display:grid; gap:10px; align-content:start">
      <div id="say"></div>
      <div class="stat">
        <span>지금 층</span><b id="sFloor">-</b>
        <span>읽은 호실</span><b id="sRooms">-</b>
        <span>이 초에 본 것</span><b id="sSeen">-</b>
        <span>장면</span><b id="sScene">-</b>
        <span>읽은 문패</span><b id="sOcr">-</b>
        <span>엘리베이터·층 표시</span><b id="sObj">-</b>
        <span>확대 재확인</span><b id="sVlm">-</b>
        <span>처리 시간</span><b id="sTime">-</b>
      </div>
      <div id="log"></div>
    </div>
  </div>
  <details class="card"><summary>말한 안내 전체</summary><ol id="all"></ol></details>
</main>
<script>
const ITEMS = __ITEMS__, IMGS = __IMGS__, META = __META__;
const $ = id => document.getElementById(id);
const steps = ITEMS.filter(x => x.kind === "step");
let cur = 0, timer = null;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const label = it => it.kind === "step" ? `구간${it.seg + 1} ${it.sec}초` : (it.wall || "").slice(11);
$("tgt").textContent = META.target;
$("seek").max = ITEMS.length - 1;
$("sum").innerHTML = [
  ["시작", META.start || "-"], ["안내한 시간", steps.length + "초"], ["영상 구간", META.segments + "개"],
  ["말한 안내", META.says + "번"], ["Claude 호출", META.calls + "번"], ["크레딧", META.credit],
  ["인식 시간 중앙값", META.median + "초"], ["갱신 간격 중앙값", META.gap + "초"], ["결과", META.arrived ? "도착 선언" : "도착 선언 없음"],
].map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("");
$("all").innerHTML = steps.flatMap(s => s.say.map(t => `<li><span class="muted">${label(s)}</span> ${esc(t)}</li>`)).join("");
ITEMS.forEach((it, i) => {
  const x = (i / Math.max(1, ITEMS.length - 1) * 100) + "%";
  const mk = c => { const m = document.createElement("i"); m.className = c; m.style.left = x; m.title = label(it); $("bar").appendChild(m); };
  if (it.kind === "event" && it.event !== "start") mk("ev");
  if (it.kind === "step" && it.say.length) mk("say");
  if (it.kind === "step" && it.recheck) mk("vlm");
});

function render(i, speakIt) {
  cur = Math.max(0, Math.min(ITEMS.length - 1, i)); $("seek").value = cur;
  $("pos").style.left = (cur / Math.max(1, ITEMS.length - 1) * 100) + "%";
  let lastStep = null, saidAt = null, said = "목적지를 넣고 시작하세요.", rooms = [], vlm = 0, paused = false, arrived = false;
  const logs = [];
  for (let k = 0; k <= cur; k++) {
    const it = ITEMS[k], t = `<span class="t">${esc(label(it))}</span>`;
    if (it.kind === "event") {
      if (it.event === "pause") paused = true;
      if (it.event === "resume") paused = false;
      const extra = it.event === "start" ? ` (${it.mode === "file" ? "영상 파일" : "카메라"}, 들은 말 "${it.heard || "-"}", 입력 ${it.source || "-"}${it.fake ? ", 가짜 응답" : ""})` : it.event === "destination" ? ` → ${it.target}호` : it.event === "stale" ? ` #${it.seq}` : "";
      logs.push(`<div>${t}<span class="ev">${esc((META.ev[it.event] || it.event) + extra)}</span></div>`);
      continue;
    }
    lastStep = it;
    for (const r of it.rooms) if (!rooms.includes(r)) rooms.push(r);
    if (it.recheck) logs.push(`<div>${t}<span class="vlm">확대 재확인(${esc(it.recheck.why)}): ${esc(it.recheck.asked)} → ${esc(it.recheck.read || "못 읽음")}${it.recheck.match === false ? " (다름, 버림)" : ""}</span></div>`);
    for (const s of it.say) { said = s; saidAt = k; logs.push(`<div>${t}<span class="${it.arrived ? "arrived" : ""}">${esc(s)}</span></div>`); }
    if (it.arrived) arrived = true;
  }
  const st = arrived ? ["arrived", "도착"] : paused ? ["paused", "일시 정지"] : ["running", "안내 중"];
  $("state").dataset.s = st[0]; $("state").textContent = st[1]; $("veil").hidden = !paused;
  $("say").textContent = said; $("say").classList.toggle("fresh", saidAt === cur);
  $("log").innerHTML = logs.reverse().join("");
  $("sRooms").textContent = rooms.length ? rooms.join(", ") : "-";
  if (lastStep) {
    $("img").src = IMGS[lastStep.img] || "";
    $("clock").textContent = `구간${lastStep.seg + 1} · ${lastStep.sec}초` + (lastStep.wall ? " · " + lastStep.wall.slice(11) : "");
    $("sFloor").textContent = lastStep.floor ? lastStep.floor + "층" + (lastStep.wrong_floor ? " (목적지 층 아님)" : "") : "모름";
    $("sScene").textContent = (lastStep.scene || "-") + " · 행동: " + lastStep.action;
    $("sOcr").textContent = lastStep.plates.length ? lastStep.plates.join(", ") : "없음";
    $("sObj").textContent = [lastStep.elev, lastStep.fs ? "층 표시 " + lastStep.fs : ""].filter(Boolean).join(" · ") || "없음";
    $("sVlm").textContent = lastStep.recheck ? `${lastStep.recheck.asked} → ${lastStep.recheck.read || "못 읽음"}` : "-";
    $("sSeen").textContent = lastStep.pending ? "목적지 확인 중 (호출 2번 전)" : "-";
    $("sTime").textContent = `인식 ${lastStep.latency}초` + (lastStep.lag != null ? ` · 찍은 뒤 반영까지 ${lastStep.lag}초` : "") + (lastStep.heading != null ? ` · 방위 ${Math.round(lastStep.heading)}°` : "") + (lastStep.steps != null ? ` · 걸음 ${lastStep.steps}` : "");
  }
  if (speakIt && saidAt === cur && $("speak").checked && window.speechSynthesis) {
    const u = new SpeechSynthesisUtterance(said); u.lang = "ko-KR"; speechSynthesis.speak(u);
  }
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "재생"; }
$("play").onclick = () => {
  if (timer) return stop();
  if (cur >= ITEMS.length - 1) render(0);
  $("play").textContent = "멈춤";
  timer = setInterval(() => { if (cur >= ITEMS.length - 1) return stop(); render(cur + 1, true); }, +$("speed").value);
};
$("speed").onchange = () => { if (timer) { stop(); $("play").click(); } };
$("prev").onclick = () => { stop(); render(cur - 1); };
$("next").onclick = () => { stop(); render(cur + 1, true); };
$("nextSay").onclick = () => {
  stop();
  const j = ITEMS.findIndex((it, k) => k > cur && it.kind === "step" && it.say.length);
  render(j < 0 ? ITEMS.length - 1 : j, true);
};
$("seek").oninput = () => { stop(); render(+$("seek").value); };
$("bar").onclick = e => { stop(); const r = $("bar").getBoundingClientRect(); render(Math.round((e.clientX - r.left) / r.width * (ITEMS.length - 1))); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") $("prev").click();
  if (e.key === "ArrowRight") $("next").click();
  if (e.key === " ") { e.preventDefault(); $("play").click(); }
});
render(0);
</script></body></html>
"""


def build(d):
    """세션 폴더 → 다시보기 HTML 글자."""
    d = Path(d)
    items, imgs = load(d)
    steps = [x for x in items if x["kind"] == "step"]
    if not steps:
        return "<!doctype html><meta charset=utf-8><p>걸음 기록이 없습니다.</p>"
    lat = [x["latency"] for x in steps]
    walls = [x["wall"] for x in steps if x.get("wall")]
    meta = {"target": steps[-1]["target"], "start": next((x["wall"] for x in items if x.get("wall")), "-"),
            "segments": len({x["seg"] for x in steps}), "says": sum(len(x["say"]) for x in steps),
            "calls": sum(x["calls"] for x in steps), "credit": round(sum(x["credit"] for x in steps), 2),
            "median": round(statistics.median(lat), 2), "gap": "-", "arrived": any(x["arrived"] for x in steps), "ev": EVENT_SAY}
    ts = [x["sec"] for x in steps]
    if len(ts) > 1:
        meta["gap"] = round(statistics.median([b - a for a, b in zip(ts, ts[1:]) if b >= a] or [0]), 2)
    dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731
    return (PAGE.replace("__SID__", html.escape(d.name)).replace("__META__", dump(meta))
            .replace("__IMGS__", dump(imgs)).replace("__ITEMS__", dump(items)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", help="기본: <세션 폴더>/replay.html")
    a = ap.parse_args()
    d = Path(a.session)
    out = Path(a.out) if a.out else d / "replay.html"
    out.write_text(build(d), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
