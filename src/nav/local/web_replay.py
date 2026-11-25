"""웹 길안내 세션 하나(outputs/kyj/web_sessions/<세션>/)를 웹 화면과 같은 모양의 다시보기 HTML 한 장으로 만든다.
1초마다 대표 사진, 그때 말한 안내, 층·호실·장면, VLM 호출, 버튼 기록(일시 정지·엘리베이터·끝내기)을 순서대로 넘겨 본다.
사진은 줄여서 HTML 안에 넣으므로 파일 하나만 옮기면 어디서나 열린다.

  python web_replay.py <세션 폴더> [--out replay.html]
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

EVENT_SAY = {"start": "안내 시작", "pause": "일시 정지", "resume": "다시 시작", "segment": "엘리베이터에서 내림 (층·위치 다시 확인)",
             "end": "안내를 끝냄"}


def thumb(p, side=560):
    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def load(d):
    """log.jsonl → 화면 한 칸씩(초 기록 또는 버튼 기록). 예전 기록(버튼 기록 없음)도 읽는다."""
    items, imgs = [], {}
    dump = {}  # 예전 기록은 raw가 없으므로 web_ocr_dump.py 결과가 있으면 대신 씀
    if (d / "ocr_dump.jsonl").is_file():
        dump = {x["frame"]: x for x in map(json.loads, open(d / "ocr_dump.jsonl", encoding="utf-8"))}
    for line in open(d / "log.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "raw" not in r and r.get("key_frame") in dump:
            r["raw"] = dump[r["key_frame"]]
        if "event" in r:
            items.append({"kind": "event", "event": r["event"], "wall": r.get("wall"), "target": r.get("target"),
                          "mode": r.get("mode"), "key": r.get("key"), "quality": r.get("quality"), "camera": r.get("camera"), "pick": r.get("pick")})
            continue
        k = r["key_frame"]
        if k not in imgs and (d / k).is_file():
            imgs[k] = thumb(d / k)
        st = r["state"]
        items.append({"kind": "step", "img": k, "seg": r.get("segment", int(r["seq"].split("_")[-1])), "sec": r["sec"],
                      "wall": r.get("wall"), "say": r["say"], "rooms": r["rooms"], "floor": st.get("floor"),
                      "wrong_floor": st.get("wrong_floor"), "scene": st.get("scene"), "arrived": st.get("arrived"),
                      "vlm": [", ".join(v.get("why") or []) for v in r["vlm"]],
                      "advice": r.get("advice"), "seen": r.get("seen"), "ambiguous": r.get("ambiguous"),
                      "latency": round(r.get("latency_seconds", 0), 2), "frames": len(r["used_frames"]),
                      "target": r["route_target"],
                      # 대표 장면의 OCR 글자 전부와 YOLOE 물체(0.25 이상). 예전 기록에는 없음
                      "ocr": [[t["text"], t["score"]] for t in (r.get("raw") or {}).get("ocr", [])],
                      "objects": [[o["name"], o["conf"]] for o in (r.get("raw") or {}).get("objects", []) if o["conf"] >= 0.25],
                      "size": (r.get("raw") or {}).get("size")})
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
    <div class="muted">파란 줄 = 안내 문장, 주황 줄 = VLM 호출, 갈색 줄 = 버튼 기록 (일시 정지·엘리베이터·끝내기)</div>
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
        <span>OCR 글자</span><b id="sOcr">-</b>
        <span>찾은 물체</span><b id="sObj">-</b>
        <span>VLM 호출</span><b id="sVlm">0번</b>
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
const label = it => it.kind === "step" ? `영상${it.seg + 1} ${it.sec}초` : (it.wall || "").slice(11);
$("tgt").textContent = META.target;
$("seek").max = ITEMS.length - 1;
$("sum").innerHTML = [
  ["시작", META.start || "-"], ["안내한 시간", steps.length + "초"], ["영상 구간", META.segments + "개"],
  ["말한 안내", META.says + "번"], ["VLM 호출", META.vlm + "번"], ["일시 정지", META.pauses + "번"],
  ["처리 시간 중앙값", META.median + "초"], ["결과", META.arrived ? "도착 선언" : "도착 선언 없음"],
].map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("");
$("all").innerHTML = steps.flatMap(s => s.say.map(t => `<li><span class="muted">${label(s)}</span> ${esc(t)}</li>`)).join("");
ITEMS.forEach((it, i) => {
  const x = (i / Math.max(1, ITEMS.length - 1) * 100) + "%";
  const mk = c => { const m = document.createElement("i"); m.className = c; m.style.left = x; m.title = label(it); $("bar").appendChild(m); };
  if (it.kind === "event" && it.event !== "start") mk("ev");
  if (it.kind === "step" && it.say.length) mk("say");
  if (it.kind === "step" && it.vlm.length) mk("vlm");
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
      const extra = it.event === "start" ? ` (방식 ${it.mode || "-"}, 대표 장면 ${it.key || "-"}, 화질 ${it.quality || "-"}, 카메라 ${it.camera || "-"}, 사진 고르기 ${it.pick || "-"})` : "";
      logs.push(`<div>${t}<span class="ev">${esc((META.ev[it.event] || it.event) + extra)}</span></div>`);
      continue;
    }
    lastStep = it; vlm += it.vlm.length;
    for (const r of it.rooms) if (!rooms.includes(r)) rooms.push(r);
    for (const w of it.vlm) logs.push(`<div>${t}<span class="vlm">VLM 호출: ${esc(w)}</span></div>`);
    if (it.advice) logs.push(`<div>${t}<span class="vlm">VLM 추천: ${esc(it.advice.action || "답 없음")}${it.advice.veto ? " → 규칙이 막음 (" + esc(it.advice.veto) + ")" : ""}</span></div>`);
    for (const s of it.say) { said = s; saidAt = k; logs.push(`<div>${t}<span class="${it.arrived ? "arrived" : ""}">${esc(s)}</span></div>`); }
    if (it.arrived) arrived = true;
  }
  const st = arrived ? ["arrived", "도착"] : paused ? ["paused", "일시 정지"] : ["running", "안내 중"];
  $("state").dataset.s = st[0]; $("state").textContent = st[1]; $("veil").hidden = !paused;
  $("say").textContent = said; $("say").classList.toggle("fresh", saidAt === cur);
  $("log").innerHTML = logs.reverse().join("");
  $("sRooms").textContent = rooms.length ? rooms.join(", ") : "-";
  $("sVlm").textContent = vlm + "번";
  if (lastStep) {
    $("img").src = IMGS[lastStep.img] || "";
    $("clock").textContent = `영상${lastStep.seg + 1} · ${lastStep.sec}초` + (lastStep.wall ? " · " + lastStep.wall.slice(11) : "");
    $("sFloor").textContent = lastStep.floor ? lastStep.floor + "층" + (lastStep.wrong_floor ? " (목적지 층 아님)" : "") : "모름";
    $("sScene").textContent = lastStep.scene || "-";
    const oc = lastStep.ocr || [];
    $("sOcr").innerHTML = oc.length ? oc.map(([t, sc]) => `<span style="opacity:${sc >= 0.5 ? 1 : .45}">${esc(t)}<small class="muted">${sc.toFixed(2)}</small></span>`).join(" · ") : (lastStep.size ? "없음" : "기록 없음 (예전 기록)");
    const ob = lastStep.objects || [];
    $("sObj").textContent = ob.length ? ob.map(([n, c]) => `${n} ${c.toFixed(2)}`).join(", ") : (lastStep.size ? "없음" : "기록 없음 (예전 기록)");
    const sn = lastStep.seen || {};
    const seen = Object.entries(sn.plates || {}).map(([r, c]) => `${r}호(${c}시)`).concat(sn.landmarks || [], sn.floors || []);
    $("sSeen").textContent = seen.length ? seen.join(", ") : "-";
    $("sTime").textContent = `${lastStep.latency}초 (사진 ${lastStep.frames}장 봄${lastStep.size ? ", " + lastStep.size.join("×") : ""})` + (lastStep.ambiguous?.length ? " · 다시 봄: " + lastStep.ambiguous.join(", ") : "");
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", help="기본: <세션 폴더>/replay.html")
    a = ap.parse_args()
    d = Path(a.session)
    items, imgs = load(d)
    steps = [x for x in items if x["kind"] == "step"]
    if not steps:
        raise SystemExit(f"{d}: 초 기록이 없습니다")
    lat = sorted(x["latency"] for x in steps)
    start = next((x["wall"] for x in items if x.get("wall")), None) or "{}-{}-{} {}:{}:{}".format(d.name[:4], d.name[4:6], d.name[6:8], d.name[9:11], d.name[11:13], d.name[13:15])
    meta = {"target": steps[0]["target"], "start": start, "segments": len({x["seg"] for x in steps}),
            "says": sum(len(x["say"]) for x in steps), "vlm": sum(len(x["vlm"]) for x in steps),
            "pauses": sum(x.get("event") == "pause" for x in items), "median": lat[len(lat) // 2],
            "arrived": any(x["arrived"] for x in steps), "ev": EVENT_SAY}
    dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    out = Path(a.out) if a.out else d / "replay.html"
    out.write_text(PAGE.replace("__SID__", html.escape(d.name)).replace("__META__", dump(meta))
                   .replace("__IMGS__", dump(imgs)).replace("__ITEMS__", dump(items)), encoding="utf-8")
    print(f"{out}  ({len(steps)}초, 안내 {meta['says']}번, VLM {meta['vlm']}번, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
