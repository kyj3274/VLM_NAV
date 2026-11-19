// 휴대폰 길안내 웹 (아이폰 Chrome = WebKit 기준). claude_nav web/app.js의 권한·소리·말하기 방식을 가져와 겹쳐 보내기로 바꿈.
// - 확인 질문·두 번 누르기·시작 버튼 없음: 목적지를 알아들으면 "○호로 안내하겠습니다" 후 바로 안내, 못 알아들으면 "다시 말씀해 주세요"
// - 화면을 누르는 그 순간(터치 이벤트 안)에 카메라·방향 센서·움직임 센서·소리·화면 꺼짐 방지를 한꺼번에 요청 (iOS는 터치 안에서만 허용)
// - 음성 입력: webkitSpeechRecognition(ko-KR) 먼저, 없으면 녹음해서 서버(맥 로컬 Whisper)로
// - 겹쳐 보내기: 1.5초마다 가장 선명한 장을 응답을 기다리지 않고 보냄(동시 최대 3개). 서버가 보낸 순서대로 상태에 넣고 늦은 응답은 버림
// - 영상 파일로 시험: 카메라 대신 고른 영상을 실제 속도로 재생하며 같은 방식으로 보냄
// API 키는 여기 없다. 서버 /api만 부르고 접속 코드는 X-Access-Code 헤더로 보낸다.
'use strict';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => performance.now();

const S = {
  code: '', sid: null, target: '', name: '', running: false, loopActive: false, listening: false, prep: null,
  heading: null, oriAsked: false, oriOk: null, motionAsked: false, motionOk: null,
  stream: null, cameraDenied: false, ultrawideId: null, cameraLabel: '', wakeLock: null, zoomMin: null, camInfo: null,
  prevNet: null,
  audio: null, speechUnlocked: false, voice: null, lastSpoken: '', lastSpeech: '', priorityActive: false,
  holding: false, holdStart: 0, holdTimer: null, voiceMode: null, rec: null, recState: null,
  files: [], fileIdx: 0, t0: 0, inflight: 0, lastSeq: 0, lastAppliedAt: 0, lastRtt: null, slowSaid: false,
  failSpoken: false, sent: 0, localBusy: 0, lastReply: null, notified: {}, guide: null, lastAction: '',
  intervalMs: 1500, shots: 6, maxInflight: 3, slowWarnMs: 8000, maxHoldMs: 8000, minHoldMs: 400, minPeakRms: 0.01,
};

function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function save(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 사생활 모드 */ } }
function drop(k) { try { localStorage.removeItem(k); } catch (e) { /* 무시 */ } }

// ---------------------------------------------------------------- 음성·효과음
function pickVoice() {
  if (!('speechSynthesis' in window)) return;
  S.voice = speechSynthesis.getVoices().find((v) => (v.lang || '').toLowerCase().startsWith('ko')) || null;
}
if ('speechSynthesis' in window) { pickVoice(); speechSynthesis.addEventListener?.('voiceschanged', pickVoice); }

function showCaption(text, err = false) {
  for (const id of ['caption', 'caption0']) { $(id).textContent = text; $(id).classList.toggle('err', !!err); }
}
// opts.force: 같은 문장도 읽음 / opts.queue: 앞 음성을 끊지 않음 / opts.priority: 끝날 때까지 다른 안내가 끊지 못함
function say(text, opts = {}) {
  if (!text) return;
  $('live').textContent = text;
  if (!opts.force && text === S.lastSpoken) return;
  S.lastSpoken = text;
  if (!('speechSynthesis' in window)) { showCaption(text, opts.err); return; }
  const queue = opts.queue || (S.priorityActive && !opts.priority && !opts.force);
  if (!queue) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ko-KR';
  if (S.voice) u.voice = S.voice;
  u.onstart = () => showCaption(text, opts.err);
  if (opts.priority) { S.priorityActive = true; u.onend = u.onerror = () => { S.priorityActive = false; }; }
  speechSynthesis.speak(u);
  if (!queue) showCaption(text, opts.err);
}
function unlockAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!S.audio && AC) S.audio = new AC();
  if (S.audio && S.audio.state === 'suspended') S.audio.resume();
  if ('speechSynthesis' in window && !S.speechUnlocked) {
    const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); S.speechUnlocked = true;
  }
}
function tone(freqs, dur = 0.12) {
  if (!S.audio) return;
  let t = S.audio.currentTime + 0.01;
  for (const f of freqs) {
    const o = S.audio.createOscillator(), g = S.audio.createGain();
    o.type = 'sine'; o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.35, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(S.audio.destination);
    o.start(t); o.stop(t + dur + 0.02);
    t += dur + 0.04;
  }
}
const CUE = {
  recStart: () => tone([988], 0.09), recEnd: () => tone([740], 0.09), stage: () => tone([660, 880]),
  arrived: () => tone([523, 659, 784, 1047], 0.16), left: () => tone([784, 523], 0.18), error: () => tone([300], 0.25),
  sensor: () => tone([1175], 0.07),
};

// ---------------------------------------------------------------- 서버
async function api(path, opts = {}) {
  const headers = Object.assign({ 'X-Access-Code': S.code }, opts.headers || {});
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), opts.timeout || 25000);
  try {
    const r = await fetch(path, { method: opts.method || 'GET', body: opts.body, headers, signal: ctl.signal });
    if (r.status === 401) { const e = new Error('code'); e.code = 401; throw e; }
    if (!r.ok) throw new Error('http ' + r.status);
    return await r.json();
  } finally { clearTimeout(timer); }
}
const postJSON = (path, obj, timeout) =>
  api(path, { method: 'POST', body: JSON.stringify(obj), headers: { 'Content-Type': 'application/json' }, timeout });

// ---------------------------------------------------------------- 권한 묶음 (터치 이벤트 안에서 첫 줄에 부름)
function onOrientation(e) {
  let h = null;
  if (typeof e.webkitCompassHeading === 'number' && !isNaN(e.webkitCompassHeading)) h = e.webkitCompassHeading; // iOS
  else if (e.absolute && typeof e.alpha === 'number') h = (360 - e.alpha) % 360;
  if (h === null) return;
  S.heading = h;
  if (S.running && $('sensors').checked) sensorSay(S.guide.onHeading(h, now() / 1000));
}
function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x === null) return;
  const mag = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  const outs = S.guide.onMotion(mag, now() / 1000);
  if (S.running && $('sensors').checked) sensorSay(outs);
}
function sensorSay(outs) {
  for (const t of outs) { CUE.sensor(); say(t, { force: true, queue: S.priorityActive }); }
}
function requestOrientation() {
  if (S.oriAsked || !('DeviceOrientationEvent' in window)) return Promise.resolve(S.oriOk);
  S.oriAsked = true;
  const attach = () => {
    window.addEventListener('deviceorientation', onOrientation);
    window.addEventListener('deviceorientationabsolute', onOrientation);
    return (S.oriOk = true);
  };
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    return DeviceOrientationEvent.requestPermission()
      .then((st) => (st === 'granted' ? attach() : (S.oriOk = false))).catch(() => (S.oriOk = false));
  }
  return Promise.resolve(attach());
}
function requestMotion() {
  if (S.motionAsked || !('DeviceMotionEvent' in window)) return Promise.resolve(S.motionOk);
  S.motionAsked = true;
  const attach = () => { window.addEventListener('devicemotion', onMotion); return (S.motionOk = true); };
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    return DeviceMotionEvent.requestPermission()
      .then((st) => (st === 'granted' ? attach() : (S.motionOk = false))).catch(() => (S.motionOk = false));
  }
  return Promise.resolve(attach());
}
function fileMode() { return S.files.length > 0; }
function cameraConstraints() {
  const hi = $('quality').value === '1080';
  const base = { width: { ideal: hi ? 1920 : 1280 }, height: { ideal: hi ? 1080 : 720 } };
  return ($('ultrawide').checked && S.ultrawideId)   // 초광각 장치가 따로 보이면 그것, 아니면 열고 나서 zoom 0.5 (applyUltraZoom)
    ? Object.assign({ deviceId: { exact: S.ultrawideId } }, base)
    : Object.assign({ facingMode: { ideal: 'environment' } }, base);
}
function openCamera(force = false) {
  if (fileMode()) return Promise.resolve(true);
  if (S.stream && !force) return Promise.resolve(true);
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return Promise.resolve(false);
  const p = navigator.mediaDevices.getUserMedia({ video: cameraConstraints(), audio: false }); // await 앞에서 바로
  return p.then(async (stream) => {
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    S.stream = stream; S.cameraDenied = false;
    const v = $('video');
    v.removeAttribute('src'); v.srcObject = stream;
    try { await v.play(); } catch (e) { /* 다음 터치에서 재생 */ }
    const tr = stream.getVideoTracks()[0];
    S.cameraLabel = tr ? tr.label : '';
    await probeCameras(tr);
    await applyUltraZoom(tr);
    return true;
  }).catch((e) => { S.cameraDenied = /NotAllowed|Security|Permission/i.test((e && e.name) || ''); return false; });
}
// 초광각 확인 (2026-11-19): ① enumerateDevices에 '초광각/Ultra Wide' 후면 카메라가 따로 보이는지
// ② 지금 카메라(아이폰은 보통 '듀얼 와이드'·'트리플' 가상 카메라)가 zoom 1 미만(0.5x)을 지원하는지(getCapabilities().zoom).
// 둘 중 하나면 설정의 초광각을 켤 수 있다. 결과는 세션 기록(camera.probe)에 남긴다.
async function probeCameras(track) {
  const info = { devices: [], zoom: null, ultrawideDevice: false };
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    info.devices = devs.filter((d) => d.kind === 'videoinput').map((d) => d.label || '(이름 없음)');
    const uw = devs.find((d) => d.kind === 'videoinput' && /ultra ?wide|울트라 ?와이드|초광각/i.test(d.label));
    if (uw) { S.ultrawideId = uw.deviceId; info.ultrawideDevice = true; }
  } catch (e) { /* 무시 */ }
  try {
    const caps = track && track.getCapabilities ? track.getCapabilities() : {};
    if (caps.zoom) { info.zoom = { min: caps.zoom.min, max: caps.zoom.max }; if (caps.zoom.min < 1) S.zoomMin = caps.zoom.min; }
  } catch (e) { /* 무시 */ }
  if (S.ultrawideId || S.zoomMin) $('ultrawide').disabled = false;
  info.label = track ? track.label : '';
  S.camInfo = info;
  return info;
}
async function applyUltraZoom(track) {
  if (!track || S.ultrawideId || !S.zoomMin) return;
  try { await track.applyConstraints({ advanced: [{ zoom: $('ultrawide').checked ? S.zoomMin : 1 }] }); } catch (e) { /* 무시 */ }
}
function keepAwake() {
  if (S.wakeLock) return Promise.resolve(true);
  if (!('wakeLock' in navigator)) return Promise.resolve(false);
  return navigator.wakeLock.request('screen').then((l) => {
    S.wakeLock = l; l.addEventListener?.('release', () => { S.wakeLock = null; }); return true;
  }).catch(() => false);
}
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && S.running) keepAwake(); });

function prepareInGesture() {
  unlockAudio();
  const ori = requestOrientation();
  const mot = requestMotion();
  const cam = openCamera();
  const wake = keepAwake();
  S.prep = Promise.all([cam, ori, mot, wake]).then(([camOk, oriOk, motOk, wakeOk]) => ({ camOk, oriOk, motOk, wakeOk }));
  return S.prep;
}
function reportPermissions(p) {
  const once = (key, text) => { if (!S.notified[key]) { S.notified[key] = true; say(text, { force: true, queue: true }); } };
  if (!p.camOk) {
    say(S.cameraDenied ? '카메라 권한이 없어 안내할 수 없습니다. 설정에서 허용해 주세요.'
      : '카메라를 열 수 없어 안내할 수 없습니다.', { force: true, err: true });
    CUE.error();
    return false;
  }
  if (p.oriOk === false) once('ori', '방향 센서 권한이 없어 회전 확인 없이 안내합니다.');
  if (p.motOk === false) once('mot', '움직임 센서 권한이 없어 걸음 수 없이 안내합니다.');
  if (!p.wakeOk) once('wake', '화면 꺼짐 방지를 쓸 수 없어 화면을 켜 두셔야 합니다.');
  return true;
}

// ---------------------------------------------------------------- 목적지 말하기 (길게 누르기)
const SRCtor = () => window.SpeechRecognition || window.webkitSpeechRecognition || null;
function pickMime() {
  if (!window.MediaRecorder) return null;
  for (const m of ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']) {
    if (!MediaRecorder.isTypeSupported || MediaRecorder.isTypeSupported(m)) return m;
  }
  return '';
}
function voiceMode() { return S.voiceMode || (SRCtor() ? 'sr' : (pickMime() !== null ? 'rec' : 'none')); }

function startCapture() {
  const mode = voiceMode();
  const st = { mode, text: '', finalText: '', error: null, ended: null, chunks: [], peak: 0 };
  S.recState = st;
  if (mode === 'sr') {
    const rec = new (SRCtor())();
    rec.lang = 'ko-KR'; rec.continuous = true; rec.interimResults = true;
    st.ended = new Promise((res) => { rec.onend = res; });
    rec.onresult = (ev) => {
      let fin = '', inter = '';
      for (let i = 0; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) fin += t; else inter += t;
      }
      st.finalText = fin; st.text = (fin + inter).trim();
    };
    rec.onerror = (ev) => { st.error = ev.error || 'error'; };
    try { rec.start(); } catch (e) { st.error = 'start-failed'; }
    S.rec = rec;
  } else if (mode === 'rec') {
    const mime = pickMime();
    st.ended = navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
      .then((stream) => new Promise((resolve) => {
        if (!S.holding) { stream.getTracks().forEach((t) => t.stop()); resolve(); return; }
        const mr = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
        st.mime = mr.mimeType || mime || 'audio/mp4';
        mr.ondataavailable = (e) => { if (e.data && e.data.size) st.chunks.push(e.data); };
        mr.onstop = () => { stream.getTracks().forEach((t) => t.stop()); clearInterval(st.meter); resolve(); };
        if (S.audio) {
          const an = S.audio.createAnalyser(); an.fftSize = 2048;
          S.audio.createMediaStreamSource(stream).connect(an);
          const buf = new Float32Array(an.fftSize);
          st.meter = setInterval(() => {
            an.getFloatTimeDomainData(buf);
            let s = 0; for (const x of buf) s += x * x;
            st.peak = Math.max(st.peak, Math.sqrt(s / buf.length));
          }, 50);
        } else st.peak = 1;
        mr.start(250);
        S.rec = mr;
      }))
      .catch((e) => { st.error = (e && e.name) || 'mic-error'; });
  } else st.error = 'no-voice';
}
function onHoldDown(e) {
  if (e.target.closest('.nohold') || S.holding) return;
  if (e.button !== undefined && e.button !== 0) return;
  e.preventDefault();
  S.holding = true; S.holdStart = now(); S.listening = true;
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  prepareInGesture();   // 이 터치 안에서 권한 요청
  CUE.recStart();
  startCapture();       // 음성 인식 시작도 터치 안에서
  $('rec-overlay').hidden = false;
  S.holdTimer = setTimeout(finishHold, S.maxHoldMs);
}
function onHoldUp() { if (S.holding) finishHold(); }
async function finishHold() {
  S.holding = false;
  clearTimeout(S.holdTimer);
  $('rec-overlay').hidden = true;
  CUE.recEnd();
  const dur = now() - S.holdStart, st = S.recState;
  try { if (S.rec) S.rec.stop(); } catch (e) { /* 이미 멈춤 */ }
  S.rec = null;
  if (st && st.ended) await Promise.race([st.ended, sleep(3000)]);
  S.listening = false;
  if (!st) return;
  if (st.mode === 'sr' && st.error && st.error !== 'no-speech' && st.error !== 'aborted') {
    S.voiceMode = pickMime() !== null ? 'rec' : 'none';   // 브라우저 인식이 안 되면 다음부터 녹음 + Whisper
    say('다시 말씀해 주세요.', { force: true });
    return;
  }
  if (st.mode === 'rec' && st.error) {
    say(/NotAllowed|Security/i.test(st.error) ? '마이크 권한이 없습니다. 글자로 입력해 주세요.' : '다시 말씀해 주세요.', { force: true });
    return;
  }
  if (st.mode === 'none') { say('이 브라우저는 말로 입력할 수 없습니다. 글자로 입력해 주세요.', { force: true }); return; }
  if (dur < S.minHoldMs) { say('다시 말씀해 주세요.', { force: true }); return; }
  if (st.mode === 'sr') {
    const text = (st.finalText || st.text).trim();
    if (!text) { say('다시 말씀해 주세요.', { force: true }); return; }
    await submitText(text, 'speech_recognition', dur);
  } else {
    if (st.peak < S.minPeakRms || !st.chunks.length) { say('다시 말씀해 주세요.', { force: true }); return; }
    await uploadVoice(new Blob(st.chunks, { type: st.mime || 'audio/mp4' }));
  }
}

// ---------------------------------------------------------------- 목적지 보내기
function sessionMeta() {
  return { camera: { label: S.cameraLabel, ultrawide: $('ultrawide').checked, quality: $('quality').value, probe: S.camInfo,
    ultra_by: $('ultrawide').checked ? (S.ultrawideId ? 'device' : S.zoomMin ? `zoom ${S.zoomMin}` : 'none') : null },
    device: { ua: navigator.userAgent, orientation: S.oriOk, motion: S.motionOk, voice: voiceMode() },
    mode: fileMode() ? 'file' : 'camera', files: S.files.map((f) => f.name) };
}
async function submitText(text, source, ms) {
  try {
    const body = Object.assign({ destination: text, source, transcript_ms: ms ? Math.round(ms) : null }, sessionMeta());
    const r = S.sid && S.running ? await postJSON(`/api/session/${S.sid}/destination`, body)
      : await postJSON('/api/session', body);
    await onDestinationReply(r);
  } catch (e) { e.code === 401 ? codeRejected() : netFail(); }
}
async function uploadVoice(blob) {
  const fd = new FormData();
  const ext = /mp4|aac/.test(blob.type) ? 'm4a' : /webm/.test(blob.type) ? 'webm' : /ogg/.test(blob.type) ? 'ogg' : 'bin';
  fd.append('audio', blob, `voice.${ext}`);
  const change = S.sid && S.running;
  if (!change) fd.append('meta', JSON.stringify(sessionMeta()));
  try {
    const r = await api(change ? `/api/session/${S.sid}/voice` : '/api/voice', { method: 'POST', body: fd, timeout: 40000 });
    await onDestinationReply(r);
  } catch (e) { e.code === 401 ? codeRejected() : netFail(); }
}
async function onDestinationReply(r) {
  const text = (r.say || []).join(' ');
  if (r.status === 'retry') { say(text || '다시 말씀해 주세요.', { force: true }); return; }
  if (r.status !== 'ok') { say(text || '목적지를 정하지 못했습니다.', { force: true }); return; }
  const changed = S.running && S.sid === r.session_id;
  S.sid = r.session_id; S.target = r.target; S.name = r.name;
  if (changed) { say(text, { force: true, priority: true }); CUE.stage(); renderTop(); return; }
  const p = await (S.prep || Promise.resolve({ camOk: !!S.stream || fileMode(), oriOk: S.oriOk, motOk: S.motionOk, wakeOk: !!S.wakeLock }));
  if (!reportPermissions(p)) return;
  say(text, { force: true, priority: true });   // "410호로 안내하겠습니다" → 바로 시작
  startGuidance();
}
$('dest-form').addEventListener('submit', (e) => {
  e.preventDefault();
  prepareInGesture();
  $('dest').blur();
  const text = $('dest').value.trim();
  if (!text) { say('목적지를 입력해 주세요.', { force: true }); return; }
  submitText(text, 'keyboard', null);
});

// ---------------------------------------------------------------- 안내 화면
function startGuidance() {
  $('dest-page').hidden = true; $('setup').hidden = true; $('guide').hidden = false;
  S.t0 = now(); S.lastSeq = 0; S.inflight = 0; S.sent = 0; S.localBusy = 0; S.lastAppliedAt = now(); S.slowSaid = false;
  S.lastAction = ''; S.guide = new SensorGuide();
  const v = $('video');
  if (fileMode()) { S.fileIdx = 0; playFile(0); }
  else if (v.paused) v.play().catch(() => {});
  S.running = true;
  CUE.stage();
  renderTop();
  if (!S.loopActive) loop();
}
function playFile(i) {
  const v = $('video');
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  v.srcObject = null;
  v.src = URL.createObjectURL(S.files[i]);
  v.play().catch(() => {});
}
$('video').addEventListener('ended', async () => {
  if (!S.running || !fileMode()) return;
  if (S.fileIdx + 1 < S.files.length) {   // 다음 파일 = 엘리베이터에서 내린 뒤
    S.fileIdx += 1;
    await sendEvent('exited', { file: S.files[S.fileIdx].name });
    playFile(S.fileIdx);
  } else {
    await sleep(2500);   // 마지막 응답을 기다림
    if (S.running) { postJSON(`/api/session/${S.sid}/event`, { type: 'end', detail: { why: '영상 끝' } }).catch(() => {}); stopGuidance('영상이 끝났습니다.'); }
  }
});
function stopGuidance(msg) {
  S.running = false;
  if (msg) say(msg, { force: true });
  if (fileMode()) $('video').pause();
  $('guide').hidden = true; $('dest-page').hidden = false;
  renderModeLine();
}
function setError(text) { $('errline').hidden = !text; $('errline').textContent = text || ''; }
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function renderTop(r) {
  r = r || S.lastReply || {};
  $('t-dest').textContent = S.name || '-';
  const st = [r.action || '시작', r.floor ? r.floor + '층' : '층 모름'];
  if (r.detail && r.detail.target_pending) st.push('목적지 확인 중');
  $('t-state').textContent = st.join(' · ');
  $('t-time').textContent = S.lastRtt !== null ? `응답 ${(S.lastRtt / 1000).toFixed(1)}초` : '응답 -';
  $('busy').hidden = S.inflight === 0;
}
function renderDetail(d) {
  if (!d) return;
  const li = (arr, f) => arr && arr.length ? '<ul>' + arr.map((x) => `<li>${f(x)}</li>`).join('') + '</ul>' : '<p>없음</p>';
  const c = d.counts || {};
  $('detail').innerHTML =
    `<h3>장소</h3><p>${esc(d.place)}${d.door && d.door !== 'none' ? ' · 앞에 ' + (d.door === 'glass' ? '유리문' : '문') : ''}</p>` +
    `<h3>Claude가 읽은 문패</h3>${li(d.plates, (p) => `${p.n ? esc(p.n) + '호' : '번호 못 읽음'} ${esc(p.name)} · ${esc(p.side)}${p.m ? ' ' + p.m + 'm' : ''}`)}` +
    `<h3>층 표시 · 엘리베이터 · 화살표</h3><p>${esc(d.floor_sign || '층 표시 없음')} · ${d.elevator ? `엘리베이터(${esc(d.elevator.why)}) ${esc(d.elevator.side)}${d.elevator.m ? ' ' + d.elevator.m + 'm' : ''}` : '엘리베이터 없음'}` +
    `${(d.arrows || []).length ? ' · 화살표 ' + d.arrows.map((a) => esc(a.to) + (a.label ? '(' + esc(a.label) + ')' : '')).join(', ') : ''}</p>` +
    `<h3>확대 재확인</h3><p>${d.recheck ? `${esc(d.recheck.asked)} → ${esc(d.recheck.read || '못 읽음')} (${esc(d.recheck.why)}${d.recheck.match === false ? ', 다름' : ''})` : '안 함'}</p>` +
    `<h3>규칙 상태</h3><p>층 ${esc(d.floor || '모름')}${d.wrong_floor ? ' (목적지 층 아님)' : ''} · 행동 ${esc(d.action)} · 확정 문패 ${esc((d.confirmed || []).join(', ') || '-')}` +
    `${d.target_pending ? ' · <b>목적지 확인 중(호출 2번 전)</b>' : ''}${d.arrived ? ' · 도착' : ''}</p>` +
    `<h3>최근 기록</h3>${li((d.events || []).slice().reverse(), esc)}` +
    `<h3>보내기</h3><p>보냄 ${S.sent} · 반영 ${c.applied || 0} · 늦어서 버림 ${c.stale || 0} · 바빠서 안 보냄 ${S.localBusy + (c.busy || 0)} · ` +
    `갱신 간격 중앙값 ${d.update_interval_median ?? '-'}초 · 인식 ${d.latency_s ?? '-'}초 · 찍은 뒤 반영 ${d.lag_s ?? '-'}초 · 크레딧 ${d.credit}</p>`;
}
$('more').addEventListener('click', () => {
  const open = $('detail').hidden;
  $('detail').hidden = !open;
  $('more').setAttribute('aria-expanded', String(open));
  $('more').textContent = open ? '접기' : '자세히';
});

// ---------------------------------------------------------------- 사진 고르기 (라플라시안 분산) + 겹쳐 보내기
const small = document.createElement('canvas');
function sharpness() {
  const v = $('video');
  if (!v.videoWidth) return -1;
  const w = 160, h = Math.max(1, Math.round(v.videoHeight * w / v.videoWidth));
  small.width = w; small.height = h;
  const c = small.getContext('2d', { willReadFrequently: true });
  c.drawImage(v, 0, 0, w, h);
  const d = c.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, lap = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
    s += lap; s2 += lap * lap; n++;
  }
  return n ? s2 / n - (s / n) ** 2 : -1;
}
function grabFull() {   // 원본 해상도 그대로 (줄이지 않음: 문패 글자)
  const v = $('video'), c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  return c;
}
const toJpeg = (canvas) => new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.9));
async function bestFrame() {   // 1.5초 동안 6장을 보고 가장 선명한 장
  let best = null, bestS = -1, at = null, heading = null;
  const tb = now();
  for (let i = 0; i < S.shots && S.running && !S.listening; i++) {
    const s = sharpness();
    if (s > bestS) { bestS = s; best = grabFull(); at = (now() - S.t0) / 1000; heading = S.heading; }
    await sleep(S.intervalMs / S.shots);
  }
  return best && best.width ? { canvas: best, t: at, heading, pickMs: now() - tb, pickEnd: (now() - S.t0) / 1000 } : null;
}
async function loop() {
  S.loopActive = true;
  try {
    while (S.running) {
      if (S.listening) { await sleep(200); continue; }
      const f = await bestFrame();
      if (!S.running) break;
      if (!f || S.listening) { await sleep(200); continue; }
      if (S.inflight >= S.maxInflight) { S.localBusy++; continue; }   // 동시 3개면 이번 장은 보내지 않음
      sendFrame(f);   // 기다리지 않음 (겹쳐 보내기)
    }
  } finally { S.loopActive = false; }
}
async function quickConfirm() {   // 문 앞 빠른 확인: 다음 1.5초를 기다리지 않고 곧바로 2장 중 선명한 장을 보냄
  let best = null, bestS = -1, at = null, heading = null;
  const tb = now();
  for (let i = 0; i < 2 && S.running; i++) {
    const s = sharpness();
    if (s > bestS) { bestS = s; best = grabFull(); at = (now() - S.t0) / 1000; heading = S.heading; }
    if (i === 0) await sleep(120);
  }
  if (best && best.width && S.running) sendFrame({ canvas: best, t: at, heading, pickMs: now() - tb, pickEnd: (now() - S.t0) / 1000 }, 'confirm');
}
async function sendFrame(f, kind) {
  S.inflight++; S.sent++;
  renderTop();
  const fd = new FormData();
  const te = now();
  const blob = await toJpeg(f.canvas);
  const encMs = now() - te;
  fd.append('image', blob, 'frame.jpg');
  fd.append('t', f.t.toFixed(2));
  if (f.heading !== null) fd.append('heading', f.heading.toFixed(1));
  if (S.guide) fd.append('steps', String(S.guide.steps));
  const t0 = now();
  // 지연 측정 (2026-11-19): 휴대폰 시계(시작부터 초)로 고르기 끝·인코딩·보낸 시각, 사진 용량, 바로 전에 끝난 요청의 왕복
  fd.append('client', JSON.stringify({ pick_ms: Math.round(f.pickMs), size: [f.canvas.width, f.canvas.height], inflight: S.inflight,
    pick_end: f.pickEnd != null ? +f.pickEnd.toFixed(3) : null, enc_ms: Math.round(encMs), bytes: blob.size,
    sent_at: +((t0 - S.t0) / 1000).toFixed(3), prev: S.prevNet }));
  try {
    const r = await api(`/api/session/${S.sid}/${kind === 'confirm' ? 'confirm' : 'frame'}`, { method: 'POST', body: fd, timeout: 30000 });
    if (r.status === 'ok') S.lastRtt = Math.round(now() - t0);
    S.prevNet = { seq: r.seq, rtt_ms: Math.round(now() - t0), got_at: +((now() - S.t0) / 1000).toFixed(3), server_s: r.server_s };
    S.failSpoken = false; setError('');
    onFrameReply(r);
  } catch (e) {
    if (e.code === 401) { codeRejected(); return; }
    netFail();
  } finally { S.inflight--; renderTop(); }
}
function onFrameReply(r) {
  switch (r.status) {
    case 'ok': {
      const newer = r.seq > S.lastSeq;
      if (newer) {
        S.lastSeq = r.seq; S.lastReply = r; S.lastAppliedAt = now(); S.slowSaid = false;
        renderTop(r); renderDetail(r.detail);
        if (r.error) setError('인식 오류: ' + String(r.error).slice(0, 40));
      }
      if (!S.running) return;
      const text = (r.say || []).join(' ');
      if (text) {
        if (r.arrived) CUE.arrived();
        else if (r.action !== S.lastAction && /뒤로|돌아가/.test(text)) CUE.left();
        else if (r.action !== S.lastAction) CUE.stage();
        say(text, { queue: !newer });   // 늦게 온(먼저 보낸) 응답은 끊지 않고 뒤에
        S.lastSpeech = text;
        if (S.guide) S.guide.onGuidance(r.say, r.action, now() / 1000);
      }
      if (newer) S.lastAction = r.action;
      if (r.arrived && newer) stopAfterArrival();
      else if (r.confirm && newer && S.running) quickConfirm();
      break;
    }
    case 'limit': case 'stopped':
      if (S.running) stopGuidance((r.say || []).join(' ') || '안내를 멈췄습니다.');
      break;
    default: break;   // busy·stale: 말하지 않음
  }
}
async function stopAfterArrival() {
  S.running = false;   // 더 보내지 않음. 도착 문장은 이미 읽는 중
  await sleep(4000);
  stopGuidance();
}
setInterval(() => {   // 응답이 오래 없으면 한 번 "잠시 멈춰 주세요"
  if (!S.running || S.listening || !$('slowwarn').checked || S.slowSaid) return;
  if (now() - S.lastAppliedAt > S.slowWarnMs) { S.slowSaid = true; say('잠시 멈춰 주세요.', { force: true }); }
}, 1000);

function netFail() {
  setError('연결 오류');
  if (!S.failSpoken) { say('연결이 불안정합니다.', { force: true, err: true }); CUE.error(); S.failSpoken = true; }
}
function codeRejected() {
  drop('nav_code'); drop('nav_setup_done');
  say('접속 코드가 맞지 않습니다. 처음 설정으로 돌아갑니다.', { force: true });
  S.running = false;
  showSetup();
}

// ---------------------------------------------------------------- 버튼
async function sendEvent(type, detail) {
  if (!S.sid) return null;
  try {
    const r = await postJSON(`/api/session/${S.sid}/event`, { type, detail: detail || {} });
    const text = (r.say || []).join(' ');
    if (text) { say(text, { force: true }); S.lastSpeech = text; }
    return r;
  } catch (e) { netFail(); return null; }
}
$('repeat').addEventListener('click', () => {
  unlockAudio();
  say(S.lastSpeech || '아직 안내가 없습니다.', { force: true });
  if (S.sid) postJSON(`/api/session/${S.sid}/event`, { type: 'repeat' }).catch(() => {});
});
$('exited').addEventListener('click', () => { unlockAudio(); CUE.stage(); sendEvent('exited'); });
$('end').addEventListener('click', async () => { unlockAudio(); await sendEvent('end'); stopGuidance(); });
for (const id of ['dest-page', 'guide']) {
  $(id).addEventListener('pointerdown', onHoldDown);
  $(id).addEventListener('contextmenu', (e) => e.preventDefault());
}
window.addEventListener('pointerup', onHoldUp);
window.addEventListener('pointercancel', onHoldUp);

// ---------------------------------------------------------------- 설정·점검
async function runChecks() {
  const items = [];
  const add = (ok, text) => items.push({ ok, text });
  add(!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    '카메라: ' + (navigator.mediaDevices ? '사용 가능 (처음 화면을 누를 때 묻습니다)' : '사용할 수 없음 (HTTPS 접속 확인)'));
  if ('speechSynthesis' in window) {
    pickVoice(); if (!S.voice) { await sleep(400); pickVoice(); }
    add(!!S.voice, '음성 안내: ' + (S.voice ? '한국어 목소리 있음' : '한국어 목소리를 찾지 못함 (기본 목소리)'));
  } else add(false, '음성 안내: 지원 안 됨');
  const m = voiceMode();
  add(m !== 'none', '목적지 말하기: ' + (m === 'sr' ? '브라우저 음성 인식' : m === 'rec' ? '녹음 후 맥 Whisper 받아쓰기' : '안 됨 (글자로 입력)'));
  add('DeviceOrientationEvent' in window, '방향 센서: ' + ('DeviceOrientationEvent' in window ? '있음 (회전 확인)' : '없음'));
  add('DeviceMotionEvent' in window, '움직임 센서: ' + ('DeviceMotionEvent' in window ? '있음 (걸음 수)' : '없음'));
  add('wakeLock' in navigator, '화면 꺼짐 방지: ' + ('wakeLock' in navigator ? '지원' : '미지원 (화면을 켜 두세요)'));
  $('checks').innerHTML = '';
  for (const it of items) {
    const el = document.createElement('li');
    el.className = it.ok ? 'ok' : 'bad'; el.textContent = it.text; $('checks').appendChild(el);
  }
  const bad = items.filter((i) => !i.ok).map((i) => i.text.split(':')[0]);
  say(bad.length ? `점검 완료. 안 되는 기능: ${bad.join(', ')}.` : '점검 완료. 모든 기능을 쓸 수 있습니다.', { force: true });
}
function renderModeLine() {
  $('mode-line').textContent = fileMode() ? `영상 파일로 시험: ${S.files.map((f) => f.name).join(' → ')}` : '';
  $('src').textContent = fileMode() ? `영상 ${S.files.length}개` : '카메라 사용';
}
function showSetup() { $('setup').hidden = false; $('dest-page').hidden = true; $('guide').hidden = true; }
function showDest() { $('setup').hidden = true; $('guide').hidden = true; $('dest-page').hidden = false; renderModeLine(); }
$('code-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  unlockAudio();
  S.code = $('code').value.trim();
  try { await api('/api/monitor'); }   // 코드 확인 (Claude 호출 없음)
  catch (err) { say(err.code === 401 ? '접속 코드가 맞지 않습니다.' : '서버에 연결할 수 없습니다.', { force: true }); return; }
  save('nav_code', S.code);
  await runChecks();
  save('nav_setup_done', '1');
});
$('cam-check').addEventListener('click', async () => {   // 카메라를 열어(권한) 후면 카메라 목록과 zoom 범위를 보여 주고 읽음
  unlockAudio();
  const ok = await openCamera(true);
  const info = S.camInfo || {};
  const lines = ok ? [`지금 카메라: ${info.label || '-'}`, `후면 카메라 목록: ${(info.devices || []).join(', ') || '없음'}`,
    `초광각 장치: ${info.ultrawideDevice ? '있음' : '없음'}`,
    `확대 범위(zoom): ${info.zoom ? `${info.zoom.min}~${info.zoom.max}` : '알 수 없음(지원 안 함)'}`] : ['카메라를 열 수 없습니다.'];
  $('cam-list').innerHTML = '';
  for (const t of lines) { const li = document.createElement('li'); li.textContent = t; $('cam-list').appendChild(li); }
  const can = !!(S.ultrawideId || S.zoomMin);
  say(ok ? (can ? '초광각을 쓸 수 있습니다. 설정에서 켤 수 있습니다.' : '초광각을 쓸 수 없습니다. 일반 카메라를 씁니다.') : '카메라를 열 수 없습니다.', { force: true });
});
$('ultrawide').addEventListener('change', () => {   // 장치가 따로면 다시 열고, zoom 방식이면 바로 바꿈
  if (!S.stream) return;
  if (S.ultrawideId) openCamera(true); else applyUltraZoom(S.stream.getVideoTracks()[0]);
});
$('to-dest').addEventListener('click', () => {
  unlockAudio();
  if (!S.code) { say('접속 코드를 먼저 넣어 주세요.', { force: true }); return; }
  showDest();
  say('화면을 길게 누른 채 목적지를 말하세요.', { force: true });
});
$('open-setup').addEventListener('click', showSetup);
$('files').addEventListener('change', () => { S.files = [...$('files').files]; renderModeLine(); });
$('clear-files').addEventListener('click', () => { S.files = []; $('files').value = ''; renderModeLine(); });
for (const id of ['quality', 'slowwarn', 'sensors']) {
  const v = load('nav_' + id);
  if (v !== null) { if ($(id).type === 'checkbox') $(id).checked = v === '1'; else $(id).value = v; }
  $(id).addEventListener('change', () => save('nav_' + id, $(id).type === 'checkbox' ? ($(id).checked ? '1' : '0') : $(id).value));
}
$('quality').addEventListener('change', () => { if (S.stream) openCamera(true); });

(async function init() {
  const code = load('nav_code');
  if (code && load('nav_setup_done') === '1') {
    S.code = code;
    try { await api('/api/monitor'); showDest(); return; } catch (e) { if (e.code !== 401) { showDest(); return; } }
  }
  if (code) $('code').value = code;
  showSetup();
})();
