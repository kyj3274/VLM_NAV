// 가짜 음성 엔진으로 안내 음성 대기열 시험 (node --test tests/web)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { SpeechQueue } = require('../../src/nav/web/speech.js');

// 시계·타이머·음성 엔진 가짜. engine.mode: 'ok'(바로 시작) | 'silent'(시작 안 함) | 'endNoStart'(바로 끝, 시작 없음)
function setup(opts, extraEnv = {}) {
  let t = 0;
  const timers = [];
  const spoken = [];   // 실제로 시작된 문장
  const logs = [];
  const engine = { mode: 'ok', cur: null, queueMode: [] };
  const synth = {
    speak(u) {
      const mode = engine.queueMode.length ? engine.queueMode.shift() : engine.mode;
      engine.cur = u;
      if (mode === 'ok') { u.onstart && u.onstart(); spoken.push(u.text); }
      else if (mode === 'endNoStart') { u.onend && u.onend(); }
    },
    cancel() { const u = engine.cur; engine.cur = null; if (u && u.onerror) u.onerror({ error: 'canceled' }); },
  };
  const env = {
    synth, makeUtterance: (text) => ({ text }), now: () => t,
    setTimeout: (f, ms) => { const h = { at: t + ms, f }; timers.push(h); return h; },
    clearTimeout: (h) => { const i = timers.indexOf(h); if (i >= 0) timers.splice(i, 1); },
    log: (k, o) => logs.push([k, o.text]),
    ...extraEnv,
  };
  const sq = new SpeechQueue(env, opts);
  const tick = (ms) => {   // 시간을 흘리며 타이머 실행
    const end = t + ms;
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      const h = timers[0];
      if (!h || h.at > end) break;
      timers.shift(); t = h.at; h.f();
    }
    t = end;
  };
  const finish = () => { const u = engine.cur; engine.cur = null; if (u && u.onend) u.onend(); };   // 지금 문장이 다 읽힘
  return { sq, engine, spoken, logs, tick, finish };
}

test('늦게 온 응답(대기열)은 앞 음성이 끝나면 앱이 이어서 읽음 (시험 3: 대기열 문장 3개가 끝내 안 나옴)', () => {
  const { sq, spoken, finish } = setup();
  sq.say('4층으로 알겠습니다. 목적지 층입니다.', { priority: true });
  sq.say('목적지 같습니다. 잠시 멈춰 주세요.', { src: 'server', seq: 107 });   // 우선 문장 중: 뒤에 섬
  assert.deepStrictEqual(spoken, ['4층으로 알겠습니다. 목적지 층입니다.']);
  finish();
  assert.deepStrictEqual(spoken, ['4층으로 알겠습니다. 목적지 층입니다.', '목적지 같습니다. 잠시 멈춰 주세요.']);
});

test('끊은 직후에는 잠깐 쉬고 말하며, 시작 없이 끝나면 한 번 다시 말함 (413 세션 도착 문장)', () => {
  const { sq, engine, spoken, logs, tick } = setup();
  sq.say('목적지 같습니다. 잠시 멈춰 주세요.');
  engine.queueMode = ['endNoStart'];   // 아이폰: 끊은 뒤 바로 넣은 문장이 시작 없이 끝남
  sq.say('왼쪽 앞에 413호 문이 있습니다.', { priority: true, force: true });
  assert.deepStrictEqual(spoken, ['목적지 같습니다. 잠시 멈춰 주세요.']);   // 아직 쉬는 중
  tick(SpeechQueue.GAP_MS);
  tick(SpeechQueue.GAP_MS);
  assert.deepStrictEqual(spoken.slice(-1), ['왼쪽 앞에 413호 문이 있습니다.']);
  assert.ok(logs.some(([k]) => k === 'say_retry'));
});

test('1초 안에 시작되지 않으면 다시 말하고, 두 번 다 안 되면 실패로 기록', () => {
  const { sq, engine, spoken, logs, tick } = setup();
  engine.mode = 'silent';
  let failed = null;
  sq.env.onFail = (item) => { failed = item.text; };
  sq.say('엘리베이터에서 내렸습니다.');
  tick(1000);
  assert.ok(logs.some(([k]) => k === 'say_retry'));
  tick(2000);
  assert.strictEqual(failed, '엘리베이터에서 내렸습니다.');
  engine.mode = 'ok';
  sq.say('다음 안내');
  tick(500);
  assert.deepStrictEqual(spoken, ['다음 안내']);   // 실패 뒤에도 다음 안내는 나옴
});

test('우선 문장(도착)은 새 안내가 끊지 못함', () => {
  const { sq, spoken, finish } = setup();
  sq.say('정면에 410호 문이 있습니다.', { priority: true });
  sq.say('복도를 따라 천천히 직진하세요.');
  sq.say('목적지 같습니다.');   // 같은 종류는 가장 새 것만 남음
  finish();
  assert.deepStrictEqual(spoken, ['정면에 410호 문이 있습니다.', '목적지 같습니다.']);
});

test('새 일반 안내는 지금 일반 안내를 끊고 오래된 대기 문장을 버림', () => {
  const { sq, spoken, tick, finish } = setup();
  sq.say('하나');
  sq.say('둘', { queue: true });
  sq.say('셋');
  tick(200);
  finish();
  assert.deepStrictEqual(spoken, ['하나', '셋']);
});

test('같은 문장: 9초 안에는 되풀이하지 않고 9초가 지나면 다시 읽음 (엘리베이터 대기 88초 무음)', () => {
  const { sq, spoken, tick, finish } = setup();
  const s = '엘리베이터를 기다리고 있습니다. 문이 열리면 타고 4층을 누르세요.';
  sq.say(s); finish();
  tick(5000); sq.say(s); finish();
  tick(5000); sq.say(s); finish();
  assert.deepStrictEqual(spoken, [s, s]);
});

test('onEnd: 문장이 끝나면 그 문장(seq)으로 알림 (옆 비추기 문장이 끝난 뒤 빠른 확인 사진, 2026-11-24 병합)', () => {
  const ends = [];
  const { sq, spoken, finish } = setup(undefined, { onEnd: (item) => ends.push(item.seq) });
  sq.say('멈춰 주세요. 410호 문이 왼쪽에 있을 것 같습니다.', { seq: 7 });
  assert.strictEqual(spoken.length, 1);
  assert.deepStrictEqual(ends, []);
  finish();
  assert.deepStrictEqual(ends, [7]);
});
