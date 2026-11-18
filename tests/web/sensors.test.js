// 가짜 센서 값으로 센서 즉시 안내 시험 (node --test tests/web)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { SensorGuide } = require('../../src/nav/web/sensors.js');

// 걷기: 0.5초마다 한 걸음(가속도 봉우리 12.5, 평소 9.8), 50Hz
function walk(g, t0, seconds) {
  const out = [];
  for (let t = t0; t < t0 + seconds; t += 0.02) {
    const ph = (t - t0) % 0.5;
    out.push(...g.onMotion(ph < 0.08 ? 12.5 : 9.8, t));
  }
  return out;
}

test('뒤로 돌기 → 방위가 135도 이상 바뀌면 "돌았습니다"', () => {
  const g = new SensorGuide();
  g.onHeading(10, 0);
  g.onGuidance(['412호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요.'], '뒤로 돌기', 0.1);
  const said = [];
  for (let k = 1; k <= 10; k++) said.push(...g.onHeading(10 + k * 17, 0.1 + k * 0.2));  // 170도까지 천천히
  assert.deepStrictEqual(said, ['돌았습니다. 이제 앞으로 천천히 가세요.']);
});

test('뒤로 돌기: 359→190(북을 넘어 돎)도 회전으로 셈', () => {
  const g = new SensorGuide();
  g.onHeading(359, 0);
  g.onGuidance(['엘리베이터로 돌아가 4층을 누르세요.'], '엘리베이터로 돌아가기', 0);
  assert.deepStrictEqual(g.onHeading(300, 1), []);
  assert.deepStrictEqual(g.onHeading(200, 2), ['돌았습니다. 이제 앞으로 천천히 가세요.']);
});

test('조금 흔들린 것(30도)은 돈 것으로 보지 않음, 시간이 지나면 기다리지 않음', () => {
  const g = new SensorGuide();
  g.onHeading(90, 0);
  g.onGuidance(['뒤로 돌아가세요.'], '뒤로 돌기', 0);
  assert.deepStrictEqual(g.onHeading(120, 1), []);
  assert.deepStrictEqual(g.onHeading(100, 20), []);
  assert.strictEqual(g.turn, null);
});

test('왼쪽으로 돌라고 했는데 오른쪽으로 돌면 한 번만 알림', () => {
  const g = new SensorGuide();
  g.onHeading(0, 0);
  g.onGuidance(['왼쪽으로 도세요.'], '왼쪽', 0);
  assert.deepStrictEqual(g.onHeading(50, 1), ['반대쪽입니다. 왼쪽으로 도세요.']);
  assert.deepStrictEqual(g.onHeading(60, 1.5), []);
  assert.deepStrictEqual(g.onHeading(290, 3), ['왼쪽으로 돌았습니다.']);
});

test('방위를 모를 때 안내가 오면 첫 방위부터 잼', () => {
  const g = new SensorGuide();
  g.onGuidance(['뒤로 돌아가세요.'], '뒤로 돌기', 0);
  assert.deepStrictEqual(g.onHeading(40, 0.5), []);
  assert.deepStrictEqual(g.onHeading(230, 2), ['돌았습니다. 이제 앞으로 천천히 가세요.']);
});

test('걸음 수: 10초 동안 0.5초마다 → 약 20걸음', () => {
  const g = new SensorGuide();
  walk(g, 0, 10);
  assert.ok(g.steps >= 19 && g.steps <= 21, `steps=${g.steps}`);
});

test('서 있을 때(작은 흔들림)는 걸음으로 세지 않음', () => {
  const g = new SensorGuide();
  for (let t = 0; t < 10; t += 0.02) g.onMotion(9.8 + 0.3 * Math.sin(t * 7), t);
  assert.strictEqual(g.steps, 0);
});

test('"약 5미터 … 그쪽으로" 뒤 4미터(약 6걸음) 걸으면 "말씀드린 곳 근처" 한 번', () => {
  const g = new SensorGuide();
  walk(g, 0, 2);
  g.onGuidance(['엘리베이터가 1시 방향 약 5미터에 있습니다. 그쪽으로 천천히 가세요.'], '직진', 2);
  const said = walk(g, 2, 6);
  assert.deepStrictEqual(said, ['말씀드린 곳 근처입니다. 천천히 가세요.']);
});

test('"멈춰 주세요" 뒤 계속 걸으면 한 번 더 "멈춰 주세요", 멈추면 말하지 않음', () => {
  const g = new SensorGuide();
  g.onGuidance(['멈춰 주세요, 확인 중입니다.'], '목적지 접근', 0);
  assert.deepStrictEqual(walk(g, 0, 4), ['멈춰 주세요.']);
  const h = new SensorGuide();
  h.onGuidance(['멈춰 주세요, 확인 중입니다.'], '목적지 접근', 0);
  for (let t = 0; t < 4; t += 0.02) h.onMotion(9.8, t);
  assert.strictEqual(h.steps, 0);
});
