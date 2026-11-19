// 센서 즉시 안내: 서버(Claude) 응답을 기다리지 않고 휴대폰 센서로 바로 말한다.
//  1) 나침반 회전 확인: "뒤로 돌아가세요"·"엘리베이터로 돌아가…" 같은 안내 뒤 방위가 충분히 바뀌면 "돌았습니다"
//  2) 걸음 수: 가속도 크기의 봉우리를 세어 걸음 수를 센다.
//     - "약 N미터에 있습니다 … 그쪽으로 가세요" 뒤 N미터만큼 걸으면 "말씀드린 곳 근처입니다"
//     - "멈춰 주세요" 뒤에도 계속 걸으면 한 번 더 "멈춰 주세요"
// 순수 계산만 한다(브라우저 API를 부르지 않음). 그래서 Node에서 가짜 센서 값으로 시험한다(tests/web/sensors.test.js).
// 맥에는 방위·가속도 센서가 없어 실제 동작은 휴대폰에서만 확인할 수 있다.
'use strict';
(function (root) {
  const STEP_M = 0.7;          // 한 걸음 길이(미터, 대략)
  const TURN_BACK = 135;       // 뒤로 돌기: 이만큼(도) 돌면 확인
  const TURN_SIDE = 65;        // 왼쪽·오른쪽: 이만큼
  const TURN_TIMEOUT = 15;     // 초: 이 안에 안 돌면 기다리지 않음(서버 안내가 다시 말함)
  const STOP_STEPS = 4;        // "멈춰 주세요" 뒤 이만큼 더 걸으면 다시 말함
  const STOP_WINDOW = 6;       // 초
  const STEP_MIN_GAP = 0.3;    // 초: 걸음 사이 최소 간격
  const STEP_RISE = 1.2;       // m/s²: 평균보다 이만큼 높은 봉우리를 걸음으로

  const diff = (a, b) => ((b - a + 540) % 360) - 180;  // a→b 회전(도), 오른쪽(시계 방향)이 +

  class SensorGuide {
    constructor() {
      this.heading = null;
      this.steps = 0;
      this.turn = null;       // {kind: back|left|right, start, t}
      this.walk = null;       // {meters, steps0}
      this.stopAt = null;     // {steps0, t, said}
      this.avg = null; this.peak = 0; this.rising = false; this.lastStepT = -1;
    }

    // 서버 안내가 왔을 때. 말할 문장들을 돌려준다(지금은 항상 빈 목록: 설정만).
    onGuidance(sayList, action, t) {
      const text = (sayList || []).join(' ');
      if (!text) return [];
      if (/뒤로 돌아|엘리베이터로 돌아가/.test(text)) this.turn = { kind: 'back', start: this.heading, t };
      else if (/왼쪽으로 도세요|왼쪽으로 돌/.test(text)) this.turn = { kind: 'left', start: this.heading, t };
      else if (/오른쪽으로 도세요|오른쪽으로 돌/.test(text)) this.turn = { kind: 'right', start: this.heading, t };
      const m = text.match(/약 (\d+(?:\.\d+)?)미터에 있습니다\.?\s*그쪽으로/);
      if (m) this.walk = { meters: parseFloat(m[1]), steps0: this.steps };
      if (/멈춰 주세요/.test(text)) this.stopAt = { steps0: this.steps, t, said: false };
      if (action === '도착') { this.walk = null; this.turn = null; }
      return [];
    }

    // 나침반 방위(도, 북=0, 시계 방향). 말할 문장 목록을 돌려준다.
    onHeading(h, t) {
      const out = [];
      if (typeof h !== 'number' || isNaN(h)) return out;
      this.heading = h;
      const tr = this.turn;
      if (!tr) return out;
      if (tr.start === null) { tr.start = h; tr.t = t; return out; }  // 안내 때 방위를 몰랐으면 지금부터
      if (t - tr.t > TURN_TIMEOUT) { this.turn = null; return out; }
      const d = diff(tr.start, h);
      if (tr.kind === 'back' && Math.abs(d) >= TURN_BACK) {
        out.push('돌았습니다. 이제 앞으로 천천히 가세요.'); this.turn = null;
      } else if (tr.kind === 'left' && d <= -TURN_SIDE) {
        out.push('왼쪽으로 돌았습니다.'); this.turn = null;
      } else if (tr.kind === 'right' && d >= TURN_SIDE) {
        out.push('오른쪽으로 돌았습니다.'); this.turn = null;
      } else if ((tr.kind === 'left' && d >= 45) || (tr.kind === 'right' && d <= -45)) {
        if (!tr.wrongSaid) { out.push(`반대쪽입니다. ${tr.kind === 'left' ? '왼쪽' : '오른쪽'}으로 도세요.`); tr.wrongSaid = true; }
      }
      return out;
    }

    // 가속도(중력 포함) 크기 m/s². 걸음이면 세고, 말할 문장 목록을 돌려준다.
    onMotion(mag, t) {
      const out = [];
      if (typeof mag !== 'number' || isNaN(mag)) return out;
      this.avg = this.avg === null ? mag : this.avg * 0.95 + mag * 0.05;  // 느린 평균(중력 + 자세)
      const high = mag > this.avg + STEP_RISE;
      if (high && !this.rising && t - this.lastStepT >= STEP_MIN_GAP) {
        this.rising = true; this.steps += 1; this.lastStepT = t;
        if (this.walk && (this.steps - this.walk.steps0) * STEP_M >= Math.max(1, this.walk.meters - 1)) {
          out.push('말씀드린 곳 근처입니다. 천천히 가세요.'); this.walk = null;
        }
        const s = this.stopAt;
        if (s && !s.said && t - s.t <= STOP_WINDOW && this.steps - s.steps0 >= STOP_STEPS) {
          out.push('멈춰 주세요.'); s.said = true;
        }
      } else if (!high) this.rising = false;
      return out;
    }
  }

  SensorGuide.diff = diff;
  if (typeof module !== 'undefined' && module.exports) module.exports = { SensorGuide };
  else root.SensorGuide = SensorGuide;
})(typeof window !== 'undefined' ? window : globalThis);
