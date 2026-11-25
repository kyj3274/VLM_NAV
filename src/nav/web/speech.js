// 안내 음성 대기열 (2026-11-23 사용자 지시 1·2).
// 브라우저(speechSynthesis)의 대기열에 맡기지 않고 앱이 한 문장씩 직접 재생한다.
//  - 2026-11-23 시험: 앞 음성이 나오는 중에 넣은(대기열) 문장 3개가 끝내 시작되지 않았고,
//    앞 음성을 끊고(cancel) 곧바로 넣은 도착 문장이 7ms 만에 시작 없이 끝났다(아이폰 WebKit).
//  - 그래서: 끊은 직후 GAP_MS 쉬고 말한다. START_MS 안에 시작되지 않거나 시작 없이 끝나면 한 번 다시 말한다.
//    우선 문장(목적지 확인·도착)은 다른 안내가 끊지 못하고 그 뒤로 줄을 선다.
//  - 같은 문장은 REPEAT_MS 안에서만 되풀이하지 않는다(전에는 계속 안 읽어 엘리베이터 대기 중 88초 무음).
// 순수 계산 + 주입한 음성 엔진만 쓴다 → Node에서 가짜 엔진으로 시험 (tests/web/speech.test.js).
'use strict';
(function (root) {
  const GAP_MS = 150;       // 끊은 직후 쉬는 시간
  const START_MS = 1000;    // 이 안에 시작되지 않으면 다시 말함
  const REPEAT_MS = 9000;   // 같은 문장은 이 시간이 지나면 다시 읽음
  const RETRIES = 1;

  class SpeechQueue {
    // env: { synth: {speak(u), cancel()}, makeUtterance(text) → u, now() ms, setTimeout, clearTimeout,
    //        log(kind, obj), onStart(item), onEnd(item), onFail(item) }
    constructor(env, opts = {}) {
      this.env = env;
      this.gapMs = opts.gapMs ?? GAP_MS;
      this.startMs = opts.startMs ?? START_MS;
      this.repeatMs = opts.repeatMs ?? REPEAT_MS;
      this.q = []; this.cur = null; this.gapUntil = 0; this.timer = null; this.watch = null;
      this.lastText = ''; this.lastAt = -1e12; this.n = 0;
    }

    priorityActive() { return !!(this.cur && this.cur.prio) || this.q.some((i) => i.prio); }

    // opts: force(같은 문장도), queue(앞 음성을 끊지 않음), priority(끊기지 않음), src, seq, err
    say(text, opts = {}) {
      if (!text) return false;
      const t = this.env.now();
      if (!opts.force && text === this.lastText && t - this.lastAt < this.repeatMs) {
        this.log('say_skip_same', { text, src: opts.src, seq: opts.seq ?? null, since_ms: Math.round(t - this.lastAt) });
        return false;
      }
      this.lastText = text; this.lastAt = t;
      const item = { id: ++this.n, text, src: opts.src || 'local', seq: opts.seq ?? null, prio: !!opts.priority,
        err: !!opts.err, tries: 0, started: false, cancelled: false };
      const curPrio = this.cur && this.cur.prio;
      if (opts.queue || (curPrio && !item.prio)) {
        // 늦게 온 응답·우선 문장이 나오는 중: 뒤에 선다 (우선 문장 뒤에 들어온 새 안내는 오래된 같은 종류를 밀어냄)
        if (curPrio && !opts.queue) this.q = this.q.filter((i) => i.prio);
        this.q.push(item);
        this.log('say_queued', { text, src: item.src, seq: item.seq, queue: true, prio: item.prio });
      } else {
        this.q = this.q.filter((i) => i.prio);   // 새 안내가 오래된 일반 안내를 대신함
        if (item.prio) this.q.unshift(item); else this.q.push(item);
        this.log('say_queued', { text, src: item.src, seq: item.seq, queue: false, prio: item.prio });
        if (this.cur && !curPrio) this.interrupt();
      }
      this.pump();
      return true;
    }

    interrupt() {
      const c = this.cur;
      if (!c) return;
      c.cancelled = true;
      this.cur = null;
      this.env.clearTimeout(this.watch);
      try { this.env.synth.cancel(); } catch (e) { /* 무시 */ }
      this.gapUntil = this.env.now() + this.gapMs;
      this.log('say_cancel', { text: c.text, src: c.src, seq: c.seq, started: c.started });
    }

    stop() {
      this.q = [];
      this.interrupt();
    }

    pump() {
      if (this.cur || !this.q.length) return;
      const wait = this.gapUntil - this.env.now();
      if (wait > 0) {
        if (!this.timer) this.timer = this.env.setTimeout(() => { this.timer = null; this.pump(); }, wait);
        return;
      }
      this.start(this.q.shift());
    }

    start(item) {
      this.cur = item;
      const u = this.env.makeUtterance(item.text);
      u.onstart = () => {
        if (item !== this.cur || item.started) return;
        item.started = true;
        this.env.clearTimeout(this.watch);
        this.log('say_start', { text: item.text, src: item.src, seq: item.seq, try: item.tries });
        if (this.env.onStart) this.env.onStart(item);
      };
      u.onend = () => {
        if (item.cancelled || item !== this.cur) return;
        if (!item.started) { this.retry(item, 'end_without_start'); return; }
        this.env.clearTimeout(this.watch);
        this.log('say_end', { text: item.text, src: item.src, seq: item.seq });
        if (this.env.onEnd) this.env.onEnd(item);
        this.cur = null;
        this.pump();
      };
      u.onerror = (ev) => {
        if (item.cancelled || item !== this.cur) return;
        this.retry(item, 'error:' + ((ev && ev.error) || '?'));
      };
      this.env.clearTimeout(this.watch);
      this.watch = this.env.setTimeout(() => {
        if (item === this.cur && !item.started) {
          try { this.env.synth.cancel(); } catch (e) { /* 무시 */ }
          this.retry(item, 'no_start_' + this.startMs + 'ms');
        }
      }, this.startMs);
      this.env.synth.speak(u);
    }

    retry(item, why) {
      this.env.clearTimeout(this.watch);
      this.cur = null;
      this.gapUntil = this.env.now() + this.gapMs;
      if (item.tries < RETRIES) {
        item.tries += 1;
        this.log('say_retry', { text: item.text, src: item.src, seq: item.seq, why });
        this.q.unshift(item);
      } else {
        this.log('say_fail', { text: item.text, src: item.src, seq: item.seq, why });
        if (this.env.onFail) this.env.onFail(item);
      }
      this.pump();
    }

    log(kind, obj) { if (this.env.log) this.env.log(kind, obj); }
  }

  SpeechQueue.GAP_MS = GAP_MS; SpeechQueue.START_MS = START_MS; SpeechQueue.REPEAT_MS = REPEAT_MS;
  if (typeof module !== 'undefined' && module.exports) module.exports = { SpeechQueue };
  else root.SpeechQueue = SpeechQueue;
})(typeof window !== 'undefined' ? window : globalThis);
