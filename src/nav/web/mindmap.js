// 머릿속 지도 그리기: Navigator.mind_map()이 주는 것을 SVG로 (웹 관찰자 화면 monitor.html과 시뮬레이터 재생 화면 src/sim/view_floor.py가 같이 씀).
// 좌표는 휴대폰 걸음 수·방위로 잡은 것 (x 동, y 북, m). 걸음 수·방위가 없으면 위치를 몰라 그리지 않는다.
//   시야 두 겹: 진한 부채꼴 = 문패를 읽을 수 있는 거리 안에서 본 곳, 옅은 부채꼴 = 문·장소는 보이지만 글자는 못 읽는 거리
//   파란 점 = 읽은 문패, 주황 ? = 문은 보이는데 번호를 못 읽은 곳(가까이 가서 확인할 후보), ★ = 목적지 추정 위치
//   ● = 갈림길, 실선 = 가 본 가지, 끝 막대 = 끝까지 가 봄, 점선 = 안 가 본 가지, 보라 선 = 걸은 길, 삼각형 = 지금 위치와 방향
'use strict';
const MindMap = (() => {
  const C = { read: '#2f6fdb', see: '#2f6fdb', trail: '#9c6ade', junc: '#2b8a3e', edge: 'currentColor', dead: '#888',
              cur: '#2f6fdb', plate: '#2f6fdb', unread: '#e8590c', target: '#d9480f', user: 'currentColor', mute: '#888' };
  const V = (deg) => [Math.sin(deg * Math.PI / 180), Math.cos(deg * Math.PI / 180)];
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function box(m, pad, seeM) {
    const pts = [[0, 0]].concat(m.trail || [], m.junc || [], Object.values(m.plates || {}), m.unread || [],
      m.spot ? [m.spot] : [], m.pos ? [m.pos] : []);
    for (const v of m.views || []) { const [dx, dy] = V(v[2]); pts.push([v[0] + dx * seeM * 0.6, v[1] + dy * seeM * 0.6]); }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad];
  }

  function svg(m, opt = {}) {
    const readM = opt.readM ?? 5, seeM = opt.seeM ?? 10, fov = opt.fov ?? 60, S = opt.scale ?? 14, P = 24;
    const b = opt.box || box(m, 4, seeM);
    const X = (x) => P + (x - b[0]) * S, Y = (y) => P + (b[3] - y) * S;
    const W = (b[2] - b[0]) * S + 2 * P, H = (b[3] - b[1]) * S + 2 * P;
    let s = `<svg${opt.id ? ` id="${opt.id}"` : ''} viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">`;
    if (!m || !m.pos) {
      return s + `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${C.mute}" font-size="14">걸음 수·방위가 아직 없음 (위치 모름)</text></svg>`;
    }
    const wedge = (v, r) => {
      const [x, y, h] = v, pts = [];
      for (let a = -fov / 2; a <= fov / 2; a += 10) { const [dx, dy] = V(h + a); pts.push(`${X(x + dx * r)},${Y(y + dy * r)}`); }
      return `<polygon points="${X(x)},${Y(y)} ${pts.join(' ')}"/>`;
    };
    const views = m.views || [];
    s += `<g fill="${C.see}" opacity=".08">${views.map((v) => wedge(v, seeM)).join('')}</g>`;
    s += `<g fill="${C.read}" opacity=".2">${views.map((v) => wedge(v, readM)).join('')}</g>`;
    if ((m.trail || []).length > 1) {
      s += `<polyline points="${m.trail.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="${C.trail}" stroke-width="2.5" stroke-opacity=".75" stroke-linejoin="round"/>`;
    }
    const labeled = new Set();
    for (const [j, c, k] of m.br || []) {
      const ed = m.edges[k], a = m.junc[j], v = V(c);
      let q = null;  // 가지 방향에 있는 다른 끝 갈림길
      for (const o of ed.ends) {
        if (o === j) continue;
        const d = [m.junc[o][0] - a[0], m.junc[o][1] - a[1]], dot = d[0] * v[0] + d[1] * v[1];
        if (dot > 0 && (!q || dot > q[1])) q = [o, dot];
      }
      const here = !q && k === m.edge;  // 지금 걷는 가지: 갈림길에서 지금 위치까지
      const end = q ? m.junc[q[0]] : here ? m.pos : [a[0] + v[0] * 5, a[1] + v[1] * 5];
      const col = ed.dead ? C.dead : k === m.edge ? C.cur : C.edge;
      s += `<line x1="${X(a[0])}" y1="${Y(a[1])}" x2="${X(end[0])}" y2="${Y(end[1])}" stroke="${col}" stroke-width="3" stroke-opacity=".7"${!q && !here && !ed.dead ? ' stroke-dasharray="6 5"' : ''}/>`;
      if (!q && ed.dead) s += `<line x1="${X(end[0]) - 6 * v[1]}" y1="${Y(end[1]) - 6 * v[0]}" x2="${X(end[0]) + 6 * v[1]}" y2="${Y(end[1]) + 6 * v[0]}" stroke="${col}" stroke-width="3"/>`;
      if (!q && !here && !ed.dead) s += `<text x="${X(end[0])}" y="${Y(end[1]) - 6}" text-anchor="middle" font-size="10" fill="${C.mute}">안 가 봄</text>`;
      if (ed.nums && !labeled.has(k)) {
        labeled.add(k);
        s += `<text x="${(X(a[0]) + X(end[0])) / 2 + 6}" y="${(Y(a[1]) + Y(end[1])) / 2 + 14}" font-size="10" fill="${C.mute}">${ed.nums[0]}~${ed.nums[1]}</text>`;
      }
    }
    (m.junc || []).forEach((p) => { s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="6" fill="${C.junc}"/>`; });
    for (const p of m.unread || []) {
      s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="6" fill="none" stroke="${C.unread}" stroke-width="2"/><text x="${X(p[0])}" y="${Y(p[1]) + 4}" text-anchor="middle" font-size="10" font-weight="700" fill="${C.unread}">?</text>`;
    }
    for (const [n, p] of Object.entries(m.plates || {})) {
      const tg = n === m.target;
      s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${tg ? 5 : 3.5}" fill="${tg ? C.target : C.plate}"/><text x="${X(p[0]) + 6}" y="${Y(p[1]) - 5}" font-size="${tg ? 12 : 10}" font-weight="${tg ? 700 : 400}" fill="${tg ? C.target : C.plate}">${esc(n)}</text>`;
    }
    if (m.spot) s += `<text x="${X(m.spot[0])}" y="${Y(m.spot[1]) + 6}" text-anchor="middle" font-size="18" fill="${C.target}">★</text>`;
    if (m.head !== null && m.head !== undefined) {
      const u = m.head * Math.PI / 180, tri = [[0, 11], [-7, -7], [7, -7]].map(([dx, f]) =>
        `${X(m.pos[0]) + f * Math.sin(u) + dx * Math.cos(u)},${Y(m.pos[1]) - f * Math.cos(u) + dx * Math.sin(u)}`).join(' ');
      s += `<polygon points="${tri}" fill="${C.user}"/>`;
    } else s += `<circle cx="${X(m.pos[0])}" cy="${Y(m.pos[1])}" r="6" fill="${C.user}"/>`;
    return s + '</svg>';
  }

  function caption(m) {
    if (!m || !m.dir) return '';
    return `목적지 방향: ${m.dir}${m.p_ahead != null ? ` (앞일 확률 ${m.p_ahead})` : ''}`;
  }

  // 시뮬레이터처럼 사진마다 그때 것만 받으면 k번째까지 쌓아 full과 같은 모양으로
  // (가까이 와서 읽은 문패 근처의 '못 읽은 문'은 지움: navigator.py MIND_SAME_M과 같은 2.5m)
  function accumulate(snaps, k, same = 2.5) {
    const out = Object.assign({}, snaps[k], { trail: [], views: [], plates: {}, unread: [] });
    const far = (u, ps) => ps.every((q) => Math.hypot(u[0] - q[0], u[1] - q[1]) > same);
    for (let j = 0; j <= k; j++) {
      const s = snaps[j];
      if (!s) continue;
      if (s.view) { out.trail.push([s.view[0], s.view[1]]); out.views.push(s.view); }
      const ps = Object.values(s.plates || {});
      Object.assign(out.plates, s.plates || {});
      out.unread = out.unread.filter((u) => far(u, ps)).concat(s.unread || []);
    }
    return out;
  }

  return { svg, box, caption, accumulate };
})();
if (typeof module !== 'undefined') module.exports = MindMap;
