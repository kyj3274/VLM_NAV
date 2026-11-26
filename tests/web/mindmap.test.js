// 머릿속 지도 그리기 시험 (node --test tests/web)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const MindMap = require('../../src/nav/web/mindmap.js');

// 시야 부채꼴 꼭짓점 중 시작점에서 가장 먼 거리 (m)
function farthest(view) {
  const m = { pos: [0, 0], head: 0, views: [view], trail: [[0, 0]] };
  const box = [-20, -20, 20, 20], S = 10, P = 24;
  const svg = MindMap.svg(m, { box, scale: S, readM: 5, seeM: 10 });
  const poly = svg.match(/<polygon points="([^"]+)"/)[1].split(' ').slice(1)
    .map((p) => p.split(',').map(Number)).map(([x, y]) => [(x - P) / S + box[0], box[3] - (y - P) / S]);
  return Math.max(...poly.map(([x, y]) => Math.hypot(x, y)));
}

test('깊이 칸별 거리가 있으면 부채꼴을 벽에서 자름, 없으면 10m까지', () => {
  assert.ok(Math.abs(farthest([0, 0, 0]) - 10) < 0.01);
  assert.ok(Math.abs(farthest([0, 0, 0, [2, 2, 3, 4, 3, 2, 2]]) - 4) < 0.01);  // 복도: 옆은 벽, 앞 4m에서 막힘
});
