"""시뮬레이터 스트레스: 지금 판단을 고치지 않고 새 조건에서 돌려 본다 (층 v3 + 섬 있는 로비).

  .venv/bin/python src/sim/stress.py            # 조건 5개 × (층 번호/이름, 로비 번호/이름)
  .venv/bin/python src/sim/stress.py --n 10

조건 (시뮬레이터 쪽만 바꿈, 판단 코드는 그대로):
  기준       지금까지와 같음
  없는 목적지 층에 없는 방·장소를 찾음 (번호 499, 이름 '도서관'). 좋은 끝 = '찾지 못했습니다' (안전한 실패)
  센서 오차   휴대폰 방위: 사람마다 치우침 N(0, BIAS°) + 사진마다 흔들림 N(0, JITTER°) (건물 안 자기장 왜곡),
             걸음 길이: 사람마다 ×N(1, STRIDE) (걸음 수 × 0.6m가 실제 걸은 거리와 다름)
  인식 놓침   보이는 문패·목적지를 사진마다 MISS 확률로 못 읽음
  센서+놓침   있는 목적지에 센서 오차와 인식 놓침을 함께
  모두       위 셋을 한꺼번에 (없는 목적지 + 센서 오차 + 인식 놓침)
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_floor  # noqa: E402
import floor as fw  # noqa: E402
import lobby  # noqa: E402
import navigator as nv  # noqa: E402
import world  # noqa: E402

BIAS, JITTER, STRIDE, MISS = 8.0, 10.0, 0.1, 0.2
MISSING = {"number": "499", "name": "도서관"}
COND = {"기준": set(), "없는 목적지": {"missing"}, "센서 오차": {"sensor"}, "인식 놓침": {"miss"},
        "센서+놓침": {"sensor", "miss"}, "모두": {"missing", "sensor", "miss"}}

ON = set()
_rng = random.Random(0)
_orig = {"fobs": fw.observe, "lobs": lobby.observe, "fsen": fw.FloorWalker.sensor, "lsen": lobby.LobbyWalker.sensor,
         "init": nv.Navigator.__init__}


def _noisy(sensor):
    def f(self):
        s = sensor(self)
        if "sensor" not in ON:
            return s
        if not hasattr(self, "_err"):
            self._err = (_rng.gauss(0, BIAS), max(0.7, _rng.gauss(1, STRIDE)))
        bias, stride = self._err
        h = s["heading"]
        return {"steps": int(s["steps"] * stride), "heading": None if h is None else (h + bias + _rng.gauss(0, JITTER)) % 360}
    return f


def _target(t):
    return MISSING["number" if t.isdigit() else "name"] if "missing" in ON else t


def _observe(orig):
    def f(world_, w, target):
        obs = orig(world_, w, _target(target))
        if "miss" in ON:
            obs["plates"] = [p for p in obs["plates"] if _rng.random() >= MISS]
            if obs.get("dest", {}).get("seen") and _rng.random() < MISS:
                obs["dest"] = {"seen": False, "side": "none", "m": None, "why": "none"}
        return obs
    return f


def _init(self, runner, target, *a, **k):
    _orig["init"](self, runner, _target(target), *a, **k)


fw.observe, lobby.observe = _observe(_orig["fobs"]), _observe(_orig["lobs"])
fw.FloorWalker.sensor, lobby.LobbyWalker.sensor = _noisy(_orig["fsen"]), _noisy(_orig["lsen"])
nv.Navigator.__init__ = _init


def summarize(name, rows):
    n = len(rows)
    if "missing" in ON:  # 좋은 끝은 '못 찾음'. 도착이라 하면 거짓 도착
        gave = [r for r in rows if r["end"] == "못 찾음"]
        false = sum(r["end"] in ("도착", "가능성 높음") for r in rows)
        t = statistics.median(r["t"] for r in gave) if gave else float("nan")
        return (f"{name:12s} 못 찾음 알림 {len(gave)}/{n} (중앙 {t:.0f}초), 거짓 도착 {false}, "
                f"끝없이 헤맴(600초) {sum(r['end'] == '시간 초과' for r in rows)}")
    ok = [r for r in rows if r["success"]]
    t = statistics.median(r["t"] for r in ok) if ok else float("nan")
    ends = {}
    for r in rows:
        if not r["success"]:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
    return f"{name:12s} 성공 {len(ok)}/{n}, 잘못 도착 {sum(r['wrong_arrival'] for r in rows)}, 시간(중) {t:.0f}초, 실패 {ends or '없음'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cond", nargs="*", default=list(COND))
    a = ap.parse_args()
    world.RANGE = 5.0
    for c in a.cond:
        ON.clear()
        ON.update(COND[c])
        _rng.seed(0)
        print(f"== {c}")
        for mode in ("number", "name"):
            rows = [bench_floor.episode(k, s, mode) for k in fw.FLOORS for s in range(a.n)]
            print("  " + summarize(f"층 {'번호' if mode == 'number' else '이름'}", rows))
        for mode in ("number", "name"):
            rows = [lobby.episode(W, s, mode, islands=3) for W in lobby.SIZES for s in range(a.n)]
            print("  " + summarize(f"로비섬 {'번호' if mode == 'number' else '이름'}", rows))


if __name__ == "__main__":
    main()
