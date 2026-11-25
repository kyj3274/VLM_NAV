"""공통 판단 core/nav_graph.py: 휴대폰 방위로 돌았는지 판단, 끝까지 가 본 쪽 (2026-11-25 시뮬레이터 v1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nav" / "core"))

from arrival import TargetTracker  # noqa: E402
from nav_graph import CorridorGraph  # noqa: E402


def walk(g, t0, rooms, side, heading):
    """1초마다 문패 하나를 3m 앞 같은 쪽 벽에서 봄."""
    for i, r in enumerate(rooms):
        t = t0 + i
        g.sense_heading(t, heading)
        g.see(t, r, 1.0, side, 3.0)
        g.set_ok(r)
    return t0 + len(rooms)


def test_heading_confirms_turn_without_guessing():
    """'뒤로 돌아' 뒤 방위가 180° 바뀌면 그 사진부터 돈 것으로 본다 (가설 없이). 돈 뒤 같은 문패가 반대쪽에 보여도 같은 벽."""
    g = CorridorGraph()
    t = walk(g, 0, ["312", "311", "310"], "왼쪽", 0)
    assert g.direction("314", now=t)["direction"] == "뒤쪽"
    g.turned(t)
    t = walk(g, t + 1, ["310", "311"], "오른쪽", 180)
    d = g.direction("314", now=t)
    assert d["direction"] == "앞쪽" and g.turn_t is None and len(g.sense_flips) == 1


def test_dead_end_from_depth_not_from_turning():
    """끝까지 가 봤는지는 '스스로 돎'이 아니라 깊이(앞이 4m 안에서 막힘)로 안다. 끝을 본 쪽에 목적지가 있다는 판단은 낮춘다.
    실험 건물 3층: 왼쪽 벽 313→314→315만 보고 '309는 뒤쪽(끝 쪽)'이라 하던 것 → 그쪽 끝을 본 뒤엔 그쪽으로 보내지 않음."""
    g = CorridorGraph()
    t = walk(g, 0, ["313", "314", "315"], "왼쪽", 0)
    assert g.direction("309", now=t)["direction"] == "뒤쪽"
    g.turned(t)
    g.sense_heading(t + 1, 180)           # 안내대로 돎
    g.sense_heading(t + 20, 0)            # 이유 없이 다시 돎: 끝으로 보지 않음
    assert g.ends == set()
    g.sense_heading(t + 21, 180)          # 다시 끝 쪽으로
    g.see_depth(t + 22, 3.0, 270)         # 옆을 비춤(방위 90° 차이): 세지 않음
    g.see_depth(t + 23, 3.0, 270)
    assert g.ends == set()
    g.see_depth(t + 24, 3.0, 180)         # 걷는 방향으로 앞이 막힘 2장 연속 → 끝
    g.see_depth(t + 25, 3.0, 180)
    assert g.ends == {-1}
    g.sense_heading(t + 26, 0)
    assert g.direction("309", now=t + 27)["direction"] != "뒤쪽"


def test_missed_target_between_neighbors_is_behind_not_near():
    """목적지(407)를 놓치고 405·409를 지나 한참 왔으면 '근처'가 아니라 '뒤쪽' (2026-11-25 시뮬레이터 v2, 인식 놓침).
    전에는 가장 가까운 문패 3개의 번호 사이라서 계속 '근처'라고 했다."""
    g = CorridorGraph()
    t = walk(g, 0, ["403", "405", "409", "411"], "왼쪽", 0)
    assert g.direction("407", now=t)["direction"] == "근처"
    assert g.direction("407", now=t + 10)["direction"] == "뒤쪽"


def test_turn_after_seeing_target_sends_back():
    """목적지를 3m 앞에서 본 뒤 돌았으면 옆을 비추게 하지 않고 뒤로 돌게 함 (core/arrival.py)."""
    from arrival import TargetTracker
    a = TargetTracker("419")
    a.seen(0, 3.0, "왼쪽", heading=0)
    assert a.unseen(2, heading=180)[0].startswith("419호는 뒤쪽") and not a.asking()
    assert a.unseen(3, heading=180) == []          # 한 번만 말함
    assert "왼쪽" in a.unseen(5, heading=0)[0] and a.asking()  # 다시 돌아 걸으면 이어서 추정 → 옆 확인


def test_side_check_keeps_asked_side():
    """옆을 비추는 중에 본 목적지는 화면 속 위치가 아니라 비추라고 한 쪽 (왼쪽으로 돌리면 문패가 화면 오른쪽에 보임)."""
    a = TargetTracker("406")
    a.ask_t, a.ask_side = 10.0, "왼쪽"
    a.seen(11.0, 2.2, "오른쪽", steps=20, heading=0)
    assert a.mem["side"] == "왼쪽"


def test_far_side_target_in_open_space_gives_clock_not_side_check():
    """넓은 곳에서 12m 밖 화면 왼쪽에 본 목적지: '12m 가면 옆'이 아니라 그쪽 시계 방향 (평면 위치, 시뮬레이터 로비)."""
    a = TargetTracker("카페")
    a.seen(0.0, 12.0, "왼쪽", steps=0, heading=0)
    out = []
    for k in range(1, 21):  # 북쪽으로 12m 곧장
        out += a.unseen(k, steps=k, heading=0)
    text = " ".join(out)
    assert "시 방향" in text and "가능성이 높습니다" not in text


def test_corridor_target_still_side_check():
    """복도(옆 1.2m)에서는 전처럼 다가가면 그쪽 옆을 비추게 함."""
    a = TargetTracker("410")
    a.seen(0.0, 4.0, "오른쪽", steps=0, heading=90)
    out = []
    for k in range(1, 8):
        out += a.unseen(k, steps=k, heading=90)
    assert any("오른쪽으로 천천히 돌려" in x for x in out)


def test_front_target_side_check_says_sweep_not_turn_front():
    """1차원(걸음 수만, 방위 없음): 정면에서 본 목적지는 '정면으로 돌려'가 아니라 좌우로 비추게 (휴대폰 시험 3 재생)."""
    tr = TargetTracker("410")
    tr.seen(0.0, 3.0, "정면", 0)
    out = tr.unseen(1.0, 5)
    assert out and "좌우로 천천히 비춰" in out[0] and "정면으로" not in out[0]


def test_standing_still_is_not_passing_when_steps_known():
    """걸음 수가 있으면 위치는 걸음으로 옮긴다: 목적지를 3m 앞에 보고 "멈춰 주세요"로 서 있으면 시간이 지나도 '지나침'이 아님.
    전에는 1초 = 한 칸으로 옮겨, 서서 확인하는 동안 지나쳤다고 보고 되돌려 보냄 (시뮬레이터 인식 오류 약함, 2.6m 앞에서 왕복).
    걸음 수가 없으면 전처럼 시간으로 (서 있는지 모름)."""
    for steps, want in ((40, "보임"), (None, "뒤쪽")):
        g = CorridorGraph()
        t = walk(g, 0, ["414", "415"], "왼쪽", 0)
        g.see(t, "416", 1.0, "왼쪽", 3.0)
        g.set_ok("416")
        for k in range(12):
            g.move(t + k, steps)
        assert g.direction("416", now=t + 11)["direction"] == want
