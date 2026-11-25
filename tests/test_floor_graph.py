"""공통 판단 core/floor_graph.py: 갈림길에서 어느 복도로 갈지 (2026-11-25 시뮬레이터 v3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nav" / "core"))

from floor_graph import FloorGraph, mentions  # noqa: E402


def walk(g, steps0, n, heading):
    """heading 쪽으로 n걸음."""
    for k in range(1, n + 1):
        g.move(steps0 + k, heading)
    return steps0 + n


def at_t_junction(target, signs=None):
    """북쪽(0°)으로 10걸음 걸어 앞 2m에 T자 갈림길 (왼쪽·오른쪽만 트임)."""
    g = FloorGraph(target)
    g.move(0, 0)
    s = walk(g, 0, 10, 0)
    j = g.see_junction(2.0, ["left", "right"], 0, signs)
    return g, j, s


def test_no_evidence_explores_unvisited_not_back():
    """근거가 없으면 안 가 본 가지 (온 길로 되돌아가지 않음). 이름만 있는 건물이 이 경우."""
    g, j, _ = at_t_junction("세미나실")
    assert g.choose(j, 0)["rel"] in ("left", "right")


def test_sign_points_the_way():
    g, j, _ = at_t_junction("세미나실", [{"dir": "left", "items": ["회의실"]}, {"dir": "right", "items": ["세미나실", "자료실"]}])
    assert g.choose(j, 0)["rel"] == "right"
    g, j, _ = at_t_junction("415", [{"dir": "left", "items": ["401~410"]}, {"dir": "right", "items": ["411~420"]}])
    assert g.choose(j, 0)["rel"] == "right"
    assert mentions("411~420", "415") and not mentions("401~410", "415")


def test_dead_branch_is_not_chosen_again():
    """오른쪽 가지를 끝까지 가 봤으면 (목적지 못 봄) 돌아와서는 왼쪽."""
    g, j, s = at_t_junction("세미나실")
    s = walk(g, s, 4, 0)            # 갈림길에 들어섬
    s = walk(g, s, 10, 90)          # 오른쪽(동쪽) 가지로
    g.dead_end()
    s = walk(g, s, 8, 270)          # 돌아옴
    j2 = g.see_junction(1.2, ["ahead", "left"], 270)
    assert j2 == j
    assert g.choose(j, 270)["rel"] == "ahead"   # 서쪽으로 곧장 = 처음의 왼쪽 가지


def test_turn_without_junction_starts_new_line():
    """갈림길 없이 다른 축으로 걸으면 (꺾인 복도, 로비 벽을 따라 돎) 번호 흐름을 새로 (시뮬레이터 로비)."""
    g = FloorGraph("세미나실")
    g.move(0, 0)
    s = walk(g, 0, 5, 0)
    assert not g.corridor_turn
    g.move(s + 1, 90)
    assert g.corridor_turn


def test_start_inside_junction_then_turn_is_corridor_turn():
    """갈림길 안에서 출발해 바로 옆 복도로 꺾어도 다른 복도로 들어선 것 (첫 사진 방위 기준)."""
    g = FloorGraph("세미나실")
    g.move(0, 0)
    j = g.see_junction(0.5, ["left", "right"], 0)
    assert j is not None
    turned = False
    for k in range(1, 8):
        g.move(k, 90)
        turned = turned or g.corridor_turn
    assert turned


def test_turn_near_known_junction_closes_loop():
    """아는 갈림길에서 2m 못 미쳐(AT_M 1.5m 밖) 꺾어도 그 갈림길에서 꺾은 것으로 본다. 갈림길 위치는 '앞 몇 m' 눈대중이라
    전에는 ㅁ자 층을 한 바퀴 돌아와도 같은 갈림길로 안 봐 고리가 안 닫히고, 없는 목적지를 600초 찾음 (2026-11-26 시뮬레이터 스트레스)."""
    g, j, s = at_t_junction("499")            # 북쪽으로 6m, 갈림길은 2m 앞
    came = g.edge
    walk(g, s, 1, 90)                         # 들어서기 전에 오른쪽(동쪽)으로 꺾음
    assert g.at is None and g.corridor_turn
    assert g.branch[(j, 90)] == g.edge != came and g.branch[(j, 180)] == came


def test_near_plates_merge_same_corridor_far_plates_do_not():
    """가까이(3m 안) 본 확정 문패 2개가 다른 가지와 같으면 같은 복도로 합침. 멀리서 본 문패는 갈림길 건너편일 수 있어 안 씀."""
    g = FloorGraph("499")
    g.move(0, 0)
    first = g.edge
    for n in ("401", "402"):
        g.see_number(n, 2.0, 0)
    g.edge = g._new_edge()                    # 갈림길을 모르고 다른 가지로 기록됨 (출발 복도를 반대쪽에서 다시 들어옴)
    g.branch[(0, 0)], g.branch[(0, 1)] = first, g.edge
    g.see_number("401", 5.0, 180)             # 멀리서 본 것만으로는 안 합침
    g.see_number("402", 5.0, 180)
    assert g.branch[(0, 0)] == first
    g.see_number("401", 2.0, 180)
    g.see_number("402", 2.0, 180)
    assert g.branch[(0, 0)] == g.edge
