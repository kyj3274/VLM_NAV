"""가짜 인식으로 규칙 흐름 시험 (API 호출 없음)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nav"))

import claude_perceiver as cp  # noqa: E402
import navigator as nv  # noqa: E402
from claude_client import FakeClient  # noqa: E402


def O(plates=(), place="corridor", door="none", elev=None, floor_sign="", say="", arrows=(), unread=()):
    return {"plates": [dict(n=n, name="", side=s, m=m, on=on, box=[40, 30, 60, 40]) for n, s, m, on in plates],
            "unread": [dict(n="", name=nm, side="L", m=5.0, on="door", box=[10, 20, 20, 25]) for nm in unread],
            "floor_sign": floor_sign, "floor": cp.floor_of_sign(floor_sign), "place": place, "door_ahead": door,
            "elev": elev or {"seen": False, "why": "none", "side": "none", "m": None, "btn": "none"},
            "arrows": list(arrows), "say": say, "fixes": []}


EL = {"seen": True, "why": "button", "side": "C", "m": 3.0, "btn": "R"}


class ScriptRunner:
    def __init__(self, script, rechecks=None):
        self.script, self.i, self.rechecks = script, 0, rechecks or {}

    def observe(self, frames):
        obs = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return {"key": "f.jpg", "pick_seconds": 0, "obs": obs, "cached": False,
                "call": {"seconds": 1.0, "credit": 0.1, "input_tokens": 1, "output_tokens": 1}}

    def recheck(self, frame, plate):
        n = self.rechecks.get(plate["n"] or plate["name"], plate["n"] or None)
        return {"n": n, "name": "", "match": (n == plate["n"]) if plate["n"] and n else None, "cached": False,
                "call": {"seconds": 0.5, "credit": 0.05}}


def exited(nav):
    """첫 구간을 지나 엘리베이터에서 내린 상태로 만든다."""
    nav.new_segment("1")
    nav.sec = 1
    nav.new_segment("2")


def run(nav, k, seg=None):
    if seg is not None:
        nav.new_segment(seg)
    return [nav.step(["f.jpg"]) for _ in range(k)]


def test_route_a_like():
    far = {**EL, "side": "R", "m": 5.0}
    script = [O(door="glass", place="entrance")] * 2 + [O()] * 3 + [O(elev=far)] * 2 + [O(elev=EL, place="elev_hall")] * 2
    nav = nv.Navigator(ScriptRunner(script), "314")
    nav.new_segment("1")
    recs = run(nav, 9)
    assert recs[0]["action"] == "문 통과" and "유리문" in recs[0]["say"][0]
    assert recs[3]["action"] == "직진"  # 문이 2초 안 보임
    assert recs[6]["action"] == "직진" and recs[6]["say"] == ["엘리베이터가 1시 방향 약 5미터에 있습니다. 그쪽으로 천천히 가세요."]
    assert recs[7]["action"] == "엘리베이터 타기" and "3층을 누르세요" in " ".join(recs[7]["say"])
    assert "오른쪽 벽" in " ".join(recs[7]["say"])
    nav.runner = ScriptRunner([O([("317", "C", 2.5, "door")])] * 2 + [O([("314", "C", 1.5, "door")])] * 3)
    recs = run(nav, 4, seg="2")
    assert recs[0]["state"]["floor"] == "3" and recs[0]["action"] == "직진"
    assert recs[2]["action"] == "목적지 접근" and recs[2]["say"] == ["멈춰 주세요, 확인 중입니다."]
    assert recs[3]["action"] == "도착" and recs[3]["say"] == ["정면에 314호 문이 있습니다."]


def test_wrong_floor_after_exit():
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 6.0, "door")])] * 3), "314")
    exited(nav)
    # 1번 읽음 + 문 옆 + 확대 재확인 일치 = 3점 → 첫 초에 확정
    recs = run(nav, 1)
    assert recs[0]["state"]["wrong_floor"] and recs[0]["action"] == "엘리베이터로 돌아가기"
    assert recs[0]["recheck"]["why"] == "층을 정할 번호"


def test_recheck_mismatch_blocks():
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 6.0, "door")])], rechecks={"410": "416"}), "314")
    exited(nav)
    r = run(nav, 1)[0]
    assert r["rooms"] == [] and r["state"]["floor"] is None


def test_unread_plate_read_by_recheck():
    nav = nv.Navigator(ScriptRunner([O(unread=["강의실3"])] * 2, rechecks={"강의실3": "410"}), "314")
    exited(nav)
    recs = run(nav, 2)
    assert recs[0]["recheck"]["read"] == "410"
    assert recs[1]["state"]["floor"] == "4"  # 2번 읽음 + 문 옆


def test_away_same_wall():
    script = [O([("317", "L", 3.0, "door")])] * 2 + [O([("316", "L", 3.0, "door")])] * 2
    nav = nv.Navigator(ScriptRunner(script), "318")
    exited(nav)
    recs = run(nav, 4)
    assert recs[2]["action"] == "뒤로 돌기" and "멀어지고" in recs[2]["say"][0]


def test_no_trend_from_numbers_far_from_target():
    # 경로 A 3층: 315 다음 307 → 307은 목적지 314에서 3보다 멀어 방향 판단에 쓰지 않음
    script = [O([("315", "L", 3.0, "door")])] * 2 + [O([("307", "L", 3.0, "door")])] * 2
    nav = nv.Navigator(ScriptRunner(script), "314")
    exited(nav)
    recs = run(nav, 4)
    assert all(r["action"] != "뒤로 돌기" for r in recs) and "trend" not in recs[2]


def test_floor_conflict_needs_two_seconds():
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O([("417", "C", 2.0, "door")])] + [O()] + \
             [O([("418", "C", 2.0, "door")])] * 2
    nav = nv.Navigator(ScriptRunner(script), "314")
    exited(nav)
    recs = run(nav, 6)
    assert recs[2]["state"]["floor"] == "3" and recs[3]["state"]["floor"] == "3"
    assert recs[5]["state"]["floor"] == "4" and recs[5]["action"] == "엘리베이터로 돌아가기"


def test_say_filter_and_quiet():
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O(say="왼쪽에 엘리베이터가 있습니다.")] * 11 + [O(say="복도가 앞으로 이어집니다.")] * 12
    nav = nv.Navigator(ScriptRunner(script), "314")
    exited(nav)
    recs = run(nav, 25)
    assert recs[5]["say_check"] == "엘리베이터 미확인"
    spoken = [r for r in recs if r.get("say_from") == "claude"]
    assert spoken and spoken[0]["say"] == ["복도가 앞으로 이어집니다."]
    fallback = [r for r in recs[:13] if "계속 앞으로 천천히 직진해 주세요." in r["say"]]
    assert fallback  # 엘리베이터를 말한 say는 쓰지 않고 틀 문장


def test_elevator_needs_evidence_two_seconds():
    nav = nv.Navigator(ScriptRunner([O(door="glass")] * 3 + [O(elev=EL), O(), O(elev=EL), O()]), "314")
    nav.new_segment("1")
    recs = run(nav, 7)
    assert all(r["action"] != "엘리베이터 타기" for r in recs)


def test_clean_side_from_box_and_unread():
    data = {"plates": [{"n": "313", "name": "특수가공실", "side": "C", "m": 4, "on": "door", "box": [70, 39, 82, 46]},
                       {"n": "", "name": "강의실3", "side": "L", "m": 8, "on": "door", "box": [10, 20, 15, 23]},
                       {"n": "", "name": "", "side": "L", "m": -1, "on": "wall", "box": []}],
            "floor_sign": "3F", "place": "corridor", "door_ahead": "none",
            "elev": {"seen": True, "why": "none", "side": "C", "m": 3, "btn": "none"}, "arrows": [], "say": "x"}
    o = cp.clean(data)
    assert o["plates"][0]["side"] == "R" and o["unread"][0]["name"] == "강의실3"
    assert o["floor"] == "3" and o["elev"]["seen"] is False


def test_perceiver_cache(tmp_path):
    from PIL import Image
    f = tmp_path / "seg" / "i0000.jpg"
    f.parent.mkdir()
    Image.new("RGB", (72, 128)).save(f)
    data = {"plates": [], "floor_sign": "", "place": "corridor", "door_ahead": "none",
            "elev": {"seen": False, "why": "none", "side": "none", "m": -1, "btn": "none"}, "arrows": [], "say": ""}
    fc = FakeClient(lambda kind, content, n: data, credit=0.1)
    p = cp.Perceiver(fc, cache_path=tmp_path / "c.jsonl")
    assert p.see(f)["cached"] is False
    p2 = cp.Perceiver(fc, cache_path=tmp_path / "c.jsonl")
    assert p2.see(f)["cached"] is True and len(fc.calls) == 1


def test_passed_target_then_neighbor_is_away():
    script = [O([("410", "C", 3.0, "door")])] * 2 + [O([("407", "R", 3.0, "door")])] * 2 + \
             [O([("412", "L", 3.0, "door")])] * 2 + [O(say="복도가 앞으로 이어집니다.")] * 12
    nav = nv.Navigator(ScriptRunner(script), "410")
    exited(nav)
    recs = run(nav, 18)
    assert recs[1]["action"] == "목적지 접근"
    assert recs[2]["action"] == "뒤로 돌기"
    assert all(r["action"] == "뒤로 돌기" for r in recs[2:])  # 412(가까워짐)에도 '방향이 맞습니다' 안 함
    assert not any(r.get("say_from") == "claude" for r in recs)
    assert ["목적지와 멀어지고 있습니다. 뒤로 돌아가세요."] in [r["say"] for r in recs]
