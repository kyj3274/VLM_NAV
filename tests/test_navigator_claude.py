"""가짜 인식으로 규칙 흐름 시험 (API 호출 없음)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nav"))

import claude_perceiver as cp  # noqa: E402
import navigator as nv  # noqa: E402
from claude_client import FakeClient  # noqa: E402


def O(plates=(), place="corridor", door="none", elev=None, floor_sign="", arrows=(), unread=()):
    return {"plates": [dict(n=n, name="", side=s, m=m, on=on, box=[40, 30, 60, 40]) for n, s, m, on in plates],
            "unread": [dict(n="", name=nm, side="L", m=5.0, on="door", box=[10, 20, 20, 25]) for nm in unread],
            "floor_sign": floor_sign, "floor": cp.floor_of_sign(floor_sign), "place": place, "door_ahead": door,
            "elev": elev or {"seen": False, "why": "none", "side": "none", "m": None, "btn": "none"},
            "arrows": list(arrows), "fixes": []}


EL = {"seen": True, "why": "button", "side": "C", "m": 3.0, "btn": "R"}


class ScriptRunner:
    def __init__(self, script, rechecks=None):
        self.script, self.i, self.rechecks = script, 0, rechecks or {}

    def observe(self, frames):
        obs = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return {"key": "f.jpg", "keys": ["f.jpg"], "pick_seconds": 0, "obs": obs, "cached": False,
                "call": {"seconds": 1.0, "credit": 0.1, "input_tokens": 1, "output_tokens": 1}}

    def recheck(self, frame, plate):
        n = self.rechecks.get(plate["n"] or plate["name"], plate["n"] or None)
        return {"n": n, "name": "", "match": (n == plate["n"]) if plate["n"] and n else None, "cached": False,
                "call": {"seconds": 0.5, "credit": 0.05}}


def exited(nav):
    """첫 구간을 지나 엘리베이터에서 내린 상태로 만든다."""
    nav.new_segment("1")
    nav.seg_n = 1
    nav.new_segment("2")


def run(nav, k, seg=None):
    if seg is not None:
        nav.new_segment(seg)
    return [nav.step(["f.jpg"]) for _ in range(k)]


def test_route_a_like():
    """출발 층을 말로 받은 경우("1층인데 314호"): 전의 1층 가정 흐름과 같다 (2026-11-23부터 1층 가정 없음)."""
    far = {**EL, "side": "R", "m": 5.0}
    script = [O(door="glass", place="entrance")] * 2 + [O()] * 3 + [O(elev=far)] * 2 + [O(elev=EL, place="elev_hall")] * 2
    nav = nv.Navigator(ScriptRunner(script), "314", start_floor="1")
    nav.new_segment("1")
    recs = run(nav, 9)
    assert recs[0]["action"] == "문 통과" and "지금 1층입니다. 앞에 유리문이 있습니다" in recs[0]["say"][0]
    assert recs[3]["action"] == "직진"  # 문이 2초 안 보임
    assert recs[6]["action"] == "직진" and recs[6]["say"] == ["엘리베이터가 1시 방향 약 5미터에 있습니다. 그쪽으로 천천히 가세요."]
    assert recs[7]["action"] == "엘리베이터 타기" and "3층을 누르세요" in " ".join(recs[7]["say"])
    assert "오른쪽 벽" in " ".join(recs[7]["say"])
    nav.runner = ScriptRunner([O([("317", "C", 2.5, "door")])] * 2 + [O([("314", "C", 1.5, "door")])] * 4)
    recs = run(nav, 5, seg="2")
    assert recs[0]["state"]["floor"] == "3" and recs[0]["action"] == "직진"
    # 목적지 번호: 첫 호출(재확인 일치 포함)은 '확인 중'까지만, 두 번째 호출에서 확정.
    # 2026-11-19: 두 호출 모두 2m 안이면 두 번째에 확정과 도착이 함께 (전에는 세 번째에 도착)
    assert recs[2]["action"] == "목적지 접근" and recs[2]["say"] == ["목적지 같습니다. 멈춰 주세요, 확인 중입니다."]
    assert recs[2]["rooms"] == [] and recs[2]["recheck"]["match"] is True
    assert recs[3]["action"] == "도착" and recs[3]["say"] == ["정면에 314호 문이 있습니다."]
    assert recs[4]["action"] == "도착" and recs[4]["say"] == []  # 도착 안내는 한 번만
    assert recs[0]["recheck"]["match"] is True  # 317: 읽기 + 재확인 일치 → 한 번에 확정


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


def test_quiet_repeats_rule_sentence():
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O()] * 12
    nav = nv.Navigator(ScriptRunner(script), "314")
    exited(nav)
    recs = run(nav, 14)
    quiet = [r for r in recs if r["say"] == ["3층 복도입니다. 문패를 찾고 있습니다."]]  # 2026-11-19: 지금 상황 문장
    assert quiet and quiet[0]["t"] - recs[0]["t"] == 10


def test_floor_sign_one_call():
    nav = nv.Navigator(ScriptRunner([O(floor_sign="4F")]), "314")
    exited(nav)
    r = run(nav, 1)[0]
    assert r["state"]["floor"] == "4" and r["action"] == "엘리베이터로 돌아가기"


def test_arrival_side_within_2m():
    script = [O([("410", "L", 1.8, "door")])] * 3
    nav = nv.Navigator(ScriptRunner(script), "410")
    exited(nav)
    recs = run(nav, 3)
    assert recs[0]["say"][-1] == "목적지 같습니다. 멈춰 주세요, 확인 중입니다."
    assert recs[1]["action"] == "도착" and recs[1]["say"][-1] == "왼쪽 앞에 410호 문이 있습니다."  # 2026-11-19: 한 걸음 빨라짐
    assert recs[2]["say"] == []


def test_target_needs_two_calls_but_floor_plate_one():
    """415 오독 같은 경우: 일반 문패는 한 번의 호출(읽기+재확인 일치)로 확정, 목적지 번호는 다른 호출에서 한 번 더."""
    nav = nv.Navigator(ScriptRunner([O([("412", "C", 2.5, "door"), ("410", "L", 3.0, "door")]), O(), O(),
                                     O([("410", "L", 3.0, "door")])]), "410")
    exited(nav)
    recs = run(nav, 4)
    assert recs[0]["rooms"] == ["412"] and recs[0]["state"]["floor"] == "4"
    assert "목적지 같습니다, 확인 중입니다." in recs[0]["say"] and recs[0]["target_pending"]
    assert recs[3]["rooms"] == ["410"] and "410호 문패가 11시 방향 약 3미터에 있습니다." in recs[3]["say"]
    assert nav.mem.get("410")["calls"] == 2


def test_live_time_clock():
    """겹쳐 보내기: 걸음 간격 1.5초. 무안내 10초는 시간으로, 층 충돌 2번 연속은 걸음으로."""
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O()] * 7 + [O([("417", "C", 2.0, "door")])] * 2
    runner = ScriptRunner(script)
    nav = nv.Navigator(runner, "314")
    nav.new_segment("1")
    nav.step(["f.jpg"], t=0.0)
    nav.new_segment("2")
    recs = [nav.step(["f.jpg"], t=1.0 + 1.5 * i) for i in range(11)]
    assert [r["say"] for r in recs].count(["3층 복도입니다. 문패를 찾고 있습니다."]) == 1
    assert recs[-1]["state"]["floor"] == "4"


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
            "elev": {"seen": True, "why": "none", "side": "C", "m": 3, "btn": "none"}, "arrows": []}
    o = cp.clean(data)
    assert o["plates"][0]["side"] == "R" and o["unread"][0]["name"] == "강의실3"
    assert o["floor"] == "3" and o["elev"]["seen"] is False


def test_perceiver_cache(tmp_path):
    from PIL import Image
    f = tmp_path / "seg" / "i0000.jpg"
    f.parent.mkdir()
    Image.new("RGB", (72, 128)).save(f)
    data = {"plates": [], "floor_sign": "", "place": "corridor", "door_ahead": "none",
            "elev": {"seen": False, "why": "none", "side": "none", "m": -1, "btn": "none"}, "arrows": []}
    fc = FakeClient(lambda kind, content, n: data, credit=0.1)
    p = cp.Perceiver(fc, cache_path=tmp_path / "c.jsonl")
    assert p.see(f)["cached"] is False
    p2 = cp.Perceiver(fc, cache_path=tmp_path / "c.jsonl")
    assert p2.see(f)["cached"] is True and len(fc.calls) == 1


def test_passed_target_then_neighbor_is_away():
    script = [O([("410", "C", 3.0, "door")])] * 2 + [O([("407", "R", 3.0, "door")])] * 2 + \
             [O([("412", "L", 3.0, "door")])] * 2 + [O()] * 12
    nav = nv.Navigator(ScriptRunner(script), "410")
    exited(nav)
    recs = run(nav, 18)
    assert recs[1]["action"] == "목적지 접근"
    assert recs[2]["action"] == "뒤로 돌기"
    assert all(r["action"] == "뒤로 돌기" for r in recs[2:])  # 412(가까워짐)에도 '방향이 맞습니다' 안 함
    assert ["목적지와 멀어지고 있습니다. 뒤로 돌아가세요."] in [r["say"] for r in recs]


# ---------------------------------------------------------------- 2026-11-19 계단·확인 겹치기·빠른 확인
def test_stairs_floor_move_one_read_and_stop_elevator():
    """1층으로 정한 뒤 계단 장면 → '층 이동 가능'. 그 뒤 다른 층 확정 문패 한 번이면 바로 층 이동, 엘리베이터 안내 멈춤."""
    script = [O([("105", "C", 2.5, "door")]), O(elev=EL, place="elev_hall"), O(elev=EL, place="elev_hall"),
              O(place="stairs"), O(place="stairs"), O([("405", "L", 4.0, "door")])]
    nav = nv.Navigator(ScriptRunner(script), "410")
    nav.new_segment("1")
    recs = run(nav, 6)
    assert recs[0]["state"]["floor"] == "1" and recs[2]["action"] == "엘리베이터 타기"
    assert recs[3]["state"]["floor_move"]["why"] == "계단"
    assert recs[4]["action"] == "계단으로 올라가기" and recs[4]["say"] == ["계단입니다. 4층까지 올라가세요."]
    assert not recs[4]["state"]["elevator_said"]  # 계단을 고르면 엘리베이터 안내를 멈춤
    assert recs[5]["state"]["floor"] == "4" and not recs[5]["state"]["wrong_floor"]
    assert recs[5]["state"]["floor_move"] is None and recs[5]["action"] == "직진"
    assert "목적지 층입니다. 복도를 따라 문패를 찾겠습니다." in recs[5]["say"]
    assert any("연속 2번 안 기다림" in e["text"] for e in nav.mem.events)


def test_floor_conflict_without_stairs_still_two():
    """계단·층 표시가 없으면 다른 층 문패는 지금처럼 2번 연속이어야 층이 바뀐다."""
    nav = nv.Navigator(ScriptRunner([O([("105", "C", 2.5, "door")])] + [O([("405", "L", 4.0, "door")])] * 2), "410")
    nav.new_segment("1")
    recs = run(nav, 3)
    assert recs[1]["state"]["floor"] == "1" and recs[2]["state"]["floor"] == "4"


def test_floor_sign_after_stairs_one_read():
    nav = nv.Navigator(ScriptRunner([O([("105", "C", 2.5, "door")]), O(place="stairs"), O(floor_sign="4F")]), "410")
    nav.new_segment("1")
    recs = run(nav, 3)
    assert recs[2]["state"]["floor"] == "4"


def test_confirm_and_arrive_overlap():
    """2m 안에서 서로 다른 호출 2번이 목적지를 읽으면 두 번째에 확정과 도착이 함께 (전에는 세 번째)."""
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 1.8, "door")])] * 3), "410")
    exited(nav)
    recs = run(nav, 2)
    assert recs[0]["action"] == "목적지 접근" and recs[0]["say"][-1] == "목적지 같습니다. 멈춰 주세요, 확인 중입니다."
    assert recs[0]["confirm_now"] and recs[0]["rooms"] == []
    assert recs[1]["state"]["arrived"] and recs[1]["say"][-1] == "왼쪽 앞에 410호 문이 있습니다."


def test_far_first_read_still_needs_two_close():
    """첫 읽기가 2m 밖이면 도착 연속에 넣지 않는다."""
    nav = nv.Navigator(ScriptRunner([O([("410", "C", 5.0, "door")]), O([("410", "C", 1.8, "door")]),
                                     O([("410", "C", 1.5, "door")])]), "410")
    exited(nav)
    recs = run(nav, 3)
    assert not recs[1]["state"]["arrived"] and recs[1]["say"][-1] == "멈춰 주세요, 확인 중입니다."
    assert recs[2]["state"]["arrived"]


def test_quick_confirm_step():
    """빠른 확인 사진(문패만)이 목적지를 2m 안에서 읽으면 그 걸음에 도착. 확대 재확인은 부르지 않는다."""
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 1.8, "door")])]), "410")
    exited(nav)
    r0 = nav.step(["f.jpg"])
    assert r0["confirm_now"]
    o = {"key": "q.jpg", "keys": ["q.jpg"], "pick_seconds": 0, "cached": False, "call": {"seconds": 1.0, "credit": 0.1},
         "obs": nv.Navigator.quick_obs([dict(n="410", name="", side="L", m=1.6, on="door", box=[10, 30, 20, 40])], r0["obs"])}
    r1 = nav.step(t=None, observed=o, quick=True)
    assert r1["state"]["arrived"] and r1["recheck"] is None and r1["claude_calls"] == 1


def test_quick_confirm_miss_resets():
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 1.8, "door")])]), "410")
    exited(nav)
    nav.step(["f.jpg"])
    o = {"key": "q.jpg", "keys": ["q.jpg"], "pick_seconds": 0, "cached": False, "call": {"seconds": 1.0, "credit": 0.1},
         "obs": nv.Navigator.quick_obs([], None)}
    r1 = nav.step(observed=o, quick=True)
    assert not r1["state"]["arrived"] and r1["state"]["arr_run"] == 0 and not r1["state"]["quick_asked"]


# ---------------------------------------------------------------- 2026-11-19 (2): 무안내·계단 층 표시·엘리베이터 두 단계·지나침
def SO(place="stairs", ss="", fs=""):
    o = O(place=place, floor_sign=fs)
    o["stair_sign"], o["stair_floors"] = ss, cp.floors_of_stair_sign(ss)
    return o


def EC(why="doors", side="L", m=8.0):
    return {"seen": why in ("button", "display"), "cand": True, "why": why, "side": side, "m": m, "btn": "none"}


def start_1f(script, target="410"):
    nav = nv.Navigator(ScriptRunner([O([("105", "C", 2.5, "door")])] + script), target)
    nav.new_segment("1")
    nav.step(["f.jpg"])
    return nav


def test_quiet_any_state_elevator_and_stairs():
    nav = start_1f([O(elev=EL, place="elev_hall")] * 2 + [O(elev=EL, place="elev_hall")] * 12)
    recs = run(nav, 14)
    assert ["엘리베이터를 타고 4층을 누르세요."] in [r["say"] for r in recs]  # 엘리베이터 타기에서도 10초면 말함
    nav = start_1f([SO()] * 14)
    recs = run(nav, 14)
    assert recs[1]["say"] == ["계단입니다. 4층까지 올라가세요."]
    assert ["계단입니다. 4층까지 올라가세요."] in [r["say"] for r in recs[2:]]


def test_stair_sign_estimate_exit_and_temp_floor():
    nav = start_1f([SO(), SO(), SO(ss="4F 3F"), SO(), SO(), SO(), SO(), SO(), SO(), SO(), SO(), SO(), SO(ss="4F")]
                   + [O()] * 4 + [O([("410", "L", 5.0, "door")])])
    recs = run(nav, 18)
    assert "지금 약 3층과 4층 사이입니다" in " ".join(" ".join(r["say"]) for r in recs[:12])
    assert recs[12]["say"] == ["4층입니다. 계단에서 나가세요."] and recs[12]["action"] == "계단에서 나가기"
    assert recs[16]["state"]["floor"] == "4" and recs[16]["state"]["floor_temp"]
    assert recs[16]["say"] == ["4층으로 보입니다. 복도를 따라 문패를 찾겠습니다."]


def test_temp_floor_overridden_by_plate():
    """계단 표시로 임시 4층 → 문패 305가 확정되면 바로 3층 (문패 우선)."""
    nav = start_1f([SO(), SO(ss="4F 3F")] + [O()] * 4 + [O([("305", "C", 2.5, "door")])])
    recs = run(nav, 7)
    assert recs[5]["state"]["floor"] == "4" and recs[5]["state"]["floor_temp"]
    assert recs[6]["state"]["floor"] == "3" and recs[6]["state"]["wrong_floor"]


def test_elevator_candidate_then_confirm():
    nav = start_1f([O(elev=EC("doors", "L", 12.0)), O(elev={**EC("button", "L", 9.0), "btn": "R"})])
    recs = run(nav, 2)
    assert recs[0]["say"] == ["왼쪽 앞에 엘리베이터가 있는 것 같습니다. 천천히 가세요."]
    assert recs[1]["say"] == ["엘리베이터가 11시 방향 약 9미터에 있습니다. 그쪽으로 천천히 가세요."]


def test_elevator_candidate_cancel_quietly():
    nav = start_1f([O(elev=EC("hall", "C", 15.0)), O(), O(), O()])
    recs = run(nav, 4)
    assert recs[0]["say"] and recs[1]["say"] == [] and recs[2]["say"] == []
    assert any("후보 취소" in e["text"] for e in nav.mem.events)


def test_elevator_passed():
    nav = start_1f([O(elev=EC("doors", "R", 4.0)), O(elev=EC("button", "R", 2.5)), O(), O()])
    recs = run(nav, 4)
    assert recs[3]["say"] == ["엘리베이터를 지나쳤습니다. 뒤로 돌아 11시 방향입니다."]
    # 4m 앞 오른쪽에서 사라짐 = 그쪽으로 돈 것: 말하지 않음
    nav = start_1f([O(elev=EC("display", "R", 8.0)), O(elev=EC("display", "R", 4.0)), O(), O()])
    recs = run(nav, 4)
    assert "지나쳤습니다" not in " ".join(" ".join(r["say"]) for r in recs)
    nav = start_1f([O(elev=EC("doors", "R", 4.0)), O(elev=EC("button", "R", 2.5)), O(), O()])
    recs = run(nav, 4)
    assert recs[3]["say"] == ["엘리베이터를 지나쳤습니다. 뒤로 돌아 11시 방향입니다."]
    assert recs[3]["action"] == "엘리베이터로 돌아가기"


def test_elevator_lost_on_stairs_is_quiet():
    nav = start_1f([O(elev=EC("button", "L", 2.0)), O(elev=EC("button", "L", 2.0)), SO(), SO()])
    recs = run(nav, 4)
    assert "지나쳤습니다" not in " ".join(" ".join(r["say"]) for r in recs)
    assert recs[3]["say"] == ["계단입니다. 4층까지 올라가세요."]


def test_clean_stair_sign_and_elevator_kinds():
    o = cp.clean({"plates": [], "fs": "", "ss": "4F 3F", "place": "stairs", "door": "none",
                  "el": [{"why": "hall", "side": "C", "m": 14, "btn": "none"}], "arrows": []})
    assert o["stair_floors"] == ["3", "4"] and o["elev"]["cand"] and not o["elev"]["seen"] and o["elev"]["m"] == 14


# ---------------------------------------------------------------- 2026-11-23 (A) 1층 가정 없앰·출발 층 말하기
C = nv.CHECK_SAY


def says(recs):
    return " ".join(" ".join(r["say"]) for r in recs)


def test_no_floor_assumption_no_elevator_guidance():
    """층을 모르면 1층으로 가정하지 않는다: 출입문·엘리베이터가 보여도 엘리베이터 안내 없이 층 확인, 15초 뒤 층 질문(20초에 한 번)."""
    script = [O(door="glass", place="entrance")] * 2 + [O()] * 3 + [O(elev=EL, place="elev_hall")] * 40
    nav = nv.Navigator(ScriptRunner(script), "410")
    nav.new_segment("1")
    recs = run(nav, 45)
    assert recs[0]["action"] == "문 통과"
    assert "엘리베이터" not in says(recs) and "1층" not in says(recs)
    assert all(r["action"] not in ("엘리베이터 타기", "직진") for r in recs[3:])
    assert recs[3]["say"] == ["문을 지났습니다. " + C] and recs[3]["action"] == "멈추고 둘러보기"
    asks = [r["t"] for r in recs if nv.ASK_SAY in r["say"]]
    assert asks == [15, 35] and recs[15]["ask_floor"]
    assert all(r["state"]["floor"] is None for r in recs)


def test_start_4f_no_door_plate_sets_floor():
    """4층에서 시작(410·405): 위치 파악 5초 동안 층을 모르면 1층이라고 하지 않고, 확정 문패를 읽으면 바로 4층."""
    for target in ("410", "405"):
        script = [O(elev=EL, place="elev_hall")] * 6 + [O([("412", "C", 2.5, "door")])] + [O(elev=EL)] * 3
        nav = nv.Navigator(ScriptRunner(script), target)
        nav.new_segment("1")
        recs = run(nav, 10)
        assert recs[4]["say"] == ["여기는 엘리베이터 앞인 것 같습니다. " + C]
        assert "엘리베이터를 타고" not in says(recs) and "1층" not in says(recs)
        assert recs[6]["state"]["floor"] == "4" and not recs[6]["state"]["wrong_floor"]
        assert recs[6]["say"][0].startswith("지금 위치는 4층") is False  # 위치 파악은 끝났음
        assert any("4층이 맞습니다" in x for x in recs[6]["say"]) and recs[6]["action"] == "직진"
        assert all(r["action"] != "엘리베이터 타기" for r in recs)


def test_start_floor_said_with_destination():
    """"4층에서 410호": 말한 층으로 바로 시작 (위치 파악·질문 없음, 엘리베이터 안내 없음)."""
    nav = nv.Navigator(ScriptRunner([O(elev=EL, place="elev_hall")] * 20), "410", start_floor="4")
    nav.new_segment("1")
    recs = run(nav, 20)
    assert recs[0]["say"] == ["410호로 안내를 시작합니다. 410호는 4층입니다. 지금 4층입니다. 목적지 층입니다. "
                              "앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]
    assert recs[0]["action"] == "직진" and recs[0]["state"]["floor_src"] == "말"
    assert "엘리베이터를" not in says(recs) and "엘리베이터가" not in says(recs) and nv.ASK_SAY not in says(recs)
    assert all(r["action"] == "직진" for r in recs)


def test_4f_start_to_1f_destination():
    """4층에서 1층 목적지(105): 층을 모를 때는 엘리베이터 안내 없음. 4층으로 정해진 뒤에야 '1층을 누르세요'."""
    script = [O(elev=EL, place="elev_hall")] * 6 + [O([("412", "C", 2.5, "door")])] + [O(elev=EL, place="elev_hall")] * 2
    nav = nv.Navigator(ScriptRunner(script), "105")
    nav.new_segment("1")
    recs = run(nav, 9)
    assert "엘리베이터를 타고" not in says(recs[:6]) and "목적지 층입니다" not in says(recs[:6])
    assert recs[6]["state"]["floor"] == "4" and recs[6]["state"]["wrong_floor"]
    assert "엘리베이터를 찾아 1층으로" in says(recs[6:7])
    assert "엘리베이터를 타고 1층을 누르세요" in says(recs[7:]) and recs[-1]["action"] == "엘리베이터 타기"
    # 출발 층을 말로 받으면 첫 걸음부터
    nav = nv.Navigator(ScriptRunner([O(elev=EL, place="elev_hall")] * 3), "105", start_floor="4")
    nav.new_segment("1")
    recs = run(nav, 3)
    assert "105호는 1층에 있습니다" in recs[0]["say"][0] and "엘리베이터를 타고 1층을 누르세요" in says(recs)


def test_said_floor_then_two_other_plates():
    """안내 중 "여기 4층이야": 바로 반영. 다른 층 확정 문패 한 번으로는 안 바뀌고, 연속 2번이면 문패를 따른다."""
    nav = nv.Navigator(ScriptRunner([O()] * 6 + [O([("315", "C", 2.5, "door")])] * 2), "410")
    nav.new_segment("1")
    run(nav, 6)
    out = nav.set_floor("4")
    assert out[0].startswith("4층으로 알겠습니다. 목적지 층입니다") and nav.st["floor"] == "4" and nav.st["act"] == "직진"
    recs = run(nav, 2)
    assert recs[0]["state"]["floor"] == "4" and recs[1]["state"]["floor"] == "3"


def test_floor_sign_at_start_sets_floor_at_once():
    nav = nv.Navigator(ScriptRunner([O(floor_sign="4F", place="elev_hall")] + [O()] * 3), "410")
    nav.new_segment("1")
    recs = run(nav, 2)
    assert recs[0]["state"]["floor"] == "4" and "4층" in says(recs[:1]) and "1층" not in says(recs)


def test_in_elevator_display_sets_floor_and_carries_on_exit():
    """엘리베이터 안 층 표시창(판 4 ed)이 또렷하면 바로 그 층. 내리기 직전에 읽었으면 내린 층의 임시 층 →
    다른 층 확정 문패 한 번이면 문패를 따른다."""
    ie = O(place="in_elev")
    ie4 = dict(ie, elev_floor="4")
    nav = nv.Navigator(ScriptRunner([ie] * 6 + [ie4] * 2), "410")
    nav.new_segment("1")
    recs = run(nav, 8)
    assert recs[6]["state"]["floor"] == "4" and recs[6]["say"] == ["엘리베이터 표시창이 4층입니다. 목적지 층입니다."]
    nav.runner = ScriptRunner([O(), O([("315", "C", 2.5, "door")])])
    recs = run(nav, 2, seg="2")
    assert recs[0]["state"]["floor"] == "4" and recs[0]["state"]["floor_temp"]
    assert recs[0]["say"][0] == "엘리베이터에서 내렸습니다. 표시창으로는 4층입니다. 문패를 확인하며 천천히 이동하세요."
    assert recs[1]["state"]["floor"] == "3" and recs[1]["state"]["wrong_floor"]


def test_clean_elevator_display():
    o = cp.clean({"plates": [], "fs": "", "ss": "", "place": "in_elev", "door": "none", "el": [], "arrows": [], "ed": "4"})
    assert o["elev_floor"] == "4"
    o = cp.clean({"plates": [], "fs": "", "ss": "", "place": "corridor", "door": "none", "el": [], "arrows": [], "ed": "4"})
    assert o["elev_floor"] is None
    assert cp.floor_of_display("B1") == "B1" and cp.floor_of_display("12F") == "12" and cp.floor_of_display("▲") is None


# ---------------------------------------------------------------- 2026-11-23 (B) PR #1 리뷰
def test_review1_same_floor_sign_does_not_arm_floor_move():
    """4층에서 '4F'를 본 뒤 215 오독 한 번(읽기+재확인 일치로 확정)으로 층이 바뀌지 않는다."""
    script = [O([("412", "C", 2.5, "door")]), O(floor_sign="4F"), O(), O([("215", "C", 2.5, "door")]), O()]
    nav = nv.Navigator(ScriptRunner(script), "410")
    exited(nav)
    recs = run(nav, 5)
    assert recs[1]["state"]["floor"] == "4" and recs[1]["state"]["floor_move"] is None
    assert "215" in nav.mem.confirmed()
    assert recs[3]["state"]["floor"] == "4" and recs[4]["state"]["floor"] == "4"
    assert "엘리베이터로 돌아가" not in says(recs)


def test_review1_floor_move_expires_after_stairs():
    """계단을 벗어나고 계단 단서 없이 FLOOR_MOVE_HOLD초가 지나면 '층 이동 가능'을 끈다 → 다른 층 문패는 다시 2번 연속."""
    script = [O([("105", "C", 2.5, "door")]), O(place="stairs"), O()] + [O()] * nv.FLOOR_MOVE_HOLD + \
             [O([("405", "L", 4.0, "door")]), O()]
    nav = nv.Navigator(ScriptRunner(script), "110")
    nav.new_segment("1")
    recs = run(nav, len(script))
    assert recs[1]["state"]["floor_move"]["why"] == "계단"
    off = next(i for i, r in enumerate(recs) if i > 1 and r["state"]["floor_move"] is None)
    assert recs[off]["t"] - recs[1]["t"] == nv.FLOOR_MOVE_HOLD
    assert recs[-2]["state"]["floor"] == "1" and recs[-1]["state"]["floor"] == "1"
    assert any("층 이동 가능 끔" in e["text"] for e in nav.mem.events)


def test_review3_target_change_needs_two_calls_again():
    """410으로 가던 중 한 번의 호출로 확정된 315(읽기+재확인 일치)는 목적지를 315로 바꾸면 확정이 풀린다."""
    nav = nv.Navigator(ScriptRunner([O([("315", "C", 2.5, "door")]), O()] + [O()] * 7 + [O([("315", "C", 2.5, "door")])]), "410")
    exited(nav)
    recs = run(nav, 2)
    assert "315" in nav.mem.confirmed() and nav.mem.get("315")["calls"] == 1
    nav.set_target("315")
    assert "315" not in nav.mem.confirmed()
    recs = run(nav, 1)
    assert recs[0]["rooms"] == [] and not recs[0]["state"]["target_seen"]
    # 6초(FORGET) 넘게 지난 뒤 다시 읽힌 한 번은 첫 호출로 다시 셈 → 아직 '확인 중'
    recs = run(nav, 7)
    assert recs[-1]["rooms"] == [] and recs[-1].get("target_pending") and "315" not in nav.mem.confirmed()


def test_review3_target_change_second_call_confirms():
    nav = nv.Navigator(ScriptRunner([O([("315", "C", 2.5, "door")])] * 3), "410")
    exited(nav)
    run(nav, 1)
    nav.set_target("315")
    assert "315" not in nav.mem.confirmed()
    r = run(nav, 1)[0]
    assert r["rooms"] == ["315"] and nav.mem.get("315")["calls"] == 2  # 서로 다른 호출 2번 → 확정