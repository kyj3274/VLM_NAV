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
    assert recs[0]["state"]["floor"] == "3" and recs[0]["action"] == "멈추고 둘러보기"  # 2026-11-23 B6: 내린 직후 둘러보기
    # 목적지 번호: 첫 호출(재확인 일치 포함)은 '확인 중'까지만, 두 번째 호출에서 확정.
    # 2026-11-19: 두 호출 모두 2m 안이면 두 번째에 확정과 도착이 함께 (전에는 세 번째에 도착)
    assert recs[2]["action"] == "목적지 접근" and recs[2]["say"] == ["목적지 같습니다. 잠시 멈춰 주세요."]
    assert recs[2]["rooms"] == [] and recs[2]["recheck"] is None  # 빠른 확인이 켜져 있어 목적지 재확인 건너뜀 (2026-11-23)
    assert recs[3]["action"] == "도착" and recs[3]["say"] == ["정면에 314호 문이 있습니다."]
    assert recs[4]["action"] == "도착" and recs[4]["say"] == []  # 도착 안내는 한 번만
    # 2026-11-23 사용자 지시 5: 내린 층은 엘리베이터로 이동한 층(3층)으로 시작 → 317이 같은 층이라 확정 근거로 바뀜
    assert recs[0]["recheck"] is None and recs[0]["state"]["floor_src"] == "317호" and not recs[0]["state"]["assumed"]


def test_wrong_floor_after_exit():
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 6.0, "door")])] * 3), "314")
    exited(nav)
    # 2026-11-23 사용자 지시 5: 내린 층은 3층으로 보고, 다른 층(4층) 문패가 서로 다른 사진 2장에서 읽혀야 고침
    recs = run(nav, 2)
    assert recs[0]["state"]["floor"] == "3" and recs[0]["action"] == "멈추고 둘러보기"
    assert recs[1]["state"]["floor"] == "4" and recs[1]["state"]["wrong_floor"] and recs[1]["action"] == "엘리베이터로 돌아가기"
    assert recs[1]["say"] == ["층이 다릅니다. 여기는 4층입니다. 엘리베이터로 돌아가 3층을 누르세요. 엘리베이터는 내린 곳 뒤쪽에 있습니다."]


def test_recheck_mismatch_blocks():
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 6.0, "door")])], rechecks={"410": "416"}), "314")
    exited(nav)
    r = run(nav, 1)[0]
    # 내린 층(3층)으로 보는 중 다른 층 문패 사진 1장: 아직 바꾸지 않음 (2장이어야)
    assert r["rooms"] == [] and r["state"]["floor"] == "3" and len(r["state"]["other_seen"]["4"]) == 1


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
    recs = run(nav, 7)
    # 2026-11-23 B6·B8: 내린 직후 둘러보는 중이라 317→316(멀어짐)은 말하지 않고 방향 근거(보통)로 모았다가 6초 끝에 안내
    assert recs[2]["say"] == [] and recs[2]["action"] == "멈추고 둘러보기" and recs[2]["state"]["trend_said"] == "away"
    assert recs[6]["action"] == "뒤로 돌기" and recs[6]["say"] == ["318호는 뒤쪽일 가능성이 높습니다. 뒤로 돌아 천천히 직진하세요. 문패로 확인하겠습니다."]
    assert recs[6]["direction"]["basis"] == "문패 번호 317→316"


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
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O()] * 16
    nav = nv.Navigator(ScriptRunner(script), "314")
    exited(nav)
    recs = run(nav, 18)
    quiet = [r for r in recs if r["say"] == [nv.QUIET_GOAL]]  # 2026-11-19: 지금 상황 문장
    # 2026-11-23 B6: 내린 뒤 6초 둘러보기 끝에 방향 안내 → 그 뒤 10초
    assert recs[6]["say"] == ["앞으로 천천히 직진하세요. 문패로 방향을 확인하겠습니다."]
    assert quiet and quiet[0]["t"] - recs[6]["t"] == 10


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
    assert recs[0]["say"][-1] == "목적지 같습니다. 잠시 멈춰 주세요."
    assert recs[1]["action"] == "도착" and recs[1]["say"][-1] == "왼쪽 앞에 410호 문이 있습니다."  # 2026-11-19: 한 걸음 빨라짐
    assert recs[2]["say"] == []


def test_target_needs_two_calls_but_floor_plate_one():
    """415 오독 같은 경우: 일반 문패는 한 번의 호출(읽기+재확인 일치)로 확정, 목적지 번호는 다른 호출에서 한 번 더."""
    nav = nv.Navigator(ScriptRunner([O([("412", "C", 2.5, "door"), ("410", "L", 3.0, "door")]), O(), O(),
                                     O([("410", "L", 3.0, "door")])]), "410")
    exited(nav)
    recs = run(nav, 4)
    assert recs[0]["rooms"] == ["412"] and recs[0]["state"]["floor"] == "4"
    assert nv.QUICK_SAY in recs[0]["say"] and recs[0]["target_pending"]  # 2026-11-23: 5m 안 후보 → 빠른 확인 문장
    assert recs[3]["rooms"] == ["410"] and "410호 문패가 11시 방향 약 3미터에 있습니다." in recs[3]["say"]
    assert nav.mem.get("410")["calls"] == 2


def test_live_time_clock():
    """겹쳐 보내기: 걸음 간격 1.5초. 무안내 10초는 시간으로, 층 충돌 2번 연속은 걸음으로."""
    script = [O([("317", "C", 2.0, "door")])] * 2 + [O()] * 12 + [O([("417", "C", 2.0, "door")])] * 2
    runner = ScriptRunner(script)
    nav = nv.Navigator(runner, "314")
    nav.new_segment("1")
    nav.step(["f.jpg"], t=0.0)
    nav.new_segment("2")
    recs = [nav.step(["f.jpg"], t=1.0 + 1.5 * i) for i in range(16)]
    assert recs[4]["t"] == 7.0 and recs[4]["direction"]["basis"] == "단서 없음 (앞으로)"  # 둘러보기 6초 끝 (B6)
    assert [r["say"] for r in recs].count([nv.QUIET_GOAL]) == 1
    assert [r["t"] for r in recs if r["say"] == [nv.QUIET_GOAL]] == [17.5]
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
              O(place="stairs"), O(place="stairs"), O([("405", "L", 4.0, "door")]), O([("405", "L", 4.0, "door")])]
    nav = nv.Navigator(ScriptRunner(script), "410")
    nav.new_segment("1")
    recs = run(nav, 7)
    recs = recs[:5] + recs[6:]  # 2026-11-23: 405는 재확인 없이 두 번째 읽기에 확정(옆 4m라 정면 가까이 아님)
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
    nav = nv.Navigator(ScriptRunner([O([("105", "C", 2.5, "door")])] + [O([("405", "L", 4.0, "door")])] * 3), "410")
    nav.new_segment("1")
    recs = run(nav, 4)  # 405 확정(두 번째 읽기) 뒤 층 충돌 2번 연속
    assert recs[2]["state"]["floor"] == "1" and recs[3]["state"]["floor"] == "4"


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
    assert recs[0]["action"] == "목적지 접근" and recs[0]["say"][-1] == "목적지 같습니다. 잠시 멈춰 주세요."
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
    assert not r1["state"]["arrived"] and r1["state"]["arr_run"] == 0 and r1["state"]["quick"]["got"] == 1


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
    # 엘리베이터 타기에서도 10초면 말함, 기다리는 중이면 그 상태 문장 (2026-11-23)
    assert ["엘리베이터를 기다리고 있습니다. 문이 열리면 타고 4층을 누르세요."] in [r["say"] for r in recs]
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
    """실내에서 시작하면 1층으로 가정하지 않는다: 복도의 유리문을 지나고 엘리베이터가 보여도 엘리베이터 안내 없이 층 확인,
    15초 뒤 층 질문(20초에 한 번). (첫 사진이 '건물 출입문'이 아니라 복도)"""
    script = [O(door="glass", place="corridor")] * 2 + [O()] * 3 + [O(elev=EL, place="elev_hall")] * 40
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
        assert recs[4]["say"] == ["여기는 엘리베이터 앞인 것 같습니다. " + nv.ASK_SAY]  # 위치 파악 끝에 바로 물음
        assert "엘리베이터를 타고" not in says(recs) and "1층" not in says(recs)
        assert recs[6]["state"]["floor"] == "4" and not recs[6]["state"]["wrong_floor"]
        assert recs[6]["say"][0].startswith("지금 위치는 4층") is False  # 위치 파악은 끝났음
        # 2026-11-23 시험 2: 목적지 층이면 내렸을 때와 같은 방향 정하기(둘러보기)부터
        assert recs[6]["say"] == [f"4층입니다. 정면에 412호가 보입니다. {target}호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."]
        assert recs[6]["action"] == "멈추고 둘러보기"
        assert all(r["action"] != "엘리베이터 타기" for r in recs)


def test_start_floor_said_with_destination():
    """"4층에서 410호": 말한 층으로 바로 시작 (위치 파악·질문 없음, 엘리베이터 안내 없음)."""
    nav = nv.Navigator(ScriptRunner([O(elev=EL, place="elev_hall")] * 20), "410", start_floor="4")
    nav.new_segment("1")
    recs = run(nav, 20)
    assert recs[0]["say"] == ["410호로 안내를 시작합니다. 410호는 4층입니다. 지금 4층입니다. 목적지 층입니다. "
                              "410호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."]
    assert recs[0]["action"] == "멈추고 둘러보기" and recs[0]["state"]["floor_src"] == "말"
    assert "엘리베이터를" not in says(recs) and "엘리베이터가" not in says(recs) and nv.ASK_SAY not in says(recs)
    assert recs[6]["say"] == ["앞으로 천천히 직진하세요. 문패로 방향을 확인하겠습니다."]
    assert all(r["action"] == "멈추고 둘러보기" for r in recs[:6]) and all(r["action"] == "직진" for r in recs[6:])


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
    assert out[0].startswith("4층으로 알겠습니다. 목적지 층입니다") and nav.st["floor"] == "4"
    assert out[0].endswith("410호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요.") and nav.st["act"] == "멈추고 둘러보기"
    recs = run(nav, 2)
    assert recs[0]["state"]["floor"] == "4" and recs[1]["state"]["floor"] == "3"


def test_floor_sign_at_start_sets_floor_at_once():
    nav = nv.Navigator(ScriptRunner([O(floor_sign="4F", place="elev_hall")] + [O()] * 3), "410")
    nav.new_segment("1")
    recs = run(nav, 2)
    assert recs[0]["state"]["floor"] == "4" and "4층" in says(recs[:1]) and "1층" not in says(recs)


def test_in_elevator_display_sets_floor_and_carries_on_exit():
    """엘리베이터 안 층 표시창(판 4 ed)이 또렷하면 바로 그 층. 내리기 직전에 읽었으면 내린 층 →
    다른 층 문패가 서로 다른 사진 2장에서 읽히면 문패를 따른다 (2026-11-23 사용자 지시 5, 전에는 한 번)."""
    ie = O(place="in_elev")
    ie4 = dict(ie, elev_floor="4")
    nav = nv.Navigator(ScriptRunner([ie] * 6 + [ie4] * 2), "410")
    nav.new_segment("1")
    recs = run(nav, 8)
    assert recs[6]["state"]["floor"] == "4" and recs[6]["say"] == ["엘리베이터 표시창이 4층입니다. 목적지 층입니다."]
    nav.runner = ScriptRunner([O(), O([("315", "C", 2.5, "door")]), O([("315", "C", 2.5, "door")])])
    recs = run(nav, 3, seg="2")
    assert recs[0]["state"]["floor"] == "4" and recs[0]["state"]["floor_src"] == "엘리베이터 표시창"
    assert recs[0]["say"][0] == "엘리베이터에서 내렸습니다. 4층에 내렸습니다. 410호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."
    assert recs[1]["state"]["floor"] == "4" and recs[2]["state"]["floor"] == "3" and recs[2]["state"]["wrong_floor"]


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


def test_review4_target_change_recomputes_action_when_floor_unknown():
    """층을 모를 때 410 → 105: 행동을 다시 계산 → '엘리베이터를 타고 1층을 누르세요'가 나오지 않는다."""
    nav = nv.Navigator(ScriptRunner([O()] * 30), "410")
    nav.new_segment("1")
    run(nav, 6)
    nav.st["act"] = "엘리베이터 타기"  # 전의 1층 가정 상태(리뷰 재현): 층 모름 + 엘리베이터 타기
    out = nav.set_target("105")
    assert out == [C] and nav.st["act"] == "멈추고 둘러보기"
    recs = run(nav, 24)
    assert "엘리베이터를 타고" not in says(recs) and "1층을 누르세요" not in says(recs)
    assert C in says(recs)


def test_review4_target_change_with_known_floor():
    nav = nv.Navigator(ScriptRunner([O([("412", "C", 2.5, "door")])] + [O()] * 3), "410")
    exited(nav)
    run(nav, 1)
    out = nav.set_target("105")
    assert nav.st["wrong_floor"] and nav.st["act"] == "엘리베이터로 돌아가기"
    assert out == ["지금 4층입니다. 105호는 1층에 있습니다. 엘리베이터로 돌아가 1층을 누르세요."]


# ---------------------------------------------------------------- 2026-11-23 (2) 출입문 근거 되살림
def test_entrance_from_outside_is_weak_first_floor():
    """밖에서 건물 출입문을 지나 들어오면 1층(약한 근거): 엘리베이터 안내가 나온다 (판단 지점 1~4와 같은 장면)."""
    far = {**EL, "side": "R", "m": 5.0}
    script = [O(door="glass", place="entrance")] * 2 + [O()] * 3 + [O(elev=far)] * 2 + [O(elev=EL, place="elev_hall")] * 2
    nav = nv.Navigator(ScriptRunner(script), "410")
    nav.new_segment("1")
    recs = run(nav, 9)
    assert recs[0]["action"] == "문 통과" and recs[0]["state"]["floor"] is None
    assert recs[3]["say"] == ["문을 지났습니다. 건물 입구로 들어왔으니 1층으로 보고 안내하겠습니다. 410호는 4층에 있습니다. "
                              "엘리베이터를 찾아 4층으로 가겠습니다. 앞으로 천천히 직진해 주세요."]
    assert recs[3]["state"]["floor"] == "1" and recs[3]["state"]["floor_temp"] and recs[3]["state"]["floor_src"] == "출입문"
    assert recs[8]["action"] == "엘리베이터 타기" and "엘리베이터를 타고 4층을 누르세요" in says(recs)


def test_entrance_floor_yields_to_plate_sign_and_speech():
    """출입문 근거는 문패·층 표시·말로 받은 층보다 약하다: 다른 층 확정 문패 한 번, 층 표시 한 번, 말한 층이면 그쪽."""
    door = [O(door="glass", place="entrance")] * 2 + [O()] * 2
    for extra, want in (([O([("412", "C", 2.5, "door")])], "4"), ([O(floor_sign="3F")], "3")):
        nav = nv.Navigator(ScriptRunner(door + extra), "410")
        nav.new_segment("1")
        recs = run(nav, 5)
        assert recs[3]["state"]["floor"] == "1" and recs[4]["state"]["floor"] == want
    nav = nv.Navigator(ScriptRunner(door), "410")
    nav.new_segment("1")
    run(nav, 4)
    nav.set_floor("4")
    assert nav.st["floor"] == "4" and not nav.st["floor_temp"] and nav.st["floor_src"] == "말"


def test_door_mid_session_is_not_entrance_evidence():
    """처음엔 실내(로비)였다가 나중에 유리문이 보여도 1층 근거로 쓰지 않는다 (4층 세션의 안쪽 유리문)."""
    script = [O(place="lobby")] * 6 + [O(door="glass", place="entrance")] * 2 + [O()] * 3 + [O(elev=EL, place="elev_hall")] * 4
    nav = nv.Navigator(ScriptRunner(script), "410")
    nav.new_segment("1")
    recs = run(nav, len(script))
    assert all(r["state"]["floor"] is None for r in recs) and "엘리베이터를 타고" not in says(recs)


# ---------------------------------------------------------------- 2026-11-23 (3) 판단 시간 줄이기·빠른 확인 개선
def qo(t, plates):
    """빠른 확인 사진 한 장 (찍은 시각 t)."""
    return {"key": f"q{t}.jpg", "keys": [f"q{t}.jpg"], "pick_seconds": 0, "cached": False, "call": {"seconds": 1.0, "credit": 0.05},
            "obs": nv.Navigator.quick_obs([dict(n=x[0], name="", side=x[1], m=x[2], on=(x[3] if len(x) > 3 else "door"),
                                                box=[40, 30, 50, 40]) for x in plates])}


def nav4(script, target="410"):
    nav = nv.Navigator(ScriptRunner([O([("412", "C", 2.5, "door")])] + script), target)
    exited(nav)
    nav.step(["f.jpg"], t=0.0)
    return nav


def test_recheck_only_target_candidate_or_unread():
    nav = nv.Navigator(ScriptRunner([O()]), "410")
    exited(nav)
    nav.st["floor"] = None
    assert nav.recheck_pick(O([("317", "L", 5.0, "door")]))[1] == "층을 정할 번호"   # 층 모름: 일반 문패도
    nav.st["floor"] = "4"
    assert nav.recheck_pick(O([("317", "L", 5.0, "door")])) == (None, None)       # 층을 안 뒤: 안 함
    assert nav.recheck_pick(O([("411", "L", 5.0, "door")]))[1] == "목적지 근처 번호"
    assert nav.recheck_pick(O(unread=["강의실3"]))[1] == "번호 안 읽힌 문패"


def test_quick_trigger_5m_two_photos_confirm_and_arrive():
    """목적지 후보를 5m 안에서 처음 읽으면 confirm. 빠른 확인 사진이 2m 안에서 목적지를 읽으면 서로 다른 사진 2장으로 확정 + 그 자리 도착."""
    # 믿음 점수가 낮은 경우: 벽 문패, 옆, 가까운 확정 번호 없음, 재확인 못 읽음 → 점수만으로는 확정 안 됨
    nav = nv.Navigator(ScriptRunner([O([("410", "L", 4.5, "wall")])], rechecks={"410": ""}), "410")
    exited(nav)
    r = nav.step(["f.jpg"], t=1.5)
    assert r["confirm_now"] and r.get("target_pending") and r["say"][-1] == nv.QUICK_SAY
    assert r["recheck"] is None or r["recheck"]["asked"] != "410"  # 빠른 확인이 켜져 있으면 목적지 재확인 건너뜀
    r1 = nav.step(t=2.2, observed=qo(2.2, [("410", "L", 1.8, "wall")]), quick=True)
    assert r1["state"]["arrived"] and r1["say"] == ["왼쪽 앞에 410호 문이 있습니다."] and r1["action"] == "도착"
    assert any("빠른 확인, 서로 다른 사진 2장" in e["text"] for e in nav.mem.events)  # 문 옆이 아니어도(벽) 2장이면 확정
    r2 = nav.step(t=2.6, observed=qo(2.6, [("410", "L", 1.7)]), quick=True)
    assert r2["say"] == []  # 도착 안내는 한 번만


def test_quick_far_photo_confirms_without_arrival():
    nav = nav4([O([("410", "C", 5.0, "door")])])
    nav.step(["f.jpg"], t=1.5)
    r1 = nav.step(t=2.2, observed=qo(2.2, [("410", "C", 4.0)]), quick=True)
    r2 = nav.step(t=2.6, observed=qo(2.6, []), quick=True)
    assert r1["rooms"] == ["410"] and not r1["state"]["arrived"] and "410호 문패가 12시 방향 약 4미터에 있습니다." in r1["say"]
    assert r2["state"]["quick"] is None


def test_quick_both_other_numbers_cancel_candidate_and_max_two_rounds():
    """2장 모두 다른 번호(412)를 읽으면 후보 취소. 목적지 하나에 빠른 확인은 최대 2번."""
    nav = nav4([O([("410", "L", 4.0, "door")]), O([("410", "L", 4.0, "door")]), O([("410", "L", 3.5, "door")])])
    r = nav.step(["f.jpg"], t=1.5)
    assert r["confirm_now"]
    nav.step(t=2.2, observed=qo(2.2, [("412", "L", 3.0)]), quick=True)
    r2 = nav.step(t=2.6, observed=qo(2.6, [("412", "L", 3.0)]), quick=True)
    assert r2["say"] == ["목적지가 아닙니다. 계속 찾겠습니다."] and r2["action"] == "직진"
    assert nav.mem.get("410")["calls"] == 0 and "410" not in nav.mem.confirmed()
    r = nav.step(["f.jpg"], t=9.0)  # 나중에 다시 읽힘: 두 번째 빠른 확인
    assert r["confirm_now"] and r["state"]["quick_n"] == 2
    nav.step(t=9.5, observed=qo(9.5, [("412", "L", 3.0)]), quick=True)
    nav.step(t=9.9, observed=qo(9.9, [("412", "L", 3.0)]), quick=True)
    r = nav.step(["f.jpg"], t=16.0)
    assert not r.get("confirm_now")  # 3번째는 없음


def test_quick_same_photo_is_not_second_call():
    """처음 본 사진과 같은 사진(같은 찍은 시각)은 두 번째 호출로 치지 않는다."""
    nav = nav4([O([("410", "L", 1.8, "wall")])])
    nav.step(["f.jpg"], t=1.5)
    r = nav.step(t=1.5, observed=qo(1.5, [("410", "L", 1.8)]), quick=True)
    assert not r["state"]["arrived"] and nav.mem.get("410")["calls"] == 1


def test_quick_off_and_timeout():
    nav = nav4([O([("410", "L", 4.0, "door")])] * 3)
    nav.quick_max = 0
    assert not nav.step(["f.jpg"], t=1.5).get("confirm_now")
    nav = nav4([O([("410", "L", 4.0, "door")])] + [O()] * 8)
    nav.step(["f.jpg"], t=1.5)
    assert nav.st["quick"] is not None
    nav.step(["f.jpg"], t=1.5 + nv.QUICK_WAIT + 0.5)  # 사진이 안 옴(흐림) → 기다리지 않음
    assert nav.st["quick"] is None


def test_perceiver_send_quality_and_passthrough(tmp_path):
    import base64
    from PIL import Image
    f = tmp_path / "seg" / "i0000.jpg"
    f.parent.mkdir()
    Image.new("RGB", (72, 128), "white").save(f, quality=70)
    seen = []
    data = {"plates": [], "fs": "", "ss": "", "place": "corridor", "door": "none", "el": [], "arrows": [], "ed": ""}
    fc = FakeClient(lambda kind, content, n: (seen.append(content), data)[1], credit=0.1)
    p = cp.Perceiver(fc, quality=70)
    p.see(f)
    assert base64.b64decode(seen[0][0]["source"]["data"]) == f.read_bytes()   # 작은 JPEG는 그대로
    assert p._key("see", f).endswith("|q70") and "|q70" not in p._key("recheck", f, "1,2,3,4")
    assert cp.Perceiver(fc)._key("see", f).endswith("|")   # 품질 90은 이전 캐시 키 그대로


# ---------------------------------------------------------------- 2026-11-23 (B) 엘리베이터에서 내린 뒤
IE = O(place="in_elev")
RIDE = [IE] * nv.EXIT_IN
HALLS = [O(place="elev_hall")] * nv.EXIT_OUT
EXIT_SAY = "엘리베이터에서 내렸습니다. 4층에 내렸습니다. 410호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."


def steps(nav, obs_list, t0=0.0, dt=1.5, heading=None):
    """웹처럼 시각(t)과 나침반 방향(heading: 숫자 또는 장마다 목록)을 주고 걸음을 넣는다."""
    out = []
    for i, ob in enumerate(obs_list):
        h = heading[i] if isinstance(heading, list) else heading
        o = {"key": "f.jpg", "keys": ["f.jpg"], "pick_seconds": 0, "obs": ob, "cached": False,
             "call": {"seconds": 1.0, "credit": 0.1}}
        out.append(nav.step(t=t0 + dt * i, observed=o, rechecked=(None, None, None), heading=h))
    return out


def live_nav(target="410", start_floor="1"):
    nav = nv.Navigator(ScriptRunner([O()]), target, start_floor=start_floor)
    nav.announce = False
    nav.new_segment("live_0")
    return nav


def test_auto_exit_after_in_elevator_then_hall():
    """엘리베이터 안 2장 → 복도·홀 2장 연속이면 내린 것으로 (버튼과 같은 처리, 문장 한 번)."""
    nav = live_nav()
    recs = steps(nav, [O()] + RIDE + HALLS + [O()])
    k = 1 + nv.EXIT_IN + nv.EXIT_OUT - 1  # 마지막 복도 사진
    assert nav.si == 1 and recs[k].get("auto_exit") and recs[k]["seq"] == "live_1"
    assert not any(r.get("auto_exit") for r in recs[:k])
    assert recs[k]["say"] == [EXIT_SAY] and recs[k]["action"] == "멈추고 둘러보기"
    assert recs[k]["state"]["floor"] == "4" and recs[k]["state"]["floor_src"] == "엘리베이터 이동"  # 내린 층 = 이동한 층
    assert nav.seg_start == recs[1 + nv.EXIT_IN]["t"]  # 구간 시작 = 첫 복도 사진
    assert recs[k + 1]["say"] == []  # 되풀이하지 않음
    # 그 뒤 버튼 "내렸어요" → 같은 하차 (다시 처리하지 않음)
    assert nav.exit_elevator("버튼") is None and nav.si == 1


def test_no_auto_exit_without_two_in_elevator_or_two_halls():
    nav = live_nav()
    steps(nav, [IE] * (nv.EXIT_IN - 1) + HALLS + RIDE + HALLS[:-1] + [O(place="unknown")] + HALLS[:-1])
    assert nav.si == 0  # 안 장면이 모자람 / 복도·홀이 연속이 아님


def test_button_exit_then_auto_exit_is_same():
    nav = live_nav()
    steps(nav, [O()] + RIDE)
    assert nav.exit_elevator("버튼") == [EXIT_SAY] and nav.si == 1
    recs = steps(nav, RIDE + HALLS, t0=6.0)  # 일찍 눌러서 아직 안 장면 → 20초 안에 자동 감지돼도 같은 하차
    assert nav.si == 1 and not any(r.get("auto_exit") for r in recs)


def test_in_elevator_display_then_auto_exit_carries_floor():
    """B9: 엘리베이터 안 층 표시창(4)을 읽고 자동 하차 → 내린 층 임시 4층, 문장에 표시창 층, 다른 층 문패 한 번이면 문패."""
    nav = live_nav()
    ie4 = dict(IE, elev_floor="4")
    recs = steps(nav, [IE, ie4, ie4] + HALLS)
    assert recs[1]["state"]["floor"] == "4" and recs[1]["say"] == ["엘리베이터 표시창이 4층입니다. 목적지 층입니다."]
    r = recs[-1]
    assert r.get("auto_exit") and r["state"]["floor"] == "4" and r["state"]["floor_temp"]
    assert r["say"][0] == EXIT_SAY and r["state"]["floor_src"] == "엘리베이터 표시창"
    r = steps(nav, [O([("315", "C", 2.5, "door")])] * 2, t0=12.0)[-1]
    assert r["state"]["floor"] == "3" and r["state"]["wrong_floor"] and r["action"] == "엘리베이터로 돌아가기"


def test_auto_exit_keeps_floor_from_first_hall_photo():
    """첫 복도 사진에서 이미 다른 층 문패로 층을 바꿨으면, 두 번째 복도 사진에서 내림을 알아도 그 층을 이어받는다."""
    nav = live_nav()
    recs = steps(nav, RIDE + [O([("405", "C", 2.5, "door")])] + HALLS[1:])
    k = nv.EXIT_IN
    assert recs[k]["state"]["floor"] == "4"
    assert recs[-1].get("auto_exit") == {"t": recs[k]["t"], "kept_floor": "4", "walked": None} and recs[-1]["state"]["floor"] == "4"


def test_exit_wrong_floor_ends_scan_without_direction():
    """다른 층에서 내림: 둘러보기를 바로 끝내고 엘리베이터로 돌아가기 (방향 안내 없음)."""
    nav = live_nav()
    recs = steps(nav, RIDE + HALLS + [O([("310", "C", 2.5, "door")]), O([("309", "L", 3.0, "door")])])
    assert recs[-2]["state"]["floor"] == "4"  # 다른 층 문패 사진 1장: 아직 안 바꿈
    r = recs[-1]
    assert r["state"]["floor"] == "3" and r["action"] == "엘리베이터로 돌아가기" and "direction" not in r
    assert r["state"]["scan"]["done"] and r["state"]["scan"]["why"] == "다른 층"
    assert r["say"] == ["층이 다릅니다. 여기는 3층입니다. 엘리베이터로 돌아가 4층을 누르세요. 엘리베이터는 내린 곳 뒤쪽에 있습니다."]


def test_scan_holds_progress_guidance_up_to_6s():
    """내린 직후 6초 동안은 진행 안내를 하지 않는다 (층을 알아도). 6초 끝에 단서가 없으면 앞으로."""
    nav = nv.Navigator(ScriptRunner([O([("405", "C", 2.5, "door")])] + [O()] * 8), "410")
    exited(nav)
    recs = run(nav, 8)
    assert recs[0]["state"]["floor"] == "4" and recs[0]["action"] == "멈추고 둘러보기"
    assert all(r["action"] == "멈추고 둘러보기" for r in recs[:6]) and not any("직진" in x for r in recs[:6] for x in r["say"])
    assert recs[6]["say"] == ["앞으로 천천히 직진하세요. 문패로 방향을 확인하겠습니다."] and recs[6]["action"] == "직진"
    assert recs[6]["direction"] == {"rel": "F", "strength": "없음", "basis": "단서 없음 (앞으로)"}
    assert any("방향 정면 (근거 없음: 단서 없음 (앞으로))" in e["text"] for e in nav.mem.events)


def test_room_range_board_gives_direction():
    """B7: 호실 범위 안내판에 목적지가 들면 바로 그 방향 ('○호는 오른쪽입니다. 오른쪽으로 가세요'), 둘러보기 끝."""
    nav = live_nav()
    exited_obs = RIDE + HALLS[1:] + [dict(O(), ranges=[{"lo": 401, "hi": 410, "to": "R"}, {"lo": 411, "hi": 420, "to": "L"}])]
    recs = steps(nav, exited_obs)
    r = recs[-1]
    assert r["say"] == [EXIT_SAY, "410호는 오른쪽입니다. 오른쪽 복도로 천천히 직진하세요."] and r["action"] == "오른쪽"
    assert r["direction"] == {"rel": "R", "strength": "강", "basis": "호실 범위 안내판 401~410"}
    assert r["state"]["scan"]["why"] == "강한 방향 단서"


def test_board_direction_uses_compass_when_user_turns():
    """나침반이 있으면 안내판을 본 방향을 절대 방향으로 두었다가, 지금 사용자 기준으로 말한다."""
    nav = live_nav()
    steps(nav, RIDE + HALLS, heading=0.0)  # 내림, 둘러보는 중
    # 동쪽(90도)을 보며 '←' 안내판(목적지 범위) → 북쪽(0도)이 목적지 쪽. 지금 동쪽을 보므로 왼쪽
    board = dict(O(), ranges=[{"lo": 405, "hi": 412, "to": "L"}])
    r = steps(nav, [board], t0=9.0, heading=90.0)[0]
    assert r["state"]["dir"]["abs"] == 0.0 and r["say"] == ["410호는 왼쪽입니다. 왼쪽 복도로 천천히 직진하세요."]
    # 북쪽으로 돌아 같은 안내판을 다시 봄(이번엔 '↑'로 보임): 같은 근거라 다시 말하지 않음
    r = steps(nav, [dict(O(), ranges=[{"lo": 405, "hi": 412, "to": "F"}])], t0=11.0, heading=0.0)[0]
    assert r["say"] == []
    # 둘러보기 뒤 다른 안내판이 반대쪽을 가리키면 다시 안내 (나중 근거를 따름)
    r = steps(nav, [dict(O(), arrows=[{"to": "B", "label": "410"}])], t0=18.0, heading=0.0)[0]
    assert r["say"] == ["410호는 뒤쪽입니다. 뒤로 돌아 천천히 직진하세요."] and r["action"] == "뒤로 돌기"


def test_arrow_with_target_number():
    nav = live_nav()
    r = steps(nav, RIDE + HALLS[1:] + [dict(O(), arrows=[{"to": "L", "label": "401~415"}])])[-1]
    assert r["action"] == "왼쪽" and r["direction"]["basis"] == "화살표 표지 '401~415'"


def test_scan_wall_plates_with_compass_two_votes():
    """B8b: 둘러보며 나침반 방향과 문패를 모아 같은 쪽 벽끼리 비교. 두 벽이 같은 결론(2개) → '○쪽입니다'.
    북(0도)을 볼 때 왼쪽 벽 405(2m)→407(6m): 북쪽이 목적지 쪽. 남(180도)을 볼 때 왼쪽 벽 404(2m)→402(6m): 역시 북쪽.
    지금 남쪽을 보고 있으므로 '뒤쪽'."""
    nav = live_nav()
    steps(nav, RIDE + HALLS, heading=0.0)
    recs = steps(nav, [O([("405", "L", 2.0, "door"), ("407", "L", 6.0, "door")]),
                       O([("404", "L", 2.0, "door"), ("402", "L", 6.0, "door")])], t0=9.0, heading=[0.0, 180.0])
    r = recs[1]
    assert r["direction"]["strength"] == "강" and r["direction"]["basis"].startswith("같은 쪽 벽 문패 비교")
    assert r["say"][-1] == "410호는 뒤쪽입니다. 뒤로 돌아 천천히 직진하세요." and r["action"] == "뒤로 돌기"
    assert r["state"]["scan"]["why"] == "강한 방향 단서"


def test_scan_wall_plates_one_vote_waits_then_maybe():
    """비교가 1개뿐이면 6초 끝에 '○쪽일 가능성이 높습니다' (보통)."""
    nav = live_nav()
    steps(nav, RIDE + HALLS, heading=0.0)  # 내림을 안 때(밖 3장째 7.5초)부터 둘러보기 → 13.5초에 끝
    recs = steps(nav, [O([("405", "L", 2.0, "door"), ("407", "L", 6.0, "door")]), O(), O(), O()], t0=9.0, heading=0.0)
    assert recs[0]["say"] == [] or "가능성" not in " ".join(recs[0]["say"])
    r = recs[3]
    assert r["t"] == 13.5 and r["say"] == ["410호는 정면일 가능성이 높습니다. 앞으로 천천히 직진하세요. 문패로 확인하겠습니다."]
    assert r["direction"]["strength"] == "보통"


def test_scan_without_compass_compares_same_photo_only():
    """나침반이 없으면 다른 사진의 문패끼리는 비교하지 않는다 (몸을 돌린 정도를 모름)."""
    nav = nv.Navigator(ScriptRunner([O([("405", "L", 2.0, "door")]), O([("407", "L", 6.0, "door")])] + [O()] * 6), "410")
    nav.st["floor"] = "4"
    exited(nav)
    recs = run(nav, 7)
    assert recs[6]["direction"]["basis"] == "단서 없음 (앞으로)"
    nav2 = nv.Navigator(ScriptRunner([O([("405", "L", 2.0, "door"), ("407", "L", 6.0, "door")])] + [O()] * 6), "410")
    exited(nav2)
    recs = run(nav2, 7)
    assert recs[6]["direction"]["basis"].startswith("같은 쪽 벽 문패 비교 405→407") and recs[6]["action"] == "직진"


def test_previous_floor_same_tail_is_weak():
    """B8c: 3층에서 내려 엘리베이터에서 걸어가며 310을 동쪽(90도)에서 봄 → 4층에서 단서가 없으면 410은 그쪽일 수 있음 (약)."""
    nav = live_nav()
    steps(nav, RIDE + HALLS, heading=90.0)
    steps(nav, [O([("310", "C", 2.5, "door")])] * 2, t0=9.0, heading=90.0)  # 3층 (다른 층) → 엘리베이터로 돌아가기
    assert nav.plate_dirs and nav.plate_dirs[0]["n"] == "310" and nav.plate_dirs[0]["h"] == 90.0
    steps(nav, RIDE, t0=40.0, heading=0.0)
    recs = steps(nav, [O()] * 9, t0=45.0, heading=270.0)  # 4층, 서쪽을 봄
    r = next(r for r in recs if "direction" in r)
    assert r["direction"] == {"rel": "B", "strength": "약", "basis": "3층에서 310호를 본 방향 (끝자리 비교)"}
    assert r["say"] == ["410호는 뒤쪽일 가능성이 있습니다. 뒤로 돌아 천천히 직진하세요. 문패로 방향을 확인하겠습니다."]


def test_long_way_when_no_clue():
    """B8d: 단서가 없으면 복도가 길게 보이는 쪽."""
    nav = live_nav()
    recs = steps(nav, RIDE + HALLS + [dict(O(), long_way="R")] * 5)
    r = next(r for r in recs if "direction" in r)
    assert r["say"] == ["복도가 길게 보이는 오른쪽 복도로 천천히 직진하세요. 문패로 방향을 확인하겠습니다."] and r["action"] == "오른쪽"


def test_far_plates_same_wall_one_step():
    """2026-11-23 사용자 지시: 목적지 ±3 밖 같은 쪽 벽 문패 2개가 멀어지면 뒤로, 가까워지면 방향이 맞습니다."""
    plates = [O([("420", "L", 3.0, "door")])] * 2 + [O([("422", "L", 3.0, "door")])] * 2
    nav = nv.Navigator(ScriptRunner([O()] * 7 + plates), "410")
    exited(nav)
    nav.st["floor"] = "4"
    recs = run(nav, 11)
    assert recs[9]["say"] == ["422호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요."] and recs[9]["action"] == "뒤로 돌기"
    assert recs[9]["trend"] == "같은 쪽 벽 420→422 멀어짐 (±3 밖)"
    plates = [O([("420", "R", 3.0, "door")])] * 2 + [O([("416", "R", 3.0, "door")])] * 2 + [O([("418", "L", 3.0, "door")])] * 2
    nav = nv.Navigator(ScriptRunner([O()] * 7 + plates), "410")
    exited(nav)
    nav.st["floor"] = "4"
    recs = run(nav, 13)
    assert recs[9]["say"] == ["416호를 지나고 있습니다. 목적지 방향이 맞습니다. 계속 직진하세요."] and recs[9]["action"] == "직진"
    assert not any("멀어지고" in x for r in recs for x in r["say"])  # 다른 쪽 벽(418) 하나로는 비교하지 않음

def test_recheck_of_name_plate_reads_same_number():
    """이름만 읽힌 문패를 확대해 읽은 번호가 이 사진의 번호와 같으면 그 번호의 재확인 일치 (q70 판단 지점 A6)."""
    nav = nv.Navigator(ScriptRunner([O([("317", "L", 3.0, "wall")], unread=["소재분석실"])],
                                    rechecks={"소재분석실": "317"}), "314")
    exited(nav)
    r = run(nav, 1)[0]
    assert r["recheck"]["same_as"] == "317" and r["state"]["floor"] == "3"


# ---------------------------------------------------------------- 2026-11-23 시험 2: 4층에서 시작
def _start_4f(target, after):
    """시험 2와 같은 흐름: 처음 5초 문패 없음(위치 파악) → 407(오른쪽 8m)로 4층 → 문패 없는 복도 → after."""
    script = [O(place="unknown")] + [O()] * 5 + [O([("407", "R", 8.0, "door")])] + [dict(O(), long_way="C")] * 17 + after
    nav = nv.Navigator(ScriptRunner(script), target)
    nav.new_segment("1")
    return nav, run(nav, len(script))


def test_start_on_goal_floor_405_scans_then_guides():
    nav, recs = _start_4f("405", [O([("406", "R", 3.0, "door")])] * 2)
    r = recs[6]
    assert r["state"]["floor"] == "4" and r["action"] == "멈추고 둘러보기"
    assert r["say"] == ["4층입니다. 오른쪽에 407호가 보입니다. 405호를 찾겠습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."]
    d = next(x for x in recs if "direction" in x)
    assert d["t"] == r["t"] + 6 and d["say"] == ["복도가 길게 보이는 앞쪽 복도로 천천히 직진하세요. 문패로 방향을 확인하겠습니다."]
    assert d["action"] == "직진"
    # 목적지 층에서는 행동이 비거나 '멈추고 둘러보기'(모름)로 남지 않고, 무안내 10초 문장에 행동이 들어감
    after = [x for x in recs if x["t"] > d["t"]]
    assert all(x["action"] in ("직진", "목적지 접근") for x in after)
    quiet = [x for x in after if x["say"] == [nv.QUIET_GOAL]]
    assert quiet and quiet[0]["t"] - d["t"] == 10
    assert recs[-2]["say"] == ["406호를 지나고 있습니다. 목적지 방향이 맞습니다. 계속 직진하세요."]


def test_start_on_goal_floor_410_toward_and_far_away():
    nav, recs = _start_4f("410", [O([("408", "R", 3.0, "door")])] * 2)
    assert "410호를 찾겠습니다" in recs[6]["say"][0]
    assert recs[-2]["say"] == ["408호를 지나고 있습니다. 목적지 방향이 맞습니다. 계속 직진하세요."]  # 407→408 (±3 안)
    # ±3 밖만 보이는 경우: 같은 쪽 벽 413 → 415면 뒤로
    nav, recs = _start_4f("410", [O([("414", "L", 3.0, "door")])] * 2 + [O([("416", "L", 3.0, "door")])] * 2)
    assert recs[-2]["say"] == ["416호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요."] and recs[-2]["action"] == "뒤로 돌기"


def test_start_on_goal_floor_when_target_plate_itself_gives_floor():
    """목적지 문패로 층을 알면 둘러보기 없이 목적지 안내."""
    nav = nv.Navigator(ScriptRunner([O([("405", "R", 4.0, "door")])] * 3), "405")
    nav.new_segment("1")
    recs = run(nav, 2)
    assert not any("잠시 멈춰서 휴대폰" in x for r in recs for x in r["say"]) and recs[0]["state"]["scan"] is None


def test_quick_photo_does_not_drive_trend():
    """시험 2: 405 앞에서 멈춘 빠른 확인 사진이 맞은편 408을 읽어 '뒤로 돌아가세요' → 빠른 확인 사진은 진행 방향에 안 씀."""
    nav = nv.Navigator(ScriptRunner([O()] * 7 + [O([("406", "R", 3.0, "door")])] * 2 + [O([("405", "C", 3.0, "door")])]), "405")
    exited(nav)
    nav.st["floor"] = "4"
    recs = run(nav, 10)
    assert recs[9]["confirm_now"]
    r = nav.step(t=10.4, observed=qo(10.4, [("408", "R", 2.0)]), quick=True)
    assert r["action"] == "목적지 접근" and not any("멀어지고" in x for x in r["say"]) and "trend" not in r


# ---------------------------------------------------------------- 2026-11-23 엘리베이터 안·밖 (판 6)
def test_closed_doors_after_inside_stay_inside():
    """시험 1: 칸 안에서 닫힌 문을 본 사진이 elev_hall로 읽혀 잘못 내림 → 직전까지 안이면 닫힌 문만 보이는 장면(cd)은 안."""
    nav = live_nav()
    cd = dict(O(place="elev_hall"), closed_doors=True)
    recs = steps(nav, RIDE + [cd] * 6 + HALLS)
    assert not any(r.get("auto_exit") for r in recs[:-1]) and recs[-1].get("auto_exit")
    assert any("계속 안으로 봄" in e for r in recs for e in r["events"])
    # 밖에서 기다리며 본 닫힌 문(탄 적 없음)은 안으로 보지 않음
    nav = live_nav()
    recs = steps(nav, [cd] * 4 + HALLS)
    assert nav.st["elev_in"] is False and not any(r.get("auto_exit") for r in recs)


def test_floor_button_panel_is_inside():
    nav = live_nav()
    panel = dict(O(place="elev_hall"), elev={"seen": False, "cand": False, "why": "panel", "side": "R", "m": 0.5, "btn": "none"})
    recs = steps(nav, RIDE + [dict(panel, closed_doors=False, panel=True)] * 3 + HALLS)
    assert recs[-1].get("auto_exit") and not any(r.get("auto_exit") for r in recs[:-1])
    nav = live_nav()  # 타기 전 패널(잘못 봄)은 '탔음'이 아님
    recs = steps(nav, [dict(panel, panel=True)] * 3 + HALLS)
    assert nav.st["elev_in"] is False


def test_close_elevator_doors_after_boarding_stay_inside():
    """시험 1·11-19: 칸 안에서 닫힌 문(1~2m)을 elev_hall로 읽어도 탄 뒤에는 안으로. 2m보다 멀거나 문이 없으면 밖."""
    nav = live_nav()
    door = dict(O(place="elev_hall"), elev={"seen": False, "cand": True, "why": "doors", "side": "C", "m": 1.0, "btn": "none"})
    far = dict(O(place="lobby"))
    recs = steps(nav, RIDE + [door] * 8 + [far] * 3)
    assert not any(r.get("auto_exit") for r in recs[:-1]) and recs[-1].get("auto_exit")


def test_exit_needs_steps_when_step_counter_works():
    """걸음 수가 들어오는 세션: 밖 장면 3장이어도 3걸음 이상 걸어야 내림. 걸음 수가 늘 0(센서 꺼짐)이면 장면만."""
    nav = live_nav()
    out = []
    for i, (ob, st_n) in enumerate(zip(RIDE + HALLS + [O(), O()], [5, 5, 5, 5, 6, 6, 7, 9])):
        o = {"key": "f.jpg", "keys": ["f.jpg"], "pick_seconds": 0, "obs": ob, "cached": False, "call": {"seconds": 1.0}}
        out.append(nav.step(t=1.5 * i, observed=o, rechecked=(None, None, None), steps=st_n))
    assert not any(r.get("auto_exit") for r in out[:7])     # 밖 3장·4장째까지 1~2걸음
    assert out[7]["auto_exit"]["walked"] == 4                # 밖 5장째, 첫 밖 사진(5걸음)부터 4걸음
    nav = live_nav()
    recs = []
    for i, ob in enumerate(RIDE + HALLS):
        o = {"key": "f.jpg", "keys": ["f.jpg"], "pick_seconds": 0, "obs": ob, "cached": False, "call": {"seconds": 1.0}}
        recs.append(nav.step(t=1.5 * i, observed=o, rechecked=(None, None, None), steps=0))
    assert recs[-1]["auto_exit"]["walked"] is None


def test_auto_exit_scan_starts_when_exit_is_known():
    """자동 하차는 첫 밖 사진보다 늦게 알게 되므로, 둘러보기 6초는 내림을 안 때부터 (내림 문장과 방향 문장이 한꺼번에 나오지 않게)."""
    nav = live_nav()
    door = dict(O(place="elev_hall"), elev={"seen": False, "cand": True, "why": "doors", "side": "C", "m": 1.0, "btn": "none"})
    recs = steps(nav, RIDE + [door] * 2 + [O(place="lobby")] + [O(place="lobby")] * 2, dt=3.0)
    r = recs[-1]
    assert r.get("auto_exit") and r["say"] == [EXIT_SAY] and "direction" not in r
    assert r["state"]["scan"]["t"] == r["t"] and nav.seg_start < r["t"]


# ---------------------------------------------------------------- 2026-11-23 사용자 지시: 내린 뒤 층 = 엘리베이터로 이동한 층
def test_exit_floor_is_elevator_target_and_no_floor_question():
    """내린 층은 '4층을 누르세요'의 4층. 내린 직후 "4층에 내렸습니다"와 방향 정하기, 층을 묻지 않음."""
    nav = live_nav()
    recs = steps(nav, RIDE + HALLS + [O()] * 14)
    k = nv.EXIT_IN + nv.EXIT_OUT - 1
    assert recs[k]["say"] == [EXIT_SAY] and recs[k]["state"]["floor"] == "4" and recs[k]["state"]["assumed"]
    assert any("direction" in r for r in recs[k:]) and nv.ASK_SAY not in says(recs) and nv.CHECK_SAY not in says(recs)


def test_exit_floor_corrected_by_sign_once():
    nav = live_nav()
    recs = steps(nav, RIDE + HALLS + [O(floor_sign="3F")])
    r = recs[-1]
    assert r["state"]["floor"] == "3" and r["action"] == "엘리베이터로 돌아가기"
    assert r["say"] == ["층이 다릅니다. 여기는 3층입니다. 엘리베이터로 돌아가 4층을 누르세요. 엘리베이터는 내린 곳 뒤쪽에 있습니다."]


def test_exit_floor_one_misread_does_not_change_and_same_floor_confirms():
    nav = live_nav()
    recs = steps(nav, RIDE + HALLS + [O([("315", "C", 2.5, "door")]), O(), O(), O([("407", "C", 2.5, "door")]), O()])
    assert all(r["state"]["floor"] == "4" for r in recs[nv.EXIT_IN + nv.EXIT_OUT - 1:])   # 315 한 번(오독)으로는 안 바뀜
    assert recs[-2]["state"]["assumed"] is False and recs[-2]["state"]["floor_src"] == "407호"   # 같은 층 문패로 확정
    assert not any("층이 다릅니다" in x for r in recs for x in r["say"])


def test_exit_floor_correction_points_back_to_elevator_with_compass():
    """나침반: 내릴 때 북(0도)을 봤으면 엘리베이터는 남쪽. 북동(30도)을 보면 뒤쪽 → 돌고 나서 11시 방향."""
    nav = live_nav()
    steps(nav, RIDE + HALLS, heading=0.0)
    recs = steps(nav, [O([("310", "C", 2.5, "door")]), O([("309", "L", 3.0, "door")])], t0=10.0, heading=30.0)
    assert recs[-1]["say"] == ["층이 다릅니다. 여기는 3층입니다. 엘리베이터로 돌아가 4층을 누르세요. 뒤로 돌아 11시 방향입니다."]
    assert nav.elevator_way(90.0) == "엘리베이터는 3시 방향입니다."    # 동쪽을 보면 남쪽은 오른쪽
    assert nav.elevator_way(180.0) == "엘리베이터는 12시 방향입니다."


def test_call_button_after_boarding_cancels_and_near_elevator_not_exit():
    """시험 3: 1층에서 기다리며 닫힌 문을 '안'으로 3장 본 뒤 호출 버튼(1m) 장면 → 탄 것 취소, 하차 아님."""
    nav = live_nav()
    btn = dict(O(place="elev_hall"), elev={"seen": True, "cand": True, "why": "button", "side": "R", "m": 1.0, "btn": "R"})
    recs = steps(nav, RIDE + [btn] * 3 + [O(place="lobby")] * 3)
    assert not any(r.get("auto_exit") for r in recs) and nav.si == 0
    assert any("탄 것이 아님" in e for r in recs for e in r["events"])
    nav = live_nav()   # 탄 뒤 2m 안 표시창 장면(elev_hall)은 하차 근거가 아님
    disp = dict(O(place="elev_hall"), elev={"seen": True, "cand": True, "why": "display", "side": "C", "m": 1.5, "btn": "none"})
    recs = steps(nav, RIDE + [disp] * 3 + [O(place="lobby")] * 2)
    assert not any(r.get("auto_exit") for r in recs)


def test_locate_end_asks_floor_at_once():
    nav = nv.Navigator(ScriptRunner([O()] * 8), "410")
    nav.new_segment("1")
    recs = run(nav, 8)
    assert recs[4]["say"] == ["여기는 복도인 것 같습니다. " + nv.ASK_SAY] and recs[4]["ask_floor"]
    assert nv.ASK_SAY not in says(recs[5:])   # 20초 안에는 다시 묻지 않음


def test_quiet_sentence_waiting_and_inside_elevator():
    nav = start_1f([O(elev=EL, place="elev_hall")] * 14)
    recs = run(nav, 14)
    assert ["엘리베이터를 기다리고 있습니다. 문이 열리면 타고 4층을 누르세요."] in [r["say"] for r in recs]
    nav.st["elev_in"] = True
    nav.st["last_say"] = nav.clock - 20
    r = nav.step(["f.jpg"])
    assert r["say"] == ["엘리베이터 안입니다. 4층에 도착하면 내리세요."]
