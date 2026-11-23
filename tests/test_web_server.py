"""웹 서버 시험: 가짜 응답(크레딧 0)으로 목적지 해석·겹쳐 보내기·한도·받아쓰기·기록을 확인한다."""
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nav"))

import navigator as nv  # noqa: E402
import web_server as wsrv  # noqa: E402
from claude_client import FakeClient, Ledger  # noqa: E402
from dest_parse import parse_room  # noqa: E402
from fake_scene import fake_client  # noqa: E402
from stt import FakeTranscriber  # noqa: E402

CODE = "test-code"
H = {"X-Access-Code": CODE}


def jpg():
    b = io.BytesIO()
    Image.new("RGB", (72, 128), "gray").save(b, "JPEG")
    return b.getvalue()


def app_with(tmp_path, factory=None, transcriber=None, **kw):
    cfg = wsrv.Settings(sessions_dir=tmp_path, access_code=CODE, **kw)
    calls = []

    def default(target, limit):
        c = fake_client(target, delay=0, limit=limit)
        calls.append(c)
        return c
    app = wsrv.create_app(cfg, client_factory=factory or default, transcriber=transcriber)
    return TestClient(app), calls


def frame(c, sid, t):
    return c.post(f"/api/session/{sid}/frame", headers=H, files={"image": ("f.jpg", jpg(), "image/jpeg")},
                  data={"t": str(t), "heading": "90", "steps": "3"}).json()


@pytest.mark.parametrize("text,room", [
    ("410호로 가고 싶어", "410"), ("410%", "410"), ("410 % 로 가 줘", "410"), ("사백십호", "410"), ("사백 십 호", "410"),
    ("사일공호", "410"), ("4 1 0 호", "410"), ("삼백십칠호", "317"), ("삼백 십칠 호로 데려다줘", "317"),
    ("몰라", None), ("4층", None), ("화장실", None), ("", None)])
def test_parse_room(text, room):
    assert parse_room(text) == room


def test_code_required_and_no_claude_call(tmp_path):
    c, calls = app_with(tmp_path)
    assert c.post("/api/session", json={"destination": "410호"}).status_code == 401
    assert c.post("/api/session", json={"destination": "410호"}, headers={"X-Access-Code": "wrong"}).status_code == 401
    assert calls == []
    assert c.get("/api/monitor", headers=H).status_code == 200


def test_start_says_and_retry(tmp_path):
    c, calls = app_with(tmp_path)
    r = c.post("/api/session", headers=H, json={"destination": "410%"}).json()
    assert r["status"] == "ok" and r["target"] == "410" and r["say"] == ["410호로 안내하겠습니다."]
    r2 = c.post("/api/session", headers=H, json={"destination": "음 그러니까"}).json()
    assert r2["status"] == "retry" and r2["say"] == ["다시 말씀해 주세요."]
    assert len(calls) == 1  # 못 알아들은 말은 세션을 만들지 않음 (Claude 호출 없음)


def test_fake_route_to_arrival(tmp_path):
    c, calls = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "사백십호", "mode": "file", "files": ["4F.mp4"]}).json()["session_id"]
    says, last = [], None
    for k in range(20):
        r = frame(c, sid, 1.5 * k)
        assert r["status"] == "ok" and r["seq"] == k + 1
        says += r["say"]
        last = r
        if r["arrived"]:
            break
    assert says[0].startswith("410호는 4층입니다. 앞에 유리문이 있습니다")  # "안내를 시작합니다"는 웹이 먼저 말해서 뺌
    assert "목적지 같습니다, 확인 중입니다." in says
    assert last["arrived"] and last["say"] == ["정면에 410호 문이 있습니다."]
    d = tmp_path / sid
    lines = [json.loads(x) for x in (d / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "start" and lines[0]["mode"] == "file"
    assert lines[1]["heading"] == 90.0 and lines[1]["steps"] == 3 and lines[1]["web_seq"] == 1
    assert (d / "memory.json").is_file() and len(list((d / "frames").glob("*.jpg"))) == k + 1
    rec = c.get("/api/records", headers=H).json()["records"][0]
    assert rec["session_id"] == sid and rec["arrived"] and rec["target"] == "410"
    html = c.get(f"/api/records/{sid}/replay?code={CODE}").text
    assert "안내 기록" in html and "정면에 410호 문이 있습니다." in html
    m = c.get(f"/api/session/{sid}", headers=H).json()
    assert m["detail"]["arrived"] and "410" in m["detail"]["confirmed"]
    assert c.get(f"/api/session/{sid}/latest-frame?code={CODE}").status_code == 200


def slow_factory(delays):
    """호출 순서마다 기다리는 초를 정한 가짜 (겹쳐 보내기 시험)."""
    made = []

    def factory(target, limit):
        lock, n = threading.Lock(), [0]

        def responder(kind, content, _n):
            if kind == "see":
                with lock:
                    i = n[0]
                    n[0] += 1
                time.sleep(delays[min(i, len(delays) - 1)])
            return {"plates": [], "fs": "", "place": "corridor", "door": "none", "el": [], "arrows": []} \
                if kind == "see" else {"n": "", "name": ""}
        cl = FakeClient(responder, credit=0.18)
        cl.ledger = Ledger(limit)
        made.append(cl)
        return cl
    return factory, made


def test_busy_when_two_in_flight(tmp_path):
    """2026-11-23: 동시 처리 최대 3 → 2."""
    factory, _ = slow_factory([0.6])
    c, _ = app_with(tmp_path, factory=factory)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    out = [None] * 5

    def go(i):
        out[i] = frame(c, sid, i * 0.1)
    ths = [threading.Thread(target=go, args=(i,)) for i in range(5)]
    for th in ths:
        th.start()
        time.sleep(0.02)
    for th in ths:
        th.join()
    st = [r["status"] for r in out]
    assert st.count("busy") == 3 and st.count("ok") + st.count("stale") == 2


def test_stale_dropped_and_order_kept(tmp_path):
    factory, _ = slow_factory([0.8, 0.05])  # 먼저 보낸 것이 늦게 옴
    c, _ = app_with(tmp_path, factory=factory)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    out = {}
    t1 = threading.Thread(target=lambda: out.__setitem__(1, frame(c, sid, 0.0)))
    t1.start()
    time.sleep(0.1)
    out[2] = frame(c, sid, 1.5)
    t1.join()
    assert out[2]["status"] == "ok" and out[2]["seq"] == 2
    assert out[1]["status"] == "stale" and out[1]["seq"] == 1
    m = c.get(f"/api/session/{sid}", headers=H).json()["detail"]
    assert m["counts"]["stale"] == 1 and m["counts"]["applied"] == 1


def test_session_credit_limit(tmp_path):
    c, _ = app_with(tmp_path, credit_per_session=0.5)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    st = [frame(c, sid, i)["status"] for i in range(5)]
    assert st[:3] == ["ok", "ok", "ok"] and st[3] == "limit" and st[4] == "stopped"
    daily = json.loads((tmp_path / "daily_credit.json").read_text())
    assert abs(sum(daily.values()) - 0.54) < 1e-6  # 3번 × 0.18 (확대 재확인 없음)


def test_daily_limit_blocks_new_session(tmp_path):
    import datetime as dt
    (tmp_path / "daily_credit.json").write_text(json.dumps({dt.date.today().isoformat(): 300.0}))
    c, calls = app_with(tmp_path)
    r = c.post("/api/session", headers=H, json={"destination": "410"}).json()
    assert r["status"] == "limit" and calls == []


def test_voice_whisper_and_retry(tmp_path):
    c, _ = app_with(tmp_path, transcriber=FakeTranscriber("사백십호"))
    r = c.post("/api/voice", headers=H, files={"audio": ("v.m4a", b"x" * 100, "audio/mp4")},
               data={"meta": json.dumps({"mode": "camera"})}).json()
    assert r["status"] == "ok" and r["target"] == "410" and r["heard"] == "사백십호"
    c2, _ = app_with(tmp_path, transcriber=FakeTranscriber("410%"))
    assert c2.post("/api/voice", headers=H, files={"audio": ("v.m4a", b"x", "audio/mp4")}).json()["target"] == "410"
    c3, _ = app_with(tmp_path, transcriber=FakeTranscriber("시청해 주셔서 감사합니다"))
    assert c3.post("/api/voice", headers=H, files={"audio": ("v.m4a", b"x", "audio/mp4")}).json()["status"] == "retry"


def test_exited_before_any_frame_is_new_segment(tmp_path):
    c, _ = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()
    assert r["segment"] == 1 and r["say"] == ["엘리베이터에서 내렸습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."]
    # 2026-11-23: 버튼 응답에서 말했으므로 첫 사진에서 되풀이하지 않음 (전에는 두 번 말함)
    f = frame(c, sid, 0.0)
    assert not any(x.startswith("엘리베이터에서 내렸습니다") for x in f["say"]) and f["action"] == "멈추고 둘러보기"


def test_exited_and_destination_change(tmp_path):
    c, _ = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    frame(c, sid, 0.0)
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()
    assert r["segment"] == 1 and "내렸습니다" in r["say"][0]
    r = c.post(f"/api/session/{sid}/destination", headers=H, json={"destination": "삼백십칠호"}).json()
    # 리뷰 4: 목적지를 바꾸면 다음 할 일도 말함 (내린 뒤 층 모름 → 층 확인)
    assert r["status"] == "ok" and r["target"] == "317" and r["say"] == ["317호로 안내하겠습니다.", nv.CHECK_SAY]
    assert c.get(f"/api/session/{sid}", headers=H).json()["target"] == "317"
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "end"}).json()
    assert frame(c, sid, 3.0)["status"] == "stopped"


def test_static_pages(tmp_path):
    c, _ = app_with(tmp_path)
    for p in ("/", "/monitor", "/records", "/app.js", "/sensors.js", "/style.css"):
        assert c.get(p).status_code == 200, p
    assert "BAZE" not in c.get("/app.js").text  # 키·키 이름이 웹 페이지에 없음


def test_sensor_js_with_fake_sensors():
    """센서 즉시 안내(sensors.js)를 Node에서 가짜 센서 값으로 시험 (Node가 없으면 건너뜀)."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node 없음")
    f = Path(__file__).resolve().parent / "web" / "sensors.test.js"
    r = subprocess.run([node, "--test", str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def quick(c, sid, t):
    return c.post(f"/api/session/{sid}/confirm", headers=H, files={"image": ("q.jpg", jpg(), "image/jpeg")},
                  data={"t": str(t)}).json()


def test_confirm_endpoint_two_photos_then_arrival(tmp_path):
    """빠른 확인(기본 켬): 목적지를 5m 안에서 읽으면 confirm → 사진 2장(/confirm). 두 번째 번(2m 안)에서 그 자리 도착."""
    c, calls = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410호"}).json()["session_id"]
    rounds, t, arrived = 0, 0.0, None
    for k in range(30):
        r = frame(c, sid, t)
        t += 1.5
        if r["arrived"]:
            arrived = ("frame", r)
            break
        if r["confirm"]:
            rounds += 1
            q1, q2 = quick(c, sid, t - 1.0), quick(c, sid, t - 0.6)
            assert q1["status"] == "ok" and q2["status"] == "ok" and q1["quick"] and q2["quick"]
            if q1["arrived"] or q2["arrived"]:
                arrived = ("quick", q1 if q1["arrived"] else q2)
                break
    assert rounds == 2 and arrived and arrived[0] == "quick"
    assert arrived[1]["say"][-1].endswith("410호 문이 있습니다.")
    log = [json.loads(x) for x in (tmp_path / sid / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for x in log if x.get("quick")) == 4 and log[0]["quick_confirm"] and log[0]["send_quality"] == 70


def test_quick_confirm_off_by_setting(tmp_path):
    c, calls = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410호", "quick_confirm": False}).json()["session_id"]
    rs = [frame(c, sid, 1.5 * k) for k in range(25)]
    assert not any(r.get("confirm") for r in rs) and any(r["arrived"] for r in rs)
    c2, _ = app_with(tmp_path / "off", quick_confirm=False)
    sid = c2.post("/api/session", headers=H, json={"destination": "410호"}).json()["session_id"]
    assert not any(frame(c2, sid, 1.5 * k).get("confirm") for k in range(25))


# ---------------------------------------------------------------- 2026-11-23
def scripted_factory(scenes, delays):
    """호출 순서마다 장면과 기다리는 초를 정한 가짜. scenes[i]는 짧은 키 형식 인식 결과."""
    def factory(target, limit):
        lock, n = threading.Lock(), [0]

        def responder(kind, content, _n):
            if kind != "see":
                return {"n": "", "name": ""}
            with lock:
                i = n[0]
                n[0] += 1
            time.sleep(delays[min(i, len(delays) - 1)])
            return scenes[min(i, len(scenes) - 1)]
        cl = FakeClient(responder, credit=0.18)
        cl.ledger = Ledger(limit)
        return cl
    return factory


BASE = {"plates": [], "fs": "", "place": "corridor", "door": "none", "el": [], "arrows": []}


def test_review2_exited_drops_photos_from_before(tmp_path):
    """'내렸어요' 때 처리 중이던 엘리베이터 안 사진(1F 표시)은 새 구간 첫 걸음으로 들어가지 않는다.
    눌렀을 때 이미 찍었지만 늦게 보낸 사진(seg=0)도 버린다."""
    inside = {**BASE, "place": "in_elev", "fs": "1F"}
    factory = scripted_factory([inside, {**BASE, "fs": "4F"}], [0.6, 0.0])
    c, _ = app_with(tmp_path, factory=factory)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    out = {}
    th = threading.Thread(target=lambda: out.__setitem__(1, frame(c, sid, 0.0)))
    th.start()
    time.sleep(0.15)
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()
    assert r["seg"] == 1 and r["say"][0].startswith("엘리베이터에서 내렸습니다")
    th.join()
    assert out[1]["status"] == "stale" and out[1]["reason"] == "old_segment"
    late = c.post(f"/api/session/{sid}/frame", headers=H, files={"image": ("f.jpg", jpg(), "image/jpeg")},
                  data={"t": "0.5", "seg": "0"}).json()
    assert late["status"] == "stale" and late["reason"] == "old_segment"
    new = c.post(f"/api/session/{sid}/frame", headers=H, files={"image": ("f.jpg", jpg(), "image/jpeg")},
                 data={"t": "2.0", "seg": "1"}).json()
    assert new["status"] == "ok" and new["seg"] == 1
    assert new["say"][0] == "4층이 맞습니다." and new["action"] == "멈추고 둘러보기"  # 새 구간의 첫 걸음은 이 사진
    m = c.get(f"/api/session/{sid}", headers=H).json()["detail"]
    assert m["counts"]["old_segment"] == 2 and m["counts"]["applied"] == 1
    ws = c.app.state.sessions[sid]
    assert ws.nav.st["floor"] == "4" and ws.nav.records[-1]["state"]["floor"] == "4"
    assert all(r["seq"] != "live_0" or r["web_seq"] != 1 for r in ws.nav.records if "web_seq" in r)


def test_start_floor_from_speech_and_here_floor(tmp_path):
    c, _ = app_with(tmp_path)
    r = c.post("/api/session", headers=H, json={"destination": "4층인데 410호 가고 싶어"}).json()
    assert r["status"] == "ok" and r["target"] == "410" and r["floor"] == "4"
    assert r["say"] == ["410호로 안내하겠습니다. 지금 4층에서 출발합니다."]
    sid = r["session_id"]
    f = frame(c, sid, 0.0)
    assert f["floor"] == "4" and "지금 4층입니다. 앞에 유리문이 있습니다" in f["say"][0]
    # 안내 중 길게 눌러 "여기 3층이야" → 지금 층으로 바로
    r = c.post(f"/api/session/{sid}/destination", headers=H, json={"destination": "여기 3층이야"}).json()
    assert r["status"] == "ok" and r["kind"] == "floor" and r["floor"] == "3" and r["target"] == "410"
    assert r["say"][0].startswith("3층으로 알겠습니다. 410호는 4층에 있습니다")
    # "4층에서 105호": 목적지와 지금 층을 함께
    r = c.post(f"/api/session/{sid}/destination", headers=H, json={"destination": "4층에서 105호"}).json()
    assert r["target"] == "105" and r["floor"] == "4" and "1층" in " ".join(r["say"])
    # 층도 호실도 없으면 다시 말해 달라고
    assert c.post(f"/api/session/{sid}/destination", headers=H, json={"destination": "음"}).json()["status"] == "retry"


def test_voice_here_floor_whisper(tmp_path):
    c, _ = app_with(tmp_path, transcriber=FakeTranscriber("사층이요"))
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    r = c.post(f"/api/session/{sid}/voice", headers=H, files={"audio": ("v.m4a", b"x" * 100, "audio/mp4")}).json()
    assert r["kind"] == "floor" and r["floor"] == "4" and r["heard"] == "사층이요"


@pytest.mark.parametrize("text,here", [
    ("4층인데 410호 가고 싶어", "4"), ("4층에서 410호", "4"), ("사층에서 사백십호", "4"), ("일층인데 사백십호", "1"),
    ("4층 410호로 가고 싶어", None), ("410호로 가고 싶어", None), ("지하1층에서 105호", "B1")])
def test_parse_here_floor_with_destination(text, here):
    from dest_parse import parse_destination
    assert parse_destination(text)["here"] == here


def test_auto_exit_then_button_is_same_exit(tmp_path):
    """2026-11-23 B5: 엘리베이터 안 2장 → 복도 2장이면 서버가 스스로 내림 처리(응답 exited). 그 뒤 "내렸어요"는 같은 하차."""
    inside = {**BASE, "place": "in_elev"}
    hall = {**BASE, "place": "elev_hall"}
    factory = scripted_factory([inside] * 3 + [hall] * 3 + [{**BASE, "rr": [["401", "410", "R"]]}], [0.0] * 7)
    c, _ = app_with(tmp_path, factory=factory)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    rs = [frame(c, sid, 1.5 * i) for i in range(6)]
    assert not rs[4]["exited"] and rs[5]["exited"] and rs[5]["action"] == "멈추고 둘러보기"
    assert rs[5]["say"][0] == "엘리베이터에서 내렸습니다. 잠시 멈춰서 휴대폰을 천천히 좌우로 비춰 주세요."
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()
    assert r["segment"] == 1 and r["say"] == ["엘리베이터에서 내린 것을 이미 확인했습니다."]
    ws = c.app.state.sessions[sid]
    assert ws.nav.si == 1 and ws.seg == 1  # 휴대폰 구간 번호와 맞추려고 seg는 올림
    r = c.post(f"/api/session/{sid}/frame", headers=H, files={"image": ("f.jpg", jpg(), "image/jpeg")},
               data={"t": "9.0", "heading": "90", "seg": "1"}).json()
    assert r["say"] == ["410호는 오른쪽입니다. 오른쪽으로 가세요."] and r["direction"]["strength"] == "강"
    assert ws.nav.records[-1]["heading"] == 90.0


def test_session_log_has_everything_for_analysis(tmp_path):
    """2026-11-23 시험 전 기록 보강: 사진마다 시각·사진 정보·센서·Claude 원본 출력·판단 사건, 휴대폰 기록, 버튼 응답 문장."""
    c, _ = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    import re as _re
    assert _re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", sid)  # 세션 이름 = 시작 시각
    cl = {"size": [720, 1280], "bytes": 1234, "sharp": 400, "early": True, "kind": "frame", "retries": 0, "beta": 80.5,
          "gamma": -2.0, "shot_epoch": 1.0, "sent_epoch": 1.2, "inflight": 1}
    r = c.post(f"/api/session/{sid}/frame", headers=H, files={"image": ("f.jpg", jpg(), "image/jpeg")},
               data={"t": "0.0", "heading": "123.4", "steps": "7", "client": json.dumps(cl), "seg": "0"}).json()
    assert r["status"] == "ok"
    c.post(f"/api/session/{sid}/clientlog", headers=H, json={"events": [
        {"k": "got", "t": 1.5, "epoch": 2.0, "seq": 1, "status": "ok"},
        {"k": "say_start", "t": 1.6, "text": "안내", "src": "server", "seq": 1}, {"k": "say_end", "t": 3.0, "text": "안내"}]})
    e = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "repeat"}).json()
    d = tmp_path / sid
    assert (d / "frames" / "00001.jpg").exists()
    lines = [json.loads(x) for x in (d / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    rec = next(x for x in lines if x.get("web_seq") == 1 and "event" not in x)
    tm = rec["timing"]
    for k in ("arrive_s", "recv_s", "recv_epoch", "see_start_s", "see_end_s", "ready_s", "apply_start_s", "applied_s"):
        assert tm[k] is not None, k
    assert tm["arrive_s"] <= tm["recv_s"] <= tm["see_start_s"] <= tm["see_end_s"] <= tm["applied_s"]
    assert rec["img_size"] and rec["img_bytes"] > 0 and rec["inflight_server"] == 1
    assert rec["client"]["beta"] == 80.5 and rec["client"]["sharp"] == 400 and rec["client"]["early"] is True
    assert rec["heading"] == 123.4 and rec["steps"] == 7
    see = rec["calls_log"][0]
    assert see["kind"] == "see" and see["raw_text"] and "attempts" in see and "timed_out" in see
    assert "events" in rec and rec["obs"]["place"] and "floor_src" in rec["state"] and rec["say"] is not None
    kinds = [x.get("kind") for x in lines if x.get("event") == "client"]
    assert kinds == ["got", "say_start", "say_end"]
    ev = next(x for x in lines if x.get("event") == "repeat")
    assert "epoch" in ev and "say" in ev
    assert any(x.get("event") == "start" for x in lines)
