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


def test_busy_when_three_in_flight(tmp_path):
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
    assert st.count("busy") == 2 and st.count("ok") + st.count("stale") == 3


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
    assert c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()["segment"] == 1
    assert frame(c, sid, 0.0)["say"][0].startswith("엘리베이터에서 내렸습니다")


def test_exited_and_destination_change(tmp_path):
    c, _ = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410"}).json()["session_id"]
    frame(c, sid, 0.0)
    r = c.post(f"/api/session/{sid}/event", headers=H, json={"type": "exited"}).json()
    assert r["segment"] == 1 and "내렸습니다" in r["say"][0]
    r = c.post(f"/api/session/{sid}/destination", headers=H, json={"destination": "삼백십칠호"}).json()
    assert r["status"] == "ok" and r["target"] == "317" and r["say"] == ["317호로 안내하겠습니다."]
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


def test_confirm_endpoint_quick_arrival(tmp_path):
    """가짜 장면: 목적지 2m 안 첫 응답에 confirm → /confirm 짧은 확인으로 바로 도착 (빠른 확인 켬)."""
    c, calls = app_with(tmp_path, quick_confirm=True)
    sid = c.post("/api/session", headers=H, json={"destination": "410호"}).json()["session_id"]
    r = None
    for k in range(25):
        r = frame(c, sid, 1.5 * k)
        if r["confirm"] or r["arrived"]:
            break
    assert r["confirm"] and not r["arrived"] and r["say"][-1].endswith("멈춰 주세요, 확인 중입니다.")
    q = c.post(f"/api/session/{sid}/confirm", headers=H, files={"image": ("q.jpg", jpg(), "image/jpeg")},
               data={"t": str(1.5 * k + 0.8)}).json()
    assert q["status"] == "ok" and q["quick"] and q["arrived"]
    log = [json.loads(x) for x in (tmp_path / sid / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["quick"] and log[-1]["frame"].endswith("_q.jpg")


def test_quick_confirm_off_by_default(tmp_path):
    c, calls = app_with(tmp_path)
    sid = c.post("/api/session", headers=H, json={"destination": "410호"}).json()["session_id"]
    rs = [frame(c, sid, 1.5 * k) for k in range(25)]
    assert not any(r.get("confirm") for r in rs) and any(r["arrived"] for r in rs)


# ---------------------------------------------------------------- 2026-11-23
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
