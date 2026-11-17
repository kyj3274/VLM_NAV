"""시간 순서 기억(nav_memory의 사건 기록)을 그대로 넣어 텍스트 모델이 다음 행동을 판단하게 한다 (Mac에서 실행, 모델은 서버 Ollama).
입력: data/video/results/memory_snapshots.json (서버 memory_snapshots.py가 만든, 걸음마다의 기억), data/video/checkpoints.json (판단 지점 17개)
  series      사건 기록에서 사실만 (문패 확정, 장소, 엘리베이터 하차) + 지금 장면
  series+say  위 + 지금까지 시스템이 한 안내 문장 (규칙의 판단이 섞여 있음)

  .venv/bin/python src/kyj/exp/4_reason_2611/reason_series.py --url http://127.0.0.1:11437 --models qwen3:14b --think off
"""

import argparse
import json
import statistics
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4]
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착"]
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string", "maxLength": 80},
    "on_track": {"type": "string", "enum": ["예", "아니요", "모름"]},
    "next_action": {"type": "string", "enum": ACTIONS}}, "required": ["reason", "on_track", "next_action"]}
PROMPT = ("너는 시각장애인 실내 길안내 보조다. 사용자는 처음 와 보는 건물에서 {target}호를 찾아 걷고 있다. "
          "아래는 시스템이 시간 순서로 기억한 기록이다. 사진은 없다.\n"
          "일반 상식: 호실 번호의 첫 자리는 보통 층이다. 복도의 호실 번호는 순서대로 이어지므로, 걸으며 차례로 본 번호가 "
          "목적지 번호에 가까워지면 맞는 방향이고 멀어지면 반대 방향이다. 가끔 맞은편 벽의 번호나 잘못 읽은 번호가 섞인다. "
          "다른 층이면 엘리베이터로 가야 한다.\n"
          "- reason: 판단 근거 한 줄.\n- on_track: 지금까지의 이동이 목적지로 가는 데 맞는가.\n- next_action: 사용자가 지금 할 행동 하나.\n\n{memory}")
THINK = None


def memory_text(snap, kind):
    lines = [f"목적지: {snap['target']}호", "시간 순서 기록:"]
    for e in snap["events"]:
        if e["text"].startswith("안내:") and kind == "series":
            continue
        lines.append(f"- {e['t']}초: {e['text']}")
    now = f"지금 ({snap['t']}초): 보이는 문패 {', '.join(x + '호' for x in snap['rooms']) or '없음'}, 장소 인식 {snap['place'] or '확실하지 않음'}"
    floor = (f"{snap['floor']}층 (문패로 확인)" if snap["floor"] else "1층으로 가정 (확인 안 됨)" if snap["assume1"] else "모름")
    lines += [now, f"지금 층: {floor}"]
    return "\n".join(lines)


def ask(url, model, prompt):
    payload = {"model": model, "stream": False, "keep_alive": "30m", "format": SCHEMA,
               "options": {"temperature": 0, "seed": 42, "num_predict": 2000 if THINK else 160},
               "messages": [{"role": "user", "content": prompt}]}
    if THINK is not None:
        payload["think"] = THINK
    body = requests.post(f"{url}/api/chat", json=payload, timeout=900).json()
    try:
        ans = json.loads(body["message"]["content"])
    except Exception:  # noqa: BLE001
        ans = None
    return ans, (body.get("total_duration", 0) - body.get("load_duration", 0)) / 1e9


def main():
    global THINK
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3:14b")
    ap.add_argument("--kinds", default="series,series+say")
    ap.add_argument("--url", default="http://127.0.0.1:11437")
    ap.add_argument("--think", choices=["on", "off"])
    ap.add_argument("--out", default=str(ROOT / "data/video/results/reason_series.jsonl"))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    THINK = {"on": True, "off": False, None: None}[a.think]
    tag = f"+think_{a.think}" if a.think else ""
    cps = json.load(open(ROOT / "data/video/checkpoints.json", encoding="utf-8"))["checkpoints"]
    snaps = {(s["route"], s["seg"], s["sec"]): s for s in json.load(open(ROOT / "data/video/results/memory_snapshots.json", encoding="utf-8"))}
    if a.show:
        for cp in cps:
            print(f"===== {cp['id']} 정답 {cp['action']}\n{memory_text(snaps[(cp['route'], cp['seg'], cp['sec'])], 'series+say')}\n")
        return
    out, summary = open(a.out, "a", encoding="utf-8"), []
    for model in a.models.split(","):
        ask(a.url, model, "ok")
        for kind in a.kinds.split(","):
            ok, secs = 0, []
            for cp in cps:
                snap = snaps[(cp["route"], cp["seg"], cp["sec"])]
                ans, sec = ask(a.url, model, PROMPT.format(target=snap["target"], memory=memory_text(snap, kind)))
                hit = bool(ans) and ans["next_action"] in cp["action"]
                ok += hit
                secs.append(sec)
                out.write(json.dumps({"model": model + tag, "kind": kind, "id": cp["id"], "gold": cp["action"], "answer": ans,
                                      "hit": hit, "seconds": round(sec, 2)}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{model + tag:22s} {kind:10s} {cp['id']:>3} {'O' if hit else 'X'} {sec:5.1f}s 정답 {cp['action']} → "
                      f"{json.dumps(ans, ensure_ascii=False)}", flush=True)
            summary.append((model + tag, kind, ok, len(cps), statistics.median(secs), max(secs)))
    print("\n| 모델 | 입력 | 다음 행동 맞음 | 시간 중앙값 | 최대 |\n|---|---|---|---|---|")
    for m, k, o, n, med, mx in summary:
        print(f"| {m} | {k} | {o}/{n} | {med:.1f}초 | {mx:.1f}초 |")


if __name__ == "__main__":
    main()
