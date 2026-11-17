"""3초마다 텍스트 모델이 '지금 맞게 가고 있나 + 다음 행동'을 판단할 수 있는지 잰다 (Mac에서 실행, 모델은 서버 Ollama).

판단 지점 17개(data/video/checkpoints.json)마다 그 시점까지의 기억을 글로 만들어 모델에 넣는다. 사진은 넣지 않는다.
  raw   : 사건 + 본 문패(시각·방향) + 지금 장면(VLM 장면, Places365, 지금 문패)
  facts : raw + 기억 층이 미리 계산한 사실(지금 층, 목적지 층인가, 문패 번호와 목적지 번호 차이의 흐름)
입력은 서버 tmp/locate 시험 결과(guide_314_*_situ_sharp.jsonl, 문패 방향·Places 포함)를 Mac에 복사한 것.

  ssh -S /tmp/kyj-cnu.sock -O forward -L 11436:127.0.0.1:11436 cnu   # 서버 Ollama를 Mac으로
  .venv/bin/python src/kyj/exp/4_reason_2611/reason_memory.py --models gemma3:12b,qwen3-vl:4b-instruct --out data/video/results/reason_memory.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4]
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착"]
SCENE_KO = {"CORRIDOR": "복도", "LOBBY": "로비", "ELEVATOR_HALL": "엘리베이터 앞", "ENTRANCE": "건물 출입문",
            "INTERSECTION": "복도 갈림길", "ROOM_DOOR": "방 문 앞", "INSIDE_ELEVATOR": "엘리베이터 안", "UNKNOWN": "모름"}
SCHEMA = {"type": "object", "properties": {
    "on_track": {"type": "string", "enum": ["예", "아니요", "모름"]},
    "next_action": {"type": "string", "enum": ACTIONS},
    "say": {"type": "string", "maxLength": 40}}, "required": ["on_track", "next_action", "say"]}
PROMPT = ("너는 시각장애인 실내 길안내 보조다. 사용자는 처음 와 보는 건물에서 {target}호를 찾아가고 있다. "
          "아래는 지금까지 시스템이 기억한 사실이다. 사진은 없다. 사실만 근거로 판단하라.\n"
          "- on_track: 지금까지의 이동이 목적지로 가는 데 맞는가.\n"
          "- next_action: 사용자가 지금 할 행동 하나.\n"
          "- say: 사용자에게 들려줄 짧은 안내 한 문장 (40자 이내).\n\n{memory}")


def load(name):
    return [json.loads(line) for line in open(ROOT / "data/video/results/locate" / name, encoding="utf-8")]


ROUTE_FILE = {"A": "guide_314_1_2b_situ_sharp.jsonl", "C": "guide_314_1_2b_situ_sharp.jsonl",
              "B": ("guide_314_1_2b_situ_sharp.jsonl", "guide_314_2a_situ_sharp.jsonl")}


def records(route, seg):
    """구간별 초 기록. B의 하차 구간은 2a만 따로 돌린 결과를 쓴다."""
    if route == "B" and seg == 1:
        return [r for r in load(ROUTE_FILE["B"][1]) if r["seq"].startswith("2a")]
    f = ROUTE_FILE[route] if route != "B" else ROUTE_FILE["B"][0]
    return [r for r in load(f) if r["seq"].startswith("1_s" if seg == 0 else "2b")]


def memory(cp, target, kind):
    lines = [f"목적지: {target}호"]
    seg0 = records(cp["route"], 0)
    last = seg0 if cp["seg"] == 0 else records(cp["route"], 1)
    upto = [r for r in last if r["sec"] <= cp["sec"]]
    ev = ["0초: 경로 시작"]
    for r in (seg0 if cp["seg"] == 1 else []) + (upto if cp["seg"] == 0 else []):
        for v in r["vlm"]:
            s = (v.get("answer") or {}).get("scene_type")
            if s:
                ev.append(f"출발 구간 {r['sec']}초: 장면 판단 '{SCENE_KO[s]}'")
    if cp["seg"] == 1:
        ev.append("엘리베이터를 타고 이동한 뒤 내림 (누른 층은 확인되지 않음)")
    seen, seq = [], []
    for r in (upto if cp["seg"] == 1 else []):
        for v in r["vlm"]:
            s = (v.get("answer") or {}).get("scene_type")
            if s:
                ev.append(f"하차 후 {r['sec']}초: 장면 판단 '{SCENE_KO[s]}'")
        for x in r["rooms"]:
            if x not in seen:
                seen.append(x)
                d = r.get("seen", {}).get("plates", {}).get(x, "")
                ev.append(f"하차 후 {r['sec']}초: {x}호 문패를 봄" + (f" ({d} 방향)" if d else ""))
                if "-" not in x:
                    seq.append(x)
    lines.append("지금까지 일어난 일:\n" + "\n".join(f"- {e}" for e in ev))
    now = upto[-1] if upto else None
    if now:
        pg = now["raw"].get("place_groups", {})
        best = max(pg, key=pg.get) if pg else None
        place = f"{best} {pg[best]:.2f}" if best and pg[best] >= 0.3 else "확실하지 않음"
        lines.append(f"지금 ({'하차 후 ' if cp['seg'] else '출발 구간 '}{now['sec']}초): 보이는 문패 "
                     f"{', '.join(x + '호' for x in now['rooms']) or '없음'}, 장소 인식 {place}")
    if kind == "facts":
        f = []
        floors = [x[0] for x in seq]
        if floors:
            fl = max(set(floors), key=floors.count)
            f.append(f"지금 층: {fl}층 (문패 번호 첫 자리로 판단). 목적지 {target}호는 {target[0]}층 → "
                     + ("목적지 층이 맞음" if fl == target[0] else "다른 층임"))
            same = [x for x in seq if x[0] == fl]
            if fl == target[0] and same:
                f.append("본 문패 순서와 목적지 번호와의 차이: "
                         + " → ".join(f"{x}호(차이 {abs(int(x) - int(target))})" for x in same)
                         + ". 복도 양쪽 벽 문패가 섞여 있을 수 있음")
        else:
            f.append("지금 층: 아직 모름 (본 문패 없음)")
        if now and target in now["rooms"]:
            f.append(f"목적지 {target}호 문패가 지금 보임")
        lines.append("기억 층이 계산한 사실:\n" + "\n".join(f"- {x}" for x in f))
    return "\n\n".join(lines)


THINK = None  # None = 모델 기본, True/False = 생각 모드 켬/끔 (qwen3 등)


def ask(url, model, prompt):
    payload = {"model": model, "stream": False, "keep_alive": "30m", "format": SCHEMA,
               "options": {"temperature": 0, "seed": 42, "num_predict": 1500 if THINK else 120},
               "messages": [{"role": "user", "content": prompt}]}
    if THINK is not None:
        payload["think"] = THINK
    body = requests.post(f"{url}/api/chat", json=payload, timeout=600).json()
    try:
        ans = json.loads(body["message"]["content"])
    except Exception:  # noqa: BLE001
        ans = None
    return ans, body.get("total_duration", 0) / 1e9, body.get("load_duration", 0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-vl:4b-instruct,gemma3:12b")
    ap.add_argument("--kinds", default="raw,facts")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--out", default=str(ROOT / "data/video/results/reason_memory.jsonl"))
    ap.add_argument("--show", action="store_true", help="모델을 부르지 않고 입력 글만 보여 줌")
    ap.add_argument("--think", choices=["on", "off"], help="생각 모드 (qwen3 등). 안 주면 모델 기본")
    a = ap.parse_args()
    global THINK
    THINK = {"on": True, "off": False, None: None}[a.think]
    tag = f"+think_{a.think}" if a.think else ""
    spec = json.load(open(ROOT / "data/video/checkpoints.json", encoding="utf-8"))
    cps = spec["checkpoints"]
    if a.show:
        for cp in cps:
            print(f"===== {cp['id']} 정답 {cp['action']}\n{memory(cp, spec['routes'][cp['route']]['target'], 'facts')}\n")
        return
    out = open(a.out, "a", encoding="utf-8")
    summary = []
    for model in a.models.split(","):
        ask(a.url, model, "ok")  # 모델 올리기 (시간에서 뺌)
        for kind in a.kinds.split(","):
            ok, secs = 0, []
            for cp in cps:
                target = spec["routes"][cp["route"]]["target"]
                ans, sec, load_s = ask(a.url, model, PROMPT.format(target=target, memory=memory(cp, target, kind)))
                hit = bool(ans) and ans["next_action"] in cp["action"]
                ok += hit
                secs.append(sec - load_s)
                out.write(json.dumps({"model": model + tag, "kind": kind, "id": cp["id"], "gold": cp["action"], "answer": ans,
                                      "hit": hit, "seconds": round(sec - load_s, 2)}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{model:22s} {kind:5s} {cp['id']:>3} {'O' if hit else 'X'} {sec - load_s:4.1f}s 정답 {cp['action']} "
                      f"→ {json.dumps(ans, ensure_ascii=False)}", flush=True)
            summary.append((model + tag, kind, ok, len(cps), statistics.median(secs), max(secs)))
    print("\n| 모델 | 입력 | 다음 행동 맞음 | 시간 중앙값 | 최대 |\n|---|---|---|---|---|")
    for m, k, o, n, med, mx in summary:
        print(f"| {m} | {k} | {o}/{n} | {med:.1f}초 | {mx:.1f}초 |")


if __name__ == "__main__":
    main()
