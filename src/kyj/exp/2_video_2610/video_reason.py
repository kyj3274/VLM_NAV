"""판단 지점에서 VLM의 맥락 추론을 따로 평가한다 (형식은 객관식으로 고정, 추론만 VLM이).

입력
  data/video/checkpoints.json  판단 지점 17곳의 정답 (모델 결과를 보기 전에 작성)
  guide_314_1_2b.jsonl, guide_314_1_2a.jsonl  video_tracker.py 결과 (초마다 확정 호실, 장면 VLM 답)
조건
  rule      규칙 (video_tracker의 판단 방식): 호실 첫 자리 = 층, 가까운 문패 번호 차이로 방향
  img       VLM: 사진 + 관찰 기록, 추론 규칙 힌트 없음      ← VLM의 추론 능력
  text      VLM: 관찰 기록만 (사진 없음)                    ← 사진이 도움이 되는가
  img_hint  VLM: img + 일반 상식 힌트                         ← 힌트(프롬프트) 효과
관찰 기록은 사실만 담는다: 사건(경로 시작, 엘리베이터 하차), 본 호실 번호와 시각, 지금 장면의 OCR·YOLOE 결과.

사용 (서버):
  ./gpu_lock.sh 3 ./gpu_lock.sh 7 env CUDA_VISIBLE_DEVICES=3 .venv/bin/python video_reason.py \
     --frames /workspace/capstone/outputs/kyj/video/frames_8fps --dir /workspace/capstone/outputs/kyj/video
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

from PIL import Image

if "--from-inputs" in sys.argv:  # Mac에서 API 모델만 부를 때: OCR·YOLOE 없이 (입력은 서버에서 --dump-inputs로 만듦)
    for _m in ("exp_a_ocr", "pipeline", "cv2"):
        sys.modules.setdefault(_m, types.ModuleType(_m))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nav"))  # 서비스 모듈 (src/kyj/nav)

import ocr_reader as exp_a_ocr
import perception as pipeline
import vlm_client as vc
import navigator as vt

FLOORS = ["1층", "2층", "3층", "4층", "5층", "모름"]
YN = ["예", "아니요", "모름"]
PROGRESS = ["가까워짐", "멀어짐", "모름"]
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착"]
SIDES = ["왼쪽", "오른쪽", "정면", "모름"]
GUIDE_FILE = {"2b_after_elevator_졸프_영상2": "guide_314_1_2b.jsonl", "2a_after_elevator_졸프_영상1": "guide_314_1_2a.jsonl"}

GUIDE = (
    "너는 시각장애인 실내 길안내 보조다. 사용자의 목적지는 {target}호다.\n"
    "아래는 지금까지 시스템이 관찰한 사실 기록과 지금 장면의 인식 결과다. 이 사실{img}을 근거로 현재 상황을 판단하고, "
    "사용자가 지금 할 다음 행동을 골라라. 모든 칸은 주어진 선택지 중에서만 고른다.\n"
    "- current_floor: 사용자가 지금 있는 층. 근거가 없으면 모름.\n"
    "- on_goal_floor: 지금 층이 목적지가 있는 층인가.\n"
    "- progress: 지금까지의 이동이 목적지에 가까워지는 중인가, 멀어지는 중인가.\n"
    "- next_action: 사용자가 지금 할 행동.\n"
    "- target_side: 목적지 문이 지금 보이면 그 방향, 안 보이면 모름.\n"
    "- evidence: 판단의 근거가 된 기록 번호 (지금 장면은 0).\n"
    "- confidence: 판단의 확신 정도."
)
HINT = (
    "\n\n일반 상식: 건물 호실 번호의 첫 자리는 보통 층을 뜻한다. 복도의 호실 번호는 대체로 순서대로 이어지므로, "
    "지나온 문패 번호가 목적지 번호에 가까워지면 맞는 방향이고 멀어지면 반대 방향이다. "
    "엘리베이터 문 위의 숫자는 엘리베이터가 지금 있는 층이다."
)


def schema(n):
    return {"type": "object", "properties": {
        "current_floor": {"type": "string", "enum": FLOORS},
        "on_goal_floor": {"type": "string", "enum": YN},
        "progress": {"type": "string", "enum": PROGRESS},
        "next_action": {"type": "string", "enum": ACTIONS},
        "target_side": {"type": "string", "enum": SIDES},
        "evidence": {"type": "array", "items": {"type": "integer", "enum": list(range(n + 1))}, "maxItems": 4},
        "confidence": {"type": "string", "enum": ["높음", "중간", "낮음"]},
    }, "required": ["current_floor", "on_goal_floor", "progress", "next_action", "target_side", "evidence", "confidence"]}


def load_guides(d):
    out = {}
    for seg, name in GUIDE_FILE.items():
        recs = [json.loads(l) for l in open(Path(d) / name, encoding="utf-8")]
        out[seg] = [r for r in recs if r["seq"] == seg]
        out["1_start_졸프_영상3"] = [r for r in recs if r["seq"] == "1_start_졸프_영상3"]
    return out


def observation_log(route, cp, guides):
    """사실 기록 (지금 시점 이전까지). 반환: [(번호, 문장)]"""
    log = [f"경로 시작 (목적지 {route['target']}호 안내 시작)"]
    if cp["seg"] >= 1:
        log.append("엘리베이터를 타고 이동한 뒤 내림 (누른 층은 확인되지 않음)")
        seen = []
        for r in guides[route["segments"][1]]:
            if r["sec"] >= cp["sec"]:
                break
            new = [x for x in r["rooms"] if x not in seen]
            if new:
                seen += new
                log.append(f"하차 후 {r['sec']}초: {', '.join(x + '호' for x in new)} 문패를 봄 (가까운 순)")
    return list(enumerate(log, 1))


def rule_answer(route, cp, guides):
    """video_tracker와 같은 판단 규칙."""
    tf = route["target"][0]
    if cp["seg"] == 0:
        scene = None
        for r in guides["1_start_졸프_영상3"]:
            if r["sec"] > cp["sec"]:
                break
            for v in r["vlm"]:
                a = v.get("answer") or {}
                if a.get("scene_type"):
                    scene = a
        act = "직진"
        if scene and (scene["scene_type"] == "ENTRANCE" or scene["door_ahead"] == "GLASS_DOOR") and cp["sec"] < 3:
            act = "문 통과"
        elif scene and scene["scene_type"] == "ELEVATOR_HALL":
            act = "엘리베이터 타기"
        return {"current_floor": "모름", "on_goal_floor": "모름", "progress": "모름", "next_action": act, "target_side": "모름"}
    recs = [r for r in guides[route["segments"][1]] if r["sec"] <= cp["sec"]]
    rooms_now = recs[-1]["rooms"] if recs else []
    near, floor = [], None
    for r in recs:
        if r["rooms"]:
            floor = floor or r["rooms"][0][0]
            if r["rooms"][0][0] == floor and (not near or near[-1] != r["rooms"][0]):
                near.append(r["rooms"][0])
    if floor is None:
        return {"current_floor": "모름", "on_goal_floor": "모름", "progress": "모름",
                "next_action": "멈추고 둘러보기", "target_side": "모름"}
    out = {"current_floor": f"{floor}층", "on_goal_floor": "예" if floor == tf else "아니요", "target_side": "모름"}
    if floor != tf:
        return {**out, "progress": "모름", "next_action": "엘리베이터로 돌아가기"}
    ds = [abs(int(x) - int(route["target"])) for x in near if "-" not in x]
    prog = "모름" if len(ds) < 2 else ("가까워짐" if ds[-1] < ds[-2] else ("멀어짐" if ds[-1] > ds[-2] else "모름"))
    act = "직진"
    if route["target"] in rooms_now:
        act = "목적지 접근"
        tgt = recs[-1].get("target_side")
        out["target_side"] = tgt or "모름"
        if recs[-1]["state"].get("arrived"):
            act = "도착"
    elif len(ds) >= 3 and ds[-1] > ds[-2] > ds[-3]:
        act = "뒤로 돌기"
    return {**out, "progress": prog, "next_action": act}


def best_frame(runner, frames_dir, seg, sec):
    files = sorted((Path(frames_dir) / seg).glob("i*.jpg"))[sec * 8:(sec + 1) * 8]
    scored = []
    for p in files:
        r, img = runner.s1(p)
        scored.append((len(r["result"]["rooms"]), vt.blur(img), r, img))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2], scored[0][3]


class Runner:
    def __init__(self):
        cwd = os.getcwd()
        self.yoloe = pipeline.load_yoloe()
        os.chdir(cwd)
        self.ocr = exp_a_ocr.load_ocr("m", {})
        self.cache = {}

    def s1(self, p):
        if p not in self.cache:
            img = Image.open(p).convert("RGB")
            self.cache[p] = (vc.stage1(self.yoloe, self.ocr, img), img)
        return self.cache[p]


THINK = ("\n\n답하기 전에 thinking 칸에, 기록과 지금 장면에서 알 수 있는 것과 목적지까지의 관계를 "
         "단계별로 짧게 적은 뒤 나머지 칸을 채워라.")
SUB_TARGET = "건물 안 목적지는 {target}호다. 이 호실은 몇 층에 있을 가능성이 가장 높은가?"
SUB_FLOOR = "다음은 사용자가 건물 안을 이동하며 관찰한 기록과 지금 장면 인식 결과다. 사용자는 지금 몇 층에 있는가? 근거가 없으면 모름.\n\n{facts}\n\n{now}"
SUB_PROG = ("목적지는 {target}호다. 다음은 같은 층 복도를 걸으며 본 문패 기록이다. 지금까지의 이동은 목적지에 "
            "가까워지는 중인가, 멀어지는 중인가? 판단할 근거가 부족하면 모름.\n\n{facts}\n\n{now}")
SUB_SCENE = "이 사진은 실내를 걷는 사용자 시점이다. 지금 장면은 어떤 곳인가?"


def shuffled(sch, key):
    """선택지 순서를 지점마다 고정된 방식으로 섞는다 (순서 편향 시험)."""
    import random
    rnd = random.Random(key)
    sch = json.loads(json.dumps(sch))
    for name, prop in sch["properties"].items():
        if "enum" in prop and name != "evidence":
            rnd.shuffle(prop["enum"])
    return sch


def with_think(sch):
    sch = json.loads(json.dumps(sch))
    sch["properties"] = {"thinking": {"type": "string", "maxLength": 300}, **sch["properties"]}
    sch["required"] = ["thinking"] + sch["required"]
    return sch


def one(prop):
    return {"type": "object", "properties": {"a": prop}, "required": ["a"]}


def decomposed(args, route, cp, facts, now, confirmed, img, model):
    """질문을 쪼개 VLM에 하나씩 묻고, 행동은 그 답으로 정한다."""
    t = route["target"]
    calls = []
    def q(prompt, prop, image=None):
        r = vc.ask(args.url, model, prompt, one(prop), image, 24 * args.np_scale)
        calls.append(r)
        return (r.get("answer") or {}).get("a")
    tf = q(SUB_TARGET.format(target=t), {"type": "string", "enum": FLOORS})
    cur = q(SUB_FLOOR.format(facts=facts, now=now), {"type": "string", "enum": FLOORS})
    prog = q(SUB_PROG.format(target=t, facts=facts, now=now), {"type": "string", "enum": PROGRESS}) if cur not in (None, "모름") else "모름"
    scene = q(SUB_SCENE, {"type": "string", "enum": vc.SCENES}, img)
    if cur in (None, "모름"):
        act = {"ENTRANCE": "문 통과", "ELEVATOR_HALL": "엘리베이터 타기", "INSIDE_ELEVATOR": "엘리베이터 타기",
               "CORRIDOR": "직진", "LOBBY": "직진"}.get(scene, "멈추고 둘러보기")
        on = "모름"
    elif cur != tf:
        act, on = "엘리베이터로 돌아가기", "아니요"
    else:
        on = "예"
        act = "목적지 접근" if t in confirmed else ("뒤로 돌기" if prog == "멀어짐" else "직진")
    ans = {"target_floor": tf, "current_floor": cur, "on_goal_floor": on, "progress": prog, "scene": scene,
           "next_action": act, "target_side": "모름"}
    return {"answer": ans, "vlm_seconds": sum(c["vlm_seconds"] for c in calls), "calls": len(calls),
            "error": next((c["error"] for c in calls if c["error"]), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--dir", required=True, help="guide_*.jsonl이 있는 폴더, 결과도 여기에")
    ap.add_argument("--checkpoints", default=str(Path(__file__).resolve().parent / "checkpoints.json"))
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--conds", default="rule,img,text,img_hint")
    ap.add_argument("--big", default="qwen2.5vl:7b,gemma3:12b", help="d 시험용 큰 모델")
    ap.add_argument("--out", default="reason.jsonl")
    ap.add_argument("--think", default="default", choices=["default", "on", "off"], help="생각 모드 (qwen3.5 등)")
    ap.add_argument("--np-scale", type=int, default=1, help="생각 모드면 출력 길이 배수")
    ap.add_argument("--dump-inputs", help="판단 지점 입력(기록·장면·대표 사진)만 저장하고 끝 (서버)")
    ap.add_argument("--from-inputs", help="--dump-inputs로 만든 입력으로 VLM만 부름 (Mac, API 모델)")
    args = ap.parse_args()
    vc.THINK = {"default": None, "on": True, "off": False}[args.think]
    tag = "" if args.model == "qwen3-vl:4b-instruct" and args.think == "default" else f"@{args.model}" + (f"+think_{args.think}" if args.think != "default" else "")
    conds = args.conds.split(",")
    spec = json.load(open(args.checkpoints, encoding="utf-8"))
    guides = load_guides(args.dir)
    if args.from_inputs:
        src = Path(args.from_inputs)
        cps = {c["id"]: c for c in spec["checkpoints"]}
        inputs = [(cps[d["id"]], spec["routes"][cps[d["id"]]["route"]], Image.open(src.parent / d["image"]).convert("RGB"),
                   [tuple(x) for x in d["log"]], d["facts"], d["now"], d["confirmed"])
                  for d in json.load(open(src, encoding="utf-8"))]
    else:
        inputs = None
        runner = Runner()
    if not args.dump_inputs:
        vc.ask(args.url, args.model, "ok", None, Image.new("RGB", (64, 64)), 4)

    # 큰 모델은 바꿔 올리는 시간을 줄이려고 모델별로 묶어서 돈다
    plan = [(c, args.model) for c in conds if not c.startswith("d_")]
    for m in args.big.split(","):
        plan += [(c, m) for c in conds if c.startswith("d_")]

    for cp in (spec["checkpoints"] if inputs is None else []):
        inputs = inputs or []
        route = spec["routes"][cp["route"]]
        seg = route["segments"][cp["seg"]]
        s1, img = best_frame(runner, args.frames, seg, cp["sec"])
        log = observation_log(route, cp, guides)
        guide_rec = next((r for r in guides.get(seg, []) if r["sec"] == cp["sec"]), None)
        confirmed = guide_rec["rooms"] if guide_rec else s1["result"]["rooms"]
        facts = "\n".join(f"#{i} {t}" for i, t in log)
        now = f"#0 지금 장면: 확정된 호실 {', '.join(confirmed) or '없음'}\n" + vc.hints(s1)
        inputs.append((cp, route, img, log, facts, now, confirmed))
    if args.dump_inputs:
        dump = Path(args.dump_inputs)
        (dump.parent / "reason_inputs").mkdir(parents=True, exist_ok=True)
        rows = []
        for cp, route, img, log, facts, now, confirmed in inputs:
            name = f"reason_inputs/{cp['id']}.jpg"
            img.save(dump.parent / name, quality=92)
            rows.append({"id": cp["id"], "image": name, "log": log, "facts": facts, "now": now, "confirmed": confirmed})
        dump.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(len(rows), "inputs →", dump)
        return

    out = open(Path(args.dir) / args.out, "w", encoding="utf-8")
    for cond, model in plan:
        for cp, route, img, log, facts, now, confirmed in inputs:
            base = {"id": cp["id"], "route": cp["route"], "sec": cp["sec"], "log": log, "now": now, "model": model}
            if cond == "rule":
                row = {**base, "cond": cond, "answer": rule_answer(route, cp, guides), "vlm_seconds": 0}
            elif cond == "c_decomp":
                row = {**base, "cond": cond + tag, **decomposed(args, route, cp, facts, now, confirmed, img, model)}
            else:
                use_img = cond in ("img", "img_hint", "a_img", "b_img")
                prompt = GUIDE.format(target=route["target"], img="과 사진" if use_img else "")
                if cond == "img_hint":
                    prompt += HINT
                sch = schema(len(log))
                if cond.startswith("a_"):
                    sch = shuffled(sch, cp["id"])
                think = cond in ("b_img", "b_text", "d_text_think")
                if think:
                    sch = with_think(sch)
                    prompt += THINK
                prompt += f"\n\n관찰 기록:\n{facts}\n\n{now}"
                r = vc.ask(args.url, model, prompt, sch, img if use_img else None, (480 if think else 160) * args.np_scale)
                row = {**base, "cond": (cond if not cond.startswith("d_") else f"{cond}:{model}") + tag, **r}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{cp['id']:>4} {row['cond']:>22} {row.get('vlm_seconds', 0):.1f}s {json.dumps(row.get('answer'), ensure_ascii=False)[:220]}", flush=True)
    out.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
