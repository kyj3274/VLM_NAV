"""영상 길안내 시연: 규칙 추적기 + 애매하면 프레임 더 보기 + VLM은 최후 수단.

목적지(예: 314호)를 받아, 1초마다 사용자에게 할 안내 문장을 만든다.
  1층  규칙 추적기 (VLM 0회): OCR 호실 확정 → 위치 갱신, 층 = 호실 첫 자리, 근거가 없으면 유지
  2층  애매하면 그 1초 구간의 8장을 더 보고 OCR 투표 (1단계만, 장당 약 0.15초)
         애매 = 흐림 / 깨진 번호 / 문패 한글은 있는데 번호 없음 / OCR 점수 중간인 호실 후보
  3층  VLM (최후 수단)
         맥락 사건: 경로 시작, 엘리베이터에서 내린 직후, 큰 엘리베이터 문, 근거 없이 10초, 층 충돌
         남은 애매함: 여러 장을 봐도 가까운 문패 번호가 안 읽힐 때 문패 조각만 물음
안내 문장은 상태와 VLM 답(빈칸 채우기)으로 틀에 맞춰 만든다. VLM이 문장을 직접 쓰지 않는다.

거리: 문 높이 2.1m 가정, 초점거리 = 사진 긴 변 × 0.72 (휴대폰 기본 카메라 약 70° 화각 가정) → 거리 = 0.72 × 2.1 / 문 상자 높이 비율.
      문 상자가 화면 위·아래에 닿으면 문 전체가 안 보일 만큼 가깝다고 본다. 대략값이다.

프레임: frames_8fps/<순서>_<영상>/i0000.jpg (1초에 8장, i = 초×8 + k). 1초의 대표 = k=0.

사용 (서버):
  ../tools/gpu_lock.sh 3 ../tools/gpu_lock.sh 7 env CUDA_VISIBLE_DEVICES=3 ../.venv/bin/python navigator.py \
     --frames /workspace/capstone/outputs/kyj/video/frames_8fps --out-dir /workspace/capstone/outputs/kyj/video \
     --route "314=1_start_졸프_영상3,2b_after_elevator_졸프_영상2" --route "314=1_start_졸프_영상3,2a_after_elevator_졸프_영상1"
"""

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import nav_memory
import ocr_reader
import perception
import room_rules
import scene_places
import vlm_client as vc

DENSE = 8
NEAR_H = 0.012  # 방향 판단에 쓰는 '가까운 문패'의 글자 높이 (사진 높이 대비, room_rules.AMBIG near_h와 같음)
F_REL = 0.72
DOOR_M = 2.1
BLUR_TH = 30
RECHECK_SIDE = 1280  # 애매해서 다시 보는 장은 긴 변 1280(보통 화질)으로 줄여 읽음. 대표 1장만 원래 크기 (1080 세션 1초 2.0초 → 줄이려고)
EL_DOOR = 0.5  # Places365 elevator/door 확률이 이 이상이면 엘리베이터 문 장면 (세 영상: 진짜 12초 중 7초, 헛것 73초 중 3초. YOLOE는 헛것 38초)
VOTE_SCORE, VOTE_NEAR, VOTE_NEAR_SCORE = 0.95, 10, 0.8  # situ 문패 다수결: 크고(NEAR_H) 점수 0.95 이상이거나, 이미 인정한 같은 층 번호와 10 이내(점수 0.8 이상)면 바로 인정. 아니면 최근 3걸음 중 2번
QUIET_SAY = 10  # situ: 이 초 동안 새 안내가 없으면 "계속 직진"을 말함 (말하지 않는 VLM 추천 대신)
LOCATE_SEC = 5  # 경로 시작: 이 시간 동안 위치(층·장소)를 먼저 파악하고 말한 뒤 다음 행동을 안내 (그동안 엘리베이터 안내는 막음)
LOCATE_ASKS = 2  # 시간이 지나도 층을 모르면 "주변을 더 보여 주세요"를 몇 번까지 (번마다 LOCATE_SEC 더 기다림)
PLACE_MIN = 0.35  # 위치 파악 동안 Places365 묶음 평균 확률이 이 이상이면 그 장소로 봄
PLACE_KO = {"복도": "복도", "엘리베이터": "엘리베이터 앞", "로비·입구": "로비", "계단": "계단 근처"}
SCENE_KO = {"CORRIDOR": "복도", "LOBBY": "로비", "ELEVATOR_HALL": "엘리베이터 앞", "ENTRANCE": "건물 출입문 앞",
            "INTERSECTION": "복도 갈림길", "ROOM_DOOR": "방 문 앞", "INSIDE_ELEVATOR": "엘리베이터 안"}
SURE_SCORE = 0.95  # 목적지 근처라도 대표 장면이 목적지 아닌 호실을 이 점수 이상으로 읽었으면 다시 보지 않음
HANGUL = re.compile(r"[가-힣]")
BROKEN = re.compile(r"^\d{2}$|^\d{4,5}$")
SIDE = lambda cx: "왼쪽" if cx < 0.4 else ("오른쪽" if cx > 0.6 else "정면")  # noqa: E731
PATH_KO = {"FORWARD": "앞쪽으로", "LEFT": "왼쪽으로", "RIGHT": "오른쪽으로", "FORWARD_LEFT": "앞과 왼쪽으로",
           "FORWARD_RIGHT": "앞과 오른쪽으로", "LEFT_RIGHT": "왼쪽과 오른쪽으로", "ALL": "여러 방향으로",
           "DEAD_END": "", "UNKNOWN": ""}

CONTEXT_SCHEMA = {"type": "object", "properties": {
    "scene_type": {"type": "string", "enum": vc.SCENES},
    "visible_path": {"type": "string", "enum": vc.PATHS},
    "decision_point": {"type": "boolean"},
    "door_ahead": {"type": "string", "enum": ["GLASS_DOOR", "DOOR", "NONE"]},
    "elevator_visible": {"type": "boolean"},
    "call_button_side": {"type": "string", "enum": ["LEFT", "RIGHT", "NONE"]},
    "evidence": {"type": "string", "maxLength": 60},
}, "required": ["scene_type", "visible_path", "decision_point", "door_ahead", "elevator_visible",
                "call_button_side", "evidence"]}

CONTEXT_GUIDE = (
    "너는 시각장애인 실내 길안내 보조다. 사용자가 걸으며 찍은 영상의 지금 장면이다. 사진에 실제로 보이는 것만 근거로 답하라.\n"
    "- scene_type: CORRIDOR 복도, LOBBY 로비·넓은 홀, ELEVATOR_HALL 엘리베이터 앞, ENTRANCE 건물 출입문, "
    "INTERSECTION 복도 갈림길, ROOM_DOOR 특정 방 문 바로 앞, INSIDE_ELEVATOR 엘리베이터 안, UNKNOWN 판단 불가.\n"
    "- visible_path: 걸어갈 수 있게 보이는 길의 방향.\n"
    "- decision_point: 방향을 골라야 하는 지점(갈림길, 엘리베이터 앞, 출입문)이면 true.\n"
    "- door_ahead: 바로 앞에 지나가야 할 문이 있으면 GLASS_DOOR(유리문) 또는 DOOR, 없으면 NONE.\n"
    "- elevator_visible: 엘리베이터 문이 보이면 true. 일반 철문과 구분하라(옆 벽에 위/아래 호출 버튼이 있는지).\n"
    "- call_button_side: 엘리베이터 호출 버튼이 엘리베이터 문의 왼쪽 벽이면 LEFT, 오른쪽 벽이면 RIGHT, 안 보이면 NONE.\n"
    "- evidence: 판단 근거로 본 것 (짧게)."
)

PLATE_SCHEMA = {"type": "object", "properties": {"room": {"type": "string", "maxLength": 5}}, "required": ["room"]}
PLATE_GUIDE = "이 조각은 문패를 확대한 것이다. 문패의 호실 번호를 그대로 적어라. 읽을 수 없으면 '모름'."


# ---------------------------------------------------------------- 1단계 + 애매 판정

def blur(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ambiguity(s1, img):
    reasons = []
    if blur(img) < BLUR_TH:
        reasons.append("흐림")
    good = [t for t in s1["texts_all"] if t["score"] >= 0.5]
    if any(BROKEN.match(t["text"].replace("호", "")) for t in good):
        reasons.append("깨진 번호")
    if not s1["result"]["rooms"] and any(HANGUL.search(t["text"]) and t["box"][3] - t["box"][1] >= 0.012 for t in good):
        reasons.append("문패 한글만")
    cands = room_rules.candidates(s1["texts_all"], img.size)
    if any(0.3 <= c["score"] < 0.9 for c in cands):
        reasons.append("호실 점수 중간")
    return reasons


def room_boxes(s1, img_size):
    """확정 호실의 글자 상자 (가장 큰 것)."""
    out = {}
    for c in room_rules.candidates(s1["texts_all"], img_size):
        if c["room"] in s1["result"]["rooms"]:
            out[c["room"]] = c
    return out


def door_distance(s1, plate):
    """문패 아래·옆의 문 상자로 거리(m) 추정. (거리, 근거) 또는 (None, 이유)."""
    doors = [d for d in s1["detections"] if d["name"] in ("문", "엘리베이터") and d["conf"] >= 0.15]
    best = None
    for d in doors:
        x0, y0, x1, y1 = d["box"]
        if x0 - 0.12 <= plate["cx"] <= x1 + 0.12 and y0 - 0.15 <= plate["cy"] <= (y0 + y1) / 2:
            gap = abs((x0 + x1) / 2 - plate["cx"])
            if best is None or gap < best[0]:
                best = (gap, d)
    if best is None:
        return None, "문 상자 없음"
    x0, y0, x1, y1 = best[1]["box"]
    if y0 <= 0.01 or y1 >= 0.99:
        return 1.5, "문이 화면에 다 안 들어옴 (아주 가까움)"
    return round(F_REL * DOOR_M / (y1 - y0), 1), "문 높이로 추정"


# ---------------------------------------------------------------- 실행

class Runner:
    def __init__(self, args):
        self.args = args
        cwd = os.getcwd()
        self.yoloe = perception.load_yoloe()
        os.chdir(cwd)
        self.ocr = ocr_reader.load_ocr("m", {})
        self.places = scene_places.load()  # 모델 파일이 없으면 None → 예전처럼 YOLOE 엘리베이터로 판단
        self.cache = {}

    def s1(self, path, max_side=None):
        if (path, max_side) not in self.cache:
            img = Image.open(path).convert("RGB")
            if max_side:
                img.thumbnail((max_side, max_side), Image.LANCZOS)
            r = vc.stage1(self.yoloe, self.ocr, img)
            self.cache[(path, max_side)] = (r, img)
        return self.cache[(path, max_side)]

    def warm(self, path):
        img = Image.open(path).convert("RGB")
        for _ in range(2):
            vc.stage1(self.yoloe, self.ocr, img)
        vc.ask(self.args.url, self.args.model, "ok", None, Image.new("RGB", (64, 64)), 4)

    def observe(self, frames, force=False, target=None):
        """1초 구간 관찰. 대표 장면 → 애매하면(또는 목적지 근처면) 나머지 장도 보고 투표.
        VLM에 보낼 장면은 1초 안에서 가장 선명한 장 (OCR 없이 선명도만 계산)."""
        t = 0.0
        key = frames[0]
        if getattr(self.args, "key", "sharp") == "sharp":  # lsh 방식: 1초 안에서 가장 선명한 장을 대표로 (긴 변 640으로 줄여 잼)
            tb = time.perf_counter()
            key = max(frames, key=lambda p: blur(Image.open(p).convert("RGB").reduce(2)))
            t += time.perf_counter() - tb
        r, img = self.s1(key)
        t += r["stage1_seconds"]
        # 대표 장면에서 OCR이 읽은 글자 전부(점수 낮은 것 포함)와 YOLOE 물체 — 무엇을 읽는지 보려고 기록만 함
        raw = {"size": list(img.size), "ocr_input_size": r.get("ocr_input_size"),
               "ocr": [{"text": x["text"], "score": round(x["score"], 2), "box": x["box"]} for x in r["texts_all"]],
               "objects": [{"name": d["name"], "conf": round(d["conf"], 2), "box": d["box"]} for d in r["detections"]]}
        elev_door = None
        if self.places:  # 대표 장면 한 장 전체가 어떤 장소인지 (엘리베이터 문 판단용)
            tb = time.perf_counter()
            top, groups, elev_door = scene_places.classify(self.places, img)
            t += time.perf_counter() - tb
            raw["place"] = top[:3]
            raw["place_groups"] = {k: round(v, 3) for k, v in groups.items()}
        reasons = ambiguity(r, img)
        used, rooms, best = [key.name], list(r["result"]["rooms"]), (r, img)
        src = {c["room"]: (c, r) for c in room_rules.candidates(r["texts_all"], img.size) if c["room"] in rooms}
        sure = rooms and target not in rooms and all(src[x][0]["score"] >= SURE_SCORE for x in rooms if x in src)
        if force and not (sure and not reasons):  # 목적지 아닌 호실을 확실히 읽었으면 목적지 근처라도 다시 안 봄
            reasons = reasons + ["목적지 근처"]
        if getattr(self.args, "key", "sharp") == "sharp":
            sharp = img  # 대표 장면이 이미 가장 선명한 장
        else:
            tb = time.perf_counter()
            sharp = max((Image.open(p).convert("RGB") for p in frames), key=blur)
            t += time.perf_counter() - tb
        if reasons and len(frames) > 1:  # 1장만 받았으면 투표할 게 없음 (웹: 다음 초에 4장을 달라고 함)
            votes, top, score_hi = Counter(), {}, set()
            results = []
            for p in frames:
                rr, im = self.s1(p) if p == key else self.s1(p, RECHECK_SIDE)
                if p != key:  # 시간은 캐시와 관계없이 실제 1단계 시간을 더한다
                    used.append(p.name)
                    t += rr["stage1_seconds"]
                results.append((rr, im))
                for c in room_rules.candidates(rr["texts_all"], im.size):
                    if c["room"] in rr["result"]["rooms"]:
                        votes[c["room"]] += 1
                        if c["room"] not in src or c["h"] > src[c["room"]][0]["h"]:
                            src[c["room"]] = (c, rr)  # 번호마다 가장 크게 읽힌 장면
                        if c["score"] >= 0.9:
                            score_hi.add(c["room"])
            rooms = [x for x, n in votes.most_common() if n >= 2 or x in score_hi]
            floors = Counter(x[0] for x in rooms)
            if floors:  # 투표 결과에도 첫 자리 다수결
                digit = floors.most_common(1)[0][0]
                rooms = [x for x in rooms if x[0] == digit]
            best = max(results, key=lambda x: (len(set(x[0]["result"]["rooms"]) & set(rooms)), blur(x[1])))
            hs = {c["room"]: c["h"] for c in room_rules.candidates(best[0]["texts_all"], best[1].size)}
            rooms.sort(key=lambda x: -hs.get(x, 0))  # 가까운(큰 글자) 순
        return {"rooms": rooms, "s1": best[0], "img": best[1], "sharp": sharp, "src": src, "ambiguous": reasons, "used": used,
                "stage1_seconds": t, "raw": raw, "elev_door": elev_door}

    def context(self, img, s1, event):
        prompt = CONTEXT_GUIDE + f"\n\n상황: {event}\n\n" + vc.hints(s1)
        return vc.ask(self.args.url, self.args.model, prompt, CONTEXT_SCHEMA, img, 200)

    def sub(self, prompt, enum, img=None):
        """c 방식: 한 칸짜리 객관식 질문 하나."""
        sch = {"type": "object", "properties": {"a": {"type": "string", "enum": enum}}, "required": ["a"]}
        r = vc.ask(self.args.url, self.args.model, prompt, sch, img, 24)
        return (r.get("answer") or {}).get("a"), r

    def advise(self, prompt, img):
        """situ: 다음 행동 추천 (한 칸 객관식). --advise-url/--advise-model로 다른 모델(예: Gemini 중계)을 쓸 수 있음."""
        url = getattr(self.args, "advise_url", None) or self.args.url
        model = getattr(self.args, "advise_model", None) or self.args.model
        sch = {"type": "object", "properties": {"a": {"type": "string", "enum": ACTIONS}}, "required": ["a"]}
        r = vc.ask(url, model, prompt, sch, img, 24)
        return (r.get("answer") or {}).get("a"), r

    def plate(self, img, box):
        x0, y0, x1, y1 = box
        w, h = img.size
        pad_x, pad_y = (x1 - x0) * 1.5 + 0.03, (y1 - y0) * 2 + 0.02
        crop = img.crop((int(max(0, x0 - pad_x) * w), int(max(0, y0 - pad_y) * h),
                         int(min(1, x1 + pad_x) * w), int(min(1, y1 + pad_y) * h)))
        return vc.ask(self.args.url, self.args.model, PLATE_GUIDE, PLATE_SCHEMA, crop, 16)


# c 방식(질문 쪼개기) 질문. video_reason.py와 같은 문장
SUB_TARGET = "건물 안 목적지는 {target}호다. 이 호실은 몇 층에 있을 가능성이 가장 높은가?"
SUB_FLOOR = "다음은 사용자가 건물 안을 이동하며 관찰한 기록과 지금 장면 인식 결과다. 사용자는 지금 몇 층에 있는가? 근거가 없으면 모름.\n\n{facts}\n\n{now}"
SUB_SCENE = ("이 사진은 실내를 걷는 사용자 시점이다. 지금 장면은 어떤 곳인가? CORRIDOR 복도, LOBBY 로비·넓은 홀, "
             "ELEVATOR_HALL 엘리베이터 앞, ENTRANCE 건물 출입문, INTERSECTION 복도 갈림길, ROOM_DOOR 특정 방 문 바로 앞, "
             "INSIDE_ELEVATOR 엘리베이터 안, UNKNOWN 판단 불가.")
FLOORS = ["1층", "2층", "3층", "4층", "5층", "모름"]
# situ: VLM이 사실 기록과 지금 장면을 보고 다음 행동을 추천 (규칙은 확인된 사실과 모순되는 추천만 막음)
ACTIONS = ["문 통과", "직진", "왼쪽", "오른쪽", "뒤로 돌기", "엘리베이터 타기", "엘리베이터로 돌아가기",
           "멈추고 둘러보기", "목적지 접근", "도착"]
ADVISE = ("너는 시각장애인 실내 길안내 보조다. 사용자의 목적지는 {target}호({tf}층으로 추정)다. 아래는 지금까지 확인된 사실과 "
          "지금 장면(사진과 인식 결과)이다. 처음 보는 건물일 수 있다. 사진에 보이는 것과 사실만 근거로, 사용자가 지금 해야 할 "
          "다음 행동 하나를 골라라.\n\n확인된 사실:\n{facts}\n\n{now}")
ACTION_SAY = {"문 통과": "앞에 문이 있습니다. 문을 지나 들어가세요.", "직진": "앞으로 계속 가세요.",
              "왼쪽": "왼쪽으로 도세요.", "오른쪽": "오른쪽으로 도세요.", "뒤로 돌기": "뒤로 돌아서 가세요.",
              "멈추고 둘러보기": "잠시 멈춰서 주변을 천천히 비춰 주세요."}
EL_WAIT = 6  # 엘리베이터 앞에서 층 단서(문패)를 기다리는 초. 그 안에 층을 알면 '타세요'를 하지 않음 (3층 영상은 첫 문패가 4~5초)


class Navigator:
    """길안내 한 번(목적지 하나)의 상태. 1초 구간 프레임을 step()에 넣으면 그 초의 기록(안내 문장 포함)을 돌려준다.
    영상 실행(run_route)과 웹(web_server.py)이 같이 쓴다.
    mode: rule = 층은 규칙(호실 첫 자리), 장면은 VLM 큰 질문 / c = 목적지 층·지금 층·장면을 VLM 한 칸 질문으로
          lean = rule에서 규칙으로 대신할 수 있는 VLM 호출(하차 직후, 근거 없이 N초)을 뺀 것. 경로 시작과 엘리베이터 확인만 남김."""

    def __init__(self, runner, target, mode="rule"):
        self.runner, self.target, self.mode = runner, target, mode
        self.tf = target[0]
        self.first_vlm = []
        if mode == "c":
            ans, r = runner.sub(SUB_TARGET.format(target=target), FLOORS)
            self.first_vlm.append({"why": ["c: 목적지 층"], **r})
            self.tf = ans[0] if ans and ans[0].isdigit() else target[0]
        self.st = {"floor": None, "room": None, "since": 0, "scene": None, "wrong_floor": False, "arrived": False,
                   "floor_rooms": [], "last_vlm": -99, "target_d": None, "elevator_said": False, "trend_said": None,
                   "el_run": 0, "pending": None, "near_d": [], "el_ready": None, "button": None, "quiet": 0,
                   "loc": None, "assume1": False, "room_hist": [], "ok_rooms": []}  # assume1: situ 경로 시작은 1층으로 가정 (문패를 읽으면 문패가 우선)  # 경로 시작 위치 파악 중이면 {"until", "asks", "places"(초마다 Places365 묶음), "vlm_scene", "path"}
        self.records, self.clock = [], 0
        self.mem = nav_memory.Memory(target, NEAR_H)  # 시간 순서 기억 (situ에서 문패 확정·사건 기록에 씀)
        self.mem.event(0, f"안내 시작 (목적지 {target}호)")
        self.events = [f"경로 시작 (목적지 {target}호 안내 시작)"]
        self.si, self.sec, self.seq, self.seg_start = 0, 0, "", 0

    def vote(self, rooms, src):
        """이번 걸음에 읽은 번호를 기억에 넣고, 믿음 점수가 찬(확정된) 번호만 돌려준다 (nav_memory.Memory.see)."""
        out = []
        for x in rooms:
            c, ps1 = src[x] if x in src else (None, None)
            if c is None:
                continue
            door = door_distance(ps1, c)[1] != "문 상자 없음"
            if self.mem.see(self.clock, x, c["score"], c["h"], c["cx"], door)["ok"]:
                out.append(x)
        self.st["ok_rooms"] = self.mem.confirmed()
        return out

    @staticmethod
    def place_name(loc):
        """위치 파악 동안 모은 장소. VLM이 Places365에 없는 장면(출입문·갈림길 등)이라 하면 그걸, 아니면 Places365 평균, 아니면 VLM."""
        v = loc["vlm_scene"]
        if v in ("ENTRANCE", "INTERSECTION", "INSIDE_ELEVATOR"):  # 방 문 앞(ROOM_DOOR)은 Places365 복도가 더 안정적이라 뺌
            return SCENE_KO[v]
        recent = loc["places"][-LOCATE_SEC:]  # 걸으며 장소가 바뀌므로 최근 몇 초만 평균
        if recent:
            tot = Counter()
            for g in recent:
                tot.update(g)
            g, p = max(((k, x / len(recent)) for k, x in tot.items()), key=lambda kv: kv[1])
            if p >= PLACE_MIN:
                return PLACE_KO[g]
        return SCENE_KO.get(v)

    def locate_end(self, loc):
        """위치 파악 시간이 끝났는데 층을 모름: 주변을 더 보여 달라고 하거나(LOCATE_ASKS번까지), 장소로 다음 행동."""
        st, tf = self.st, self.tf
        place = self.place_name(loc)
        if self.mode == "situ" and self.si == 0:  # 경로 시작은 1층으로 가정하고 바로 다음 행동
            st["loc"], st["assume1"] = None, True
            head = (f"여기는 {place}인 것 같습니다. " if place else "") + "1층으로 보고 안내하겠습니다. "
            if tf == "1":
                return [head + "목적지 층입니다. 앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]
            if place == "엘리베이터 앞":
                st["elevator_said"] = True
                return [head + f"앞에 엘리베이터가 있는 것 같습니다. 엘리베이터를 타고 {tf}층을 누르세요."]
            return [head + f"{self.target}호는 {tf}층이니 엘리베이터를 찾겠습니다. 앞으로 천천히 직진해 주세요."]
        if loc["asks"] < LOCATE_ASKS and place != "건물 출입문 앞":
            loc["asks"] += 1
            loc["until"] = self.clock + 1 + LOCATE_SEC
            first = (f"여기는 {place}인 것 같습니다. 층을 알 수 있는 문패나 층 표시는 아직 안 보입니다. " if place
                     else "아직 여기가 어디인지 잘 모르겠습니다. ")
            return [(first if loc["asks"] == 1 else "") + "주변을 조금 더 보여 주세요. 휴대폰을 천천히 좌우로 돌려 주세요."]
        st["loc"] = None
        head = f"여기는 {place}인 것 같습니다. 층은 아직 모릅니다. " if place else "지금 위치와 층을 아직 모릅니다. "
        if place == "건물 출입문 앞":
            return [head + "앞의 문을 지나 들어가세요."]
        if place == "엘리베이터 앞":
            st["el_ready"] = self.clock - EL_WAIT + 3  # 3초 더 층 단서를 찾고 없으면 기존처럼 '타세요'
            return [head + "층 표시를 조금 더 찾아보겠습니다."]
        path = PATH_KO.get(loc["path"] or "UNKNOWN", "")
        return [head + (f"길이 {path} 이어집니다. " if path else "") + "앞으로 천천히 직진해 주세요. 문패를 찾아보겠습니다."]

    def new_segment(self, seq=""):
        """구간 시작. 첫 구간이 아니면 엘리베이터를 타고 내린 것으로 본다 (층·위치는 모름으로)."""
        if self.records or self.sec:
            self.si += 1
            self.mem.new_segment(self.clock)
        self.seq, self.sec, self.seg_start = seq, 0, self.clock
        if self.si > 0:
            self.events = self.events[:1] + ["엘리베이터를 타고 이동한 뒤 내림 (누른 층은 확인되지 않음)"]
            self.st.update(floor=None, wrong_floor=False, room=None, since=0, floor_rooms=[], scene=None, target_d=None, trend_said=None,
                           pending=None, near_d=[], el_ready=None, button=None, quiet=0, elevator_said=False, el_run=0, assume1=False,
                           room_hist=[], ok_rooms=[])

    def step(self, frames):
        """frames: 1초 구간의 사진 파일 경로들 (첫 장이 대표). 반환: 이 초의 기록."""
        runner, st, target, tf, mode = self.runner, self.st, self.target, self.tf, self.mode
        si, sec, seq, seg_start, clock = self.si, self.sec, self.seq, self.seg_start, self.clock
        events = self.events
        near_target = st["floor"] == tf and any(abs(int(r) - int(target)) <= 2 for _, r in st["floor_rooms"]
                                                 if "-" not in r)
        obs = runner.observe(frames, force=near_target, target=target)
        s1, img = obs["s1"], obs["img"]
        rooms = obs["rooms"]
        if mode == "situ":
            self.mem.place(clock, obs["raw"].get("place_groups") or {})
            rooms = self.vote(rooms, obs["src"])
        say, vlm, vlm_why, note = [], [], [], {}
        if self.first_vlm:
            vlm, self.first_vlm = self.first_vlm, []
        boxes = room_boxes(s1, img.size)
        if sec == 0 and si == 0 and mode == "situ":
            st["loc"] = {"until": clock + LOCATE_SEC, "asks": 0, "places": [], "vlm_scene": None, "path": None}
        loc = st["loc"]
        if loc is not None and obs["raw"].get("place_groups"):
            loc["places"].append(obs["raw"]["place_groups"])

        # VLM 호출 사건 판단
        if sec == 0 and (si == 0 or mode != "lean") and mode != "situ":  # lean: 하차 직후 층은 문패로 / situ: 시작·하차 모두 Places365·문패로
            vlm_why.append("경로 시작" if si == 0 else "엘리베이터에서 내린 직후")
        # 엘리베이터 문 장면이 2초 연속이면. Places365가 있으면 그걸로, 없으면 큰 YOLOE 엘리베이터 후보로
        # (YOLOE '엘리베이터'는 철문·창문·입간판도 잡아 헛것이 많음)
        if obs["elev_door"] is not None:
            big_el = obs["elev_door"] >= EL_DOOR
        else:
            big_el = [d for d in s1["detections"] if d["name"] == "엘리베이터" and d["conf"] >= 0.3
                      and d["box"][3] - d["box"][1] > 0.4]
        st["el_run"] = st["el_run"] + 1 if big_el else 0
        el_say = None  # situ: Places365로 엘리베이터 문을 2초 연속 보면 VLM(지금 15초 넘게 걸림)을 기다리지 않고 바로 말함
        if mode == "situ" and st["el_run"] >= 2 and loc is None and not st["elevator_said"] and not (si > 0 and sec < 4):  # 막 내린 직후 뒤의 엘리베이터는 무시
            if st["wrong_floor"] or (st["floor"] is None and st["assume1"] and tf != "1"):
                el_say = f"앞에 엘리베이터가 있는 것 같습니다. 엘리베이터를 타고 {tf}층을 누르세요."
                st["elevator_said"] = True
            elif st["floor"] is None and st["el_ready"] is None:
                el_say = "앞에 엘리베이터가 있는 것 같습니다. 지금 몇 층인지 먼저 확인하겠습니다. 주변을 천천히 비춰 주세요."
                st["el_ready"] = clock
        if (mode != "situ" and st["el_run"] >= 2 and loc is None and clock - st["last_vlm"] >= 3 and not st["elevator_said"]
                and st["floor"] is None and (si == 0 or mode == "situ") and st["el_ready"] is None):
            vlm_why.append("엘리베이터 후보가 계속 보임")
        # 층 충돌: 한 번 정한 층과 첫 자리가 다른 번호는 2초 연속일 때만 인정 (한 번은 오독으로 보고 버림)
        other = [x for x in rooms if st["floor"] and x[0] != st["floor"]]
        if other:
            digit = other[0][0]
            if st["pending"] and st["pending"][0] == digit and st["pending"][1] == clock - 1:
                vlm_why.append(f"층 충돌 지속 {st['floor']}→{digit}")
            else:
                st["pending"] = (digit, clock)
                rooms = [x for x in rooms if x[0] == st["floor"]]
        if not rooms:
            st["since"] += 1
            if st["since"] in (10, 20, 30) and clock - st["last_vlm"] >= 5 and mode not in ("lean", "situ"):
                vlm_why.append(f"근거 없이 {st['since']}초")
        # 남은 애매함 (최후): 여러 장을 봐도 가까운 문패 한글만 있고 번호가 없을 때
        near_plate = None
        if not rooms and "문패 한글만" in obs["ambiguous"] and len(obs["used"]) > 1 and st["floor"] == tf:
            near = [t for t in s1["texts_all"] if t["score"] >= 0.5 and HANGUL.search(t["text"])
                    and t["box"][3] - t["box"][1] >= 0.015]
            if near:
                near_plate = max(near, key=lambda t: t["box"][3] - t["box"][1])

        ctx = None
        if vlm_why and mode == "c":
            scene, r = runner.sub(SUB_SCENE, vc.SCENES, obs["sharp"])
            vlm.append({"why": vlm_why + ["c: 장면"], **r})
            ctx = {"answer": {"scene_type": scene}} if scene else None
            st["last_vlm"] = clock
        elif vlm_why:
            ctx = runner.context(obs["sharp"], s1, ", ".join(vlm_why))
            vlm.append({"why": vlm_why, **ctx})
            st["last_vlm"] = clock
        if near_plate:
            ans = runner.plate(img, near_plate["box"])
            vlm.append({"why": ["문패 번호 확인(최후)"], **ans})
            v = ((ans.get("answer") or {}).get("room") or "").replace("호", "")
            if room_rules.ROOM.match(v) and v[0] == st["floor"]:
                rooms = [v]

        a = (ctx or {}).get("answer") or {}
        if a:
            st["scene"] = a["scene_type"]

        # ---- 상태 갱신과 안내 문장
        if el_say:
            say.append(el_say)
        if loc is not None and a:
            loc["vlm_scene"], loc["path"] = a.get("scene_type"), a.get("visible_path")
        if sec == 0 and si == 0 and mode == "situ":
            say.append(f"{target}호로 안내를 시작합니다. {target}호는 {tf}층입니다. "
                       "지금 위치를 확인하겠습니다. 휴대폰을 앞으로 들고 천천히 좌우로 비춰 주세요.")
        elif sec == 0 and si == 0:
            door = (a.get("scene_type") == "ENTRANCE" if mode == "situ"
                    else a.get("door_ahead") in ("GLASS_DOOR", "DOOR") or a.get("scene_type") == "ENTRANCE")
            if door:
                kind = "유리문" if a.get("door_ahead") == "GLASS_DOOR" else "문"
                say.append(f"{target}호로 안내를 시작합니다. {target}호는 {tf}층입니다. 앞에 {kind}이 있습니다. 문을 열고 들어가세요.")
            else:
                say.append(f"{target}호로 안내를 시작합니다. {target}호는 {tf}층입니다.")
        if sec == 0 and si > 0:
            path = PATH_KO.get(a.get("visible_path", "UNKNOWN"), "")
            say.append(f"엘리베이터에서 내렸습니다. " + (f"복도가 {path} 이어집니다. " if path else "")
                       + "문패를 확인하며 천천히 이동하세요.")
        if mode == "situ":
            if loc is None and a.get("scene_type") == "ELEVATOR_HALL" and st["floor"] is None and st["el_ready"] is None:
                st["el_ready"] = clock
                st["button"] = {"LEFT": "왼쪽", "RIGHT": "오른쪽"}.get(a.get("call_button_side"))
                say.append("엘리베이터 근처로 보입니다. 지금 몇 층인지 먼저 확인하겠습니다. 주변을 천천히 비춰 주세요.")
        elif ((a.get("elevator_visible") or mode == "c") and a.get("scene_type") == "ELEVATOR_HALL" and not st["elevator_said"]
                and st["floor"] is None and si == 0):
            side = {"LEFT": "왼쪽", "RIGHT": "오른쪽"}.get(a.get("call_button_side"))
            say.append("앞에 엘리베이터가 있습니다. " + (f"호출 버튼은 엘리베이터 문 {side} 벽에 있습니다. " if side else "")
                       + f"엘리베이터를 타고 {tf}층을 누르세요.")
            st["elevator_said"] = True
        if any(w.startswith("근거 없이") for w in vlm_why) and a and not st["wrong_floor"]:
            if a.get("scene_type") in ("CORRIDOR", "LOBBY") and (mode == "c" or a.get("visible_path") in ("FORWARD", "ALL", "FORWARD_LEFT", "FORWARD_RIGHT")):
                say.append("복도가 앞으로 이어집니다. 계속 직진하세요.")

        if rooms and mode == "c" and (st["floor"] is None or any(w.startswith("층 충돌") for w in vlm_why)):
            # c: 지금 층을 VLM에 묻는다 (사실 기록 + 지금 장면). 모름이면 이번 초는 층을 정하지 않음
            seen = [f"하차 후 {c - seg_start}초: {x}호 문패를 봄" for c, x in st["floor_rooms"]]
            facts = "\n".join(f"#{i} {t}" for i, t in enumerate(events + seen, 1))
            now = f"#0 지금 장면: 확정된 호실 {', '.join(rooms)}\n" + vc.hints(s1)
            ans, r = runner.sub(SUB_FLOOR.format(facts=facts, now=now), FLOORS)
            vlm.append({"why": ["c: 지금 층"], **r})
            vlm_floor = ans[0] if ans and ans[0].isdigit() else None
        else:
            vlm_floor = None
        if rooms:
            st["since"] = 0
            floor = rooms[0][0] if mode != "c" else (vlm_floor or st["floor"])
        if rooms and floor is None:  # c에서 VLM이 층을 모른다고 함: 호실만 기록
            for x in rooms:
                if x not in [r for _, r in st["floor_rooms"]]:
                    st["floor_rooms"].append((clock, x))
        elif rooms:
            new = [x for x in rooms if x not in [r for _, r in st["floor_rooms"]]]
            if st["floor"] is None and loc is not None:  # 위치 파악 중 문패로 층을 앎 → 위치를 말하고 다음 행동
                st["floor"], st["loc"] = floor, None
                place = self.place_name(loc)
                plate = obs["src"].get(rooms[0], (None,))[0]
                where = f"{SIDE(plate['cx'])}에 " if plate else ""
                head = f"지금 위치는 {floor}층" + (f" {place}" if place else "") + f"입니다. {where}{rooms[0]}호가 보입니다. "
                if floor == tf:
                    say.append(head + "목적지 층입니다. 앞으로 천천히 직진해 주세요. 문패를 확인하며 안내하겠습니다.")
                else:
                    st["wrong_floor"] = True
                    say.append(head + f"{target}호는 {tf}층에 있습니다. 엘리베이터로 {tf}층에 가야 합니다. 앞으로 천천히 직진하며 엘리베이터를 찾겠습니다.")
            elif st["floor"] is None:
                st["floor"], assumed = floor, st["assume1"]
                st["assume1"] = False
                if assumed and floor != "1":
                    say.append(f"문패를 보니 1층이 아니라 {floor}층입니다.")
                if floor == tf:
                    say.append(f"{tf}층이 맞습니다. 지금 {rooms[0]}호 근처입니다."
                           + (" 엘리베이터를 탈 필요 없습니다. 복도를 따라 문패를 찾겠습니다." if st["el_ready"] is not None else ""))
                else:
                    st["wrong_floor"] = True
                    say.append(f"여기는 {floor}층입니다. {target}호는 {tf}층에 있습니다. "
                               + (f"엘리베이터로 돌아가 {tf}층을 누르세요." if si > 0 else
                                  ("" if st["elevator_said"] else f"엘리베이터를 찾아 {tf}층으로 가겠습니다. 앞으로 천천히 직진해 주세요.")))
            elif floor != st["floor"]:
                say.append(f"층 판단이 바뀌었습니다: {st['floor']}층 → {floor}층.")
                st["floor"], st["wrong_floor"] = floor, floor != tf
            st["room"] = rooms[0]
            for x in new:
                st["floor_rooms"].append((clock, x))
            # 방향: 가장 가까운 문패(큰 글자)의 번호가 목적지 번호에 가까워지는지
            near = rooms[0]
            near_h = obs["src"][near][0]["h"] if near in obs["src"] else 0
            if (not st["wrong_floor"] and target not in rooms and near[0] == tf and "-" not in near
                    and near_h >= NEAR_H
                    and (not st["near_d"] or st["near_d"][-1][0] != near)):
                d = abs(int(near) - int(target))
                prev = st["near_d"][-1][1] if st["near_d"] else None
                st["near_d"].append((near, d))
                ds = [x[1] for x in st["near_d"]]
                if prev is not None and d < prev:
                    say.append(f"{near}호를 지나고 있습니다. 목적지 방향이 맞습니다. 계속 직진하세요.")
                    st["trend_said"] = None
                elif len(ds) >= 3 and ds[-1] > ds[-2] > ds[-3] and st["trend_said"] != "away":
                    say.append(f"{near}호입니다. 목적지와 멀어지고 있습니다. 뒤로 돌아가세요.")
                    st["trend_said"] = "away"
            # 목적지 발견
            if target in rooms and not st["wrong_floor"]:
                plate, ps1 = obs["src"].get(target, (None, None))
                d, how = door_distance(ps1, plate) if plate else (None, "글자 상자 없음")
                side = SIDE(plate["cx"]) if plate else ""
                if d is not None and d <= 2.0:
                    say.append(f"도착했습니다. {side}의 문이 {target}호입니다.")
                    st["arrived"] = True
                elif st["target_d"] is None or (d and st["target_d"] - d >= 2):
                    say.append(f"목적지 {target}호가 {side} 앞에 보입니다." + (f" 약 {d:.0f}m 남았습니다." if d else ""))
                note = {"target_distance_m": d, "distance_basis": how, "target_side": side}
                st["target_d"] = d
        if st["loc"] is not None and clock + 1 >= st["loc"]["until"] and not st["arrived"]:
            say.extend(self.locate_end(st["loc"]))
        advice = None
        if mode == "situ" and not st["arrived"]:
            if st["el_ready"] is not None and not st["elevator_said"]:
                if st["floor"] is not None:  # 기다리는 사이 층을 알았다
                    st["el_ready"] = None
                elif clock - st["el_ready"] >= EL_WAIT:
                    say.append("앞에 엘리베이터가 있습니다. 지금 층을 알 수 있는 표시는 보이지 않습니다. "
                               + (f"호출 버튼은 엘리베이터 문 {st['button']} 벽에 있습니다. " if st["button"] else "")
                               + f"엘리베이터를 타고 {tf}층을 누르세요.")
                    st["elevator_said"] = True
            # 추천을 물을 때: 경로 시작, 엘리베이터에서 내린 직후, 15초 동안 안내할 말이 없을 때
            st["quiet"] = 0 if say else st["quiet"] + 1
            ask_why = ("경로 시작" if sec == 0 and si == 0 else "엘리베이터에서 내린 직후" if sec == 0
                       else "15초 동안 새 안내 없음" if st["quiet"] >= 15 else None)
            if (not getattr(runner.args, "advise_model", None) and st["loc"] is None and not st["arrived"]
                    and st["quiet"] >= QUIET_SAY and st["el_ready"] is None):
                seeking = st["wrong_floor"] or (st["floor"] is None and st["assume1"] and tf != "1")
                if not seeking:
                    say.append("계속 앞으로 천천히 직진해 주세요.")
                elif not st["elevator_said"]:
                    say.append("엘리베이터를 찾고 있습니다. 천천히 직진하며 주변을 비춰 주세요.")
                st["quiet"] = 0
            if ask_why and not st["wrong_floor"] and getattr(runner.args, "advise_model", None):
                floor_fact = f"지금 층: {st['floor']}층 (문패 번호로 확인)" if st["floor"] else "지금 층: 아직 모름"
                seen = [f"{x}호 문패를 봄" for _, x in st["floor_rooms"]][-6:]
                facts = "\n".join(f"- {t}" for t in events + [floor_fact] + seen)
                now = f"지금 장면: 확정된 호실 {', '.join(rooms) or '없음'}\n" + vc.hints(s1)
                act, r = runner.advise(ADVISE.format(target=target, tf=tf, facts=facts, now=now), obs["sharp"])
                vlm.append({"why": [f"다음 행동 추천: {ask_why}"], **r})
                st["last_vlm"], st["quiet"] = clock, 0
                # 규칙은 확인된 사실과 모순되는 추천만 막는다
                veto = None
                if act in ("엘리베이터 타기", "엘리베이터로 돌아가기") and st["floor"] == tf:
                    veto = f"이미 {tf}층(목적지 층)"
                elif act in ("목적지 접근", "도착") and target not in rooms:
                    veto = "목적지 문패를 아직 못 봄"
                elif act == "엘리베이터로 돌아가기" and not st["wrong_floor"]:
                    veto = "다른 층이라는 근거 없음"
                elif act == "엘리베이터 타기" and st["floor"] is None and st["el_ready"] is None:
                    veto = "엘리베이터가 확인되지 않음"
                elif act is None:
                    veto = "답 없음"
                advice = {"action": act, "why": ask_why, "veto": veto}
                # 추천을 말로 하는 건 추천 모델을 따로 줬을 때만 (4B는 다음 행동 고르기가 약해서 기록만 남김, docs/10 3-3절)
                speak = bool(getattr(runner.args, "advise_model", None))
                advice["spoken"] = False
                if speak and veto is None and act in ACTION_SAY and not any(("엘리베이터" in x or "돌아" in x) for x in say):
                    advice["spoken"] = True
                    say.append(ACTION_SAY[act])
        for x in say:
            self.mem.event(clock, "안내: " + x)
        rec = {**note,
            "route_target": target, "seq": seq, "sec": sec, "t": clock, "key_frame": obs["used"][0],
            "used_frames": obs["used"], "ambiguous": obs["ambiguous"], "rooms": rooms,
            "vlm": vlm, "vlm_called": len(vlm), "state": json.loads(json.dumps({k: v for k, v in st.items() if k != "floor_rooms"})),
            "say": say, "advice": advice, "stage1_seconds": obs["stage1_seconds"],
            # 팀 형식(시각정보) 내보내기용: 확정 호실의 방향, 대표 장면의 층표시·랜드마크 (video_team_format.py)
            "seen": {"plates": {x: perception.clock(obs["src"][x][0]["cx"]) for x in rooms if x in obs["src"]},
                     "floors": s1["result"]["floors"], "landmarks": s1["result"]["landmarks"],
                     "directions": s1["result"]["directions"]},
            "vlm_seconds": sum(v.get("vlm_seconds", 0) for v in vlm),
            "raw": obs["raw"],
        }
        rec["latency_seconds"] = rec["stage1_seconds"] + rec["vlm_seconds"]
        print(f"[{target}] {seq[:3]} {sec:02d}s 장{len(obs['used'])} VLM{len(vlm)} {rooms} "
              f"{rec['latency_seconds']:.2f}s | {' / '.join(say)}", flush=True)
        self.records.append(rec)
        self.clock += 1
        self.sec += 1
        return rec


def run_route(runner, root, target, seq_names, out, mode="rule"):
    nav = Navigator(runner, target, mode)
    for seq in seq_names:
        nav.new_segment(seq)
        files = sorted((Path(root) / seq).glob("i*.jpg"))
        for frames in [files[i:i + DENSE] for i in range(0, len(files), DENSE)]:
            if nav.st["arrived"]:
                break
            nav.step(frames)
    records = nav.records
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = [r["latency_seconds"] for r in records]
    print(f"[{target}] {len(records)}초, VLM {sum(r['vlm_called'] for r in records)}번, 다시 본 구간 "
          f"{sum(len(r['used_frames']) > 1 for r in records)}, 시간 중앙값 {statistics.median(lat):.2f}s 최대 {max(lat):.2f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--route", action="append", required=True, help="목적지=순서1,순서2 …")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11436")
    ap.add_argument("--mode", default="rule", choices=["rule", "c", "lean", "situ"])
    ap.add_argument("--advise-url", help="situ 다음 행동 추천에 쓸 다른 VLM 주소 (기본: --url)")
    ap.add_argument("--advise-model", help="situ 다음 행동 추천에 쓸 다른 모델 (기본: --model)")
    ap.add_argument("--key", default="sharp", choices=["first", "sharp"],
                    help="1초 대표 장면: first = 첫 장, sharp = 가장 선명한 장 (lsh 방식)")
    args = ap.parse_args()
    runner = Runner(args)
    first = sorted(Path(args.frames).glob("*/i0000.jpg"))[0]
    runner.warm(first)
    for spec in args.route:
        target, seqs = spec.split("=")
        seq_names = seqs.split(",")
        name = "_".join(s.split("_")[0] for s in seq_names)
        suffix = "" if args.mode == "rule" else f"_{args.mode}"
        suffix += "_sharp" if args.key == "sharp" else ""
        run_route(runner, args.frames, target, seq_names, Path(args.out_dir) / f"guide_{target}_{name}{suffix}.jsonl",
                  args.mode)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
