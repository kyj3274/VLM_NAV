"""목적지 도착 추정 (인식과 무관한 공통 판단, Claude 버전과 로컬 버전이 같이 쓴다).

문패는 문 앞에 가까이 갈수록 화면 밖으로 빠져서(옆벽·문 위) 계속 보이지 않는다 (11/20 410호: 3m에서 본 뒤 사라짐).
그래서 목적지를 본 거리·방향을 기억하고, 그 뒤 걸은 거리로 남은 거리를 추정한다.
  남은 거리 ≤ CONFIRM_AT → "멈춰 주세요. ○호 문이 왼쪽에 있을 것 같습니다. 왼쪽을 비춰 주세요" (확인 중에는 도착 거리를 넉넉히)
  ASK_WAIT 안에 문패가 안 읽히면 → "○호일 가능성이 높습니다"
'근처'라고 한 뒤 목적지를 한 번도 못 보고 NOT_FOUND가 지나면 → "찾지 못했습니다 … 주변 분께 물어봐 주세요" (11/20 105호는 없는 방)
걸은 거리: 휴대폰 걸음 수가 있으면 걸음 × STEP_M, 없으면 시간 × WALK_MPS.
휴대폰 방위·걸음 수가 있으면 평면 위치로 (2026-11-25 시뮬레이터 로비): 본 방위·거리로 목적지 점을 찍고, 걸음마다 방위대로
위치를 옮겨 앞·옆 거리를 잰다. 복도에서는 옆 거리가 늘 작아 위와 같고, 넓은 곳에서 옆으로 REACH보다 멀면
'N시 방향 약 M미터'로 그쪽으로 가게 한다 (일직선 추정은 12m 밖 10시 방향 간판을 '12m 가면 옆'으로 잘못 봄).
"""

import math

NOT_FOUND = 40    # '근처'라고 한 뒤 이 초 동안 목적지 문패가 안 보이면 못 찾았다고 알림
WALK_MPS = 0.8    # 센서가 없을 때 걷는 속도 가정 (m/s)
STEP_M = 0.6      # 걸음 하나 거리 (휴대폰 걸음 수가 있을 때)
CONFIRM_AT = 1.5  # 남은 거리(추정)가 이 안이면 멈추고 옆을 비춰 확인하게 함
ASK_WAIT = 8      # 옆을 비춰 달라고 한 뒤 이 초 안에 문패가 안 읽히면 '가능성 높음'으로 알림
ARRIVE_M = 2.0    # 도착 거리 (확인 중에는 ARRIVE_ASK_M)
ARRIVE_ASK_M = 2.5
TURN_DEG = 135    # 휴대폰 방위가 이만큼 넘게 바뀌면 돈 것으로 봄 (옆을 비추는 120°까지는 안 돎, nav_graph도 같이 씀)
SIDE_DEG = {"왼쪽": -20, "오른쪽": 20}  # 화면 왼쪽·오른쪽에 보인 목적지는 카메라 방향에서 대략 이만큼 (정면·옆은 0)
ALONG_AT = 1.0    # 평면 위치로는 앞으로 남은 거리가 이 안이면 확인 요청 (CONFIRM_AT은 옆 거리까지 든 직선 거리라 복도 옆 1.2m면 앞 약 0.9m)
SIDE_MIN = 0.6    # 평면 위치의 옆 거리가 이보다 작으면 왼쪽·오른쪽을 단정하지 않음 (마주 본 벽의 문: 방위가 10°만 흔들려도 부호가 바뀜,
                  # 시뮬레이터 스트레스 '센서 오차'에서 반대쪽을 말함). 복도 문은 옆 약 1.2m라 해당 없음
REACH = 2.5       # 목적지가 걷는 방향에서 옆으로 이 안이면 '지나갈 때 옆에 있음'(복도), 밖이면 그쪽으로 가게 함 (넓은 곳)


def josa(word, a, b):
    """받침이 있으면 a(은·이·을), 없으면 b(는·가·를). 한글로 끝나지 않으면 b ('410호'는 '호'로 끝남)."""
    c = word[-1:] or " "
    return a if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else b


def _unit(deg):
    r = math.radians(deg)
    return math.sin(r), math.cos(r)


class TargetTracker:
    def __init__(self, target, event=None):
        self.target = target
        self.ho = "호" if target.isdigit() else ""  # '410호', 이름 목적지(세미나실)는 그대로
        self.event = event or (lambda text: None)
        self.reset()

    def reset(self):
        self.mem = None          # 목적지를 마지막으로 본 때 {"t", "steps", "d", "side"}
        self.ask_t = None        # 옆을 비춰 달라고 한 때
        self.ask_side = None     # 그때 비춰 달라고 한 쪽 (몸 기준 '왼쪽'/'오른쪽'/'옆')
        self.done = False        # 확인 요청을 끝냄 (가능성 안내까지 함)
        self.near_t = None       # '근처'라고 처음 말한 때
        self.notfound_said = False
        self.pos, self.steps = (0.0, 0.0), None  # 걸음 수·방위로 옮긴 평면 위치 (m)

    def _move(self, steps, heading):
        """걸음 수가 늘었으면 그만큼 지금 방위로 위치를 옮김 (멈춰 옆을 비추는 동안은 걸음이 안 늘어 안 옮김)."""
        if steps is None or heading is None:
            return
        if self.steps is not None and steps > self.steps:
            ux, uy = _unit(heading)
            d = (steps - self.steps) * STEP_M
            self.pos = (self.pos[0] + ux * d, self.pos[1] + uy * d)
        self.steps = max(steps, self.steps or 0)

    @staticmethod
    def walked(mem, now, steps):
        """mem 뒤로 걸은 거리(m): 걸음 수가 있으면 그것, 없으면 시간 × 걷는 속도."""
        if steps is not None and mem.get("steps") is not None:
            return (steps - mem["steps"]) * STEP_M
        return (now - mem["t"]) * WALK_MPS

    def arrive_m(self):
        """도착으로 볼 거리: 옆을 비춰 확인 중이면 조금 넉넉히."""
        return ARRIVE_ASK_M if self.ask_t is not None else ARRIVE_M

    def seen(self, now, d, side, steps=None, sure=True, heading=None):
        """목적지 문패를 이 거리·방향(왼쪽/오른쪽/정면)에서 봄. sure=False: 확정 전 읽기 (위치는 기억하지만
        옆을 비춰도 안 읽히면 '가능성 높음'으로 도착을 말하지 않음, 2026-11-25 시뮬레이터 v2)."""
        if self.near_t is not None:
            self.near_t = now  # 못 찾음 시계는 마지막으로 본 때부터
        if d is not None:
            sure = sure or bool(self.mem and self.mem.get("sure"))
            spot = None
            if steps is not None and heading is not None:
                self._move(steps, heading)
                ux, uy = _unit(heading + SIDE_DEG.get(side, 0))  # heading은 사진 때 휴대폰 방위 (옆을 비추면 그쪽)
                spot = (self.pos[0] + ux * d, self.pos[1] + uy * d)
            if self.asking() and self.ask_side in ("왼쪽", "오른쪽"):
                side = self.ask_side  # 옆을 비추는 중: 화면 속 위치가 아니라 비추라고 한 몸 기준 쪽 (시뮬레이터 로비에서 반대로 말함)
            self.mem = {"t": now, "steps": steps, "d": d, "side": side, "sure": sure, "heading": heading, "spot": spot}

    def _unseen_2d(self, now, steps, heading):
        """평면 위치로: 목적지 점까지 앞(along)·옆(lat, 오른쪽 +) 거리."""
        out, t, mem = [], self.target, self.mem
        self._move(steps, heading)
        dx, dy = mem["spot"][0] - self.pos[0], mem["spot"][1] - self.pos[1]
        ux, uy = _unit(heading)
        along, lat, r = dx * ux + dy * uy, dx * uy - dy * ux, math.hypot(dx, dy)
        if self.ask_t is None and along < 0 and r > CONFIRM_AT + 1:  # 등 뒤 (본 뒤 돌았거나 지나침)
            if mem.get("said") != "back":
                out.append(f"{t}{self.ho}{josa(t + self.ho, '은', '는')} 뒤쪽 약 {max(1, round(r))}미터입니다. 뒤로 돌아 천천히 가세요.")
                self.event("목적지가 등 뒤 (평면 위치) → 뒤로 돌게 함")
                mem["said"] = "back"
            return out
        if self.ask_t is None and abs(lat) > REACH:  # 걷는 줄에서 옆으로 멀리: 그쪽으로 가게 함 (넓은 곳)
            k = round(math.degrees(math.atan2(lat, along)) / 30) % 12 or 12
            if mem.get("said") != k:
                out.append(f"{t}{self.ho}{josa(t + self.ho, '은', '는')} {k}시 방향 약 {max(1, round(r))}미터입니다. 그쪽으로 천천히 가세요.")
                self.event(f"목적지가 옆으로 {abs(lat):.1f}m (평면 위치) → {k}시 방향")
                mem["said"] = k
            return out
        side = ("오른쪽" if lat > 0 else "왼쪽") if abs(lat) >= SIDE_MIN else "정면"
        if self.ask_t is None and along <= ALONG_AT:
            if side == "정면":  # 옆 거리가 작음: 쪽을 단정하지 않고 좌우를 다 비추게 함 (모름 = 안전한 실패)
                out.append(f"멈춰 주세요. {t}{self.ho} 문이 바로 앞 가까이에 있을 것 같습니다. 휴대폰을 좌우로 천천히 비춰 주세요.")
            else:
                out.append(f"멈춰 주세요. {t}{self.ho} 문이 {side}에 있을 것 같습니다. 휴대폰을 {side}으로 천천히 돌려 문패를 비춰 주세요.")
            self.ask_t, self.ask_side = now, side
            self.event(f"목적지 위치 추정: 앞 {along:.1f}m, 옆 {lat:+.1f}m → 확인 요청")
        elif self.ask_t is not None and now - self.ask_t >= ASK_WAIT:
            if mem.get("sure", True):
                out.append(f"{t}{self.ho} 문패를 확인하지 못했습니다. {self.ask_side} 바로 앞의 문이 {t}{self.ho}일 가능성이 높습니다.")
                self.done = True
            else:
                out.append(f"{t}{self.ho} 문패를 확인하지 못했습니다. 문패를 확인하며 천천히 가세요.")
                self.mem, self.ask_t = None, None
        return out

    def cancel(self):
        """목적지를 지나쳤다고 판단함: 위치 추정을 버림 (다시 보면 새로 시작)."""
        self.mem, self.ask_t, self.done = None, None, False

    def retry(self):
        """옆을 비췄더니 목적지가 아직 앞에 있음: 확인 요청을 풀고 다시 걷게 함 (가까워지면 다시 요청)."""
        self.ask_t, self.ask_side, self.done = None, None, False

    def asking(self):
        """옆을 비춰 확인해 달라고 했음 (사용자는 멈춰 있음, '가능성 높음'까지 말한 뒤도 포함). 다시 보거나 cancel()이면 풀림."""
        return self.ask_t is not None

    def near(self, now):
        """'근처'라고 말함."""
        if self.near_t is None:
            self.near_t = now

    def unseen(self, now, steps=None, heading=None):
        """이번 걸음에 목적지 문패가 안 보임. 할 말 목록. heading: 지금 휴대폰 방위 (본 때와 비교해 돌았는지)."""
        out, t = [], self.target
        if self.mem is not None and not self.done and self.mem.get("spot") and steps is not None and heading is not None:
            return self._unseen_2d(now, steps, heading)
        if self.mem is not None and not self.done:
            left = self.mem["d"] - self.walked(self.mem, now, steps)
            side = self.mem["side"] or "옆"
            h0 = self.mem.get("heading")
            if h0 is not None and heading is not None and abs((heading - h0 + 180) % 360 - 180) > TURN_DEG:
                # 목적지를 본 뒤 돌았음: 목적지는 이제 뒤쪽. 옆을 비추게 하면 반대쪽 벽을 보게 된다 (시뮬레이터, 이유 없이 도는 사용자).
                # 지금 자리를 기준으로 남은 거리를 다시 잡고, 다시 돌아 원래 방향이 되면 이어서 추정한다.
                if not self.mem.get("back_said"):
                    out.append(f"{t}{self.ho}{josa(t + self.ho, '은', '는')} 뒤쪽 약 {max(1, round(max(left, 0)))}미터입니다. 뒤로 돌아 천천히 가세요.")
                    self.event("목적지를 본 뒤 돈 것을 확인 → 뒤로 돌게 함")
                self.mem = {**self.mem, "t": now, "steps": steps, "d": max(left, 0), "back_said": True}
                self.ask_t, self.ask_side = None, None
                return out
            self.mem.pop("back_said", None)
            if self.ask_t is None and left <= CONFIRM_AT:
                if side in ("정면", "앞"):  # 정면에서 본 문: '정면으로 돌려'가 아니라 좌우로 비추게 (휴대폰 시험 3 재생)
                    out.append(f"멈춰 주세요. {t}{self.ho} 문이 바로 앞 가까이에 있을 것 같습니다. 휴대폰을 좌우로 천천히 비춰 주세요.")
                else:
                    out.append(f"멈춰 주세요. {t}{self.ho} 문이 {side}에 있을 것 같습니다. 휴대폰을 {side}으로 천천히 돌려 문패를 비춰 주세요.")
                self.ask_t, self.ask_side = now, side
                self.event(f"목적지 위치 추정: {self.mem['d']:.1f}m에서 본 뒤 약 {self.mem['d'] - left:.1f}m 걸음 → 확인 요청")
            elif self.ask_t is not None and now - self.ask_t >= ASK_WAIT:
                if self.mem.get("sure", True):
                    out.append(f"{t}{self.ho} 문패를 확인하지 못했습니다. {side} 바로 앞의 문이 {t}{self.ho}일 가능성이 높습니다.")
                    self.done = True
                else:  # 확정 전 읽기뿐: 도착이라 하지 않고 다시 찾게 함
                    out.append(f"{t}{self.ho} 문패를 확인하지 못했습니다. 문패를 확인하며 천천히 가세요.")
                    self.mem, self.ask_t = None, None
        elif self.mem is None and self.near_t is not None and not self.notfound_said and now - self.near_t >= NOT_FOUND:
            out.append(f"{t}{self.ho} 문패를 아직 찾지 못했습니다. 이 근처에 없을 수도 있습니다. "
                       "양쪽 벽을 천천히 비춰 보시거나 주변 분께 물어봐 주세요.")
            self.notfound_said = True
        return out
