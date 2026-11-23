"""목적지 도착 추정 (인식과 무관한 공통 판단, Claude 버전과 로컬 버전이 같이 쓴다).

문패는 문 앞에 가까이 갈수록 화면 밖으로 빠져서(옆벽·문 위) 계속 보이지 않는다 (11/20 410호: 3m에서 본 뒤 사라짐).
그래서 목적지를 본 거리·방향을 기억하고, 그 뒤 걸은 거리로 남은 거리를 추정한다.
  남은 거리 ≤ CONFIRM_AT → "멈춰 주세요. ○호 문이 왼쪽에 있을 것 같습니다. 왼쪽을 비춰 주세요" (확인 중에는 도착 거리를 넉넉히)
  ASK_WAIT 안에 문패가 안 읽히면 → "○호일 가능성이 높습니다"
'근처'라고 한 뒤 목적지를 한 번도 못 보고 NOT_FOUND가 지나면 → "찾지 못했습니다 … 주변 분께 물어봐 주세요" (11/20 105호는 없는 방)
걸은 거리: 휴대폰 걸음 수가 있으면 걸음 × STEP_M, 없으면 시간 × WALK_MPS.
"""

NOT_FOUND = 40    # '근처'라고 한 뒤 이 초 동안 목적지 문패가 안 보이면 못 찾았다고 알림
WALK_MPS = 0.8    # 센서가 없을 때 걷는 속도 가정 (m/s)
STEP_M = 0.6      # 걸음 하나 거리 (휴대폰 걸음 수가 있을 때)
CONFIRM_AT = 1.5  # 남은 거리(추정)가 이 안이면 멈추고 옆을 비춰 확인하게 함
ASK_WAIT = 8      # 옆을 비춰 달라고 한 뒤 이 초 안에 문패가 안 읽히면 '가능성 높음'으로 알림
ARRIVE_M = 2.0    # 도착 거리 (확인 중에는 ARRIVE_ASK_M)
ARRIVE_ASK_M = 2.5


class TargetTracker:
    def __init__(self, target, event=None):
        self.target = target
        self.event = event or (lambda text: None)
        self.reset()

    def reset(self):
        self.mem = None          # 목적지를 마지막으로 본 때 {"t", "steps", "d", "side"}
        self.ask_t = None        # 옆을 비춰 달라고 한 때
        self.done = False        # 확인 요청을 끝냄 (가능성 안내까지 함)
        self.near_t = None       # '근처'라고 처음 말한 때
        self.notfound_said = False

    @staticmethod
    def walked(mem, now, steps):
        """mem 뒤로 걸은 거리(m): 걸음 수가 있으면 그것, 없으면 시간 × 걷는 속도."""
        if steps is not None and mem.get("steps") is not None:
            return (steps - mem["steps"]) * STEP_M
        return (now - mem["t"]) * WALK_MPS

    def arrive_m(self):
        """도착으로 볼 거리: 옆을 비춰 확인 중이면 조금 넉넉히."""
        return ARRIVE_ASK_M if self.ask_t is not None else ARRIVE_M

    def seen(self, now, d, side, steps=None):
        """목적지 문패를 이 거리·방향(왼쪽/오른쪽/정면)에서 봄."""
        if d is not None:
            self.mem = {"t": now, "steps": steps, "d": d, "side": side}

    def cancel(self):
        """목적지를 지나쳤다고 판단함: 위치 추정을 버림 (다시 보면 새로 시작)."""
        self.mem, self.ask_t, self.done = None, None, False

    def near(self, now):
        """'근처'라고 말함."""
        if self.near_t is None:
            self.near_t = now

    def unseen(self, now, steps=None):
        """이번 걸음에 목적지 문패가 안 보임. 할 말 목록."""
        out, t = [], self.target
        if self.mem is not None and not self.done:
            left = self.mem["d"] - self.walked(self.mem, now, steps)
            side = self.mem["side"] or "옆"
            if self.ask_t is None and left <= CONFIRM_AT:
                out.append(f"멈춰 주세요. {t}호 문이 {side}에 있을 것 같습니다. 휴대폰을 {side}으로 천천히 돌려 문패를 비춰 주세요.")
                self.ask_t = now
                self.event(f"목적지 위치 추정: {self.mem['d']:.1f}m에서 본 뒤 약 {self.mem['d'] - left:.1f}m 걸음 → 확인 요청")
            elif self.ask_t is not None and now - self.ask_t >= ASK_WAIT:
                out.append(f"{t}호 문패를 확인하지 못했습니다. {side} 바로 앞의 문이 {t}호일 가능성이 높습니다.")
                self.done = True
        elif self.mem is None and self.near_t is not None and not self.notfound_said and now - self.near_t >= NOT_FOUND:
            out.append(f"{t}호 문패를 아직 찾지 못했습니다. 이 근처에 없을 수도 있습니다. "
                       "양쪽 벽을 천천히 비춰 보시거나 주변 분께 물어봐 주세요.")
            self.notfound_said = True
        return out
