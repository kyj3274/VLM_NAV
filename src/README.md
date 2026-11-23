# 소스 코드

길안내 서비스와 보조 도구의 소스 코드입니다. 전체 개요는 [저장소 README](../README.md)를 참고하세요.

```
src/
├── nav/            길안내 서비스 (Claude 버전, 실행 위치)
│   ├── core/       공통 판단: 인식과 무관하며 두 버전이 같이 씁니다
│   └── local/      로컬 버전 (YOLOE + PaddleOCR + Places365, 학교 서버 GPU)
├── tools/          운영 도구
└── eval/           평가 스크립트
```

## 구조: 인식은 두 가지, 판단은 하나

```
인식 (갈아 끼움)                                   공통 판단 (core)                        안내
┌ Claude 버전: Claude Haiku 1회 (nav/)        ┐                                                    
│   문패·층 표지·장소·엘리베이터·화살표        │ →  문패 읽기 → 복도 고정 그래프 → 확률 방향 → 안내 문장
└ 로컬 버전: YOLOE + OCR + Places365 (local/) ┘     깊이(벽 쪽·거리)   뒤돌기 추정   도착 추정
```

두 인식기 모두 "문패 번호, 화면 위치(상자), 거리"를 내놓으므로 그 뒤의 판단(`core/`)을 공유합니다. 문패 확정(믿음 점수)과 층·엘리베이터·계단 규칙은 인식기에 맞춰 각 버전에 있습니다.

---

## `nav/core/` 공통 판단

| 파일 | 역할 |
|---|---|
| `nav_graph.py` | **복도 고정 그래프와 확률 방향 판단.** 문패를 복도 기준 벽(A/B)과 위치에 놓고, "걷는 방향으로 번호 증가/감소" 가설을 갱신해 P(앞쪽)을 냅니다. "뒤로 돌아 주세요" 뒤에는 "돌았다/안 돌았다" 가설로 돌았는지 판단하고, 확정한 목적지가 뒤에 있으면 지나쳤다고 봅니다 |
| `arrival.py` | **도착 추정.** 목적지를 본 거리·방향을 기억하고 걸은 거리(걸음 수 또는 시간)로 남은 거리를 추정해, 가까우면 "멈춰 주세요, 왼쪽을 비춰 주세요"라고 합니다. 확인이 안 되면 "가능성이 높습니다", 근처에서 오래 못 찾으면 "찾지 못했습니다"라고 안내합니다 |
| `depth.py` | **깊이 모델** (Depth Anything V2 Metric Indoor small). 가장 먼 열 = 복도 방향 기준으로 문패의 벽 쪽, 문패까지 거리를 잽니다 |

판단 기준값

| 값 | 뜻 |
|---|---|
| `FLOW_EPS = 0.1` | 같은 벽 이웃 문패의 증감이 실제 흐름과 반대일 확률 (번호 건너뜀·오독) |
| `DIR_SURE = 0.7` | P(앞) 0.7 이상 앞쪽, 0.3 이하 뒤쪽, 그 사이는 모름 (틀리는 대신 늦어짐) |
| `CONFIRM_AT = 1.5` | 남은 거리(추정)가 1.5m 안이면 옆을 비춰 확인 |
| `NOT_FOUND = 40` | "근처"라고 한 뒤 40초 동안 목적지를 못 보면 알림 |

## `nav/` Claude 버전 (기본)

| 파일 | 역할 |
|---|---|
| `web_mac.sh` | Mac 제어 스크립트. `start`(Claude), `start --fake`(가짜 응답, 크레딧 0), `start --depth`(깊이 켬), `stop`, `status`, `local …`(로컬 버전으로 넘김) |
| `web_server.py` | FastAPI 웹 (Mac에서 실행). 겹쳐 보내기, 접속 코드, 크레딧 한도, 관찰자 화면 `/monitor`, 기록 `/records` |
| `web/` | 휴대폰 화면 (`app.js`, `sensors.js`: 방위·걸음 수로 즉시 안내, 사진마다 걸음 수·방위를 서버로 보냄) |
| `navigator.py` | `Runner`(Claude 인식, 깊이 선택)와 `Navigator`(층·엘리베이터·계단·목적지 규칙 + `core` 판단) |
| `nav_memory.py` | 문패 믿음 점수 (여러 호출, 정면 3m, 문 옆, 가까운 번호, 확대 재확인). 읽기마다 `core` 그래프에도 넣음 |
| `claude_perceiver.py`, `claude_client.py` | Claude 인식 프롬프트·스키마, 학교 게이트웨이 호출, 캐시 |
| `dest_parse.py`, `stt.py` | 목적지 말 해석, Mac 로컬 받아쓰기 |
| `fake_scene.py`, `live_sim.py`, `web_replay.py` | 가짜 응답, 실시간 흉내, 기록 다시보기 |

실행: `.env`에 `BAZE_API_KEY`를 넣고 `bash src/nav/web_mac.sh start`. 기록·영상은 캐시(`outputs/cache`)로 크레딧 없이 다시 돌릴 수 있습니다 (`eval/score_checkpoints.py`, `eval/replay_web_session.py`).

## `nav/local/` 로컬 버전 (비교·예비)

| 파일 | 역할 |
|---|---|
| `web_mac.sh` | 학교 서버 웹 실행·종료, 포트 연결(8766), https 주소, 기록 조회. `bash src/nav/web_mac.sh local start`로도 부름 |
| `web_server.py`, `web/index.html` | 서버 웹과 휴대폰 화면 (기다리는 동안 사진 고르기, 센서 기록) |
| `navigator.py`, `nav_memory.py` | 로컬 규칙과 믿음 점수 (읽힌 횟수, 크고 또렷함, 문 옆, 가까운 번호) + `core` 판단 |
| `perception.py`, `ocr_reader.py`, `room_rules.py`, `scene_places.py`, `vlm_client.py` | YOLOE, PaddleOCR(PP-OCRv5), 호실 판정, Places365, Ollama VLM (situ 모드에서는 VLM을 부르지 않음) |
| `replay_sessions.py` | 시연 기록 사진을 지금 코드로 다시 실행 (`NAV_SESSIONS`로 기록 폴더 지정) |

GPU가 있으면 CUDA, Mac에서는 MPS·CPU로 자동 선택합니다.

## `tools/`, `eval/`

| 파일 | 역할 |
|---|---|
| `tools/claude_probe.py`, `tools/latency_probe.py` | Claude 스키마 확인, 지연 원인 측정 |
| `tools/gpu_lock.sh` | 같은 GPU에서 실험이 겹치지 않게 순서 보장 |
| `eval/score_checkpoints.py` | 판단 지점 17개 채점 (Claude 버전, 캐시) |
| `eval/replay_web_session.py`, `eval/compare_410.py` | 휴대폰 기록 재생, 410 경로 비교 |
| `eval/layout_belief.py` | 확률 방향 판단 실험 (판단 지점 13개, 기준값 비교) |
| `eval/scene_graph.py` | 기록으로 장면 그래프 그림 그리기 (로컬 버전 기록) |
| `eval/memory_snapshots.py`, `eval/reason_series.py` | 기억 저장, 텍스트 모델 판단 실험 |

## 데이터와 모델 위치

| 경로 | 내용 |
|---|---|
| `data/video/checkpoints.json` | 판단 지점 17개 정답 |
| `data/video/results/` | 실험 결과 |
| `data/video/frames_8fps/` | 영상 프레임 (저장소 미포함) |
| `local_models/` 또는 서버 `models/` | YOLOE, Places365, 깊이 모델 가중치 (저장소 미포함) |
| `outputs/` | Claude 캐시, 웹 기록 (저장소 미포함) |

## 버전 표시

| 태그 | 내용 |
|---|---|
| `building1-claude` | 정리 전 Claude 버전 (2026-11-23 main) |
| `building1-local` | 정리 전 로컬 버전 (깊이·그래프·확률 판단, 2026-11-20) |

두 태그는 한 건물(실험 건물)에 맞춰 만든 상태를 그대로 보존합니다.
