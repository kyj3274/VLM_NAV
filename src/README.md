# 소스 코드

길안내 서비스와 보조 도구의 소스 코드입니다. 전체 개요는 [저장소 README](../README.md)를 참고하세요.

```
src/
├── nav/      길안내 서비스 (실행 위치)
├── tools/    운영 도구
└── eval/     평가 스크립트
```

---

## `nav/` 길안내 서비스

서비스 모듈은 서로를 모듈 이름으로 불러오므로 `nav/` 폴더에서 실행합니다.

### 웹

| 파일 | 역할 |
|---|---|
| `web_mac.sh` | Mac에서 실행하는 제어 스크립트. 서버 웹 실행·종료, 포트 연결, 휴대폰용 https 주소 발급, 시연 기록 조회 |
| `web_server.py` | 서버 웹. 휴대폰 사진을 받아 `navigator`에 넘기고 안내 문장을 반환. 시연마다 사진, `log.jsonl`, `memory.json`을 저장 |
| `web/index.html` | 휴대폰 화면. 촬영(1초 12장 중 선명한 1장), 음성 입력(목적지·명령), 음성 안내 |
| `web_replay.py` | 시연 기록을 다시보기 HTML로 변환 |

`web_mac.sh` 명령:

| 명령 | 동작 |
|---|---|
| `start` | 서버 웹 실행 → Mac 포트 연결 → https 주소 출력 |
| `stop` | https 주소와 서버 웹 종료 (GPU 반납) |
| `status` | 실행 상태 확인 |
| `records` | 최근 시연 기록 목록 |
| `record [세션]` | 시연 기록을 다시보기 HTML로 열기 |

### 길안내

| 파일 | 역할 |
|---|---|
| `navigator.py` | 길안내 본체. `Runner`(인식 실행)와 `Navigator`(1초마다 상태 갱신·안내 문장 생성). 녹화 영상으로도 실행 가능 |
| `nav_memory.py` | 시간 순서 기억. 문패별 믿음 점수, 장소 흐름, 사건 기록 |

**문패 믿음 점수** (`nav_memory.py`)

| 근거 | 점수 |
|---|---|
| 읽힌 횟수 (최근 6걸음 안) | 회당 +1, 최대 3 |
| 크고 또렷함 (점수 0.95 이상, 가까운 글자 크기) | +1 |
| YOLOE 문 상자 옆에 있음 | +1 |
| 이미 확정한 같은 층 번호와 10 이내 | +1 |
| **확정 기준** | **3점 이상** |

### 인식

| 파일 | 역할 | 설정 |
|---|---|---|
| `perception.py` | YOLOE 물체 인식과 1단계 결과 합치기 (rooms, floors, landmarks, directions) | `yoloe-11l-seg`, 입력 1280, 기준 0.15 |
| `ocr_reader.py` | PaddleOCR 불러오기와 읽기 | PP-OCRv5 mobile 위치 모델 + 한국어 인식 모델, 긴 변 2048 |
| `room_rules.py` | OCR 글자에서 호실 번호 고르기 | 첫 자리 다수결 |
| `scene_places.py` | Places365 장소 분류 | ResNet18, 엘리베이터 문 확률 0.5 이상 2초 연속 |
| `vlm_client.py` | 1단계 실행 묶음, 인식 결과 요약 글, VLM 호출 | Ollama `qwen3-vl:4b-instruct` |

---

## `tools/` 운영 도구

| 파일 | 역할 |
|---|---|
| `replay_sessions.py` | 시연 기록의 사진을 현재 코드로 다시 실행해 안내 변화를 비교 (회귀 확인용) |
| `gpu_lock.sh` | 같은 GPU에서 실험이 동시에 돌지 않도록 순서 보장 |

## `eval/` 평가

| 파일 | 역할 |
|---|---|
| `memory_snapshots.py` | 녹화 영상을 실행하며 걸음마다 기억(사건 기록)을 저장 |
| `reason_series.py` | 저장한 기억을 텍스트 모델에 넣어 판단 지점 17개에서 다음 행동을 채점 |

---

## 데이터와 모델 위치

| 경로 | 내용 |
|---|---|
| `data/video/checkpoints.json` | 판단 지점 17개 정답 |
| `data/video/frame_gt.json` | 프레임별 정답표 |
| `data/video/results/` | 실험 결과 (jsonl, log, 데모 HTML) |
| `data/video/raw/`, `frames/`, `frames_8fps/` | 원본 영상과 프레임 (저장소 미포함) |
| `local_models/` 또는 서버 `models/` | YOLOE, Places365 가중치 (저장소 미포함) |

## 서버 경로

공용 GPU 서버에서는 팀이 정한 개인 폴더를 사용합니다.

| 구분 | 경로 |
|---|---|
| 코드 | `/workspace/capstone/src/kyj/` (이 저장소의 `src/`와 같은 구조) |
| 결과 | `/workspace/capstone/outputs/kyj/` |
| 시연 기록 | `/workspace/capstone/outputs/kyj/web_sessions/<시각>_<목적지>/` |

코드에 남아 있는 서버 경로의 `kyj`는 이 개인 폴더 이름입니다.
