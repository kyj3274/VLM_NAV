# src/kyj: 실내 길안내 (kyj)

시각장애인용 실내 길안내. 휴대폰 웹이 1초마다 사진을 서버로 보내고, 서버가 YOLOE·OCR·Places365와 규칙으로 안내 문장을 만들어 돌려준다. VLM(qwen3-vl 4B)은 층 충돌처럼 드문 사건 때만 부른다.

## 폴더

```
src/kyj/
  nav/      서비스: 시연 웹이 실제로 쓰는 코드 (여기서 실행)
  tools/    도구: GPU 잠금, Gemini 중계, 시연 기록 다시 돌리기
  exp/      실험: 시기·주제별로 묶은 지난 실험 스크립트 (기록용)
```

## nav/ 서비스

| 파일 | 하는 일 |
|---|---|
| `web_mac.sh` | **Mac에서 실행.** 서버 웹 켜기·끄기, https 주소, 기록 보기 (`start`, `stop`, `status`, `records`, `record`) |
| `web_server.py` | 서버 웹. 휴대폰 사진을 받아 `navigator`로 넘기고 안내 문장을 돌려줌. 시연마다 `web_sessions/<시각>_<목적지>/`에 사진·`log.jsonl`·`memory.json` 저장 |
| `web/index.html` | 휴대폰 화면. 촬영(12장 중 선명한 1장), 말하기(목적지·명령 음성 입력), 안내 음성 |
| `web_replay.py` | 시연 기록을 다시보기 HTML로 |
| `navigator.py` | 길안내 본체. `Runner`(인식 실행), `Navigator`(1초마다 상태 갱신과 안내 문장). 영상 경로로도 실행 가능 |
| `nav_memory.py` | 시간 순서 기억: 문패 믿음 점수(읽힌 횟수·크기·문 옆·가까운 번호), 장소 흐름, 사건 기록 |
| `perception.py` | YOLOE 물체 인식과 1단계 합치기(rooms, floors, landmarks …) |
| `ocr_reader.py` | PaddleOCR 불러오기·읽기 (한국어 mobile 모델, 긴 변 2048) |
| `room_rules.py` | OCR 글자에서 호실 번호 고르기 (첫 자리 다수결 등) |
| `scene_places.py` | Places365 장소 분류 (엘리베이터 문 판단에 씀) |
| `vlm_client.py` | 1단계 실행 묶음(`stage1`), 인식 요약 글(`hints`), VLM 호출(`ask`) |

서비스 모듈끼리 이름으로 서로 부르므로 `nav/` 안에서 실행한다.

```bash
# Mac
bash src/kyj/nav/web_mac.sh start
# 서버에서 영상 경로로 돌리기
cd /workspace/capstone/src/kyj/nav
CUDA_VISIBLE_DEVICES=3 ../.venv/bin/python navigator.py --mode situ --frames <frames_8fps> --out-dir <폴더> --route "314=1_start_졸프_영상3,2b_after_elevator_졸프_영상2"
```

모델 파일: YOLOE는 `src/kyj/models/`(서버) 또는 저장소의 `local_models/`(Mac), Places365는 `models/places365/` 또는 `local_models/places365/`.

## tools/

| 파일 | 하는 일 |
|---|---|
| `replay_sessions.py` | 시연 기록의 사진을 지금 코드에 그대로 다시 넣어 안내가 어떻게 달라지는지 확인 (서버) |
| `gpu_lock.sh` | GPU 동시 사용 막기 |
| `gemini_proxy.py` | Ollama 형식 요청을 Gemini로 중계 (Mac 전용, 키는 Mac에만) |
| `q35_server.py` | qwen3.5 시험용 서버 (10월, 버림) |

## exp/ 실험 (날짜순)

| 폴더 | 내용 | 정리 문서 |
|---|---|---|
| `1_photo_2610/` | 사진 1장 파이프라인: 모델 비교, OCR 설정(실험 A), 호실 규칙(B·C), VLM 고르기(D·E), 최종 파이프라인. 옛 이름의 모듈(`pipeline`, `exp_a_ocr`, `run_task` …)도 여기 그대로 있어 서로 부름 | `docs/06`~`09`, `docs/exp/`, 이 폴더의 `README.md` |
| `2_video_2610/` | 영상 길안내: 판단 지점 17개 평가(`video_reason`), 프레임 정답표, 팀 형식 내보내기, 데모 HTML | `docs/10`~`13` |
| `3_recog_2611/` | 3층·4층 객체인식 비교(`team_compare`), 해상도·전송량(`res_ab`, `send_ab`), 문패 잘라 읽기(`crop_reread`), YOLOE·OCR 엮기(`fuse_ab`), 다수결 모의(`vote_collect`) | `docs/14` |
| `4_reason_2611/` | 텍스트 모델 추론: 기억 글로 판단(`reason_memory`), 시계열 기억으로 판단(`reason_series`), 걸음마다 기억 뽑기(`memory_snapshots`) | `docs/15` |

`2_`~`4_`의 스크립트는 맨 위에서 `nav/`를 경로에 넣고 옛 이름을 별칭으로 쓴다(`import perception as pipeline` 등). 본문은 실험 당시 그대로다.

## 데이터 (저장소 `data/video/`)

| 경로 | 내용 |
|---|---|
| `raw/` | 원본 영상 3개 |
| `frames/`, `frames_8fps/` | 1초 1장, 1초 8장 프레임 |
| `checkpoints.json` | 판단 지점 17개 정답 |
| `frame_gt.json` | 프레임 정답표 |
| `results/` | 실험 결과(jsonl, log, 데모 HTML) |
| `deliverable/` | 공유용 경로 영상 |
