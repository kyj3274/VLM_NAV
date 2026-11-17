# kyj: OCR 중심 실내 길안내 파이프라인 (YOLOE + PaddleOCR + VLM)

사진 1장에서 길안내 단서(fields: `rooms`, `floors`, `spaces`, `landmarks`, `directions`)를 뽑는다.
**OCR이 글자를 읽고, YOLOE가 물체 위치를 찾고, VLM은 잘라낸 조각을 보고 "고르기"만 한다.**
자세한 실험 기록: `docs/07_…`, `docs/08_…`, `docs/exp/` (이 폴더 안).

---

## 1. 구조

```
사진 ─┬─ YOLOE (0.14초) ── 문·계단·화장실·비상구표시·사람·상자 위치
      └─ OCR   (0.53초) ── 글자 + 위치 → 호실 후보 (3자리 숫자)
                    │
            [규칙] 호실: 사진 안 첫 자리 다수결, 큰 글자(가까운) 순
                   방향: 상자 위치로 계산 (왼쪽 1/3=11시, 가운데=12시, 오른쪽=1시)
                    │
            [VLM, 사진당 1번] 후보 조각만 보고 고르기 (JSON schema로 답 제한)
               R 호실 후보 → 예/아니요   O 물체 조각 → 무엇인지   S 표지 → 화살표 방향
               B (시험 중) 큰 문 양옆 → 엘리베이터 호출 버튼 있나
```

## 2. 결과 요약 (qwen3-vl:4b, 평가용 117장)

| 방식 | 호실 F1 | 가까운 호실 | 랜드마크 F1 | 화장실 F1 | 시간 중앙값 |
|---|---|---|---|---|---|
| qwen3-vl 단독 (run_task.py fields) | 32 | 11/23 | - | - | 4.7초 |
| qwen2.5vl:7b 단독 | 49 | 18/23 | - | - | 8.5초 |
| **pipeline_final.py** (R, 엘리베이터 E, S) | 89 | 23/23 | 74 | 30 | 3.8초 |
| **exp_obj_vlm.py --crop margin** (R, O, S) | 90 | 23/23 | 73 | 64 | 3.5초 |
| exp_obj_vlm.py --crop margin --buttons | 90 | 23/23 | 76 | 52 | 5.0초 |

- 채점 기준(`eval/survey_v2.json`)은 **사람 정답이 아니다** (사진 조사 + OCR 후보 확인, AI가 만듦). 모델끼리 비교용.
- "가까운 호실" = 문패 숫자 글자 높이가 사진 높이의 1.5% 이상.
- 엘리베이터는 아직 어렵다 (문 생김새로는 어떤 모델도 구분 못 함 → 호출 버튼으로 시험 중).

## 3. 파일 지도

**최종으로 쓰는 것**
| 파일 | 역할 |
|---|---|
| `pipeline_final.py` | 전체 실행 (YOLOE+OCR+규칙+VLM). 사진에서 결과까지 한 번에 |
| `exp_obj_vlm.py` | VLM 부분만 다른 방식으로 (물체 조각 O, 버튼 B). 1단계 결과(jsonl)를 재사용 |
| `pipeline.py` | YOLOE·OCR 불러오기, 1단계 합치기 규칙 |
| `room_rules.py` | 호실 후보·다수결 규칙 |
| `exp_d_vlm.py` | VLM 질문(R, E, S) 만들기·반영 |
| `exp_a_ocr.py`, `exp_bc_rules.py` | OCR 설정, 규칙 재계산 (위에서 import) |
| `run_task.py` | 공통 설정, 사진 목록, VLM 단독 fields/signs/perception/navigation 실행 |
| `score_v2.py` + `eval/` | 채점 (기준 `eval/survey_v2.json`, 조정용·평가용 나누기 `eval/split.json`) |
| `gpu_lock.sh` | 같은 GPU에서 작업이 겹치지 않게 순서대로 실행 |

**실험용 (참고만)**: `exp_a_crop.py`, `exp_bc_plate.py`, `exp_e_*.py`, `exp_e_run.sh`, `make_ref_crops.py`, `score_v2_adj.py`, `run_obj.sh`, `run_pipeline.sh`, `install_vision.sh`(환경 설치 기록)

**결과**: `/workspace/capstone/outputs/kyj/final/`에 최종 결과 3개 복사본. 나머지 `outputs/kyj/exp_*`는 중간 실험.

## 4. 환경

- 전용 Python: `/workspace/capstone/src/kyj/.venv` (torch 2.3.1+cu118, ultralytics 8.4, paddleocr 3.7 GPU). 팀 공용 `.venv`와 별개
- YOLOE 가중치: `src/kyj/models/` (yoloe-11l-seg + mobileclip)
- OCR 모델: `/root/.paddlex/official_models/` (PaddleOCR이 자동으로 받음)
- GPU: YOLOE·OCR은 **GPU 3**, VLM은 **GPU 7의 Ollama 0.24.0** (`http://127.0.0.1:11436`)
  - 기본 Ollama(11434)는 드라이버 문제로 CPU 전용이라 느림

## 5. 다른 모델로 돌리기

**결과는 꼭 자기 폴더로** (`--out-dir`). 안 주면 `outputs/kyj`에 섞인다.

```bash
cd /workspace/capstone/src/kyj

# (A) 전체 파이프라인 (176장, 약 10~15분)
./gpu_lock.sh 3 ./gpu_lock.sh 7 env CUDA_VISIBLE_DEVICES=3 .venv/bin/python pipeline_final.py \
  --name final_gemma3 --model gemma3:4b --out-dir /workspace/capstone/outputs/<이니셜>

# (B) VLM 부분만 (1단계 YOLOE+OCR은 모델과 무관하니 kyj 결과 재사용, 약 10분)
./gpu_lock.sh 7 .venv/bin/python exp_obj_vlm.py --crop margin \
  --model minicpm-v:8b --name obj_margin_minicpm --out-dir /workspace/capstone/outputs/<이니셜>
#   --buttons 를 붙이면 엘리베이터를 호출 버튼으로 판단 (시험 중, 느려짐)
#   --source 기본값 = /workspace/capstone/outputs/kyj/exp_final_all.jsonl (1단계 원자료)
```

서버 Ollama에 있는 모델: `qwen3-vl:4b-instruct`, `qwen2.5vl:7b`, `qwen2.5vl:3b`, `gemma3:4b`, `gemma3:12b`, `minicpm-v:8b`, `llama3.2-vision:11b`, `qwen3.5:4b`.
→ 5초 기준은 qwen3-vl 4B로 잰 것. 8B 이상은 더 느릴 수 있다.

**qwen3.5:4b 주의 (10/26 시험)**: GPU용 Ollama 0.24.0은 qwen3.5를 못 연다(새 구조). 기본 Ollama(11434, CPU)로는 열리지만 느리다.
transformers로 GPU에서 돌리는 방법도 시험했다: `install_q35.sh`(별도 환경 `.venv_q35` + 모델 9GB 받기, 약 10분) → `q35_server.py`(Ollama와 같은 형식의 서버, 포트 11437).
영상 판단 지점 평가에서 qwen3-vl 4B보다 나은 조건이 없었고 생각 모드는 시간 안에 답을 못 내서, 환경과 모델 파일은 지웠다(스크립트만 남김).

## 6. 채점

```bash
cd /workspace/capstone/src/kyj
python3 score_v2.py --split test \
  "qwen3-vl=/workspace/capstone/outputs/kyj/final/exp_obj_margin.jsonl" \
  "gemma3=/workspace/capstone/outputs/<이니셜>/exp_obj_margin_gemma3.jsonl"
```
- `--split`: `dev`(59장, 설정 고를 때만) / `test`(117장, 마지막 비교용) / `all`
- 설정은 dev로 고르고, test는 마지막에 한 번만 보는 게 원칙.

## 7. 주의

- GPU 작업은 `gpu_lock.sh <GPU번호> 명령`으로 감싸기 (시간 측정이 서로 망가지지 않게).
- Ollama(11434, 11436) **재시작·종료 금지**, 모델 pull/삭제는 팀과 상의.
- VLM 프롬프트에 구체적인 숫자 예시(예: 호실 번호)를 넣으면 모델이 그대로 베낀다.
- 문의: kyj
