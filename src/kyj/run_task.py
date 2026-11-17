import argparse
import base64
import hashlib
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


# ===============================================================
# 공통 설정 (모든 모델/Task에 동일하게 적용)
# ===============================================================

# --ollama-url로 변경 가능 (GPU Ollama: http://127.0.0.1:11435)
OLLAMA_URL = "http://127.0.0.1:11434"

DATA_ROOT = Path("/workspace/capstone/data/cnuCS")
FLOORS = ["3rdFloor", "4thFloor", "5thFloor", "stairs"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_DIR = Path("/workspace/capstone/outputs/kyj")

# OCR은 해상도 영향이 크므로 512 대신 1024 사용 (MD 6절, sign.py와 동일)
# --max-side로 변경 가능 (해상도 비교 실험용, 0이면 원본 크기)
MAX_SIDE = 1024
JPEG_QUALITY = 85

# sign.py와 동일: 이 시간(초)을 넘으면 "인식 실패"로 기록하고 다음 이미지로
REQUEST_TIMEOUT_SECONDS = 60
FAIL_TEXT = "인식 실패"

# 파일명(3rdFloor/1.jpeg) -> 이미지 ID(3F_001)
FLOOR_IDS = {
    "3rdFloor": "3F",
    "4thFloor": "4F",
    "5thFloor": "5F",
    "stairs": "ST",
}

# 같은 이미지에 같은 답이 나오도록 샘플링 고정
OPTIONS = {
    "temperature": 0,
    "seed": 42,
}

# ---------------------------------------------------------------
# 이동 단서(fields) Task 설정 — docs/05 v2 기준
# ---------------------------------------------------------------

FIELDS_PROMPT_VERSION = "p2"

LANDMARKS = [
    "엘리베이터", "계단", "화장실표시", "화장실입구", "비상구표시",
    "안내판", "소화전", "소화기", "점자블록", "정수기", "자판기",
    "출입문", "유리문", "복도끝", "기둥", "게시판", "사물함",
    "액자", "CCTV", "소파", "노란바닥띠", "미닫이문", "셔터",
]

# RELATIVE_POSITION은 길안내 핵심 대상에만 기록 (출입문 등은 너무 많음)
POSITION_TARGETS = [
    "엘리베이터", "계단", "화장실표시", "화장실입구",
    "비상구표시", "유리문", "복도끝",
]

CLOCK_VALUES = [f"{hour}시" for hour in [12] + list(range(1, 12))]
DIRECTION_VALUES = CLOCK_VALUES + ["판단 불가"]

STRING_ARRAY = {"type": "array", "items": {"type": "string"}}


# ===============================================================
# Task 정의: schema + prompt
# 새 Task는 여기에 항목만 추가하면 된다.
# ===============================================================

TASKS = {
    # -----------------------------------------------------------
    # 상황 인식 (팀에서 정한 5개 항목)
    # -----------------------------------------------------------
    "perception": {
        "schema": {
            "type": "object",
            "properties": {
                "environment": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
                "texts": {"type": "array", "items": {"type": "string"}},
                "spatial_relations": {"type": "array", "items": {"type": "string"}},
                "obstacles": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "environment",
                "objects",
                "texts",
                "spatial_relations",
                "obstacles",
            ],
            "additionalProperties": False,
        },
        "prompt": (
            "이미지를 분석하여 현재 보이는 정보를 파악하세요. "
            "다음 항목을 이미지에서 직접 확인할 수 있는 내용만을 "
            "바탕으로 작성하세요. "

            "environment에는 현재 보이는 장소와 환경을 작성하세요. "

            "objects에는 이동이나 위치 파악에 중요할 수 있는 "
            "주요 객체를 문자열 목록으로 작성하세요. "

            "texts에는 이미지에서 실제로 읽을 수 있는 글자, "
            "호실 번호, 층수, 안내 표지 등의 텍스트를 "
            "문자열 목록으로 작성하세요. "

            "spatial_relations에는 주요 객체나 통로의 위치를 "
            "화면을 기준으로 왼쪽, 중앙, 오른쪽 등의 표현을 사용하여 "
            "문자열 목록으로 작성하세요. "

            "obstacles에는 이동에 영향을 줄 수 있는 장애물이나 "
            "위험 요소를 문자열 목록으로 작성하세요. "

            "이미지에서 직접 확인할 수 없는 내용은 추측하지 마세요. "
            "확인할 수 있는 내용이 없는 목록 항목은 빈 배열 []로, "
            "environment를 확인할 수 없으면 '확인할 수 없음'으로 작성하세요."
        ),
    },

    # -----------------------------------------------------------
    # Navigation (팀원 perception_navigation 스크립트와 동일)
    # -----------------------------------------------------------
    "navigation": {
        "schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": [
                        "12시 방향",
                        "1시 방향",
                        "2시 방향",
                        "3시 방향",
                        "4시 방향",
                        "5시 방향",
                        "6시 방향",
                        "7시 방향",
                        "8시 방향",
                        "9시 방향",
                        "10시 방향",
                        "11시 방향",
                        "판단 불가",
                    ],
                },
                "basis": {"type": "string"},
            },
            "required": ["direction", "basis"],
            "additionalProperties": False,
        },
        "prompt": (
            "이미지를 보고 현재 위치에서 410호로 가기 위한 "
            "다음 이동 방향을 판단하세요. "

            "이미지에서 직접 확인할 수 있는 호실 번호, 표지판, "
            "화살표, 복도 구조 등의 정보만 근거로 사용하세요. "

            "12시는 정면, "
            "3시는 오른쪽, "
            "6시는 뒤쪽, "
            "9시는 왼쪽입니다. "

            "그 사이의 방향은 해당 시각 방향으로 판단하세요. "

            "direction에는 정해진 방향 값 중 하나만 선택하세요. "

            "basis에는 그 방향을 선택한 이미지상의 근거를 "
            "간단히 작성하세요. "

            "이미지 하나만으로 410호의 위치 또는 "
            "410호로 연결되는 이동 방향을 확인할 수 없다면 "
            "추측하지 말고 direction을 '판단 불가'로 작성하세요."
        ),
    },

    # -----------------------------------------------------------
    # 표지판 글자 읽기 (lsh sign.py 프롬프트/스키마 그대로)
    # -----------------------------------------------------------
    "signs": {
        "schema": {
            "type": "object",
            "properties": {
                "signs": {"type": "string"},
            },
            "required": ["signs"],
            "additionalProperties": False,
        },
        "prompt": (
            "이미지에 보이는 모든 글자를 읽어서 signs에 적어줘. "

            "표지판, 호실 번호판, 안내문, 포스터 등에 적힌 "
            "한국어, 영어, 숫자를 모두 보이는 그대로 적어줘. "
            "번역하거나 요약하지 말고 원문 그대로 적어줘. "

            "서로 다른 글자 덩어리는 쉼표와 공백(', ')으로 구분해서 "
            "한 줄의 문자열로 적어줘. "
            "예: '410호, 강의실, EXIT, 비상구' "

            "흐리거나 가려져서 확실히 읽을 수 없는 글자는 "
            "추측해서 채우지 말고 제외해줘. "

            "읽을 수 있는 글자가 하나도 없으면 "
            "signs를 '글자 없음'으로 해줘."
        ),
    },

    # -----------------------------------------------------------
    # 이동 단서 추출 (docs/05 공통 프롬프트 v2, 시계 방향)
    # -----------------------------------------------------------
    "fields": {
        "schema": {
            "type": "object",
            "properties": {
                "rooms": STRING_ARRAY,
                "floors": STRING_ARRAY,
                "spaces": STRING_ARRAY,
                "landmarks": {
                    "type": "array",
                    "items": {"type": "string", "enum": LANDMARKS},
                },
                "directions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["SIGN_ARROW", "RELATIVE_POSITION"],
                            },
                            "value": {
                                "type": "string",
                                "enum": DIRECTION_VALUES,
                            },
                        },
                        "required": ["target", "type", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rooms", "floors", "spaces", "landmarks", "directions"],
            "additionalProperties": False,
        },
        "prompt": (
            "당신은 시각장애인의 실내 길안내를 돕기 위해 사진에서 이동 단서를 추출합니다. "
            "사진은 대학 건물 안에서 사람 눈높이로 찍은 것입니다.\n\n"

            "아래 5가지 항목을 JSON으로만 답하세요.\n\n"

            "1. rooms: 문패나 표지판에 적힌 호실 번호.\n"
            "- '호'는 빼고 숫자, 하이픈, 알파벳만 보이는 그대로 씁니다. "
            "(형식 예: \"999\", \"999-1\")\n"
            "- 전화번호, 날짜, 포스터 속 숫자는 넣지 않습니다.\n\n"

            "2. floors: 사진에 글자로 적힌 층 표시.\n"
            "- 'n층'은 'nF'로, 지하는 'Bn'으로 씁니다.\n"
            "- 글자로 보이지 않으면 빈 배열로 둡니다.\n\n"

            "3. spaces: 문패나 표지판에 글자로 적힌 공간 이름을 보이는 그대로 씁니다. "
            "포스터나 공지문 속 글자는 넣지 않습니다.\n\n"

            "4. landmarks: 사진에 실제로 보이는 시설이나 물체. "
            "아래 목록의 이름만 씁니다.\n"
            f"{', '.join(LANDMARKS)}\n\n"

            "5. directions: 방향 정보. 각 항목은 target, type, value를 가집니다.\n"
            "- 방향은 시계 방향으로 씁니다. "
            "12시는 정면, 3시는 오른쪽, 6시는 뒤, 9시는 왼쪽입니다.\n"
            "- type이 SIGN_ARROW이면 표지판 화살표가 가리키는 방향입니다. "
            "target은 그 화살표가 안내하는 rooms, spaces, landmarks 값입니다. "
            "value는 12시부터 11시 중 하나, 모르면 '판단 불가'입니다.\n"
            "- type이 RELATIVE_POSITION이면 대상이 사진의 어디에 있는지입니다. "
            "target은 rooms 값 또는 다음 landmarks 중 하나입니다: "
            f"{', '.join(POSITION_TARGETS)}. "
            "대상이 사진 왼쪽 1/3에 있으면 11시, 가운데 1/3이면 12시, "
            "오른쪽 1/3이면 1시, 모르면 '판단 불가'입니다.\n\n"

            "규칙:\n"
            "- 사진에서 직접 확인한 것만 씁니다. 추측하지 않습니다.\n"
            "- 확인할 수 없는 항목은 빈 배열 []로 둡니다.\n"
            "- 같은 값을 반복해서 쓰지 않습니다.\n"
            "- JSON 외의 설명은 쓰지 않습니다.\n\n"

            '{"rooms": [], "floors": [], "spaces": [], "landmarks": [], "directions": []}'
        ),
    },
}


def encode_image(path):
    with Image.open(path) as image:
        if MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE))

        buffer = BytesIO()

        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
        )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def image_sort_key(path):
    # 1, 2, ..., 10 순서로 정렬 (문자열 정렬이면 1, 10, 100, 2 ...)
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def image_id(path):
    floor = FLOOR_IDS.get(path.parent.name, path.parent.name)
    number = f"{int(path.stem):03d}" if path.stem.isdigit() else path.stem
    return f"{floor}_{number}"


def drop_duplicates(images):
    # 3rdFloor는 같은 사진이 다른 번호로 여러 장 있음 (md5 기준 첫 장만 남김)
    seen = set()
    unique = []

    for path in images:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)

    return unique


def find_images(floors):
    images = []

    for floor in floors:
        image_dir = DATA_ROOT / floor

        images.extend(
            sorted(
                (
                    path
                    for path in image_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=image_sort_key,
            )
        )

    return images


def run_one(model, task_name, image_path):
    task = TASKS[task_name]

    # 층마다 파일명(1.jpeg ...)이 겹치므로 floor도 함께 기록
    record = {
        "model": model,
        "task": task_name,
        "image_id": image_id(image_path),
        "floor": image_path.parent.name,
        "image": image_path.name,
        "image_path": str(image_path),
        "ollama_url": OLLAMA_URL,
        "max_side": MAX_SIDE,
        "options": OPTIONS,
        "result": None,
        "raw_response": None,
        "latency_seconds": None,
        "total_duration": None,
        "load_duration": None,
        "prompt_eval_count": None,
        "prompt_eval_duration": None,
        "eval_count": None,
        "eval_duration": None,
        "success": False,
        "error": None,
    }

    # signs는 팀원 결과(sign.py)와 같은 키로도 저장
    if task_name == "signs":
        record["signs"] = None

    if task_name == "fields":
        record["prompt_version"] = FIELDS_PROMPT_VERSION

    try:
        image = encode_image(image_path)

        payload = {
            "model": model,
            "format": task["schema"],
            "messages": [
                {
                    "role": "user",
                    "content": task["prompt"],
                    "images": [
                        image
                    ],
                }
            ],
            "options": OPTIONS,
            "stream": False,
        }

        request_started = time.perf_counter()

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                # (연결 대기, 응답 대기)
                timeout=(10, REQUEST_TIMEOUT_SECONDS),
            )

        except requests.exceptions.Timeout:
            record["latency_seconds"] = (
                time.perf_counter()
                - request_started
            )
            record["error"] = (
                f"Timeout: {REQUEST_TIMEOUT_SECONDS}초 초과"
            )
            if task_name == "signs":
                record["signs"] = FAIL_TEXT
            return record

        record["latency_seconds"] = (
            time.perf_counter()
            - request_started
        )

        response.raise_for_status()

        data = response.json()

        record["raw_response"] = data["message"]["content"]
        record["total_duration"] = data.get("total_duration")
        record["load_duration"] = data.get("load_duration")
        record["prompt_eval_count"] = data.get("prompt_eval_count")
        record["prompt_eval_duration"] = data.get("prompt_eval_duration")
        record["eval_count"] = data.get("eval_count")
        record["eval_duration"] = data.get("eval_duration")

        # JSON 파싱 실패도 error로 기록 (raw_response는 남겨 둠)
        record["result"] = json.loads(record["raw_response"])

        if task_name == "signs":
            record["signs"] = record["result"]["signs"]

        record["success"] = True

    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    return record


def warm_up(model):
    # 첫 요청의 모델 로딩 시간 때문에 첫 이미지가
    # 시간 제한에 걸리지 않도록 모델을 미리 올려 둔다.
    print("모델 로딩 중...")
    try:
        requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "hi"}
                ],
                "options": OPTIONS,
                "stream": False,
            },
            timeout=300,
        ).raise_for_status()
        print("모델 로딩 완료\n")
    except Exception as error:
        print(f"모델 로딩 실패: {error}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ollama VLM 일괄 실행 (Task/모델/층 선택)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:4b-instruct",
        help="Ollama 모델명 (기본: qwen3-vl:4b-instruct)",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASKS),
    )
    parser.add_argument(
        "--floor",
        default="all",
        choices=FLOORS + ["all"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 N장만 처리 (테스트용)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="N번째 이미지부터 처리 (중단된 실행 이어서 돌리기용, 1부터 시작)",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="이 이미지 ID만 처리. 쉼표 구분(4F_046,5F_010) 또는 한 줄에 하나씩 적은 파일 경로",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="같은 사진(md5 동일)은 첫 장만 처리",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama 주소 (기본: {OLLAMA_URL})",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=MAX_SIDE,
        help=f"긴 변을 이 크기로 줄여서 보냄, 0이면 원본 (기본: {MAX_SIDE})",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="최대 출력 토큰 수 (반복 루프 방지, fields 권장 512)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help=f"이미지 1장 응답 제한 시간(초) (기본: {REQUEST_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="결과 파일명 끝에 붙일 메모 (예: gpu6, p2)",
    )
    return parser.parse_args()


def load_ids(value):
    path = Path(value)
    if path.is_file():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    global OLLAMA_URL, MAX_SIDE, REQUEST_TIMEOUT_SECONDS

    args = parse_args()

    OLLAMA_URL = args.ollama_url.rstrip("/")
    MAX_SIDE = args.max_side
    REQUEST_TIMEOUT_SECONDS = args.timeout
    if args.num_predict is not None:
        OPTIONS["num_predict"] = args.num_predict

    # all = 3·4·5층 (stairs는 따로 지정할 때만)
    floors = [floor for floor in FLOORS if floor != "stairs"] if args.floor == "all" else [args.floor]

    images = find_images(floors)

    if args.dedupe:
        images = drop_duplicates(images)

    if args.ids:
        wanted = load_ids(args.ids)
        by_id = {image_id(path): path for path in images}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            print(f"찾지 못한 이미지 ID: {', '.join(missing)}")
        images = [by_id[item] for item in wanted if item in by_id]

    images = images[args.start - 1:]

    if args.limit is not None:
        images = images[:args.limit]

    if not images:
        print(f"이미지를 찾지 못했습니다: {floors}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / (
        f"{args.task}_"
        f"{args.model.replace(':', '_')}_"
        f"{args.floor}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"{'_' + args.tag if args.tag else ''}.jsonl"
    )

    warm_up(args.model)

    print(f"모델: {args.model}")
    print(f"Task: {args.task}")
    print(f"층: {', '.join(floors)}")
    print(f"Ollama: {OLLAMA_URL} | 긴 변: {MAX_SIDE or '원본'} | 옵션: {OPTIONS}")
    print(f"이미지 {len(images)}장 처리 시작")
    print(f"결과 파일: {output_path}\n")

    success_count = 0
    timeout_count = 0

    for index, image_path in enumerate(images, start=1):
        record = run_one(args.model, args.task, image_path)

        # 이미지 처리 직후 즉시 append
        with open(output_path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        latency = record["latency_seconds"]
        latency_text = (
            f"{latency:.2f}초" if latency is not None else "-"
        )

        name = record["image_id"]

        if record["success"]:
            success_count += 1
            print(
                f"[{index}/{len(images)}] {name} | "
                f"{json.dumps(record['result'], ensure_ascii=False)[:120]} | "
                f"{latency_text}"
            )
        elif record["error"] and record["error"].startswith("Timeout"):
            timeout_count += 1
            print(
                f"[{index}/{len(images)}] {name} | "
                f"{FAIL_TEXT} ({REQUEST_TIMEOUT_SECONDS}초 초과)"
            )
        else:
            print(
                f"[{index}/{len(images)}] {name} | "
                f"ERROR: {record['error']}"
            )

    print(
        f"\n완료: 성공 {success_count} / "
        f"인식 실패(시간 초과) {timeout_count} / "
        f"전체 {len(images)}"
    )
    print(f"결과 파일: {output_path}")


if __name__ == "__main__":
    main()
