"""results/*.jsonl 과 img/ 를 읽어 검수 페이지용 data.js 를 만든다.

새 결과 파일이 생기면 MODELS 에 추가하고 다시 실행:
    python3 build_review.py
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).parent
FLOORS = ["3rdFloor", "4thFloor", "5thFloor"]

# (표시 이름, {층: 결과 파일 glob})
MODELS = [
    ("qwen3-vl:4b · kyj (CPU)", {
        "3rdFloor": "kyj/signs_qwen3-vl_4b-instruct_3rdFloor_*.jsonl",
        "4thFloor": "kyj/signs_qwen3-vl_4b-instruct_4thFloor_*.jsonl",
        "5thFloor": "kyj/signs_qwen3-vl_4b-instruct_5thFloor_*.jsonl",
    }),
    ("qwen3-vl:4b · cjy (GPU)", {
        "3rdFloor": "cjy/signs_qwen3-vl_4b-instruct_3F*.jsonl",
    }),
    ("gemma3:4b · lsh (CPU)", {
        "3rdFloor": "lsh/signs_gemma3_4b_3F.jsonl",
        "4thFloor": "lsh/signs_gemma3_4b_4F.jsonl",
        "5thFloor": "lsh/signs_gemma3_4b_5F.jsonl",
    }),
    ("qwen2.5vl:7b · sjh (CPU)", {
        # 3층 파일만 이름에 층이 없음 (IMAGE_DIR=3rdFloor 로 실행된 101줄 파일)
        "3rdFloor": "sjh/signs_qwen2.5vl_7b_2026*.jsonl",
        "4thFloor": "sjh/signs_qwen2.5vl_7b_4F.jsonl",
        "5thFloor": "sjh/signs_qwen2.5vl_7b_5F.jsonl",
    }),
    ("minicpm-v:8b · cjy (GPU)", {
        "3rdFloor": "cjy/signs_minicpm-v_8b_3F_gpu.jsonl",
        "4thFloor": "cjy/signs_minicpm-v_8b_4F_gpu.jsonl",
    }),
]


def load_answers(pattern):
    paths = sorted(glob.glob(str(HERE / "results" / pattern)))
    if not paths:
        return None
    answers = {}
    # 같은 패턴에 여러 파일이 있으면 가장 최근 파일 사용
    for line in open(paths[-1], encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        error = r.get("error") or ""
        if r.get("success"):
            status = "ok"
        elif error.startswith("Timeout"):
            status = "timeout"
        else:
            status = "error"
        answers[r["image"]] = {
            "text": r.get("signs") if status == "ok" else None,
            "status": status,
            "latency": round(r["latency_seconds"], 1) if r.get("latency_seconds") else None,
            "error": error[:120] or None,
        }
    return answers


def image_key(path):
    return int(path.stem) if path.stem.isdigit() else 10**9


def main():
    dups = json.loads((HERE / "dups.json").read_text())

    models = []
    for name, files in MODELS:
        per_floor = {}
        for floor, pattern in files.items():
            answers = load_answers(pattern)
            if answers is not None:
                per_floor[floor] = answers
        models.append({"name": name, "floors": per_floor})

    items = []
    for floor in FLOORS:
        for path in sorted((HERE / "img" / floor).glob("*.jpeg"), key=image_key):
            same = sorted(
                (k.split("/")[1] for k, v in dups.items() if v == f"{floor}/{path.name}"),
                key=lambda n: int(n.split(".")[0]),
            )
            items.append({
                "id": f"{floor}/{path.name}",
                "floor": floor,
                "image": path.name,
                "src": f"img/{floor}/{path.name}",
                "duplicates": same,
                "answers": [
                    m["floors"].get(floor, {}).get(path.name)
                    for m in models
                ],
            })

    data = {
        "models": [m["name"] for m in models],
        "items": items,
    }
    (HERE / "data.js").write_text(
        "window.REVIEW_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"items {len(items)}, models {len(models)}")
    for m in models:
        print(f"  {m['name']}: " + ", ".join(f"{f} {len(a)}" for f, a in m["floors"].items()))


if __name__ == "__main__":
    main()
