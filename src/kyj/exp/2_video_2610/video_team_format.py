"""길안내 추적기 결과(guide_*.jsonl)를 팀 형식 JSON(cjy answer.py와 같은 칸)으로 1초마다 내보낸다.

VLM을 더 부르지 않는다. 추적기가 그 초에 이미 알고 있는 것만 옮긴다.
  시각정보: 확정 호실(8장 투표 후)과 그 방향, 대표 장면의 OCR 층표시, YOLOE 랜드마크
            (엘리베이터는 VLM이 장면을 엘리베이터 앞으로 봤을 때만. YOLOE 엘리베이터는 오탐이 많아서, docs/10 9절)
  상황_맥락: 장면(VLM 판단) + 층(문패 첫 자리) + 목적지와의 관계(층 비교, 목적지 문패를 봤는지)
  방향정보: 그 순간 안내하는 행동을 OBSERVE/MOVE + 시계 방향으로
  요청 인식: 목적지 호실
거리·높이는 모르면 null.

사용: python3 src/kyj/exp/2_video_2610/video_team_format.py data/video/results/results_guide_410_1_2b_2a_situ_sharp.jsonl OUT_DIR [--frames-root ROOT]
  OUT_DIR/<순서>_<초>.json 과 OUT_DIR/index.jsonl({"json", "image", "request"}) 을 만든다.
"""

import argparse
import json
from pathlib import Path

LANDMARK = {"문": "출입문", "엘리베이터": "엘리베이터", "계단": "계단", "화장실": "화장실표시", "비상구표시": "비상구표시"}
SCENE = {"CORRIDOR": "복도", "LOBBY": "로비", "ELEVATOR_HALL": "엘리베이터 홀", "ENTRANCE": "건물 출입구",
         "INTERSECTION": "복도 갈림길", "ROOM_DOOR": "방 문 앞 복도", "INSIDE_ELEVATOR": "엘리베이터 안"}
SIDE_CLOCK = {"왼쪽": 11, "가운데": 12, "오른쪽": 1}


def clock(v):
    return int(str(v).replace("시", "")) if v else 12


def convert(r, target):
    st, seen = r["state"], r.get("seen") or {}
    dirs = {d["target"]: d["value"] for d in seen.get("directions", [])}
    scene = st.get("scene")

    plates = [{"텍스트_내용": x, "공간_위치": [clock(seen.get("plates", {}).get(x)), None, None]} for x in r["rooms"]]
    floors = [{"텍스트_내용": f, "공간_위치": [clock(dirs.get(f)), None, None]} for f in seen.get("floors", [])]
    others = []
    for name in seen.get("landmarks", []):
        if name == "엘리베이터" and scene != "ELEVATOR_HALL":
            continue
        if name in LANDMARK:
            others.append({"대상 이름": LANDMARK[name], "공간_위치": [clock(dirs.get(name)), None, None]})

    tf = target[0]
    floor = st.get("floor")
    facts = []
    if scene:
        facts.append(f"장면 종류는 VLM이 판단({SCENE.get(scene, scene)})")
    if r["rooms"]:
        facts.append(f"문패 {', '.join(r['rooms'])}를 OCR로 읽음")
    if floor:
        facts.append(f"지금까지 읽은 문패 첫 자리로 {floor}층이라고 추론")
    if st.get("wrong_floor"):
        rel = f"목적지 {target}호는 {tf}층인데 지금은 {floor}층이라 엘리베이터로 돌아가야 함"
    elif target in r["rooms"]:
        d = r.get("target_distance_m")
        rel = f"목적지 {target}호 문패가 {r.get('target_side') or ''} 앞에 보임" + (f", 약 {d:.0f}m" if d else "")
    elif floor == tf:
        rel = f"목적지 층({tf}층)이 맞음. {target}호 문패는 아직 안 보임"
    elif floor is None:
        rel = "지금 층을 아직 모름 (층을 알려 주는 문패나 표시가 아직 없음)"
    else:
        rel = "판단 불가"

    # 다음 행동: 그 순간 추적기가 안내하는 행동
    if target in r["rooms"] and not st.get("wrong_floor"):
        act = ("MOVE", [SIDE_CLOCK.get(r.get("target_side"), 12), r.get("target_distance_m") or 2.0])
    elif st.get("wrong_floor"):
        act = ("MOVE", [6, 3.0])  # 뒤로 돌아 엘리베이터로
    elif st.get("el_ready") is not None and not st.get("elevator_said"):
        act = ("OBSERVE", [12, 3])  # 엘리베이터 앞에서 층 단서를 찾느라 둘러보는 중
    elif scene == "ELEVATOR_HALL" and st.get("elevator_said"):
        act = ("MOVE", [12, 1.0])  # 엘리베이터 타기
    else:
        act = ("MOVE", [12, 2.0])  # 문 통과 / 복도 따라 계속 가기
    obs = [[12, 3]] if act[0] == "MOVE" else [act[1]]
    move = [act[1]] if act[0] == "MOVE" else []

    return {
        "시각정보": {"층표시_표지판": floors, "호실_표지판": plates, "화살표": [], "others": others},
        "상황_맥락": {"공간이름": SCENE.get(scene, "판단 불가"),
                    "공간이름 판단근거": "; ".join(facts) or "근거 없음",
                    "목적지와의_관계": rel},
        "방향정보": {"관측이 필요한 방향 리스트": obs, "이동_가능_경로 리스트": move,
                  "다음_행동": {"유형": act[0], "방향": act[1]}},
        "요청 인식 내용": {"인식한_목적지_텍스트": f"{target}호"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("guide")
    ap.add_argument("out_dir")
    ap.add_argument("--frames-root", default=str(Path(__file__).resolve().parents[4] / "data" / "video" / "frames_8fps"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(a.guide, encoding="utf-8"):
        r = json.loads(line)
        target = r["route_target"]
        name = f"{r['seq'].split('_')[0]}_{r['sec']:02d}"
        p = out / f"{name}.json"
        p.write_text(json.dumps(convert(r, target), ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append({"json": str(p), "image": str(Path(a.frames_root) / r["seq"] / r["key_frame"]),
                     "request": f"{target}호로 가고 싶어", "seq": r["seq"], "sec": r["sec"]})
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(len(rows), "→", out)


if __name__ == "__main__":
    main()
