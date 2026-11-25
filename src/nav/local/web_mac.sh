#!/bin/bash
# 실내 길안내 웹 켜기/끄기 (Mac에서 실행). 학교 서버의 web_server.py + Mac의 Cloudflare 임시 https 주소.
#   bash src/nav/web_mac.sh local start    # 서버 웹 켜기 → Mac으로 포트 연결 → https 주소 출력
#   bash src/nav/web_mac.sh local stop     # https 주소 끄기 + 서버 웹 끄기 (GPU 반납)
#   bash src/nav/web_mac.sh local status
#   bash src/nav/web_mac.sh local records           # 지난 안내 기록 목록
#   bash src/nav/web_mac.sh local record [세션]     # 기록을 웹 모양 다시보기로 만들어 Mac에서 열기 (세션 생략 = 가장 최근)
# 학교 서버 비밀번호를 물으면 입력. 웹 비밀번호는 서버 outputs/kyj/nav_web_password.
# Mac이 잠자기에 들어가면 주소가 끊긴다. 주소는 켤 때마다 바뀐다.

SOCK=/tmp/kyj-cnu.sock
PORT=8766  # Claude 버전(Mac, 8765)과 같이 켤 수 있게 (2026-11-24)
STATE="$HOME/.nav_web"
mkdir -p "$STATE"
REMOTE_LOG=/workspace/capstone/outputs/kyj/nav_web.log
SESS=/workspace/capstone/outputs/kyj/web_sessions
RECORDS="$HOME/nav_web_records"
REMOTE_DIR=/workspace/capstone/src/kyj
# 로컬 버전은 서버의 nav/local/web_server.py (가상환경은 src/kyj/.venv)
REMOTE_START="cd $REMOTE_DIR/nav/local && (nohup env CUDA_VISIBLE_DEVICES=3 ../../.venv/bin/python web_server.py --port $PORT > $REMOTE_LOG 2>&1 < /dev/null &)"
REMOTE_PID="ps aux | awk '/(nav_web|web_server).py --port $PORT/ && !/awk/ {print \$2}'"

ssh_up() {  # 열어 둔 연결이 살아 있으면 그대로, 죽었으면 새로 연결 (비밀번호는 여기서 한 번만)
  ssh -S "$SOCK" -O check cnu 2>/dev/null && return 0
  rm -f "$SOCK"
  echo "학교 서버에 연결합니다 (비밀번호 입력)"
  ssh -o ControlMaster=yes -o ControlPath="$SOCK" -o ControlPersist=12h -fN cnu
}

tunnel_url() { grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$STATE/tunnel.log" 2>/dev/null | head -1; }

case "$1" in
start)
  ssh_up || { echo "서버 연결 실패"; exit 1; }
  if [ -z "$(ssh -S "$SOCK" -o BatchMode=yes cnu "$REMOTE_PID")" ]; then
    echo "서버 웹을 켭니다 (모델을 올리는 데 30초~1분)"
    ssh -S "$SOCK" -o BatchMode=yes cnu "$REMOTE_START"
    for i in $(seq 1 60); do
      ssh -S "$SOCK" -o BatchMode=yes cnu "grep -qE 'READY|Traceback' $REMOTE_LOG" && break; sleep 3
    done
    ssh -S "$SOCK" -o BatchMode=yes cnu "grep -E 'READY|Traceback' $REMOTE_LOG | tail -1"
  else
    echo "서버 웹은 이미 켜져 있습니다"
  fi
  curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/" || ssh -S "$SOCK" -O forward -L "$PORT:127.0.0.1:$PORT" cnu
  if [ -f "$STATE/tunnel.pid" ] && kill -0 "$(cat "$STATE/tunnel.pid")" 2>/dev/null; then
    echo "https 주소는 이미 켜져 있습니다"
  else
    nohup cloudflared tunnel --url "http://localhost:$PORT" > "$STATE/tunnel.log" 2>&1 &
    echo $! > "$STATE/tunnel.pid"
    for i in $(seq 1 30); do [ -n "$(tunnel_url)" ] && break; sleep 1; done
  fi
  echo "Mac에서:   http://localhost:$PORT"
  echo "휴대폰/팀: $(tunnel_url)   (주소가 열리기까지 10~30초 걸릴 수 있음)"
  ;;
stop)
  if [ -f "$STATE/tunnel.pid" ]; then kill "$(cat "$STATE/tunnel.pid")" 2>/dev/null; rm -f "$STATE/tunnel.pid"; fi
  echo "https 주소를 껐습니다"
  if ssh_up; then
    pid=$(ssh -S "$SOCK" -o BatchMode=yes cnu "$REMOTE_PID")
    [ -n "$pid" ] && ssh -S "$SOCK" -o BatchMode=yes cnu "kill $pid" && echo "서버 웹을 껐습니다 (GPU 3 반납)"
    ssh -S "$SOCK" -O cancel -L "$PORT:127.0.0.1:$PORT" cnu 2>/dev/null
  else
    echo "서버에 연결하지 못해 서버 웹은 끄지 못함 (stop을 다시 실행)"
  fi
  ;;
status)
  if [ -f "$STATE/tunnel.pid" ] && kill -0 "$(cat "$STATE/tunnel.pid")" 2>/dev/null; then
    echo "https 주소: 켜짐 $(tunnel_url)"
  else
    echo "https 주소: 꺼짐"
  fi
  if ssh -S "$SOCK" -O check cnu 2>/dev/null; then
    [ -n "$(ssh -S "$SOCK" -o BatchMode=yes cnu "$REMOTE_PID")" ] && echo "서버 웹: 켜짐" || echo "서버 웹: 꺼짐"
  else
    echo "서버 연결: 끊김"
  fi
  ;;
records)
  ssh_up || exit 1
  ssh -S "$SOCK" -o BatchMode=yes cnu "cd $SESS && for d in \$(ls -d 2*/ 2>/dev/null | tail -20); do d=\${d%/}; echo \"\$d  \$(grep -vc '\"event\"' \$d/log.jsonl 2>/dev/null)초\"; done"
  ;;
record)
  ssh_up || exit 1
  sid="$2"
  [ -z "$sid" ] && sid=$(ssh -S "$SOCK" -o BatchMode=yes cnu "ls -d $SESS/2*/ | tail -1 | xargs basename")
  [ -z "$sid" ] && { echo "기록이 없습니다"; exit 1; }
  ssh -S "$SOCK" -o BatchMode=yes cnu "cd $REMOTE_DIR/nav/local && ../../.venv/bin/python web_replay.py $SESS/$sid" || exit 1
  mkdir -p "$RECORDS"
  scp -q -o BatchMode=yes -o ControlPath="$SOCK" "cnu:$SESS/$sid/replay.html" "$RECORDS/$sid.html" && echo "Mac: $RECORDS/$sid.html" && open "$RECORDS/$sid.html"
  ;;
*)
  echo "사용: bash $0 start|stop|status|records|record [세션]"; exit 1 ;;
esac
