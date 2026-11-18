#!/bin/bash
# 실내 길안내 웹 켜기/끄기 (맥북에서만 실행. 학교 서버·ssh·GPU를 쓰지 않음).
#   bash src/nav/web_mac.sh start          # 실제 Claude (크레딧 씀: 세션 60, 하루 300 한도)
#   bash src/nav/web_mac.sh start --fake   # 가짜 응답 (크레딧 0)
#   bash src/nav/web_mac.sh stop | status
# 접속 코드: 환경변수 NAV_ACCESS_CODE, 없으면 outputs/nav_access_code 파일(없으면 새로 만듦). 화면에 한 번 보여 준다.
# 휴대폰: start가 Cloudflare 임시 HTTPS 주소(cloudflared, 2026-11-18 사용자 승인 후 Homebrew로 설치)도 켠다.
#   주소는 켤 때마다 바뀌고, 맥이 잠들면 끊긴다. 주소를 아는 사람도 접속 코드가 있어야 쓴다.
cd "$(dirname "$0")/../.." || exit 1
PORT=${PORT:-8765}
PIDF=outputs/web_server.pid
TPID=outputs/tunnel.pid
TLOG=outputs/tunnel.log
tunnel_url() { grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TLOG" 2>/dev/null | head -1; }
LOG=outputs/web_server.log
mkdir -p outputs
case "$1" in
start)
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "이미 켜져 있습니다: http://localhost:$PORT"; exit 0; fi
  if [ -z "$NAV_ACCESS_CODE" ]; then
    [ -s outputs/nav_access_code ] || (LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 8 > outputs/nav_access_code)
    NAV_ACCESS_CODE=$(cat outputs/nav_access_code)
  fi
  NAV_ACCESS_CODE="$NAV_ACCESS_CODE" nohup .venv/bin/python src/nav/web_server.py --port "$PORT" $2 > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  for i in $(seq 1 20); do grep -q READY "$LOG" && break; sleep 0.5; done
  grep -E "READY|Error" "$LOG" | tail -1
  echo "맥 브라우저: http://localhost:$PORT   관찰자: /monitor   기록: /records   접속 코드: $NAV_ACCESS_CODE"
  if command -v cloudflared >/dev/null; then
    nohup cloudflared tunnel --url "http://localhost:$PORT" > "$TLOG" 2>&1 < /dev/null &
    echo $! > "$TPID"
    for i in $(seq 1 40); do [ -n "$(tunnel_url)" ] && break; sleep 0.5; done
    echo "휴대폰: $(tunnel_url)   (열리기까지 10~30초 걸릴 수 있음)"
  fi
  ;;
stop)
  [ -f "$TPID" ] && kill "$(cat "$TPID")" 2>/dev/null && echo "휴대폰 주소를 껐습니다"; rm -f "$TPID"
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && echo "서버를 껐습니다"; rm -f "$PIDF" ;;
status)
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "서버 켜짐 http://localhost:$PORT"; else echo "서버 꺼짐"; fi
  if [ -f "$TPID" ] && kill -0 "$(cat "$TPID")" 2>/dev/null; then echo "휴대폰 주소 켜짐 $(tunnel_url)"; else echo "휴대폰 주소 꺼짐"; fi ;;
*)
  echo "사용: bash $0 start [--fake] | stop | status"; exit 1 ;;
esac
