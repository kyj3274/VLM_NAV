#!/bin/bash
# 실내 길안내 웹 켜기/끄기 (맥북에서만 실행. 학교 서버·ssh·GPU를 쓰지 않음).
#   bash src/nav/web_mac.sh start          # 실제 Claude (크레딧 씀: 세션 60, 하루 300 한도)
#   bash src/nav/web_mac.sh start --fake   # 가짜 응답 (크레딧 0)
#   bash src/nav/web_mac.sh stop | status
# 접속 코드: 환경변수 NAV_ACCESS_CODE, 없으면 outputs/nav_access_code 파일(없으면 새로 만듦). 화면에 한 번 보여 준다.
# 휴대폰에서 쓰려면 HTTPS 터널(cloudflared)이 필요하다. 아직 설치하지 않았다(설치 전에 사용자에게 묻기).
cd "$(dirname "$0")/../.." || exit 1
PORT=${PORT:-8765}
PIDF=outputs/web_server.pid
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
  ;;
stop)
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && echo "껐습니다"; rm -f "$PIDF" ;;
status)
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "켜짐 http://localhost:$PORT"; else echo "꺼짐"; fi ;;
*)
  echo "사용: bash $0 start [--fake] | stop | status"; exit 1 ;;
esac
