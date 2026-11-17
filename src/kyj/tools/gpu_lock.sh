#!/bin/bash
# 사용법: gpu_lock.sh <GPU번호> <명령...>
# 같은 GPU에서 kyj 실험이 동시에 돌지 않도록 순서대로 실행 (시간 측정 보호)
gpu=$1; shift
exec flock /tmp/kyj_gpu${gpu}.lock "$@"
