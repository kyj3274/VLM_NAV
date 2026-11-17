#!/bin/bash
# 실험 E 전체 그리드 (dev 59장 + 호실 진단). 서버에서 nohup으로 실행.
cd /workspace/capstone/src/kyj
IDS=/workspace/capstone/outputs/kyj/exp_d_ids_dev.txt
LOG=/workspace/capstone/outputs/kyj/exp_e_log.txt
for m in q3 q25; do
  [ $m = q3 ] && M=qwen3-vl:4b-instruct || M=qwen2.5vl:7b
  for l in ko en; do
    for r in none short; do
      rs=${r:0:1}
      echo "=== ${m}_${l}_${rs}_dev $(date)" >> $LOG
      ./gpu_lock.sh 7 .venv/bin/python exp_e_vlm.py --model $M --lang $l --reason $r --ids-file $IDS --name ${m}_${l}_${rs}_dev >> $LOG 2>&1
      echo "=== diag_${m}_${l}_${rs}_dev $(date)" >> $LOG
      ./gpu_lock.sh 7 .venv/bin/python exp_e_vlm.py --model $M --lang $l --reason $r --ids-file $IDS --name diag_${m}_${l}_${rs}_dev --diag-rooms >> $LOG 2>&1
    done
  done
done
echo "=== ALL DONE $(date)" >> $LOG
