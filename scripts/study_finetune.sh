#!/bin/bash
# Overnight study schedule (shared machine; launch only with enough free memory):
#  1. wait for the standard-object PPO seeds (hand_wbc_std_s*) to finish their 10M steps;
#  2. fine-tune each wide-object seed from its best checkpoint within 10M steps, with the full task's
#     post-grasp sequence as the carry phase (--task-carry) and the protocol's object-start range, 4M steps;
#  3. once the RLPD seeds have stopped (after RLPD_HOURS of training, PPO's compute to 10M steps, for the
#     wall-clock comparison) and >100 GB is free, the same fine-tune for each standard-object seed alongside,
#     3M steps with 16 environments;
#  4. when every fine-tune has finished, the protocol evaluations (scripts/evaluate_study.sh).
# Idempotent: runs that exist are not relaunched (an interrupted fine-tune resumes with train_hand.py --resume).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RLPD_HOURS=${RLPD_HOURS:-6}
alive() { pgrep -f "[r]un-id $1 " > /dev/null; }
best() { python3 - "$1" 10000000 <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f'runs/{sys.argv[1]}/metrics.jsonl') if '"evaluation"' in l]
done = [r for r in rows if r['transitions'] <= int(sys.argv[2])]
print(max(done, key=lambda r: (r['carry_success_rate'], r['transitions']))['transitions'])
PY
}
stop_rlpd() {
  for s in 0 1 2; do R=hand_rlpd_std_s$s
    started=$(python3 -c "import json,datetime; d=json.load(open('runs/$R/config.json')); import os; print(int(os.path.getmtime('runs/$R/config.json')))")
    [ $(( $(date +%s) - started )) -gt $(( RLPD_HOURS * 3600 )) ] && alive $R && pkill -f "[r]un-id $R " && echo "$(date +%H:%M) stopped $R after ${RLPD_HOURS} h"
  done
}
finetune() {   # run objects source_run steps seed [envs]
  local R=$1 OBJ=$2 SRC=$3 STEPS=$4 SEED=$5 ENVS=${6:-22}
  [ -e runs/$R/config.json ] && return
  local STEP=$(best $SRC)
  echo "$(date +%H:%M) launch $R from $SRC/$STEP"
  setsid nohup $PY scripts/train_hand.py --run-id $R --seed $SEED --algo ppo --physics realistic --robot fetch --hand-mount wrist \
    --wrist-box .08 .5 --carry --horizon 200 --objects $OBJ --init-from $SRC/$STEP --task-carry --start-shift .08 .12 \
    --critic-warmup 300000 --steps $STEPS --envs $ENVS --eval-every 500000 --eval-episodes 12 > runs/$R.log 2>&1 < /dev/null &
}
while alive hand_wbc_std_s0 || alive hand_wbc_std_s1 || alive hand_wbc_std_s2; do stop_rlpd; sleep 120; done
echo "$(date +%H:%M) standard PPO seeds finished"
finetune hand_w_wide_tc_s0 "0 102" hand_w_wide01 4000000 0
finetune hand_w_wide_tc_s1 "0 102" hand_w_wide_s1 4000000 1
finetune hand_w_wide_tc_s2 "0 102" hand_w_wide_s2 4000000 2
while alive hand_rlpd_std_s0 || alive hand_rlpd_std_s1 || alive hand_rlpd_std_s2; do stop_rlpd; sleep 60; done
echo "$(date +%H:%M) RLPD seeds stopped"
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
while [ $(free_gb) -lt 100 ]; do sleep 120; done
for s in 0 1 2; do finetune hand_wbc_std_tc_s$s "14 5 2 7 10" hand_wbc_std_s$s 3000000 $s 16; done
sleep 120
while alive hand_w_wide_tc_s0 || alive hand_w_wide_tc_s1 || alive hand_w_wide_tc_s2 \
   || alive hand_wbc_std_tc_s0 || alive hand_wbc_std_tc_s1 || alive hand_wbc_std_tc_s2; do sleep 120; done
echo "$(date +%H:%M) fine-tunes finished; protocol evaluations"
WORKERS=12 bash scripts/evaluate_study.sh
echo "$(date +%H:%M) evaluations done"
