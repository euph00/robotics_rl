#!/bin/bash
# Redesigned study (docs/project_spec.md, 2026-10-04): the standard-object training campaign.
# One recipe for both end effectors: demonstrations from each end effector's hand-written grasp
# (100 attempted episodes per object, protocol start range, 300-decision horizon, carried) ->
# BC -> PPO; and PPO from scratch. Same PPO settings, 22 workers, 10M steps, 5 seeds.
# A queue keeps at most MAX_RUNS runs alive and launches only with MIN_FREE_GB available.
# Idempotent: existing demonstration sets and runs are skipped (resume a stopped run with
# train_hand.py --resume STEP).
#   setsid nohup bash scripts/campaign_v2.sh >> runs/campaign_v2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX_RUNS=${MAX_RUNS:-5}
MIN_FREE_GB=${MIN_FREE_GB:-45}
STANDARD="14 5 2 7 10"
COMMON=(--algo ppo --physics realistic --wrist-box .08 .5 --carry --horizon 300 --start-shift .08 .12
        --steps 10000000 --envs 22 --eval-every 500000 --eval-episodes 12)
declare -A ROBOT=([gripper]="--robot fetch-gripper" [hand]="--robot fetch --hand-mount wrist")

demos() {   # effector robot
  local OUT=runs/v2_demos_$1_std
  [ -e $OUT/demonstrations.npz ] && return
  echo "$(date +%H:%M) demonstrations $1"
  $PY scripts/hand_teacher_demos.py --robot $2 --objects $STANDARD --episodes 100 --carry --horizon 300 \
      --start-shift .08 .12 --workers 24 --output $OUT > $OUT.log 2>&1
  python3 -c "import json; d=json.load(open('$OUT/summary.json')); print('   kept', d['kept'], 'of', d['episodes'], 'episodes,', d['transitions'], 'transitions')"
}
demos gripper fetch-gripper
demos hand fetch

QUEUE=()
for s in 0 1 2 3 4; do for e in gripper hand; do QUEUE+=("v2_${e}_bcppo_std_s$s $e bc $s"); done; done
for s in 0 1 2 3 4; do for e in gripper hand; do QUEUE+=("v2_${e}_ppo_std_s$s $e scratch $s"); done; done

alive_runs() { pgrep -fc "[t]rain_hand.py --run-id v2_" ; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
for item in "${QUEUE[@]}"; do
  set -- $item; R=$1; E=$2; KIND=$3; SEED=$4
  [ -e runs/$R/config.json ] && continue
  while [ "$(alive_runs)" -ge $MAX_RUNS ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 120; done
  EXTRA=()
  [ $KIND = bc ] && EXTRA=(--bc-demos runs/v2_demos_${E}_std/demonstrations.npz --bc-updates 40000 --critic-warmup 1000000)
  echo "$(date +%H:%M) launch $R"
  setsid nohup $PY scripts/train_hand.py --run-id $R --seed $SEED ${ROBOT[$E]} --objects $STANDARD "${COMMON[@]}" "${EXTRA[@]}" \
      > runs/$R.log 2>&1 < /dev/null &
  sleep 300      # let it build its environments before measuring free memory again
done
echo "$(date +%H:%M) every run launched"
