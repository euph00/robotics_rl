#!/bin/bash
# Redesigned study (docs/project_spec.md, 2026-10-04): the training campaign.
# One recipe for both end effectors: demonstrations from each end effector's hand-written grasp on the
# standard objects (300 attempted episodes per object, protocol start range, 300-decision horizon,
# carried) -> BC -> PPO; and PPO from scratch. Same PPO settings, 22 workers, 10M steps, 5 seeds.
# Recipe chosen by the equal-budget tuning sweep (scripts/tune_v2.sh, runs/tune_v2_report.json; the
# same levels won for both end effectors): learning rate 1e-4, exploration log-std -0.7 after cloning,
# 300 demonstrations per object, no interleaved cloning. From scratch: the same learning rate.
# Wide task: each end effector's best hand-written prior. The gripper has none that works (teacher 0-1/12),
# so it clones its standard-object demonstrations; the hand clones its side power-grasp script on the
# idealised rig (it slides off on the Fetch) and starts from the side pregrasp (from scratch too).
# Step 0 = the imitation cell; and PPO from scratch.
# Queue order: standard BC -> PPO, wide BC -> PPO, standard from scratch, wide from scratch.
# A queue keeps at most MAX_RUNS runs alive and launches only with MIN_FREE_GB available; SKIP (a regex of
# run ids) leaves runs out.
# Idempotent: existing demonstration sets and runs are skipped (resume a stopped run with
# train_hand.py --resume STEP).
#   setsid nohup bash scripts/campaign_v2.sh >> runs/campaign_v2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX_RUNS=${MAX_RUNS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-45}
STANDARD="14 5 2 7 10"
WIDE="0 102"
declare -A OBJECTS=([std]="$STANDARD" [wide]="$WIDE")
COMMON=(--algo ppo --physics realistic --wrist-box .08 .5 --carry --horizon 300 --start-shift .08 .12
        --steps 10000000 --envs 22 --eval-every 500000 --eval-episodes 12
        --learning-rate 1e-4
        --side-grasp-objects)     # every object starts top-down (the cloned demonstrations' pregrasp)
declare -A ROBOT=([gripper]="--robot fetch-gripper" [hand]="--robot fetch --hand-mount wrist")

demos() {   # effector robot
  local OUT=runs/v2_demos_$1_std300
  [ -e $OUT/demonstrations.npz ] && return
  echo "$(date +%H:%M) demonstrations $1"
  $PY scripts/hand_teacher_demos.py --robot $2 --objects $STANDARD --episodes 300 --carry --horizon 300 \
      --start-shift .08 .12 --workers 24 --output $OUT > $OUT.log 2>&1
  python3 -c "import json; d=json.load(open('$OUT/summary.json')); print('   kept', d['kept'], 'of', d['episodes'], 'episodes,', d['transitions'], 'transitions')"
}
demos gripper fetch-gripper
demos hand fetch

QUEUE=()
for recipe in bcppo ppo; do for task in std wide; do for s in 0 1 2 3 4; do for e in gripper hand; do
  QUEUE+=("v2_${e}_${recipe}_${task}_s$s $e $recipe $s $task")
done; done; done; done

alive_runs() { pgrep -fc "[t]rain_hand.py --run-id v2_" ; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
for item in "${QUEUE[@]}"; do
  set -- $item; R=$1; E=$2; KIND=$3; SEED=$4; TASK=$5
  [ -e runs/$R/config.json ] && continue
  [ -n "$SKIP" ] && [[ $R =~ $SKIP ]] && continue      # e.g. SKIP='^v2_hand_bcppo_wide_' (recipe under review)
  while [ "$(alive_runs)" -ge $MAX_RUNS ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 120; done
  EXTRA=()
  DEMOS=runs/v2_demos_${E}_std300
  if [ $E = hand ] && [ $TASK = wide ]; then
    # The hand's own wide-object prior: its side power-grasp script's demonstrations (idealised rig, 300 per
    # object) and the side pregrasp they start from; development seed v2dev_hand_sidebc_wide_s0 decided it.
    DEMOS=runs/v2_demos_hand_wide_rig300
    EXTRA+=(--side-grasp-objects $WIDE)
  fi
  [ $KIND = bcppo ] && EXTRA+=(--bc-demos $DEMOS/demonstrations.npz --bc-updates 40000 --bc-log-std -0.7
                               --critic-warmup 1000000)
  echo "$(date +%H:%M) launch $R"
  setsid nohup $PY scripts/train_hand.py --run-id $R --seed $SEED ${ROBOT[$E]} --objects ${OBJECTS[$TASK]} "${COMMON[@]}" "${EXTRA[@]}" \
      > runs/$R.log 2>&1 < /dev/null &
  sleep 300      # let it build its environments before measuring free memory again
done
echo "$(date +%H:%M) every run launched"
