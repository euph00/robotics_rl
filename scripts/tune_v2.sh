#!/bin/bash
# Equal tuning budget per end effector (docs/project_spec.md, fairness controls), development data only.
# A 2^(4-1) fractional factorial (generator D = ABC), 8 configurations per end effector, one seed (100),
# 4M environment steps, standard objects, otherwise the campaign's settings (scripts/campaign_v2.sh):
#   A learning rate                  3e-4 (-) | 1e-4 (+)
#   B exploration log-std after BC   -1.2 (-) | -0.7 (+)
#   C demonstrations per object      100 (-)  | 300 (+)
#   D interleaved BC during RL       off (-)  | 200 updates per rollout, decay 0.9995 (+)
# Selection rule, fixed before the runs: score = mean lift-and-carry of the last three training-time
# evaluations (3.0, 3.5, 4.0M); per factor, the level with the higher mean score over its four runs;
# the tuned recipe is that combination, per end effector (scripts/tune_v2_report.py).
#   setsid nohup bash scripts/tune_v2.sh >> runs/tune_v2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MAX_RUNS=${MAX_RUNS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-45}
STANDARD="14 5 2 7 10"
COMMON=(--algo ppo --physics realistic --wrist-box .08 .5 --carry --horizon 300 --start-shift .08 .12
        --steps 4000000 --envs 22 --eval-every 500000 --eval-episodes 12 --objects $STANDARD
        --bc-updates 40000 --critic-warmup 1000000 --seed 100)
declare -A ROBOT=([gripper]="--robot fetch-gripper" [hand]="--robot fetch --hand-mount wrist")
declare -A TEACHER=([gripper]=fetch-gripper [hand]=fetch)

for e in gripper hand; do                       # the larger demonstration sets (C+)
  OUT=runs/v2_demos_${e}_std300
  [ -e $OUT/demonstrations.npz ] && continue
  echo "$(date +%H:%M) demonstrations $e x300"
  $PY scripts/hand_teacher_demos.py --robot ${TEACHER[$e]} --objects $STANDARD --episodes 300 --carry --horizon 300 \
      --start-shift .08 .12 --workers 24 --output $OUT > $OUT.log 2>&1
  python3 -c "import json; d=json.load(open('$OUT/summary.json')); print('   kept', d['kept'], 'of', d['episodes'], 'episodes,', d['transitions'], 'transitions')"
done

#        A B C D
CONFIGS=("- - - -" "+ - - +" "- + - +" "+ + - -" "- - + +" "+ - + -" "- + + -" "+ + + +")
alive_runs() { pgrep -fc "[t]rain_hand.py --run-id v2" ; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
for k in 0 1 2 3 4 5 6 7; do for e in gripper hand; do
  R=v2tune_${e}_c$((k + 1))
  [ -e runs/$R/config.json ] && continue
  set -- ${CONFIGS[$k]}
  ARGS=(--learning-rate $([ $1 = + ] && echo 1e-4 || echo 3e-4) --bc-log-std $([ $2 = + ] && echo -0.7 || echo -1.2)
        --bc-demos runs/v2_demos_${e}_std$([ $3 = + ] && echo 300)/demonstrations.npz)
  [ $4 = + ] && ARGS+=(--bc-interleave 200 --bc-interleave-decay .9995)
  while [ "$(alive_runs)" -ge $MAX_RUNS ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 120; done
  echo "$(date +%H:%M) launch $R (${CONFIGS[$k]})"
  setsid nohup $PY scripts/train_hand.py --run-id $R ${ROBOT[$e]} "${COMMON[@]}" "${ARGS[@]}" > runs/$R.log 2>&1 < /dev/null &
  sleep 300
done; done
echo "$(date +%H:%M) every tuning run launched"
