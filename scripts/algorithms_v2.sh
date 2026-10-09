#!/bin/bash
# Redesigned study, claim C3 (docs/project_spec.md): PPO vs RLPD under equal conditions, both end effectors,
# standard objects, 5 seeds. Both start from the same imitation start (the end effector's 300-per-object
# demonstrations, 40,000 cloning updates, exploration log-std -0.7) and the same 1M-step value warm-up with the
# actor frozen; same 22 workers, reward, horizon, start range and 4M environment steps (the tuning sweep's
# budget). PPO: the campaign's tuned recipe (scripts/campaign_v2.sh). RLPD (pnr/rlpd.py): SAC with
# layer-normalised critics and the demonstrations in half of every batch, its published defaults
# (learning rate 3e-4, gamma .99, tau .005) except one update per transition and two critics (wall-clock).
# Seed 0 of each end effector and algorithm runs in turn with the machine to itself (no other training or
# evaluation of ours; load logged), so its training hours compare; seeds 1-4 and a sensitivity check per end
# effector (RLPD without the value warm-up, its own default) then run MAX_RUNS at a time and count per step
# only (2026-10-09: an isolated RLPD run takes ~14 h, PPO 1.3 h). Evaluations every 250k steps, excluded
# from train_seconds.
#   setsid nohup bash scripts/algorithms_v2.sh >> runs/algorithms_v2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STANDARD="14 5 2 7 10"
COMMON=(--physics realistic --wrist-box .08 .5 --carry --horizon 300 --start-shift .08 .12 --objects $STANDARD
        --steps 4000000 --envs 22 --eval-every 250000 --eval-episodes 12 --side-grasp-objects
        --bc-updates 40000 --bc-log-std -0.7)
declare -A ROBOT=([gripper]="--robot fetch-gripper" [hand]="--robot fetch --hand-mount wrist")
busy() { pgrep -f "[d]ev_v2.sh|[c]ampaign_v2.sh|[t]rain_hand.py --run-id|scripts/[e]val" > /dev/null; }

MAX_RUNS=${MAX_RUNS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-45}
QUEUE=()
for s in 0 1 2 3 4; do for e in hand gripper; do for a in ppo rlpd; do QUEUE+=("v2c3_${e}_${a}_s$s $e $a $s 1000000"); done; done; done
for e in hand gripper; do QUEUE+=("v2c3_${e}_rlpd_nowarmup_s0 $e rlpd 0 0"); done
alive_runs() { pgrep -fc "[t]rain_hand.py --run-id v2c3_" ; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
summary() { python3 -c "
import json; r = [json.loads(l) for l in open('runs/$1/metrics.jsonl') if '\"evaluation\"' in l]
print('    $1', ' '.join(f\"{x['carry_success_rate']:.2f}\" for x in r), '| %.1f h training' % (r[-1]['train_seconds'] / 3600))"; }

for item in "${QUEUE[@]}"; do
  set -- $item; R=$1; E=$2; A=$3; SEED=$4; WARMUP=$5
  [ -e runs/$R/config.json ] && continue
  ISOLATED=$([ $SEED = 0 ] && [ $WARMUP != 0 ] && echo 1)
  if [ -n "$ISOLATED" ]; then
    while busy; do sleep 120; done
  else
    while pgrep -f "[d]ev_v2.sh|scripts/[e]val" > /dev/null || [ "$(alive_runs)" -ge $MAX_RUNS ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 120; done
  fi
  DEMOS=runs/v2_demos_${E}_std300
  if [ $A = ppo ]; then
    EXTRA=(--algo ppo --learning-rate 1e-4)
  else
    EXTRA=(--algo rlpd --learning-rate 3e-4 --batch-size 512 --utd 1 --rlpd-demos $DEMOS/transitions.npz)
  fi
  echo "$(date '+%a %H:%M') run $R (load $(cut -d' ' -f1 /proc/loadavg)${ISOLATED:+, isolated})"
  if [ -n "$ISOLATED" ]; then
    $PY scripts/train_hand.py --run-id $R --seed $SEED ${ROBOT[$E]} "${COMMON[@]}" "${EXTRA[@]}" \
        --bc-demos $DEMOS/demonstrations.npz --critic-warmup $WARMUP > runs/$R.log 2>&1 < /dev/null
    summary $R
  else
    setsid nohup $PY scripts/train_hand.py --run-id $R --seed $SEED ${ROBOT[$E]} "${COMMON[@]}" "${EXTRA[@]}" \
        --bc-demos $DEMOS/demonstrations.npz --critic-warmup $WARMUP > runs/$R.log 2>&1 < /dev/null &
    sleep 300      # let it build its environments before measuring free memory again
  fi
done
echo "$(date '+%a %H:%M') every run launched"
