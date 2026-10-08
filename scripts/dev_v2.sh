#!/bin/bash
# Redesigned study (docs/project_spec.md): full-task development evaluations of every cell, development
# mazes only (--set dev: 52 layouts), the protocol's settings (realistic physics, planned speeds 3 / holding
# 2, seeded random object starts; wide: 3 starts per case). Completed evaluations are skipped.
#   setsid nohup bash scripts/dev_v2.sh >> runs/dev_v2.log 2>&1 < /dev/null &
# Checkpoint per learned run, fixed before these evaluations: the best training-time evaluation (lift and
# carry) within the 10M budget, the later one on ties; imitation = step 0 of the BC -> PPO run.
# Phase 1 decides the grip after the grasp per end effector and task (frozen targets vs held by the policy)
# on the BC -> PPO cells, 5 seeds each: the grip with more pooled deliveries, frozen on a tie
# (runs/dev_v2/grip_choice.json). Phase 2 evaluates every cell with that grip.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${WORKERS:-40}
OUT=runs/dev_v2
mkdir -p $OUT
STANDARD="14 5 2 7 10"; WIDE="0 102"
declare -A OBJECTS=([std]="$STANDARD" [wide]="$WIDE")
declare -A STARTS=([std]=1 [wide]=3)
declare -A EFFECTOR=([gripper]=gripper [hand]=allegro)
P=(--set dev --random-object-pose --workers $W)

report() { python3 -c "
import json; o = json.load(open('$OUT/$1/summary.json'))['overall']; print('   ', o['success'], '/', o['cases'], 'delivered')"; }
ev() {      # id, eval.py arguments (the gripper's planner)
  local id=$1; shift; [ -e $OUT/$id/summary.json ] && return; rm -rf $OUT/$id; echo "== $id $(date +%H:%M)"
  $PY scripts/eval.py --output $OUT/$id --label "dev v2 · $id" --physics realistic "${P[@]}" "$@" > $OUT/$id.log 2>&1; report $id; }
evh() {     # id, eval_hand.py arguments
  local id=$1; shift; [ -e $OUT/$id/summary.json ] && return; rm -rf $OUT/$id; echo "== $id $(date +%H:%M)"
  $PY scripts/eval_hand.py --output $OUT/$id "${P[@]}" "$@" > $OUT/$id.log 2>&1; report $id; }
best() {    # run -> step of its best training-time evaluation within 10M (later on ties)
  python3 - "$1" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f'runs/{sys.argv[1]}/metrics.jsonl') if '"evaluation"' in l]
rows = [r for r in rows if r['transitions'] <= 10_000_100]
print(max(rows, key=lambda r: (r['carry_success_rate'], r['transitions']))['transitions'])
PY
}
learned() { # id, effector, run, step, task, grip
  evh $1 --policy $3 --step $4 --effector ${EFFECTOR[$2]} --objects ${OBJECTS[$5]} --starts ${STARTS[$5]} --fingers-after-grasp $6; }

# Phase 1: the grip after the grasp, on the BC -> PPO cells.
for task in std wide; do for e in gripper hand; do for s in 0 1 2 3 4; do for grip in frozen policy; do
  R=v2_${e}_bcppo_${task}_s$s
  learned ${e}_bcppo_${task}_s${s}_$grip $e $R $(best $R) $task $grip
done; done; done; done
python3 - <<'PY'
import json
from pathlib import Path
out, choice = Path('runs/dev_v2'), {}
for e in ('gripper', 'hand'):
    for task in ('std', 'wide'):
        pooled = {grip: sum(json.loads((out / f'{e}_bcppo_{task}_s{s}_{grip}/summary.json').read_text())['overall']['success']
                            for s in range(5)) for grip in ('frozen', 'policy')}
        choice[f'{e}_{task}'] = {'delivered': pooled, 'grip': 'policy' if pooled['policy'] > pooled['frozen'] else 'frozen'}
        print(f'   grip {e} {task}: {pooled} -> {choice[f"{e}_{task}"]["grip"]}')
(out / 'grip_choice.json').write_text(json.dumps(choice, indent=1))
PY
grip() { python3 -c "import json; print(json.load(open('$OUT/grip_choice.json'))['$1_$2']['grip'])"; }

# Phase 2: every other cell with the chosen grip.
ev gripper_planner_std --controller reference --objects $STANDARD --planned-speed-scale 3.0 --holding-speed-scale 2.0
ev gripper_planner_wide --controller reference --objects $WIDE --starts 3 --planned-speed-scale 3.0 --holding-speed-scale 2.0
for task in std wide; do
  evh hand_scripted_$task --policy teacher --objects ${OBJECTS[$task]} --starts ${STARTS[$task]} --fingers-after-grasp $(grip hand $task)
done
for task in std wide; do for e in gripper hand; do G=$(grip $e $task); for s in 0 1 2 3 4; do
  learned ${e}_bc_${task}_s$s $e v2_${e}_bcppo_${task}_s$s 0 $task $G
  R=v2_${e}_ppo_${task}_s$s
  learned ${e}_ppo_${task}_s$s $e $R $(best $R) $task $G
done; done; done
echo "$(date +%H:%M) development evaluations done"
