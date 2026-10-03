#!/bin/bash
# Study evaluations under the frozen protocol (docs/project_spec.md, configs/study.json): hold-out
# mazes, random object starts, realistic physics, planned speeds 3 / holding 2. Completed runs are
# skipped. Run ids match the `evals` entries in configs/study.json.
#   WORKERS=16 bash scripts/evaluate_study.sh            # all cells whose policies exist
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W="${WORKERS:-16}"
STANDARD="14 5 2 7 10"; WIDE="0 102"
P=(--set holdout --random-object-pose --workers $W)
ev() { local id=$1; shift; [ -e runs/$id/summary.json ] && return; rm -rf runs/$id; echo "== $id $(date +%H:%M)"
       $PY scripts/eval.py --output runs/$id --label $id --physics realistic "${P[@]}" "$@" 2>&1 | grep -E "^overall"; }
evh() { local id=$1; shift; [ -e runs/$id/summary.json ] && return; rm -rf runs/$id; echo "== $id $(date +%H:%M)"
        $PY scripts/eval_hand.py --output runs/$id "${P[@]}" "$@" > /dev/null 2>&1; python3 -c "
import json; o = json.load(open('runs/$id/summary.json'))['overall']; print('   ', o['success'], '/', o['cases'], 'delivered,', o['timed'], 'in time')"; }

# Gripper: planner (the default project); imitation and imitation + SAC grasps per seed
# (scripts/train_gripper_grasp.sh writes configs/study/gripper_{bc,bcrl}_s<seed>.json; planned carry and reach).
ev p_gripper_planner_standard --controller reference --objects $STANDARD --planned-speed-scale 3.0 --holding-speed-scale 2.0
ev p_gripper_planner_wide --controller reference --objects $WIDE --planned-speed-scale 3.0 --holding-speed-scale 2.0
for s in 0 1 2; do
  for kind in bc bcrl; do
    [ -e configs/study/gripper_${kind}_s$s.json ] && ev p_gripper_${kind}_standard_s$s --candidate configs/study/gripper_${kind}_s$s.json --objects $STANDARD
  done
done
# Hand: scripted grasp (deterministic), then the learned hand policies as they become available.
evh p_hand_scripted_standard --policy teacher --objects $STANDARD
evh p_hand_scripted_wide --policy teacher --objects $WIDE
echo "Study evaluations done"
