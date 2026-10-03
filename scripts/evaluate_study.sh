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

# Gripper: planner (the default project), imitation, imitation + SAC (seed 0 = the mjr_ training run).
ev p_gripper_planner_standard --controller reference --objects $STANDARD --planned-speed-scale 3.0 --holding-speed-scale 2.0
ev p_gripper_planner_wide --controller reference --objects $WIDE --planned-speed-scale 3.0 --holding-speed-scale 2.0
ev p_gripper_bc_standard_s0 --candidate configs/imitation_only.json --objects $STANDARD
ev p_gripper_bcrl_standard_s0 --candidate configs/imitation_sac.json --objects $STANDARD
# Hand: scripted grasp (deterministic), then the learned hand policies as they become available.
evh p_hand_scripted_standard --policy teacher --objects $STANDARD
evh p_hand_scripted_wide --policy teacher --objects $WIDE
echo "Study evaluations done"
