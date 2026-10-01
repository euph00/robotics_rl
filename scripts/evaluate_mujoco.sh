#!/bin/bash
# MuJoCo evaluations behind the dashboard and docs (about 2 hours on 44 workers):
# three systems x {dev, holdout, course}, SAC-vs-imitation and planner-vs-learned stage swaps,
# hand-written carry rules, planned-speed and carry-speed sweeps. Completed runs are skipped.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SAC=configs/imitation_sac.json; BC=configs/imitation_only.json
BCGRASP=runs/mj_grasp01/candidate/grasp_zero_online.zip; BCCARRY=runs/mj_carry_init02/checkpoints/bc_initialized.zip
ev() { local id=$1; shift; [ -e runs/$id/summary.json ] && return; rm -rf runs/$id; echo "== $id"; $PY scripts/eval.py --output runs/$id --label $id "$@" 2>&1 | grep -E "^overall|failure"; }
for set in dev holdout course; do
  ev mj_planner_$set --controller reference --set $set --planned-speed-scale 3.0 --holding-speed-scale 2.0
  ev mj_imitation_$set --candidate $BC --set $set
  ev mj_imitation_sac_$set --candidate $SAC --set $set
done
ev mj_ablation_sac_grasp_bc_carry --candidate $SAC --carry $BCCARRY --set dev
ev mj_ablation_bc_grasp_sac_carry --candidate $SAC --grasp $BCGRASP --set dev
ev mj_swap_planned_carry --candidate $BC --carry-mode planned --set dev
ev mj_swap_planned_grasp --candidate $BC --grasp-mode planned --set dev
ev mj_rules_turn30 --candidate $BC --carry-max-turn-deg 30 --set dev
ev mj_rules_turn15 --candidate $BC --carry-max-turn-deg 15 --set dev
ev mj_rules_slip2mm --candidate $BC --carry-slip-drift-mm 2 --set dev
ev mj_rules_turn15_slip2mm --candidate $BC --carry-max-turn-deg 15 --carry-slip-drift-mm 2 --set dev
for s in 1.0 1.5 2.0 2.5 3.0; do ev mj_planspeed_${s/./} --candidate $SAC --planned-speed-scale $s --holding-speed-scale $s --set dev; done
ev mj_carryspeed_0035 --candidate $SAC --planned-speed-scale 1.0 --holding-speed-scale 1.0 --delta-scale 0.035 --carry-horizon 2600 --set dev
ev mj_carryspeed_005 --candidate $SAC --planned-speed-scale 1.0 --holding-speed-scale 1.0 --delta-scale 0.05 --carry-horizon 1800 --set dev
echo "Evaluations done; rubric: python scripts/rubric.py runs/mj_imitation_sac_course"
