#!/bin/bash
# MuJoCo evaluations behind the dashboard and docs (about 2 hours on 44 workers).
# PNR_PHYSICS=realistic (default; runs mjr_*, configs/) or course (runs ${P}_*, configs/course_physics/):
# three systems x {dev, holdout, course}, SAC-vs-imitation and planner-vs-learned stage swaps,
# hand-written carry rules, planned-speed and carry-speed sweeps. Completed runs are skipped.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PNR_PHYSICS="${PNR_PHYSICS:-realistic}"
P="${PREFIX:-$([ "$PNR_PHYSICS" = course ] && echo mj || echo mjr)}"
CONFIGS=$([ "$PNR_PHYSICS" = course ] && echo configs/course_physics || echo configs)
SAC=$CONFIGS/imitation_sac.json; BC=$CONFIGS/imitation_only.json
BCGRASP=runs/${P}_grasp01/candidate/grasp_zero_online.zip; BCCARRY=runs/${P}_carry_init02/checkpoints/bc_initialized.zip
ev() { local id=$1; shift; [ -e runs/$id/summary.json ] && return; rm -rf runs/$id; echo "== $id"; $PY scripts/eval.py --output runs/$id --label $id "$@" 2>&1 | grep -E "^overall|failure"; }
for set in dev holdout course; do
  ev ${P}_planner_$set --controller reference --set $set --planned-speed-scale 3.0 --holding-speed-scale 2.0 --physics $PNR_PHYSICS
  ev ${P}_imitation_$set --candidate $BC --set $set
  ev ${P}_imitation_sac_$set --candidate $SAC --set $set
done
ev ${P}_ablation_sac_grasp_bc_carry --candidate $SAC --carry $BCCARRY --set dev
ev ${P}_ablation_bc_grasp_sac_carry --candidate $SAC --grasp $BCGRASP --set dev
ev ${P}_swap_planned_carry --candidate $BC --carry-mode planned --set dev
ev ${P}_swap_planned_grasp --candidate $BC --grasp-mode planned --set dev
ev ${P}_rules_turn30 --candidate $BC --carry-max-turn-deg 30 --set dev
ev ${P}_rules_turn15 --candidate $BC --carry-max-turn-deg 15 --set dev
ev ${P}_rules_slip2mm --candidate $BC --carry-slip-drift-mm 2 --set dev
ev ${P}_rules_turn15_slip2mm --candidate $BC --carry-max-turn-deg 15 --carry-slip-drift-mm 2 --set dev
for s in 1.0 1.5 2.0 2.5 3.0; do ev ${P}_planspeed_${s/./} --candidate $SAC --planned-speed-scale $s --holding-speed-scale $s --set dev; done
ev ${P}_carryspeed_0035 --candidate $SAC --planned-speed-scale 1.0 --holding-speed-scale 1.0 --delta-scale 0.035 --carry-horizon 2600 --set dev
ev ${P}_carryspeed_005 --candidate $SAC --planned-speed-scale 1.0 --holding-speed-scale 1.0 --delta-scale 0.05 --carry-horizon 1800 --set dev
echo "Evaluations done; rubric: python scripts/rubric.py runs/${P}_imitation_sac_course"
