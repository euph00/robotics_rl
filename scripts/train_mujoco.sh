#!/bin/bash
# Full MuJoCo training recipe for the three learned skills, as run on 2026-10-01
# (about 2 hours on 48 CPUs + one GPU). Each stage writes runs/<run-id>/.
#
#   grasp:  240 scripted contact-grasp demos -> BC (24k updates) -> demo-regularized SAC+HER (12k decisions)
#   reach:  SAC+HER from reward only (20k steps)
#   carry:  A* look-ahead teacher demos and noise recoveries on 192 TRAIN mazes (foam brick),
#           plus teacher carries after learned-grasp handovers (five objects)
#           -> BC (32k updates, CNN map encoder) -> demo-regularized SAC+HER (40k decisions)
#
# Physics: PNR_PHYSICS=realistic (default) or course; run IDs start with $PREFIX (mjr / mj).
# Then: bash scripts/evaluate_mujoco.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PNR_PHYSICS="${PNR_PHYSICS:-realistic}"
P="${PREFIX:-$([ "$PNR_PHYSICS" = course ] && echo mj || echo mjr)}"

# Reach and grasp data are independent of everything else.
$PY -m pnr.train --skill reach --run-id ${P}_reach01 --steps 20000 --seed 0 --scene-seed 42 --object-idx 14 \
  --horizon 100 --num-envs 4 --device cpu --eval-every 2500 --eval-episodes 1 --goal-features &
$PY scripts/phase2_grasp_demos.py --run-id ${P}_grasp_demos01 --objects 14 5 2 7 10 \
  --scene-seeds 42 42001 42002 42003 42004 42005 --workers 10 --repetitions 2 --jitter-xy 0.01 --seed 61000 \
  --lift-range .2 .26 --grasp-proposal-mode clearance-v2 --teacher-approach-control ray-tilted

$PY scripts/phase2_train_grasp.py --run-id ${P}_grasp01 --demonstrations runs/${P}_grasp_demos01/artifacts/demonstrations.pkl \
  --workers 6 --steps 12000 --bc-updates 24000 --bc-eval-every 4000 --warmup-updates 2000 --batch-size 400 \
  --learning-rate 3e-5 --horizon 100 --eval-horizon 120 --eval-every 2000 --checkpoint-every 2000 \
  --validation-seeds 43001 43002 --seed 8101 --device cpu --bc-weight 200 --demo-batch-size 400 \
  --lift-range .2 .26 --eval-lift-m .24 --grasp-proposal-mode clearance-v2 &
GRASP=$!

# Carry teacher data (A* look-ahead follower; recovery modes inject noise and record corrections).
W=32
carry() { $PY scripts/phase2_carry_demos.py --delta-scale .075 --workers $W "$@"; }
carry --run-id ${P}_carry_demos01 --bank configs/phase2_scene_bank.json --layouts 16 --episodes 4
carry --run-id ${P}_carry_demos02 --bank configs/phase2_scene_bank.json --start-layout 16 --layouts 48 --episodes 6
carry --run-id ${P}_carry_recovery64 --bank configs/phase2_scene_bank.json --layouts 64 --episodes 6 --mode recovery --recovery-noise-std .12
carry --run-id ${P}_carry_geometric64 --bank configs/phase2_scene_bank.json --layouts 64 --episodes 4 --mode geometric-recovery --recovery-noise-std 0
carry --run-id ${P}_carry_expanded_standard01 --bank configs/phase2_scene_bank_expanded.json --start-layout 64 --layouts 128 --episodes 4 --mode standard
carry --run-id ${P}_carry_expanded_geometric01 --bank configs/phase2_scene_bank_expanded.json --start-layout 64 --layouts 128 --episodes 4 --mode geometric-recovery --recovery-noise-std .12

wait $GRASP
$PY scripts/phase2_freeze_grasp.py runs/${P}_grasp01
$PY scripts/phase2_learned_carry_demos.py --grasp runs/${P}_grasp01/candidate/grasp.zip --bank configs/phase2_scene_bank.json \
  --output runs/${P}_learned_carry_demos02 --workers 8 --collection-seed 8401
$PY - "$P" <<'PYEOF'
import sys
from pathlib import Path
from pnr.demonstrations import merge_carry_datasets
P = sys.argv[1]
sources = [Path(f'runs/{r}/artifacts/demonstrations.pkl') for r in (f'{P}_carry_demos01', f'{P}_carry_demos02', f'{P}_carry_recovery64',
           f'{P}_carry_geometric64', f'{P}_carry_expanded_standard01', f'{P}_carry_expanded_geometric01')]
sources += sorted(Path(f'runs/{P}_learned_carry_demos02').rglob('demonstrations.pkl'))
destination = Path(f'runs/{P}_carry_mix02/artifacts/demonstrations.pkl')
destination.parent.mkdir(parents=True, exist_ok=True)
print(merge_carry_datasets(sources, destination, bank_path='configs/phase2_scene_bank_expanded.json',
                           parent_bank_paths=['configs/phase2_scene_bank.json'], object_indices=[14, 5, 2, 7, 10]))
PYEOF

COMMON="--demonstrations runs/${P}_carry_mix02/artifacts/demonstrations.pkl --bank configs/phase2_scene_bank_expanded.json \
  --layouts 192 --workers 8 --seed 7401 --warmup-updates 2000 --critic-warmup-tau 0.1 --learning-rate 3e-05 \
  --delta-scale 0.075 --horizon 900 --eval-every 8000 --checkpoint-every 8000 --device cuda --validation-layouts 12 \
  --encoder cnn32 --balance-objects --preparation-grasp runs/${P}_grasp01/candidate/grasp.zip --training-layouts 8 \
  --buffer-size 600000 --bc-weight 100 --demo-batch-size 256"
$PY scripts/phase2_train_carry.py --run-id ${P}_carry_init02 $COMMON --bc-updates 32000 --steps 0
$PY scripts/phase2_train_carry.py --run-id ${P}_carry_sac02 $COMMON --initial-checkpoint runs/${P}_carry_init02/checkpoints/bc_initialized.zip \
  --bc-updates 0 --steps 40000 --balance-replay-objects
wait
CONFIGS=$([ "$PNR_PHYSICS" = course ] && echo configs/course_physics || echo configs)
$PY - "$P" "$PNR_PHYSICS" "$CONFIGS" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
prefix, physics, directory = sys.argv[1:]
final = sorted(Path(f'runs/{prefix}_carry_sac02/checkpoints').glob('step_000040000_*/model.zip'))[-1]
skills = {'imitation_sac': {'grasp': f'runs/{prefix}_grasp01/candidate/grasp.zip', 'carry': str(final),
                            'reach': f'runs/{prefix}_reach01/checkpoints/best/model.zip'},
          'imitation_only': {'grasp': f'runs/{prefix}_grasp01/candidate/grasp_zero_online.zip',
                             'carry': f'runs/{prefix}_carry_init02/checkpoints/bc_initialized.zip',
                             'reach': f'runs/{prefix}_reach01/checkpoints/best/model.zip'}}
about = {'imitation_sac': 'Behaviour-cloned grasp and carry actors fine-tuned with demonstration-regularized SAC+HER '
                          '(grasp: best online checkpoint; carry: final 40k-decision checkpoint), SAC+HER reach.',
         'imitation_only': 'Behaviour-cloned grasp and carry actors with zero online updates; SAC+HER reach.'}
execution = {'grasp_horizon': 600, 'carry_horizon': 1200, 'reach_horizon': 150, 'delta_scale': 0.075,
             'planned_speed_scale': 3.0, 'holding_speed_scale': 2.0, 'deadline_physics_ticks': 15000,
             'grasp_lift_m': 0.24, 'reach_tolerance_m': 0.08, 'grasp_proposal_mode': 'clearance-v2', 'physics': physics}
Path(directory).mkdir(parents=True, exist_ok=True)
for name, paths in skills.items():
    config = {'candidate_id': name, 'simulator': 'mujoco', 'physics': physics, 'description': about[name] + ' Planned approach and carry posture.',
              'checkpoints': {k: {'frozen_path': v, 'sha256': hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k, v in paths.items()},
              'execution': execution}
    Path(directory, f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')
print('wrote', directory)
PYEOF
