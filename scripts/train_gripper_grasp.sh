#!/bin/bash
# Gripper grasp for the study (docs/project_spec.md): planner demonstrations from random object
# starts, behaviour cloning, then demonstration-regularised SAC+HER; one training run per seed.
# Writes configs/study/gripper_{bc,bcrl}_s<seed>.json: the imitation (BC grasp) and imitation + RL
# (BC + SAC grasp) gripper systems, with planned approach, carry and reach like the hand systems.
#   bash scripts/train_gripper_grasp.sh 0 1 2
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PNR_PHYSICS=realistic
DEMOS=study_gripper_grasp_demos
if [ ! -e runs/$DEMOS/artifacts/demonstrations.pkl ]; then
  # 24 scene seeds, each with its own random object start (validation uses 43001-43002).
  $PY scripts/phase2_grasp_demos.py --run-id $DEMOS --objects 14 5 2 7 10 --scene-seeds $(seq 42001 42024) \
    --workers 16 --repetitions 2 --jitter-xy 0.01 --seed 61000 --lift-range .2 .26 \
    --grasp-proposal-mode clearance-v2 --teacher-approach-control ray-tilted --random-object-pose
fi
mkdir -p configs/study
for s in "$@"; do
  RUN=study_gripper_grasp_s$s
  if [ ! -e runs/$RUN/candidate/grasp.zip ]; then
    rm -rf runs/$RUN
    $PY scripts/phase2_train_grasp.py --run-id $RUN --demonstrations runs/$DEMOS/artifacts/demonstrations.pkl \
      --workers 6 --steps 12000 --bc-updates 24000 --bc-eval-every 4000 --warmup-updates 2000 --batch-size 400 \
      --learning-rate 3e-5 --horizon 100 --eval-horizon 120 --eval-every 2000 --checkpoint-every 2000 \
      --validation-seeds 43001 43002 --seed $((8101 + s)) --device cpu --bc-weight 200 --demo-batch-size 400 \
      --lift-range .2 .26 --eval-lift-m .24 --grasp-proposal-mode clearance-v2 --random-object-pose
    $PY scripts/phase2_freeze_grasp.py runs/$RUN
  fi
  $PY - "$s" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
seed = sys.argv[1]
for kind, base, grasp in (('bc', 'configs/imitation_only.json', 'grasp_zero_online.zip'),
                          ('bcrl', 'configs/imitation_sac.json', 'grasp.zip')):
    manifest = json.loads(Path(base).read_text())
    path = f'runs/study_gripper_grasp_s{seed}/candidate/{grasp}'
    manifest['candidate_id'] = f'study_gripper_{kind}_s{seed}'
    manifest['description'] = (f'Study gripper system, seed {seed}: {"behaviour-cloned" if kind == "bc" else "behaviour-cloned then SAC+HER"} '
                               'grasp trained from random object starts; planned approach, carry posture, A* carry and reach.')
    manifest['checkpoints']['grasp'] = {'frozen_path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    manifest['execution'].update(carry_mode='planned', reach_mode='planned')
    Path(f'configs/study/gripper_{kind}_s{seed}.json').write_text(json.dumps(manifest, indent=1) + '\n')
print('manifests written for seed', seed)
PYEOF
done
