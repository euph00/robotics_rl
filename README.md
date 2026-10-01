# Goal-conditioned RL for Pick–Nav–Reach

A Fetch robot in PyBullet must drive to a table, grasp a supplied YCB object from
its point cloud, carry it through a random maze, and hold it within 10 cm of an
elevated goal sphere. The course task is
[NUS-LinS-Lab/Pick-Nav-Reach-Default-Project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project),
pinned at `54ab2d1` ([upstream.lock.json](upstream.lock.json)); the checkout under
`upstream/` stays unchanged.

**Current goal:** a strong SAC+HER baseline on the original gripper that
generalises to unseen mazes. Later: TD3+HER, a modified gripper, and a 3-seed
comparison ([spec](docs/project_spec.md)).

**Status:** candidate 3 ([configs/candidate03.json](configs/candidate03.json)) completes
**80/100** unseen-maze tasks on a fresh hold-out bank, 79/100 within 15,000 physics
ticks. Details: [docs/project_state.md](docs/project_state.md). History of all
earlier experiments: [docs/experiment_log.md](docs/experiment_log.md).

## System

A fixed supervisor runs three learned skills from the original task start, with
no resets between skills and no object attachment:

| Stage | Controller |
|---|---|
| Approach table, pre-grasp | Planned: antipodal grasp proposal from the point cloud, IK, collision-checked motion |
| Grasp and lift | Learned: SAC+HER (BC-initialized, demonstration-regularized) |
| Carry posture | Planned arm posture |
| Carry through maze | Learned: SAC+HER over base XY from maze-geometry maps; no route oracle |
| Reach goal while holding | Learned: SAC+HER |

A classical reference (A* navigation, planned grasp and reach) is in
`pnr/reference.py` and `scripts/run_reference.py`.

## Layout

| Path | Contents |
|---|---|
| `pnr/` | Environment adapter, skills, planners, feature extractors, SAC variants |
| `scripts/eval.py` | Full-task evaluation on fixed maze sets (start here) |
| `scripts/phase2_train_{grasp,carry}.py`, `pnr/train.py` | Skill training (grasp, carry, reach) |
| `scripts/phase2_{grasp,carry,learned_carry}_demos.py` | Demonstration collection |
| `scripts/build_scene_bank.py`, `build_confirmation_bank.py` | Maze banks (train/validation/test) |
| `configs/` | Scene banks and candidate manifests (checkpoints + execution settings) |
| `runs/` | Checkpoints, datasets, evaluation results (ignored by Git) |
| `dashboard/` | Read-only live dashboard |

## Setup

```bash
source .venv/bin/activate   # Python 3.10, PyBullet, SB3 2.7, PyTorch 2.7.1+cu118
python -m pytest -q         # ~3 minutes
```

Fresh machine: clone the upstream repository into `upstream/Pick-Nav-Reach-Default-Project`,
check out `54ab2d1faa54d1aa7e2738f114531a76c6514cd3`, then
`PNR_PYTHON_BIN=python3.10 bash scripts/setup.sh`.

## Common commands

Evaluate a candidate on the 260-case development set (52 unseen mazes × 5 objects, ~5 minutes on 44 CPU workers):

```bash
python scripts/eval.py --candidate configs/phase2_candidate02.json --set dev --output runs/eval_example
# overrides: --carry runs/.../model.zip, --delta-scale 0.05 --carry-horizon 1800, --grasp-settle-calls 30
```

Sets: `validation` (60), `test1` (100), `test2` (100), `dev` (all 260). The
`reserve` set (100 cases on 20 unseen mazes) is kept sealed for a final claim.

Physical reference rollout with video:

```bash
python scripts/run_reference.py --seed 42 --object-idx 14 --video --output runs/reference_example
```

Carry training as used for the current candidate (BC-initialized actor, demonstration-regularized SAC+HER):

```bash
python scripts/phase2_train_carry.py --run-id carry_example \
  --initial-checkpoint runs/phase2_actor_spatial_freeze_plan01/artifacts/initial_actor.zip \
  --demonstrations runs/phase2_carry_learned_mix01/artifacts/demonstrations.pkl \
  --bank configs/phase2_scene_bank_expanded.json --layouts 192 --workers 8 --bc-updates 0 \
  --warmup-updates 2000 --critic-warmup-tau 0.1 --steps 40000 --learning-rate 3e-05 \
  --delta-scale 0.075 --horizon 900 --validation-layouts 12 --encoder gated48-k20-f5 \
  --preparation-grasp runs/phase2_actor_spatial_freeze_plan01/artifacts/grasp.zip \
  --training-layouts 8 --buffer-size 600000 --bc-weight 100 --balance-objects \
  --balance-replay-objects --freeze-actor-spatial
```

Dashboard: `python dashboard/server.py --port 8766`, then open http://127.0.0.1:8766
(forward the port over SSH from a remote machine).

## Documents

- [Project spec](docs/project_spec.md): task authority, learning boundary, final study design.
- [Project state](docs/project_state.md): current results and next steps.
- [Experiment log](docs/experiment_log.md): condensed history; full reports in `docs/archive/`.
- [Upstream audit](docs/upstream_audit.md): source defects and compatibility decisions.
