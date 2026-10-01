# Goal-conditioned RL for Pick–Nav–Reach (MuJoCo)

A Fetch robot must drive to a table, grasp a supplied YCB object from its point
cloud, carry it through a random maze, and hold it within 10 cm of an elevated
goal sphere. The task comes from the course project
[NUS-LinS-Lab/Pick-Nav-Reach-Default-Project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project),
pinned at `54ab2d1` ([upstream.lock.json](upstream.lock.json)). The project runs it in
**MuJoCo** (`pnr/env.py`) with the course's robot, objects, mazes, seeds, motor law,
friction coefficients and success criterion; the course checkout under `upstream/`
supplies the assets and maze generator and stays unchanged.

**Study:** how much each component contributes — a classical planner, imitation of
that planner, and online RL (SAC+HER, later TD3+HER) on top of imitation.

**Status (2026-10-01):** on 20 fresh mazes × 5 objects, the planner alone completes
98/100, imitation 92/100 and imitation + SAC 92/100. SAC fine-tuning adds nothing
measurable over imitation. Details: [docs/project_state.md](docs/project_state.md).

## System

A fixed supervisor runs each phase from the original start, with no resets between
phases and no object attachment:

| Phase | Planner only | Imitation / Imitation + SAC |
|---|---|---|
| Approach table, pre-grasp | Antipodal grasp proposal from the point cloud, IK, collision-checked motion | same (planned) |
| Grasp and lift | Planned descent, close, lift | Learned (BC, optionally SAC+HER fine-tuned) |
| Carry posture | Planned arm posture | same (planned) |
| Carry through maze | A* waypoints | Learned base controller on a maze map (no route oracle) |
| Reach goal while holding | Base and torso servo | Learned (SAC+HER from reward) |

## Layout

| Path | Contents |
|---|---|
| `pnr/env.py` | MuJoCo environment: course scene, motor law, contacts, snapshots, rendering |
| `pnr/` | Skills, planners, feature extractors, SAC variants, learned system |
| `scripts/eval.py` | Full-task evaluation on fixed maze sets (start here) |
| `scripts/train_mujoco.sh`, `scripts/evaluate_mujoco.sh` | The full training recipe and every evaluation behind the docs |
| `scripts/phase2_{grasp,carry,learned_carry}_demos.py` | Demonstration collection |
| `scripts/phase2_train_{grasp,carry}.py`, `pnr/train.py` | Skill training (grasp, carry, reach) |
| `scripts/record.py`, `scripts/dashboard_maze.py` | Episode videos/trajectories; dashboard maze figure |
| `configs/` | Maze banks and candidate manifests (`imitation_sac.json`, `imitation_only.json`) |
| `runs/` | Checkpoints, datasets, evaluation results (ignored by Git; PyBullet era in `runs/archive_pybullet/`) |
| `dashboard/` | Read-only live dashboard |

## Setup

```bash
source .venv/bin/activate   # Python 3.10, MuJoCo 3.14, SB3 2.7, PyTorch 2.7.1+cu118
python -m pytest -q         # ~2 minutes
```

Fresh machine: clone the upstream repository into `upstream/Pick-Nav-Reach-Default-Project`,
check out `54ab2d1faa54d1aa7e2738f114531a76c6514cd3`, then
`PNR_PYTHON_BIN=python3.10 bash scripts/setup.sh`. Rendering uses headless EGL on the
GPU (`MUJOCO_GL=egl`, device 0 by default).

## Common commands

Evaluate a system on the 260-case development set (52 unseen mazes × 5 objects, ~4 minutes on 44 CPU workers):

```bash
python scripts/eval.py --candidate configs/imitation_sac.json --set dev --output runs/eval_example
python scripts/eval.py --controller reference --set course --planned-speed-scale 3 --holding-speed-scale 2 --output runs/planner_example
python scripts/rubric.py runs/eval_example   # course rubric estimate per trial
```

Sets: `validation` (60), `test1` (100), `test2` (100), `dev` (all 260), `holdout` (100 on
20 fresh mazes), `course` (course seeds 1000–1019). The `reserve` set (100 cases on 20
unseen mazes) stays sealed for a final claim. Overrides: `--grasp/--carry/--reach model.zip`,
`--grasp-mode planned`, `--carry-mode planned`, `--delta-scale`, `--carry-horizon`.

Record episodes (video, preview, per-call trajectory) exactly as the evaluator runs them:

```bash
python scripts/record.py --candidate configs/imitation_sac.json --set dev \
  --cases validation_008_object_05 --output runs/recording_example
```

Retrain everything and rerun all evaluations: `bash scripts/train_mujoco.sh && bash scripts/evaluate_mujoco.sh`
(about 2 + 2 hours on 48 CPUs and one GPU).

Dashboard: `python dashboard/server.py --port 8766`, then open http://127.0.0.1:8766
(forward the port over SSH from a remote machine).

## Documents

- [Project spec](docs/project_spec.md): task authority, learning boundary, study design.
- [Project state](docs/project_state.md): current results and next steps.
- [Experiment log](docs/experiment_log.md): MuJoCo-era experiments.
- [PyBullet era](docs/archive/pybullet/): the original course simulator — results, experiment log,
  upstream audit and the dexterous-hand spike, kept for the report.
