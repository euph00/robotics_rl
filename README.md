# Where does RL add value in mobile manipulation? (Pick–Nav–Reach, MuJoCo)

A Fetch robot must drive to a table, grasp a supplied YCB object, carry it through an unseen
maze, and hold it within 10 cm of an elevated goal sphere. The task comes from the course project
[NUS-LinS-Lab/Pick-Nav-Reach-Default-Project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project),
pinned at `54ab2d1` ([upstream.lock.json](upstream.lock.json)), and runs here in **MuJoCo**
(`pnr/env.py`) with the course's robot, objects, mazes and success test, under realistic actuator
limits for both end effectors.

**Study ([spec](docs/project_spec.md), [`configs/study.json`](configs/study.json)):** how much RL adds
to the grasp, and how that depends on the end effector's complexity. Two end effectors (the Fetch
parallel gripper, a 16-joint Allegro hand on the Fetch's wrist) × four grasp controllers (scripted,
imitation, imitation + RL, RL from scratch), on standard objects and on wide, heavy cylinders,
evaluated on held-out mazes with random object starts.

**Status (2026-10-03):** on the gripper RL adds nothing measurable over imitation of the planner;
on the hand, imitation then PPO lifts and carries 97% on the test rig (imitation alone 14%, the
scripted teacher 72%, RL from scratch 0%) and delivers 49 of 60 development cases in the full task
on the Fetch (scripted hand 40, gripper planner 57; the hand is faster: 28 vs 1 within the
deadline). Protocol runs are in progress: [docs/project_state.md](docs/project_state.md).

## Systems

A fixed supervisor runs each stage from the original start, with no resets and no object attachment:

| Stage | Gripper systems | Hand systems |
|---|---|---|
| Approach table, pregrasp | Planned (antipodal grasp from the point cloud, IK, collision-checked motion) | Planned (hand pregrasp, IK) |
| Grasp and lift | Planner, or learned (BC, optionally SAC fine-tuned) | Scripted, or learned (BC, optionally PPO fine-tuned) |
| Carry posture | Planned | Planned |
| Carry through the maze | A* (planner), or learned base controller on a maze map | A* |
| Reach goal while holding | Base and torso servo, or SAC+HER | Base, torso and arm servo |

## Layout

| Path | Contents |
|---|---|
| `pnr/env.py` | MuJoCo environment: course scene, motor law, contacts, snapshots, rendering |
| `pnr/` | Skills, planners, feature extractors, SAC variants, learned system |
| `pnr/hand.py`, `pnr/hand_env.py`, `pnr/hand_teacher.py` | Allegro test rig, hand grasp task, scripted grasps |
| `pnr/fetch_hand.py`, `pnr/hand_system.py` | The hand on the Fetch; full task with the hand |
| `scripts/eval.py`, `scripts/eval_hand.py` | Full-task evaluation, gripper and hand systems (start here) |
| `scripts/train_hand.py`, `scripts/hand_teacher_demos.py` | Hand training (BC, PPO) and scripted demonstrations |
| `scripts/train_mujoco.sh`, `scripts/evaluate_mujoco.sh` | The full training recipe and every evaluation behind the docs |
| `scripts/phase2_{grasp,carry,learned_carry}_demos.py` | Demonstration collection |
| `scripts/phase2_train_{grasp,carry}.py`, `pnr/train.py` | Skill training (grasp, carry, reach) |
| `scripts/record.py`, `scripts/dashboard_maze.py` | Episode videos/trajectories; dashboard maze figure |
| `configs/` | Maze banks and candidate manifests (`imitation_sac.json`, `imitation_only.json`) |
| `runs/` | Checkpoints, datasets, evaluation results (ignored by Git; PyBullet era in `runs/archive_pybullet/`) |
| `dashboard/` | Read-only live dashboard: plan, results matrix, training, clips |

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

- [Project spec](docs/project_spec.md): question, study matrix, frozen evaluation protocol, milestones.
- [Project state](docs/project_state.md): current results and next steps.
- [Experiment log](docs/experiment_log.md): MuJoCo-era experiments.
- [PyBullet era](docs/archive/pybullet/): the original course simulator — results, experiment log,
  upstream audit and the dexterous-hand spike, kept for the report.
