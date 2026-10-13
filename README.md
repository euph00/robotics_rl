# Where does reinforcement learning pay off in mobile manipulation?

A course project (CS5478, Intelligent Robots) on the [Pick–Nav–Reach default project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project)
(pinned at `54ab2d1`, [upstream.lock.json](upstream.lock.json)): a Fetch robot drives to a table, grasps an object, carries
it through an unseen maze and holds it within 10 cm of an elevated goal. The task runs here in MuJoCo with the course's robot,
objects, mazes and success test, under realistic actuator limits.

**Start with the project site: [`site/index.html`](site/index.html)** (open it from disk, or
`python3 -m http.server 8767 --directory site`). It tells the whole story, from the SAC+HER proposal through the move to
MuJoCo and the dexterous hand to the final study, with diagrams, explainers, charts and clips.

**Answer.** RL pays off where hand-engineering runs out. Full-task delivery on 20 sealed maze layouts, 5 seeds pooled:

| Grasp controller | Parallel gripper | Allegro hand (16 joints) | Wide cylinders: gripper's best / hand |
|---|---|---|---|
| Hand-written (planner / script) | **0.85** | 0.40 | 0.27 / 0.00 |
| Imitation | 0.69 | 0.49 | 0.00 / 0.01 |
| Imitation + RL (PPO) | 0.62 | **0.76** | 0.21 / **0.68** |
| RL from scratch (PPO) | 0.61 | 0.18 | **0.39** / 0.00 |

PPO vs RLPD from the same imitation start, 4M steps, 5 seeds: lift-and-carry 0.65 vs 0.07 (hand), 0.62 vs 0.03 (gripper).

## Layout

| Path | Contents |
|---|---|
| `pnr/env.py` | MuJoCo environment: course scene, motor law, realistic physics, contacts, rendering |
| `pnr/reference.py`, `pnr/planning.py`, `pnr/clearance_grasps.py` | The gripper's classical pipeline: antipodal grasp, IK, collision-checked motion, A* carry, reach |
| `pnr/hand.py`, `pnr/fetch_hand.py`, `pnr/fetch_gripper.py` | The Allegro hand (rig and on the Fetch) and the gripper behind the same interface |
| `pnr/hand_env.py`, `pnr/hand_teacher.py` | Grasp-learning environment for both end effectors; scripted teachers |
| `pnr/hand_system.py` | The full task with a learned (or scripted) grasp and planned approach, carry and reach |
| `pnr/rlpd.py` | RLPD components (demonstration-mixed replay, layer-normalised critics) |
| `pnr/scenes.py`, `pnr/confirmation_scenes.py` | Maze banks and evaluation cases |
| `scripts/train_hand.py`, `scripts/hand_teacher_demos.py` | Training (BC, PPO, RLPD) and demonstration collection |
| `scripts/eval.py`, `scripts/eval_hand.py` | Full-task evaluation: the gripper planner; systems with a learned or scripted grasp |
| `scripts/tune_v2.sh`, `campaign_v2.sh`, `dev_v2.sh`, `algorithms_v2.sh`, `final_v2.py` | The study, in order (below) |
| `scripts/site_data.py`, `scripts/record_clips.py` | The site's data bundle and clips |
| `configs/` | Maze banks and the frozen systems (`final_v2.json`) |
| `site/` | The project site |
| `docs/` | [Spec](docs/project_spec.md), [experiment log](docs/experiment_log.md), [PyBullet era](docs/archive/pybullet/) |
| `runs/` | Checkpoints, datasets and results (not in Git) |

The PyBullet-era code, the SAC+HER skill stack of the first phase, the first study's scripts and the old live dashboard are at
git tag `pre-cleanup-2026-10-13`.

## Setup

```bash
source .venv/bin/activate   # Python 3.10, MuJoCo 3.14, SB3 2.7, PyTorch 2.7.1+cu118
python -m pytest -q         # under a minute
```

Fresh machine: clone the upstream repository into `upstream/Pick-Nav-Reach-Default-Project`, check out
`54ab2d1faa54d1aa7e2738f114531a76c6514cd3`, then `PNR_PYTHON_BIN=python3.10 bash scripts/setup.sh`. Rendering uses headless EGL
on the GPU (`MUJOCO_GL=egl`).

## Reproduce the study

Each driver is idempotent (finished runs are skipped) and keeps at most four 22-worker runs alive; launch long ones with
`setsid nohup … &`. Times are for 48 CPU cores and one A100.

```bash
bash scripts/tune_v2.sh            # equal-budget tuning sweep, 16 runs (~9 h); report: python scripts/tune_v2_report.py
bash scripts/campaign_v2.sh        # demonstrations, then 40 training runs: 2 effectors x {imitation+PPO, PPO} x 2 tasks x 5 seeds (~2 days)
bash scripts/dev_v2.sh             # full-task development evaluations, grip choice (~3 h)
python scripts/final_v2.py freeze  # manifest of the 64 frozen systems (commit it before the next step)
python scripts/final_v2.py run     # one run on the sealed reserve bank (~1 h); report: python scripts/final_v2.py report
bash scripts/algorithms_v2.sh      # PPO vs RLPD, 24 runs (~5 days, RLPD-bound)
python scripts/site_data.py        # rebuild site/data/results.js from runs/
```

Single evaluations:

```bash
python scripts/eval.py --set dev --random-object-pose --planned-speed-scale 3 --holding-speed-scale 2 --output runs/planner_example
python scripts/eval_hand.py --policy v2_hand_bcppo_std_s0 --step 9000002 --set dev --random-object-pose --objects 14 5 2 7 10 --output runs/hand_example
```

Sets: `validation` (60), `test1` (100), `test2` (100), `dev` (all 260), `holdout` (100), `course` (course seeds 1000–1019),
`reserve` (sealed; `--allow-reserve`). Wide objects: `--objects 0 102 --starts 3`.
