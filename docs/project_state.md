# Project state

Updated 2026-10-03 (evening). Plan: [project_spec.md](project_spec.md) (machine-readable:
[`configs/study.json`](../configs/study.json)); the dashboard tracks it live. Experiments:
[experiment_log.md](experiment_log.md).

**Question:** where does RL add value in mobile manipulation, and how does that depend on the end
effector's complexity? Two end effectors (Fetch parallel gripper, 16-joint Allegro hand) × four grasp
controllers (scripted, imitation, imitation + RL, RL from scratch), on a standard and a wide-object
task, evaluated on held-out mazes with random object starts under realistic physics.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Protocol frozen | done 2026-10-03 |
| 2 | Hand trained on the Fetch (wrist-end mount; standard and wide objects) | done: BC → PPO 10M, 3 seeds per task |
| 3 | Wide objects complete the full task with the hand | in progress: task-carry fine-tunes overnight |
| 4 | Matrix filled with one seed | gripper and scripted/imitation hand done; hand imitation + RL after the fine-tunes |
| 5 | Three seeds per learned condition | gripper done; hand imitation done; hand imitation + RL overnight |
| 6 | PPO vs RLPD on the hand; ablations | RLPD seeds training (6 h each) |
| 7 | Report | figures script ready (`scripts/report_figures.py`) |
| — | Stretch: orientation-constrained delivery | only after 1–6 |

## Results so far

Protocol: 20 hold-out mazes, random object starts, realistic physics. Standard objects, 100 cases
per run; delivered (within 15k ticks) · picked · delivered given picked:

| End effector · grasp controller | Delivered | Picked | Given picked |
|---|---:|---:|---:|
| Gripper · planner | **89** (19) | 91 | 98% |
| Gripper · imitation, 3 seeds | 79/79/79 (33/32/32) | 80/80/80 | 99% |
| Gripper · imitation + SAC, 3 seeds | 77/80/80 (33/32/31) | 78/81/81 | 99% |
| Hand · scripted | 44 (16) | 47 | 94% |
| Hand · imitation, 3 seeds | 53/50/48 (19/17/16) | 54/54/49 | 96% |
| Hand · imitation + PPO | after the fine-tunes; development check (seed 0, 8M, before the fine-tune): 71 (21) | 87 | 82% |

Wide objects (master chef can, soup can ×1.6), 40 cases: gripper planner 12 (none in time); scripted
hand 0 (its side grasp fails on the Fetch); hand imitation + PPO after the fine-tunes.

Reading so far: on the gripper RL adds nothing (planner 89, learned 77–80). On the hand RL is what
makes the grasp work (picked 47 scripted, ~52 imitation, 87 with PPO), and with the hand-specific
carry posture almost every picked object is delivered. Within the deadline the RL hand already
matches the gripper planner (21 vs 19).

Earlier evidence (rig and course-task studies): PPO from scratch 0% after 24M steps on the hand rig;
BC then PPO 97% (teacher 72%); on the gripper course task (dev, 260 cases) planner 250, imitation 175,
imitation + SAC 178. Object survey: only wide, heavy cylinders separate the two end effectors
(rubber pads at ~240 N let the gripper hold objects wider than its 9.8 cm opening).

## Platform

- MuJoCo 3.14 (`pnr/env.py`), the course's robot, objects, mazes and success test. **Realistic
  physics** for both end effectors: joint velocity, acceleration and effort limits (Fetch spec,
  MoveIt limits); rubber grip μ 1.0; Fetch jaws ≤ 245 N; Allegro joints ≤ 0.7 N m.
- Allegro hand (Menagerie) mounted on the Fetch wrist flange by its wrist end (`effector='allegro'`,
  `hand_mount='wrist'`). `pnr/fetch_hand.py` runs the hand task on the real arm (palm targets →
  damped least-squares arm targets; the carry drives the base). `pnr/hand_system.py` is the full task
  with the hand: planned approach and pregrasp, learned or scripted grasp, the hand's carry posture
  (`plan_tuck`), A* carry and reach; the hand policy keeps controlling the fingers after the grasp.
- Objects: the course's 15 YCB objects plus enlarged variants 102/107/110 (soup can ×1.6, apple
  ×1.45, mug ×1.35, same density).

Course-task results for the gripper systems (realistic physics, fixed object pose; delivered, in
brackets within 15k ticks): planner dev 250 (21), hold-out 99 (12), course 95 (10); imitation 175
(69), 77 (32), 84 (49); imitation + SAC 178 (73), 77 (27), 81 (56). Course physics (PyBullet-era
settings): planner 97–99%, learned 92–94%.

## How to evaluate

Protocol runs (hold-out mazes, random object starts):

```bash
python scripts/eval.py --controller reference --set holdout --random-object-pose --planned-speed-scale 3 --holding-speed-scale 2 --output runs/p_gripper_planner_standard
python scripts/eval.py --candidate configs/imitation_sac.json --set holdout --random-object-pose --output runs/p_gripper_bcrl_standard_s0
python scripts/eval_hand.py --policy hand_w_01 --step N --set holdout --objects 14 5 2 7 10 --random-object-pose --output runs/p_hand_bcrl_standard_s0
python scripts/eval_hand.py --policy teacher --set holdout --objects 0 102 --random-object-pose --output runs/p_hand_scripted_wide
```

Add each finished run id to its cell's `evals` in `configs/study.json`; the dashboard picks it up.
Hand training: `scripts/train_hand.py` (`--robot fetch --hand-mount wrist --wrist-box .08 .5`);
demonstrations: `scripts/hand_teacher_demos.py`; object survey: `scripts/object_survey.py`.

## Operations

- Dashboard: `python dashboard/server.py --port 8766` → http://127.0.0.1:8766.
- `runs/archive_pybullet/` and `runs/archive_invalid/` hold superseded runs.
