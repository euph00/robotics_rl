# Project state

Updated 2026-10-04. Plan: [project_spec.md](project_spec.md) (machine-readable:
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
| 2 | Hand trained on the Fetch (wrist-end mount; standard and wide objects) | done: BC → PPO 10M → task-carry fine-tune, 3 seeds per task |
| 3 | Wide objects complete the full task with the hand | done: 14/10/14 of 40 (gripper 12/40) |
| 4 | Matrix filled with one seed | done |
| 5 | Three seeds per learned condition | done |
| 6 | PPO vs RLPD on the hand; ablations | done: RLPD 3–12% after 6 h; fine-tune ablation |
| 7 | Report | figures (`scripts/report_figures.py`); text and clips next |
| — | Stretch: orientation-constrained delivery | only after 1–6 |

## Results

Protocol: 20 hold-out mazes, random object starts, realistic physics. Delivered (within 15k ticks) ·
picked · delivered given picked. Figures: [docs/figures/](figures/).

| End effector · grasp controller | Standard objects /100 | Wide objects /40 |
|---|---|---|
| Gripper · planner | **89** (19) · 91 · 98% | 12 (0) · 13 · 92% |
| Gripper · imitation, 3 seeds | 79/79/79 · 80/80/80 · 99% | – |
| Gripper · imitation + SAC, 3 seeds | 77/80/80 · 78/81/81 · 99% | – |
| Hand · scripted | 44 (16) · 47 · 94% | 0 · 0 (side grasp fails) |
| Hand · imitation, 3 seeds | 53/52/47 · 54/54/49 · 97% | – (no Fetch demos) |
| Hand · imitation + PPO, 3 seeds | 79/48/84 (20/14/26) · 83/59/88 · 92% | **14**/10/**14** (0) · 36/31/33 · 38% |

Reading:

- **Gripper: RL adds nothing.** The planner delivers 89; cloned and SAC-tuned grasps 77–80.
- **Hand: RL is what makes the grasp work.** Picked 47 scripted, ~52 imitation, 83–88 with PPO (two
  of three seeds; seed 1 grasps poorly from the protocol's wider starts). With the hand's own carry
  posture almost every picked object is delivered (92–97%). Best hand seeds reach 79–84 delivered,
  near the gripper planner's 89, and 20–26 within the deadline vs the planner's 19.
- **Wide objects: the hand picks, the carry drops.** The RL hand picks 31–36 of 40 heavy cylinders
  (gripper 13) but loses about half in the carry; delivered 14/10/14 vs 12.
- **Algorithms:** BC → PPO ~70% lift and carry within 2M steps (~1 h); RLPD 3–12% after 0.75M steps
  (6 h). The task-carry fine-tune did not help (ablation: PPO 10M checkpoints 84/46/88).
- **Grip after the grasp** is chosen per task on development mazes: frozen for standard objects,
  policy-held for wide ones (docs/project_spec.md).

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
  (`plan_tuck`), A* carry and reach; after the grasp the grip is frozen (standard objects) or held by
  the policy (wide objects).
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
