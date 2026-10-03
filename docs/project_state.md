# Project state

Updated 2026-10-03. Plan: [project_spec.md](project_spec.md) (machine-readable:
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
| 2 | Hand trained on the Fetch (wrist-end mount; five objects; chef can) | in progress: `hand_w_01`, `hand_w_chefcan01` |
| 3 | Wide objects complete the full task with the hand | next: side-grasp carry posture |
| 4 | Matrix filled with one seed | in progress: gripper and scripted-hand protocol runs |
| 5 | Three seeds per learned condition | next |
| 6 | PPO vs RLPD on the hand; ablations | next |
| 7 | Report | next |
| — | Stretch: orientation-constrained delivery | only after 1–6 |

## Results so far

No protocol (hold-out, random-start) results yet. Development results, course mazes with random
object starts, soup can/apple/mug, 60 cases:

| System | Delivered | Within 15k ticks | Median ticks |
|---|---:|---:|---:|
| Gripper · planner | 57 | 1 | 17.9k |
| Hand · scripted grasp | 40 | 27 | 14.5k |
| Hand · imitation + RL (`hand_f_02` 4M, old back-of-palm mount) | 49 | 28 | 14.6k |

Evidence behind the claim:

- **Hand, RL is essential.** On the test rig (hand on an idealised wrist): PPO from scratch 0% after
  24M steps; behaviour cloning of the scripted grasp 14% lift and carry; cloning then PPO 97% (soup
  can, apple, mug; the scripted teacher 72%). Master chef can: 88% (teacher 50%); on the Fetch 92%.
- **Gripper, RL adds nothing measurable.** Course task, realistic physics, dev set (260 cases):
  planner 250 delivered, imitation 175, imitation + SAC 178.
- **Where each end effector wins** (survey, random starts, 6 trials; gripper = planner's pick, hand =
  scripted grasps on the rig):

  | Object | Gripper | Hand (better grasp) |
  |---|---:|---:|
  | Master chef can (10.4 cm) | 4 | 5 |
  | Soup can ×1.6 (1.3 kg) | 2 | 1 |
  | Apple ×1.45 / mug ×1.35 | 6 / 6 | 0 / 3 |
  | Bleach bottle / bowl | 5 / 0 | 0 / 0 |
  | Soup can / apple / mug | 6 / 6 / 6 | 6 / 6 / 4 |

  Rubber pads (μ 1.0) at ~240 N let the gripper hold objects wider than its 9.8 cm opening above
  their widest line; only wide, heavy cylinders separate the two, hence the wide-object task.

## Platform

- MuJoCo 3.14 (`pnr/env.py`), the course's robot, objects, mazes and success test. **Realistic
  physics** for both end effectors: joint velocity, acceleration and effort limits (Fetch spec,
  MoveIt limits); rubber grip μ 1.0; Fetch jaws ≤ 245 N; Allegro joints ≤ 0.7 N m.
- Allegro hand (Menagerie) mounted on the Fetch wrist flange by its wrist end (`effector='allegro'`,
  `hand_mount='wrist'`). `pnr/fetch_hand.py` runs the hand task on the real arm (palm targets →
  damped least-squares arm targets; the carry drives the base). `pnr/hand_system.py` is the full task
  with the hand: planned approach and pregrasp, learned or scripted grasp, planned carry posture,
  A* carry and reach, grip frozen after the grasp.
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
