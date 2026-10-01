# Experiment log (MuJoCo)

Condensed record of MuJoCo-era experiments; current results are in
[project_state.md](project_state.md). Everything before 2026-10-01 ran on the course's
PyBullet simulator and is archived with its own log in [archive/pybullet/](archive/pybullet/)
(including the dexterous-hand spike and the upstream audit).

## 2026-10-01: porting the task to MuJoCo

Why: in PyBullet a 16-joint Allegro hand held objects worse than the two-finger gripper,
with grasps flipping on millimetre changes, and most failures of every system were
objects creeping out of the grip during the carry. The course grader is not binding for
this project, so the task moved to MuJoCo with the course scene and rules unchanged.

`pnr/env.py` (first as a second backend behind the same interface):
- Same robot URDF, YCB CoACD collision parts, table box, maze generator, seed mapping,
  point clouds, joint limits, kinematics and Jacobians (checked against PyBullet).
- Motor law: the course's position motors close a fixed fraction of the error per tick.
  Applied as explicit computed torque, τ = M(v_cmd − v)/dt + bias (base profile
  4.50/3.15/2.21 mm per tick vs PyBullet 4.47/3.17/2.22). Dead ends on the way: a force
  cap made the base overshoot 3×; stiff uncapped velocity servos tripped MuJoCo's
  instability reset on the light upper-arm roll joint; inertia-scaled implicit servos
  tracked perfectly but the constraint solver ignores implicit damping, so the fingers
  closed through objects with 10–100 kN contact forces.
- Grip: finger armature 4.09 kg sets the blocked grip to the course's (~450 N on the can;
  PyBullet 432 N). Both engines slip at ~110 N of external load (≈ μN).
- Friction: per-pair coefficients match the course (0.25 finger/object, 0.5 table/object;
  MuJoCo combines geoms by max, PyBullet by product). Elliptic cones, impratio 10 and
  noslip remove MuJoCo's soft-friction creep (default settings: 4/20 planner successes).
- Fixes found by comparison: MjSpec's unset inertial origin compiled the object's centre
  of mass 1 m above the mesh (the banana rolled off the table); the five head-camera links
  get PyBullet's default 1 kg / unit inertia.

Planner-only replication, same settings: course 97/100 (PyBullet 76), dev 252/260
(190), rubric 39.2–39.4 vs 36.2 per trial, median ticks 12.25k in both. Nearly all of
the difference is carry drops (21 → 1–2 on the course seeds): with equal grip force,
friction and slip load, PyBullet's drops come from its contact handling during the
carry's motion pulses. PyBullet-trained policies run unchanged in MuJoCo (imitation +
SAC 96/100 course, 240/260 dev), confirming the port preserves the task.

## 2026-10-01: MuJoCo-only codebase and retraining

PyBullet code, the backend switch and the hand-rig scripts were removed; `pnr/env.py`
gained headless EGL rendering with the course cameras. The full recipe was rerun on
MuJoCo data (`scripts/train_mujoco.sh`), with the same hyperparameters as the PyBullet
candidate. Simplification: the PyBullet actors came out of several BC rounds on growing
datasets; here BC runs once on the final dataset (grasp 24k, carry 32k updates).

| Stage | Result |
|---|---|
| Grasp demos (240) | 240/240 lifted and held |
| Grasp BC → SAC 12k | 100% validation at every check |
| Reach SAC+HER 20k | 100% training success at the end |
| Carry teacher data | 1,721 A* look-ahead and recovery episodes on 192 mazes |
| Learned-handover carries | 78/78 transports (PyBullet 62/80) |
| Carry BC 32k → SAC 40k | 100% of 12 validation carries at every check |

Bug found on the way: snapshot restores re-ran `mj_forward`, which recomputes contact
forces at the restored state, whereas after `mj_step` the stored forces belong to the
pre-integration state. Restored observations differed slightly, and the handover
collector rejected 37/78 restored episodes. Snapshots now copy the full `MjData`
(bit-exact; tested). Only the handover data and carry models were regenerated.

## 2026-10-01: MuJoCo evaluations

All 24 runs behind the project state and dashboard (`scripts/evaluate_mujoco.sh`):

| Run | Complete /260 dev | Timed |
|---|---:|---:|
| Planner only | 252 | 243 |
| Imitation | 235 | 212 |
| Imitation + SAC | 238 | 215 |
| SAC grasp + imitation carry / imitation grasp + SAC carry | 235 / 239 | 213 / 215 |
| Imitation grasp + A* carry / planned grasp + imitation carry | 243 / 243 | 242 / 187 |
| Carry rules: turn 30°, turn 15°, slip slowdown, both | 213, 55, 222, 82 | 193, 52, 133, 35 |
| Planned speed 1, 1.5, 2, 2.5, 3 | 242, 237, 240, 239, 235 | 179, 210, 218, 216, 212 |
| Carry 0.035 / 0.05 m per decision | 154 / 209 | 0 / 0 |

Hold-out (20 fresh mazes): planner 98, imitation 92, imitation + SAC 92. Course seeds:
97, 94, 93. Conclusions:

1. SAC fine-tuning adds nothing over imitation for grasp or carry (as in PyBullet).
2. In MuJoCo the classical planner beats imitation of it: the learned carry loses cases
   to timeouts and wall collisions rather than drops, and the learned grasp to its
   decision horizon. In PyBullet the learned carry won by dropping less.
3. Drops are rare (≤3% for any grasp × carry pairing) and concentrated on the can.
4. Episodes are chaotic at the margin: library load order alone flips the can on
   validation maze 8. Recordings therefore import the evaluator first.
