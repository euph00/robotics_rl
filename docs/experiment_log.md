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

## 2026-10-02: dexterous-hand feasibility with RL (in progress)

Rig (`pnr/hand.py`): Menagerie Allegro hand (16 joints, position actuators kp = 1 N m/rad,
roughly the hardware's torque range) on a six-axis wrist that follows the course motor law,
over the course table, with the task's physics settings. Hand–object friction equals the
Fetch fingers' (0.25). Task (`pnr/hand_env.py`): from a palm-down pregrasp, grasp and lift
15 cm while thumb and a finger touch the object; five objects. Training: `scripts/train_hand.py`.

| Experiment | Result |
|---|---|
| Scripted top-down power grasp, μ 0.25, kp 1 or 3 | 0/5 objects lifted |
| Same, μ 1.0 | foam (kp 1); banana, can, mug (kp 3) |
| SAC (8 updates per 20 transitions) | ~200 steps/s: too slow; PPO used instead |
| PPO, absolute finger targets | no learning by 3M steps (exploration noise reopens grasps) |
| PPO, delta finger targets, small lift weight | hand wraps the object and stays on the table (reward plateau) |
| PPO, lift-dominant reward, μ 0.25 | lifts foam and banana (up to 100% each at some checkpoints; 20–33% over five objects), never can, apple or mug through 14M steps |
| Holds during the lift | 15–42 N total normal force, ~1 mm median penetration: physical, not an exploit |
| Frozen grasp through 10 recorded maze carries | 1/10 retained; μ 1.0 or maximum squeeze during the carry do not help |
| Carry phase in training (policy active, synthetic course-style motions), μ 0.25 | 1/30 evaluation carries completed after 8M steps |
| Near-grasp curriculum starts | slower than without |

Throughput is ~1,400 decisions/s per run on 40 CPU workers: contact-heavy steps cost
5–10 ms (elliptic friction cones; CoACD objects with up to 98 convex parts).

The binding constraints are physical. The course gripper survives the carry because of
its 450 N squeeze (~110 N of friction at μ 0.25); a hand of realistic strength exerts
15–40 N, and the course motor law gives the carried hand ~25–30 m/s² acceleration pulses.

## 2026-10-02: realistic physics for both end effectors

Decision (user): same believable physics for the two-finger gripper and the Allegro instead
of the course settings. Parameters and sources are in `pnr/env.py` (REALISTIC) and
docs/project_state.md. Corrections found on the way: the course gripper's "450 N" was the
sum over both jaws; its squeeze (~220 N) is close to the Fetch's rated 245 N. What was
unrealistic was μ 0.25 and the unlimited motors (base jumps ~1 m/s per command; the
high-friction hand run had 118 numerical blow-ups from unbounded wrist torques).

Implementation notes:
- Rate-limiting the reference velocity (not the measured one): a blocked joint keeps its
  reference and pushes up to its effort limit; limiting from the measured velocity left the
  jaws at ~20 N.
- The planner commanded each motion for a fixed number of calls; with rate-limited joints it
  planned grasps before the base had arrived. It now waits for settling.
- Carry action = velocity. Targets 7.5 cm ahead of the measured base capped the speed at
  ~0.3 m/s (the teacher exceeded its horizon); an integrated reference overshot corners.
  The A* teacher now brakes for turns (sqrt(v_c² + 2 a d)).
- The regression suite pins course physics (tests/conftest.py); tests/test_physics.py covers
  the realistic law.

Results (24 evaluations, `PNR_PHYSICS=realistic bash scripts/evaluate_mujoco.sh`):

| Run (dev /260) | Complete | Timed | Main failures |
|---|---:|---:|---|
| Planner only | 250 | 21 | 3 grasp validation, 3 reach |
| Imitation | 175 | 69 | 56 carry collisions, 24 carry timeouts |
| Imitation + SAC | 178 | 73 | 50 collisions, 28 timeouts |
| SAC grasp + imitation carry / imitation grasp + SAC carry | 180 / 174 | 71 / 78 | collisions |
| Learned grasp + A* carry / planned grasp + learned carry | 248 / 175 | 94 / 8 | — / collisions |
| Turn limit 30° / 15° | 164 / 77 | 66 / 32 | 82 / 175 collisions |
| Planned speed 1, 1.5, 2, 2.5, 3 | 173, 172, 180, 176, 182 | 34, 59, 76, 77, 78 | collisions |
| Carry 0.035 / 0.05 m per decision | 7 / 93 | 0 / 0 | timeouts |

Hold-out: planner 99 (12 timed), imitation 77 (32), imitation + SAC 77 (27). Course seeds:
95 (10), 84 (49), 81 (56); rubric 28.5, 36.1, 36.5 per trial.

## 2026-10-02: dexterous hand under realistic physics

- RL from scratch (PPO, grasp + carry phases): no 15 cm lift by 24M steps; peak lift 1 → 7.6 cm.
- The course-physics policy that lifted the foam brick and banana lifts 0/8 under realistic
  physics: it pressed the palm onto the object with the unlimited wrist and scooped.
- Object survey (`scripts/object_survey.py`, 6 trials each): planner + gripper picks 10/15
  objects (0/6: bleach bottle, bowl, plate, large marker; 2/6 master chef can); the scripted
  Allegro power grasp picks soup can 4/6, apple 4/6, mug 2/6, nothing else (a top-down claw
  of this size cannot wrap low objects: fingertips reach ~9 cm below the palm).
- Teacher at a 200-decision horizon (120 was too short for it): can/apple/mug lifted 48/60,
  carried 43/60. Behaviour cloning on 254 successful noisy demos: 15% (compounding errors).
- Running: PPO from the cloned policy (value warm-up 1M steps), and a curriculum run
  (easy-first: foam and banana, then all five; near-grasp starts) warm-started from the
  realistic RL policy, both with a 200-decision horizon; a 160k-decision carry SAC run for the
  gripper's collisions.

Later on 2026-10-02:
- Longer carry SAC under realistic physics (`mjr_carry_sac03`, 160k decisions): validation on
  12 carries bounced 75–100%; the best-validated checkpoint (140k) completes 162/260 dev with
  77 wall collisions, worse than the 40k-decision carry (178, ~50). More SAC does not fix the
  collisions, and 12-carry validation is too noisy to select checkpoints.
- Hand, imitation → PPO (`hand_r_bc01`, lr 3e-4): 12.5% (cloned) → 42% lift-and-carry at 4M
  steps, then 4–21% (24-episode evaluations) while stochastic training success rose to ~53%.
  Objects that enter the carry are nearly always carried; the losses are in holding the lift
  for 10 steps. `hand_r_bc02` repeats it with lr 1e-4 and interleaved behaviour cloning
  (DAPG-style) to keep the policy near the teacher.
- Hand, curriculum without a teacher (`hand_r_cur_c`, warm start from the realistic RL policy):
  0% after 10M steps; stopped.
- A stateless version of the teacher (for DAgger relabelling) reaches can 17/20, mug 9/20,
  apple 2/20 carried, below the stateful teacher (16, 15, 12 of 20); DAgger deferred.

## 2026-10-02: hand feasibility under realistic physics — imitation + RL works

- **Reward exploit.** In `hand_r_bc01`/`bc02` the success rate rose and then collapsed after
  ~4M steps while peak lifts settled at 14.2–15.0 cm: the carry phase paid ~1.5 per decision,
  holding a lift just under the 15 cm success height paid up to ~7.5, so PPO learned to hover
  below the threshold. The carry phase now adds its bonus on top of the grasp-phase reward
  (`f9977b7`).
- **`hand_r_bc03`** (teacher demos → behaviour cloning → 1M-step value warm-up → PPO, lr 3e-4,
  fixed reward; can, apple, mug; 36 deterministic evaluation episodes per point):

  | Steps | 0 (cloned) | 2M | 4M | 6M | 7M | 8M |
  |---|---:|---:|---:|---:|---:|---:|
  | Lift + carry | 14% | 53% | 86% | **97%** | 94% | 89% |

  At 6M: can 92%, apple 100%, mug 100%, above the scripted teacher (72%) it learned from.
  Recorded episodes (`scripts/record_hand.py`, `runs/hand_recordings01`): all three carried.
- **`hand_r_bc04`** (same, lr 1e-4 + interleaved behaviour cloning + target KL 0.02): 14–33%
  through 9M; keeping the policy near a 14% imitator holds it back. Stopped.
- Next: `hand_r_all5_01` adds the foam brick and banana (no teacher for these; warm start from
  bc03 at 6M, PPO only).
