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

## 2026-10-02: wide objects, the hand on the Fetch, first full task with the hand

- Five objects (`hand_r_all5_01`, warm start from bc03, PPO only): 72.5% at 1–2M steps (foam brick
  75%, banana 0–13%, can/apple/mug kept). Stopped for CPU.
- Why the gripper fails (planner diagnosis, 3 trials each): bowl, plate, marker give no antipodal
  pair within its 1.2–9.8 cm width and 3.7 cm table-clearance filters; bleach bottle (lying) 24
  pairs, none feasible; master chef can 2/3 at the aperture limit.
- Side power grasp on the master chef can: the palm stopped 6 mm short, the fingertips (7–8 N
  each, thumb ~15 N) slid up the can, its weight loading the fingers' soft sideways joints. Palm
  pressed on (gap −5 mm): lifts 13–15 cm; finger servo kp 3 instead of Menagerie's 1: 0/6. From
  a side pregrasp with a 26 cm teacher lift: 12/24 carried. Demos: 66 carried of 288 noisy
  episodes (`runs/hand_demos_chefcan01`).
- Bleach bottle lying on its side, top-down claw, clearance × `along` sweep: 0/36.
- Allegro on the Fetch (dorsal mount on the wrist flange, palm frame = gripper_link). The bc03 policy
  (6M) unchanged: rig 22/24 carried, Fetch 3/24 (lifted 7/24). Holding is exact, but a 1 cm palm
  move takes ~18 decisions because each joint is rate-limited separately (the same motor law as
  the gripper systems), and finger contact deflects the wrist by ~0.05 rad.
- Full task with the hand: scripted grasp, then planned carry and reach. Mug, course seed 0:
  delivered in 16,095 ticks. Soup can: the scripted grasp knocks it over on the Fetch (wrist
  deflection). The planner now picks the pregrasp yaw across the object's narrow axis, so the
  hand has no wrist rotation to make.

## 2026-10-02 (evening): full-task comparison with the hand, randomised object starts

- Every course and bank scene starts the object in the same pose, so a grasp stage is one
  deterministic attempt per object (the first hand evaluation, 60 course cases: scripted hand
  60/60 at planner speed 1 but 20/60 at speed 3, RL hand 0/60 then 40/60; all-or-nothing per object).
  `--random-object-pose` (eval.py, eval_hand.py) seeds an object offset (x ±8 cm, y ±12 cm) and
  yaw per case; every system sees the same starts.
- Course seeds × soup can/apple/mug, randomised starts, planner speeds 3 (holding 2), realistic physics:

  | System | Delivered /60 | Within 15k ticks | Median ticks |
  |---|---:|---:|---:|
  | Planner + gripper | **57** | 1 | 17.9k |
  | Planner + scripted Allegro grasp | 40 | **27** | 14.5k |
  | Planner + RL Allegro grasp (`hand_f_02`, 1M) | 31 | 17 | 14.8k |

  The hand systems are faster (their grasp takes fewer ticks than the gripper's descend, close,
  lift and validate) but fail more grasps (19 and 27 of 60); failed cases have slightly larger
  object offsets (9.5 vs 8.3 cm mean).
- Fixes found on the way: the RL grasp drifted 28 cm and tilted to -82 deg pitch, where the rig's
  yaw/pitch/roll reading from the arm jumped (gimbal) and no carry posture was reachable: wrist
  angles now by damped least squares near the target, the grasp works in a box (±8 cm, ±0.5 rad)
  around the pregrasp, and an unreachable carry posture is retried with the palm levelled. The
  hand holds the object 10–19 cm higher than the gripper: the arm now corrects what the base and
  torso cannot at the reach.
- Chef can specialist (`hand_r_chefcan01`): 88% carried on the rig at 9M (teacher 50%, gripper
  2/6); 2/8 unchanged on the Fetch, 46% after 1M steps of Fetch fine-tuning (`hand_f_chefcan01`).

## 2026-10-03: RL hand in the full task; the chef can on the Fetch

- Same 60 randomised course cases (soup can/apple/mug), planner speeds 3/2:

  | System | Delivered /60 | Within 15k ticks | Median ticks |
  |---|---:|---:|---:|
  | Planner + gripper | **57** | 1 | 17.9k |
  | Planner + scripted Allegro grasp | 40 | 27 | 14.5k |
  | Planner + RL Allegro grasp (`hand_f_02` 4M), policy keeps the fingers in the carry | 38 | 21 | 14.7k |
  | Planner + RL Allegro grasp (`hand_f_02` 4M), grip frozen after the grasp | 49 | **28** | 14.6k |

  The policy's finger control was trained on the rig's synthetic carry; in the planner's carry it
  dropped 10 objects (frozen grip: none). Frozen is now the default. The RL hand beats the
  scripted hand on delivery (49 vs 40) and the gripper on time (28 vs 1 within the deadline);
  the gripper still delivers more (57). Remaining hand losses: 10 grasp failures.
- Chef can on the Fetch: `hand_f_chefcan01` (from the rig specialist) 92% carried at 4M steps in the
  hand task. In the full task the gripper planner delivers 14/20 (none in time, median 37k ticks);
  the hand does not yet: the side pregrasp's forearm met the table (collision-free IK seeds and a
  10 cm minimum grip height fix it), the planner's tuck swung the can loose (raise-only posture
  fixes that), but the extended side-held can then meets maze walls or drops in the carry. A side
  grasp with the back-of-palm mount holds the object well in front of the robot. Retraining the
  chef can on the corrected geometry: `hand_f_chefcan02`.

## 2026-10-03: wrist-end mount; which objects suit the hand (survey with randomised starts)

- The Allegro is now mounted on the Fetch by its wrist end (`hand_mount='wrist'`), as on real
  arms. IK check: both mounts reach side pregrasps and compact side-grasp carry postures; the
  wrist-end mount needs top-down fingers within ~90 deg of pointing away from the robot.
  Fetch fine-tunes restarted on it (`hand_w_01`, `hand_w_chefcan01`).
- Enlarged variants (same density, ~11 cm across): 102 soup can x1.6 (1.31 kg), 107 apple x1.45
  (0.75 kg), 110 mug x1.35 (0.34 kg).
- Survey, randomised starts, 6 trials (gripper: the planner's pick; hand: scripted grasps on the rig):

  | Object | Gripper | Hand top-down | Hand side |
  |---|---:|---:|---:|
  | Master chef can | 4 | 2 | 5 |
  | Bleach bottle | 5 | 0 | 0 |
  | Bowl | 0 | 0 | 0 |
  | Soup can x1.6 | 2 | 0 | 1 |
  | Apple x1.45 | 6 | 0 | 0 |
  | Mug x1.35 | 6 | 3 | 0 |
  | Soup can / apple / mug | 6/6/6 | 6/6/4 | – |

  With rubber pads (μ 1.0) and ~240 N per jaw, the gripper holds objects wider than its 9.8 cm
  opening by closing above their widest line on the sloped surface (apple x1.45: 9.7 cm chord
  3.4 cm above the centre, 477 N total normal force). The bleach bottle's 0/6 at the course pose
  was its fixed yaw. Width alone separates the end effectors only for wide, heavy cylinders.


## 2026-10-03 (evening): first protocol results; the hand's carry posture

- Protocol runs (20 hold-out mazes, random object starts, realistic physics), standard objects,
  100 cases each; delivered (within 15k ticks) · picked · delivered given picked:

  | System | Delivered | Picked | Given picked |
  |---|---:|---:|---:|
  | Gripper · planner | 89 (19) | 91 | 98% |
  | Gripper · imitation, seeds 0/1/2 | 79/79/79 (33/32/32) | 80/80/80 | 99% |
  | Gripper · imitation + SAC, seeds 0/1/2 | 77/80/80 (33/32/31) | 78/81/81 | 99% |
  | Hand · scripted | 44 (16) | 47 | 94% |
  | Hand · imitation, seeds 0/1/2 | 53/51/48 (19/17/16) | 54/54/49 | 97% |
  | Hand · BC → PPO 8M, seed 0 (development; before the task-carry fine-tune) | 71 (21) | 87 | 82% |

  Wide objects (40 cases): gripper planner 12 (0 in time); scripted hand 0 (the side grasp fails).
  The gripper's learned grasps were retrained from random starts first (imitation 63 → 79 with the
  fixed-pose grasp).
- The hand's losses after the grasp came from the planner's carry posture (flange 24 cm ahead at
  1.05 m, orientation kept), designed for the gripper. With the wrist-end mount some grasps pressed
  the object into the robot's head, which jammed the wrist roll and left the object 23 cm above the
  goal, below which the torso cannot go; the planner never checked the held object against the robot.
  `plan_tuck` (shared by training and the hand system): object 20–28 cm ahead of the base at the goal
  height, the hand turned about the vertical (the unturned hand points away from the robot, so the
  forearm would have to pass through the body), torso lowest, arm compact along x and y (the first
  version swung the elbow into maze walls), the move to it clear of table, maze and body. Scripted
  hand 23 → 44 delivered, hand imitation 11/17/11 → 53/51/48 (before the fix also a multi-seed tuck:
  12 → 23 scripted). The hand now loses objects mainly at the grasp, where RL helps (picked 54 → 87).
- Task-carry fine-tunes (`scripts/study_finetune.sh`) train the policy's hold through this real
  retreat, tuck and carry from the protocol's object-start range; they start after the PPO seeds
  reach 10M steps. The PPO policy's 12 carry drops above are what they target.
- RLPD on one shared GPU runs ~37 environment steps/s against PPO's ~510 (22 workers); the
  comparison reports both axes. Dashboard hours now exclude the OOM interruption.

## 2026-10-04 (night): fine-tunes, grip after the grasp, PPO vs RLPD, the matrix filled

- Task-carry fine-tunes (BC → PPO 10M → PPO with the full task's retreat, tuck and base carry from the
  protocol's start range; 3M standard, 4M wide). Training-time lift and carry stayed flat or fell
  (standard 0.60–0.65 → 0.38–0.57; wide 0.17–0.46). Throughput first fell to ~110 steps/s: every tuck
  search stalled all 22 vectorised environments; training now takes the first clear tuck and searches
  every hand turn from the current arm first (~300–400 steps/s).
- Unchecked straight retreat from the table: after some RL grasps the upper arm rests on the table
  edge and the checked retreat failed (4–9 of 100 per RL run).
- Grip after the grasp, chosen per task on development mazes (not the hold-out bank):

  | Development set, random starts | Policy holds | Grip frozen |
  |---|---:|---:|
  | Standard, imitation + RL seed 2 (validation, 50 cases) | 37 | **43** |
  | Standard, imitation seed 0 (validation, 50 cases) | 25 | 24 |
  | Wide, imitation + RL seed 0 (dev, 104 cases) | **29** | 11 |

  Light objects ride best in fixed servo targets (the policy's finger moves in the planned carry
  dropped them); heavy side-held cylinders slip out of fixed targets (34 slips in the retreat and
  tuck, 41 carry drops) and need the policy's active hold.
- Protocol results (hold-out, random starts), delivered (in time) · picked:

  | System | Standard /100 | Wide /40 |
  |---|---|---|
  | Gripper · planner | 89 (19) · 91 | 12 (0) · 13 |
  | Gripper · imitation, 3 seeds | 79/79/79 · 80/80/80 | – |
  | Gripper · imitation + SAC, 3 seeds | 77/80/80 · 78/81/81 | – |
  | Hand · scripted | 44 (16) · 47 | 0 · 0 |
  | Hand · imitation, 3 seeds | 53/52/47 · 54/54/49 | – |
  | Hand · imitation + PPO, 3 seeds | 79/48/84 (20/14/26) · 83/59/88 | 14/10/14 · 36/31/33 |

  Ablation, standard objects, the PPO 10M checkpoints without the task-carry fine-tune: 84/46/88
  (picked 87/55/95); the fine-tune did not help. Seed 1 grasps poorly from the protocol's wider object
  starts (training used ±3 cm): picked 55–59.
- PPO vs RLPD (standard objects, same 243 demonstrations): RLPD reached 0.75M steps in its 6 hours
  (~37 steps/s on the shared GPU against PPO's ~510 with 22 workers), lift and carry 3–12%; BC → PPO
  reached ~70% within 2M steps (about an hour). Per environment step below 1M the two are similar
  (PPO dips after cloning, 10–18% at 1M), so the demonstrations in RLPD's batches did not buy the
  expected head start here.

## 2026-10-04: redesign probes; one trainer for both end effectors

- Redesign (docs/project_spec.md): one learning recipe for both end effectors, wide objects by a
  physical rule, 5 seeds, final evaluation once on the sealed reserve bank.
- Object survey, gripper planner, 52 dev cases per object, random starts (picked): bowl 0, plate 0,
  large marker 2, soup can x1.6 6, master chef can 26, sugar box 35, mustard 36, bleach 37, potted meat 39,
  mug x1.35 39, apple 42, mug 43, apple x1.45 47, strawberry 48, foam brick 48, hammer 50, soup can 52,
  banana 52. Bowl, plate and marker fail for lack of jaw clearance in our planner (not physical); the
  wide cylinders (10.2-10.6 cm, wider than the 9.8 cm opening) slip on the lift: the wide set.
- Speed sweep, 120 dev cases, standard objects, holding speed 2/3/4: gripper 107/108/108 delivered
  (21/31/31 in time, 0 carry drops); hand (PPO seed 2) 102/105/101 (34/61/60 in time, 1 drop). Neither
  drops at speed; the hand finishes ~1,000 ticks sooner; speed saturates at 3 (joint rate limits).
- Palm-up cradle carry: the object falls out mid-roll with the current policies or a frozen grip; off
  by default until a policy is trained to roll it into the palm.
- Gripper in the hand trainer (FetchGripperRig, GripperPlannerTeacher). Demonstrators at a 300-decision
  horizon (the deployed grasp budget; 200 cut lifts short: the arm lifts ~2.5 mm per decision) and the
  protocol's start range, 12 episodes per object: gripper ~88% per standard object, 0-1 wide; hand
  ~50% standard (brick 0/12, soup can 12/12), 0-1 wide. A straight-down grasp at the far end of the
  start range is beyond the Fetch's reach from the planner's base position; the teacher, like the
  planner, takes the first reachable candidate.

## 2026-10-04 (afternoon): first campaign results (untuned recipe); tuning sweep

- Standard objects, BC -> PPO, training-time lift and carry (60 episodes), 10M steps: hand seeds
  0.35/0.40/0.43/0.42 -> 0.82/0.78/0.72 (7.5M)/0.88 (8M); gripper 0.47/0.43/0.47/0.52/0.62 -> flat
  0.38-0.55. The gripper lifts ~50% throughout (its teacher ~88% in the same environment) and carries
  what it lifts; failures at 10M: one pad first (misaligned jaws) 12/60, both pads but no lift 11/60.
  A flat curve 40 points under its own teacher reads as a pipeline problem, not a finding, so the
  campaign queue is paused for an equal-budget tuning sweep (scripts/tune_v2.sh; spec: Tuning).
  Untuned hand seed 4 stopped at 2M (resumable).
- Wide cylinders, gripper BC -> PPO (untuned): 0.21-0.33 lift and carry at 1.5-3M, although the
  planner and the teacher manage ~0. The learned gripper pinches the can's top rim with the jaws tilted
  0.5 rad (the training workspace's tilt limit) instead of approaching straight down. RL adds value to
  the gripper exactly where its hand-written controller fails; C2 must compare the RL hand with the RL
  gripper, not only with the planner.
- Invalid run: v2_hand_bcppo_wide_s0 started its wide-object episodes in the old side pregrasp (palm
  vertical beside the can) while cloning top-down standard demonstrations: 0.00 lift and carry for 8.5M
  steps. Fixed with train_hand.py --side-grasp-objects (none: every object starts top-down); the
  full-task evaluation reads the setting from the policy's training config. Archived in
  runs/archive_invalid/v2_side_start/. The gripper's wide run is unaffected (always top-down): best 0.42.
- Tuning sweep, first four gripper configurations (score = mean of the last three evaluations):
  untuned 0.44; learning rate 1e-4 + exploration log-std -0.7 0.68 (0.47 -> 0.72); interleaved cloning
  collapses the gripper (0.07, 0.11). The gripper's flat curve was a tuning problem.

## 2026-10-05: tuning sweep result; campaign restarted with the tuned recipe

- Sweep (8 configurations per end effector, 4M steps, score = mean of the last three evaluations):
  gripper 0.44 / 0.11 / 0.07 / 0.68 / 0.11 / 0.75 / 0.55 / 0.33, hand 0.42 / 0.37 / 0.38 / 0.61 / 0.31 /
  0.77 / 0.49 / 0.48 (c1-c8). Main effects, both end effectors the same direction: learning rate 1e-4
  (+0.18 / +0.16), log-std -0.7 (+0.05 / +0.02, within noise), 300 demos (+0.11 / +0.07), interleaved
  cloning harmful (-0.45 / -0.19). Recipe: lr 1e-4, log-std -0.7, 300 demos, no interleaving.
- Untuned campaign runs archived (runs/archive_superseded/v2_untuned/); campaign restarted with the
  tuned recipe and top-down starts for every object (40 runs, 4 at a time).

## 2026-10-05 (evening): tuned standard results; the hand's wide-object prior

- Standard objects, tuned BC -> PPO, 5 seeds, lift and carry (step 0 -> mean of last three evaluations):
  hand 0.42-0.58 -> 0.80-0.87 (best 0.82-0.90); gripper 0.42-0.57 -> 0.55-0.67 (best 0.68-0.80). RL adds
  ~+0.38 for the hand, ~+0.11 for the gripper, which stays below its own teacher (~0.88).
- Wide cylinders, BC of standard demos -> PPO: gripper 0.33-0.51 (seed 0 at 10M: chef can 0.42, soup can
  x1.6 0.75; tilted rim pinch); hand 0.00 for 9M steps from the top-down start (peak lift 1.5 cm, never a
  lift, no signal for PPO). Hand wide seed 1 stopped at 4M (archived); the queue skips the hand's wide
  BC -> PPO runs (SKIP) pending its prior.
- Hand side-grasp demonstrations on the idealised rig, 300 per object: chef can 79 lifted / 69 carried,
  soup can x1.6 12 / 11; 80 episodes, 24,739 transitions (runs/v2_demos_hand_wide_rig300). Development
  seed v2dev_hand_sidebc_wide_s0: BC of these -> PPO on the Fetch from the side pregrasp, tuned recipe.
- Development seed v2dev_hand_sidebc_wide_s0 (rig side-grasp demonstrations -> PPO on the Fetch from the
  side pregrasp): 0.00 to 1M, then 0.33 / 0.67 / 0.54 / 0.71 / 0.71 at 2.0-4.0M (chef can 0.83, soup can
  x1.6 0.50 at 4M). Mean over 2.5-4M 0.66 vs the gripper's 0.19-0.32 at the same steps and 0.36-0.55 at
  8.5-10M: the rule is met; the campaign's five hand wide BC -> PPO seeds use this prior (fresh seeds;
  the development run stays development evidence). Hand wide from scratch also starts from the side pregrasp.

## 2026-10-07: training campaign complete (40 runs)

Training-time lift and carry (60 episodes standard, 24 wide), step 0 -> mean of the last three
evaluations (best), 10M steps, 5 seeds:

| | Standard | Wide |
|---|---|---|
| Gripper BC -> PPO | 0.42-0.57 -> 0.55-0.67 (0.68-0.80) | 0.00-0.04 -> 0.26-0.54 (0.38-0.62) |
| Hand BC -> PPO | 0.42-0.58 -> 0.80-0.87 (0.82-0.90) | 0.00 -> 0.65-0.83 (0.83-0.88) |
| Gripper PPO from scratch | 0.53-0.67 (0.67-0.80) | 0.50-0.74 (0.67-0.79) |
| Hand PPO from scratch | seed 0 0.81; seeds 1-4 0.00 (best 0.03) | 0.00 (all five seeds) |

- The gripper learns as well from scratch as from its demonstrations on standard objects, and better
  from scratch on the wide cylinders (its standard-object prior is a top-down antipodal grasp that does
  not fit them; from scratch it finds the tilted rim pinch directly). C2 therefore compares the hand
  with the gripper's best system (planner, BC -> PPO or from scratch) on development data.
- The hand from scratch needs its prior: 1 of 5 seeds discovers a grasp on standard objects, none on the
  wide cylinders; with the prior every seed reaches 0.65-0.87.
- Smoke test of the full task, hand BC -> PPO wide seed 0 at 10M, policy-held grip, 3 dev layouts x 2
  objects x 3 starts: grasp held 18/18, delivered 9/18; all failures but one are drops in the maze
  carry (the wide-object hold, plan step 3).

## 2026-10-08: development evaluations; PPO vs RLPD set up

- Full-task development evaluations of every cell (scripts/dev_v2.sh, dev mazes, wide 3 starts per case:
  eval.py / eval_hand.py --starts). Checkpoint: the best training-time evaluation within 10M (later on
  ties), fixed before; grip after the grasp decided per end effector and task on the BC -> PPO cells.
- RLPD from the same imitation start as BC -> PPO: the SAC actor's squashed mean cloned on the same
  demonstrations (40,000 updates), log-std -0.7, then the same 1M-step value warm-up (actor and
  entropy coefficient frozen; checked: identical evaluations before and after warm-up steps).
  train_seconds (wall-clock without evaluations) and the machine load are logged per evaluation.
  scripts/algorithms_v2.sh queues the comparison behind the development evaluations.

## 2026-10-09: development evaluations complete; PPO vs RLPD under way

Full task, development mazes (standard 260 cases, wide 312 = 52 layouts x 2 objects x 3 starts), delivered,
pooled over 5 seeds [Wilson 95%] (per seed):

| | Standard | Wide |
|---|---|---|
| Gripper planner | 0.90 [0.85, 0.93] | 0.31 [0.27, 0.37] |
| Gripper BC | 0.73 [0.70, 0.75] (0.67-0.77) | 0.00 |
| Gripper BC -> PPO | 0.63 [0.61, 0.66] (0.59-0.72) | 0.23 [0.21, 0.25] (0.14-0.31) |
| Gripper PPO from scratch | 0.64 [0.61, 0.66] (0.60-0.68) | 0.42 [0.40, 0.45] (0.38-0.51) |
| Hand scripted | 0.43 [0.38, 0.50] | 0.00 |
| Hand BC | 0.50 [0.47, 0.52] (0.47-0.52) | 0.01 |
| Hand BC -> PPO | 0.78 [0.76, 0.81] (0.60-0.91) | 0.69 [0.67, 0.71] (0.54-0.78) |
| Hand PPO from scratch | 0.19 (seed 0 0.88, others 0.00-0.04) | 0.00 |

- Grip after the grasp (rule: more pooled deliveries on the BC -> PPO cells): frozen for every end effector
  and task (gripper 823 vs 789 standard, 359 vs 349 wide; hand 1020 vs 873, 1079 vs 619). The first study's
  policy-held wide grip does not carry over to the side-grasp policies.
- C2 on development data: the RL hand delivers 0.69 of the wide cases vs the gripper's best system 0.42
  (from scratch) and its planner 0.31; intervals far apart. The hand's wide failures are mostly drops in
  the maze carry (seed 0 42, seed 4 113 of 312); the planner fails at the hold (192 of 312: the can
  slips out of the jaws).
- C1 in the full task: RL adds +0.28 to the hand's imitation (0.50 -> 0.78) and nothing to the gripper's
  (0.73 -> 0.63 BC -> PPO, 0.64 from scratch), whose planner (0.90) stays best. The gripper's training-time
  gain does not transfer: grasp failures unchanged (55 -> 49-58 of 260) and more drops in the maze carry
  (3 -> 16-32), so RL's gripper grasps hold less securely than the cloned planner grasp.
- Seed spread in the full task is wider than at training time (hand BC -> PPO 0.60-0.91 vs 0.80-0.87);
  seed 1 fails at the grasp (91 of 260), seed 3 at grasp and carry.
- PPO vs RLPD, hand seed 0, isolated: PPO 0.47 -> 0.75 at 4M in 1.3 training hours; RLPD holds 0.50 through
  the warm-up, falls to 0.00 when the actor is released (1.25M) and recovers to 0.08 by 2.5M, at ~80
  steps/s (~14 h per run, GPU ~26% busy). Seeds 1-4 and the no-warm-up check now run four at a time
  after the isolated seed-0 runs (per step only); per-hour numbers come from the isolated seed 0.

## 2026-10-09 (afternoon): final evaluation on the reserve bank (frozen systems, configs/final_v2.json)

One run of all 64 frozen systems, 13:28-14:32 (runs/final_v2/, report.json). Delivered, pooled over 5 seeds
[Wilson 95%] (per seed); standard 100 cases, wide 120 (20 layouts x 2 objects x 3 starts):

| | Standard | Wide |
|---|---|---|
| Gripper planner | 0.85 [0.77, 0.91] | 0.27 [0.20, 0.35] |
| Gripper BC | 0.69 [0.65, 0.73] (0.63-0.73) | 0.00 |
| Gripper BC -> PPO | 0.62 [0.58, 0.67] (0.56-0.65) | 0.21 [0.18, 0.25] (0.10-0.25) |
| Gripper PPO from scratch | 0.61 [0.56, 0.65] (0.55-0.67) | 0.39 [0.35, 0.43] (0.35-0.45) |
| Hand scripted | 0.40 [0.31, 0.50] | 0.00 |
| Hand BC | 0.49 [0.44, 0.53] (0.43-0.55) | 0.01 |
| Hand BC -> PPO | 0.76 [0.72, 0.80] (0.59-0.92) | 0.68 [0.64, 0.71] (0.58-0.77) |
| Hand PPO from scratch | 0.18 (seed 0 0.86, others 0.00-0.05) | 0.00 |

- Every cell within a few points of its development estimate: the development-stage decisions did not
  overfit the development mazes.
- C1: RL adds +0.27 to the hand's imitation and -0.07 to the gripper's; the gripper's planner stays best
  on standard objects. RL from scratch: the gripper matches its imitation + RL, the hand fails in 4 of 5 seeds.
- C2: wide cylinders, RL hand 0.68 vs the gripper's best system 0.39 (from scratch) and its planner 0.27;
  the hand's worst seed (0.58) is above the gripper's best (0.45).
- PPO vs RLPD, seed 0, isolated: hand PPO 0.47 -> 0.75 in 1.3 training hours; RLPD 0.50 through the
  warm-up, 0.00-0.08 after the actor is released, 0.07 at 4M after 14.4 hours (74 steps/s). Gripper PPO
  0.48 -> 0.55 at 4M (best 0.73) in 0.7 hours; gripper RLPD running.

## 2026-10-13: PPO vs RLPD complete (24 runs, scripts/algorithms_v2.sh)

Standard objects, 4M steps, training-time lift and carry (60 episodes per evaluation, every 250k), 5 seeds:

| | Imitation start (step 0) | Mean of last three evaluations at 4M (per seed) | Best after the warm-up | Seed 0 training hours (isolated) |
|---|---|---|---|---|
| Hand PPO | 0.47 | 0.65 (0.76 0.71 0.72 0.54 0.53) | 0.74 | 1.3 |
| Hand RLPD | 0.47 | 0.07 (0.03 0.10 0.03 0.07 0.09) | 0.12 | 14.4 |
| Gripper PPO | 0.50 | 0.62 (0.58 0.68 0.64 0.65 0.55) | 0.75 | 0.7 |
| Gripper RLPD | 0.52 | 0.03 (0.03 0.01 0.04 0.02 0.04) | 0.18 | 13.8 |

- Equal starts confirmed: both algorithms begin at the same imitation level (0.47 / 0.47 hand, 0.50 / 0.52
  gripper). RLPD holds it while the actor is frozen (normalisation fixed) and collapses within 250k steps of
  the actor's release on every seed of both end effectors; no seed recovers above 0.18.
- No-warm-up sensitivity (seed 0): hand 0.06, gripper 0.03 at 4M (best 0.12 / 0.27 early): the collapse
  starts at step 0 instead; the warm-up is not its cause.
- Per hour: PPO finishes 4M steps in 0.7-1.3 training hours; RLPD at the same wall-clock is still in its
  warm-up (0.8 h, 0.50 hand) and ends at 0.07 / 0.02 after ~14 h. Seeds 1-4 ran four at a time (per step only).
- Hand PPO seeds 3-4 end at 0.53-0.54 after best evaluations of 0.75 / 0.68 (late drift; full curves in the report).
