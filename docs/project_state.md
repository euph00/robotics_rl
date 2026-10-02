# Project state

Updated 2026-10-02. Simulator: MuJoCo 3.14 (`pnr/env.py`) with the course task's robot,
objects, mazes, seeds, position-target interface and success criterion. Since 2026-10-02 the
default physics is **realistic**: both end effectors get the same believable physics instead
of the course's PyBullet-era settings. Experiments: [experiment_log.md](experiment_log.md);
PyBullet era: [archive/pybullet/](archive/pybullet/).

## Physics settings

| | Course (`physics='course'`, before 2026-10-02) | Realistic (default) |
|---|---|---|
| Joint motion | every joint closes 30% of its error per tick, unlimited force (base velocity jumps ~1 m/s per command) | same targets, but each joint's reference velocity is limited by its stopping distance, velocity and acceleration limits, and its torque by its effort limit |
| Base | — | 1.0 m/s, 0.75 m/s² (Fetch spec, base-controller defaults) |
| Arm, torso | — | fetch_moveit_config velocity/acceleration limits, URDF efforts |
| Gripping surfaces | μ 0.25 | rubber, μ 1.0 (Fetch fingers and palm; every Allegro surface) |
| Fetch jaws | ~220 N squeeze, fast | ≤ 5 cm/s, up to 245 N (Fetch spec) |
| Allegro joints | Menagerie kp 1, uncapped | ≤ 0.7 N m (Allegro v4 spec) |

Rate-limited motion needed three controller adaptations: the planner waits for each motion
to settle; the learned carry's action is a velocity (target held at the braking distance);
the A* carry teacher slows for corners.

## Two-finger gripper systems (realistic physics)

| Complete (within 15k ticks) | Dev /260 | Hold-out /100 | Course /100 | Rubric / trial (course) |
|---|---:|---:|---:|---:|
| Planner only | **250** (21) | **99** (12) | **95** (10) | 28.5 / 40 |
| Imitation | 175 (69) | 77 (32) | 84 (49) | 36.1 / 40 |
| Imitation + SAC | 178 (73) | 77 (27) | 81 (56) | **36.5** / 40 |

The realistic task is no longer saturated, and it splits two ways:

- **The planner is reliable but slow.** It plans and stops at every waypoint and waits for
  each motion; median 17.4k ticks against the 15k deadline. Its pick is often too slow for
  the rubric's "lift within 3000 steps".
- **The learned systems are faster but crash.** Median ~15k ticks and 2–5× more timed
  successes, but their carry fails by wall collisions (50–56 of 260 on dev) and timeouts
  (24–28). There are no drops: with rubber grips and realistic motion nothing slips.
- **SAC still adds nothing over imitation** (dev 178 vs 175; SAC on one skill only:
  180 and 174 — all within ±7).

Stage swaps on dev (imitation policies vs planner stages):

| Complete (timed) | Learned carry | A* carry |
|---|---:|---:|
| Learned grasp | 175 (69) | 248 (94) |
| Planned grasp | 175 (8) | 250 (21) |

The learned grasp is fast and reliable; the learned carry is the weak stage. Other levers:
faster planned motion raises the timed share (13% at 1×, 30% at 3×); slower carry commands
time out (0.035 / 0.05 m per decision: 7 / 93 complete); hand-written turn limits add
collisions (30°: 164; 15°: 77). The course-physics results (planner 97–99%, learned 92–94%,
drops the main failure) are in the experiment log.

## Dexterous hand (feasible; extending)

Test rig (`pnr/hand.py`): Menagerie Allegro (16 joints) on a six-axis wrist with arm-like
limits over the course table, same physics settings. Task (`pnr/hand_env.py`): grasp from a
pregrasp, lift 15 cm with thumb and a finger in contact, then hold through a realistic base
carry while the policy keeps controlling the fingers.

- **Which objects can each end effector pick?** (realistic, 6 trials each, `scripts/object_survey.py`)
  The planner + gripper picks 10 of 15 course objects; it fails the bleach bottle, bowl, plate
  and large marker (0/6) and mostly the master chef can (2/6, at its 10 cm aperture). A
  scripted Allegro power grasp (`pnr/hand_teacher.py`) picks the soup can, apple (4/6) and
  mug (2/6) and nothing else; it is too large for low objects in a top-down claw.
- **RL from scratch** (PPO): learns approach and wrap, but under realistic physics has not yet
  lifted 15 cm (24M steps; peak lift rising from 1 to 7.6 cm). Under the course physics the
  same set-up lifted the foam brick and banana by 5–8M steps, by pressing the palm down with
  the unlimited wrist — a strategy realistic effort caps remove (that policy lifts 0/8 under
  realistic physics). A curriculum without a teacher (easy objects first, near-grasp starts)
  stayed at 0% after 10M steps.
- **Imitation + RL works** (`hand_r_bc03`): the scripted teacher's 254 successful demos →
  behaviour cloning (14% lift and carry) → PPO. 86% at 4M steps, **97% at 6M** (can 92%,
  apple 100%, mug 100%; 36 episodes), above the teacher's 72%. Two things mattered: fixing a
  reward exploit (the policy hovered just under the 15 cm success height because hovering paid
  more than the carry phase) and letting PPO move away from the imitator (keeping it close
  with interleaved cloning and a KL limit held it at 14–33%). Videos are on the dashboard.
- **Wide upright objects (side grasp).** The master chef can (10 cm, 1.05 kg) is at the gripper's
  aperture limit (2/6). A top-down claw cannot hold it; a side power grasp (palm vertical against
  the can, fingers wrapped round it) can, from a side pregrasp: the scripted version carries 12 of
  24. With the palm short of the can, the 0.7 N m fingers slide up it (their soft sideways
  joints carry the weight); pressing the palm on transfers the load to palm friction.
  `hand_r_side01` trains imitation + PPO on soup can, apple, mug and master chef can.
- **Objects neither end effector picks yet.** Bleach bottle (lies on its side: 11.5 cm wide,
  0.97 kg): top-down claw 0/36 over nine settings, the fingertips cannot get below its widest
  line. Bowl, plate and large marker: the planner's grasp filter rejects them (no antipodal pair
  1.2–9.8 cm wide, at least 3.7 cm above the table). For the marker and plate this is a real
  limit of a top-down parallel gripper; the bowl could be grasped by its rim with a better planner.
- **On the Fetch.** `effector='allegro'` mounts the hand on the wrist flange in place of the
  gripper; `pnr/fetch_hand.py` runs the hand task on the real arm and base, with the same
  observation and actions as the rig. The rig policy transfers poorly (3/24 carried against 22/24
  on the rig): the arm follows small wrist targets in ~0.4 s, not ~3 ticks, and contact
  pushes its wrist off. `hand_f_01` fine-tunes it on the Fetch.
- **Full task with the hand** (`pnr/hand_system.py`, `scripts/eval_hand.py`): planned approach and
  pregrasp, then the hand policy grasps and lifts inside a box around the pregrasp, then the
  planner's carry posture, A* carry and reach (the arm corrects the last height error) with the
  grip frozen. Course seeds × soup can/apple/mug with randomised object starts (60 cases):
  planner + gripper 57 delivered (1 within 15k ticks), planner + scripted hand 40 (27), planner +
  RL hand 31 (17). The hand is faster; its grasp is not yet as reliable on the real arm.
- **Next:** more Fetch training for the hand (soup can/apple/mug and the chef can), then the
  harder task where the two end effectors differ (wide and heavy objects, randomised starts).

## How we evaluate

`scripts/eval.py --physics {realistic,course}` runs full original-start episodes in parallel;
`scripts/evaluate_mujoco.sh` reruns every number above (PNR_PHYSICS selects the settings).
Complete: object within 10 cm of the goal after the final hold. Timed: complete within
15,000 physics ticks including settling. `scripts/rubric.py` estimates per-trial rubric points.
Episodes are deterministic for a given process set-up but borderline cases are chaotic; treat
each case as a draw (±7 on dev). Sets: dev 260, hold-out 100, course 100, reserve 100 (sealed).

## Training

`scripts/train_mujoco.sh` (PNR_PHYSICS=realistic, run prefix `mjr_`; ~70 minutes): grasp
(240 scripted demos, 219 lifted under realistic physics → BC → SAC), carry (1,801 teacher
episodes on 192 TRAIN mazes → BC → SAC), reach (SAC+HER). Validation: grasp and reach 100%,
carry 75% (BC) and 75–92% (SAC). Hand: `scripts/train_hand.py`, `scripts/hand_teacher_demos.py`.

## Operations

- Dashboard: `python dashboard/server.py --port 8766` → http://127.0.0.1:8766 (hand training
  curves update live).
- `runs/archive_pybullet/` holds every PyBullet-era run. The Codex agent that built most of
  this repository is paused.
