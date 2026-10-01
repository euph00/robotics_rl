# Project state

Updated 2026-10-01. Simulator: MuJoCo 3.14 (`pnr/env.py`), with the course task's
robot, objects, mazes, seeds, motor law, friction coefficients and success criterion.
Every result below was trained and evaluated in MuJoCo; the PyBullet era is archived in
[archive/pybullet/](archive/pybullet/). Experiments: [experiment_log.md](experiment_log.md).

## Systems and results

| Complete (within 15k ticks) | Dev /260 | Hold-out /100 | Course /100 | Rubric / trial (course) |
|---|---:|---:|---:|---:|
| Planner only | **252** (243) | **98** (93) | **97** (97) | **39.4** / 40 |
| Imitation | 235 (212) | 92 (83) | 94 (92) | 38.2 / 40 |
| Imitation + SAC | 238 (215) | 92 (83) | 93 (89) | 38.1 / 40 |

- **Planner only:** antipodal grasp proposal, IK and collision-checked motion, planned
  descent/close/lift, A* carry, base/torso reach; 3× planned speed empty-handed, 2× holding.
- **Imitation** ([configs/imitation_only.json](../configs/imitation_only.json)): behaviour-cloned
  grasp and carry actors, SAC+HER reach (reach has no teacher).
- **Imitation + SAC** ([configs/imitation_sac.json](../configs/imitation_sac.json), the headline
  system): the same actors fine-tuned online with demonstration-regularized SAC+HER.

Learned systems keep the planned approach and carry posture and use the same speeds.
Hold-out: 20 fresh mazes × 5 objects, built before any MuJoCo training. Course: the
course's seed mapping (seeds 1000–1019). One standard deviation on dev is about ±7 cases.

**SAC adds nothing measurable over imitation** (421 vs 423 of 460 cases across all sets).
Per skill, on dev:

| Complete /260 (timed) | SAC carry | Imitation carry |
|---|---:|---:|
| SAC grasp | 238 (215) | 235 (213) |
| Imitation grasp | 239 (215) | 235 (212) |

**The planner is the strongest system.** Swapping single stages on dev (imitation policies
vs the planner's stages):

| Complete (carry drops) | Learned carry | A* carry |
|---|---:|---:|
| Learned grasp | 235 (7) | 243 (8) |
| Planned grasp | 243 (5) | 252 (4) |

Drops are rare for every pairing (≤3%). The learned carry loses more cases than A* to
running out of decisions (8–9) and wall collisions; the learned grasp loses 5 cases to
its decision horizon. (The planned grasp is slower: with the learned carry only 187/260
finish within 15k ticks.)

## What limits the learned systems

Imitation + SAC on dev fails 22 of 260: carry wall collisions 5, carry grasp loss 5 (all
the can), carry decision horizon 5, grasp decision horizon 5, reach horizon 2. Per object,
the can is weakest (79% dev) and the foam brick and apple strongest (98%). The planner's
8 dev failures are 4 drops, 3 reach misses and 1 blocked motion.

Execution levers (imitation + SAC unless noted, dev):

| Setting | Complete | Timed | Carry drops |
|---|---:|---:|---:|
| Planned speed 1× / 1.5× / 2× / 2.5× / 3× (holding = planned) | 242 / 237 / 240 / 239 / 235 | 179 / 210 / 218 / 216 / 212 | 5 / 7 / 5 / 5 / 9 |
| Carry 0.035 / 0.05 / 0.075 m per decision (1× planned) | 154 / 209 / 242 | 0 / 0 / 179 | 0 / 2 / 5 |
| Imitation carry + turn limit 30° / 15° per decision | 213 / 55 | 193 / 52 | 3 / 1 (35 / 196 wall hits) |
| Imitation carry + half speed after 2 mm slip / both | 222 / 82 | 133 / 35 | 1 / 0 |

Faster planned motion buys deadline margin; slower or rule-constrained carrying trades
the few drops for timeouts and wall collisions.

## How we evaluate

`scripts/eval.py` runs full original-start episodes in parallel; `scripts/evaluate_mujoco.sh`
reruns every number above (about 2 hours on 44 workers). Two numbers per run: **complete**
(retained object within 10 cm of the goal after the final hold) and **timed** (complete
within 15,000 physics ticks including settling). `scripts/rubric.py` estimates the course
rubric's per-trial points (the 60 points for implemented functions are not per trial).

Episodes are deterministic for a given process setup, but borderline cases are chaotic:
floating-point differences as small as library load order flip some outcomes (the can on
validation maze 8 is dropped under the evaluator and delivered when recorded with a
different import order). Treat each case as a draw; compare systems with the ±7 noise in mind.
`scripts/record.py` imports the evaluator first so recordings reproduce its outcomes.

| Set | Cases | Role |
|---|---:|---|
| `dev` | 260 | 12 validation + 20 first-test + 20 confirmation mazes × 5 objects |
| `holdout` | 100 | 20 fresh mazes (`configs/holdout_bank_20260929.json`) |
| `course` | 100 | Course seed mapping, seeds 1000–1019 |
| `reserve` | 100 | Sealed for the final SAC/TD3 study |

Objects: foam brick 14, banana 5, tomato can 2, apple 7, mug 10.

## Training (scripts/train_mujoco.sh, ~2 hours)

| Skill | Data | Learning |
|---|---|---|
| Grasp | 240 scripted contact-grasp episodes (24k steps), 240/240 lifted | BC 24k updates; SAC+HER 12k decisions, β=200 demo loss |
| Carry | 1,799 teacher episodes (509k steps) on 192 TRAIN mazes: A* look-ahead follower, noise recoveries, and 78 carries after learned-grasp handovers (five objects) | BC 32k updates (CNN map encoder); SAC+HER 40k decisions, β=100 demo loss, object-balanced replay |
| Reach | — | SAC+HER from reward, 20k steps |

Validation during training stayed at 100% for grasp (10 episodes per check) and carry
(12 full-goal episodes per check), before and after SAC.

## Agreed study design

Imitation-only is a baseline arm of its own. Per gripper, compare **BC only**,
**BC + SAC+HER**, **BC + TD3+HER**, with matched data and budgets. In MuJoCo the
two-finger gripper and planner nearly saturate the task, so the next step needs a
direction decision: make dexterous grasping the job learning must do (the planner has no
answer for a multi-finger hand), make the task harder in a principled way, or both.

## Operations

- Dashboard: `python dashboard/server.py --port 8766` → http://127.0.0.1:8766.
- `runs/archive_pybullet/` holds every PyBullet-era run (17 GB), out of the dashboard's view.
- The Codex agent that built most of this repository is paused.
