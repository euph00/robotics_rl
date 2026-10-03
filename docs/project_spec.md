# Project spec

Refocused 2026-10-03. Earlier plans: [archive/project_spec_2026-09-28.md](archive/project_spec_2026-09-28.md)
(SAC+HER vs TD3+HER × two grippers) and the PyBullet-era guide in [archive/](archive/).
Machine-readable version of this plan (drives the dashboard): [`configs/study.json`](../configs/study.json).

## Question

**Where does RL add value in mobile manipulation, and how does that depend on the end
effector's complexity?**

Expected claim, supported by the development results so far: RL adds little to a two-finger
gripper that a classical planner already handles well; for a 16-joint dexterous hand it is what
makes the hand usable, and the resulting system is faster than the gripper planner.

## Task

The course task, unchanged in substance: a Fetch drives from the start room to a table, picks up
a supplied object, carries it through an unseen maze and holds it within 10 cm of a goal sphere
0.9 m high. Simulated in MuJoCo (`pnr/env.py`) with **realistic physics for both end effectors**:
every joint limited in velocity, acceleration and effort; rubber grip (μ 1.0); Fetch jaws ≤ 245 N;
Allegro joints ≤ 0.7 N m. The course rubric is estimated by `scripts/rubric.py`.

## Study matrix: end effector × grasp controller

| Grasp controller | Parallel gripper (2 joints) | Allegro hand (16 joints) |
|---|---|---|
| Scripted | Planner: antipodal grasp from the point cloud, planned descent (the default project) | Scripted top-down or side power grasp |
| Imitation | Behaviour cloning of the planner's grasp | Behaviour cloning of the scripted grasp |
| Imitation + RL | BC then SAC fine-tuning | BC then PPO fine-tuning |
| RL from scratch | — (not part of the study) | PPO from scratch (rig ablation) |

**Only the grasp differs between systems.** Every system plans the rest: approach and pregrasp,
carry posture, A* maze carry, reach and final hold (refined 2026-10-03; the gripper's learned
carry and reach from the course-task study are reported separately). Learned grasps of both end
effectors train from random object starts, as the protocol evaluates them.

Hand hardware: Menagerie Allegro (right hand), mounted on the Fetch's wrist flange by its wrist
end (`hand_mount='wrist'`), as Allegro hands are mounted on arms.

## Evaluation protocol (frozen 2026-10-03)

| Item | Setting |
|---|---|
| Tasks | **Standard**: the course's five objects (foam brick, banana, soup can, apple, mug). **Wide**: master chef can and soup can ×1.6 (wide, heavy cylinders at the gripper's limit) |
| Cases | Hold-out bank: 20 unseen mazes × the task's objects (standard 100, wide 40) |
| Object start | Seeded random offset (±8 cm, ±12 cm) and yaw per case (`--random-object-pose`); the course always starts objects in one pose |
| Physics and speeds | Realistic; planned motions at speed 3 (holding 2); hand grasp ≤ 300 decisions; one settle rule for all systems (base/torso 5 mm, arm 0.02 rad, < 0.15 rad/s) |
| Primary metrics | Delivered (`complete_task_success`); delivered within 15,000 physics ticks including settling. Diagnostic: the stage where each failure happens (picked, then delivered given picked) |
| Secondary | Course rubric estimate per trial; failures by stage; median ticks |
| Seeds | 3 independently trained systems per learned condition; scripted conditions run once (deterministic) |
| Checkpoints | Chosen on development evaluations only, never on the hold-out bank. Hand: imitation = the cloned policy (step 0); imitation + RL = BC, then 10M PPO steps, then a fine-tune whose carry phase is the full task's retreat, tuck and base carry, from the protocol's object-start range (3M steps standard, 4M wide; `scripts/study_finetune.sh`): best training-time evaluation (lift and carry) within each stage's budget. Gripper: the grasp trainer's own validation choice |
| Sealed | `reserve` bank stays unused |

## Algorithms

| Where | Algorithm | Why |
|---|---|---|
| Gripper grasp | BC of the planner's grasp, then demonstration-regularised SAC+HER | Low-dimensional actions; the course-task study's recipe, now from random object starts (`scripts/train_gripper_grasp.sh`) |
| Hand grasp | BC from scripted demos, then PPO (rig pretraining, then Fetch fine-tuning) | 22-D actions, dense reward, cheap parallel simulation: the standard dexterous-manipulation recipe; SAC ran ~10× slower per wall-clock second here |
| Hand algorithm comparison | PPO (BC, then PPO) vs RLPD (SAC with layer-normalised critics, the same 243 Fetch demonstrations in half of every batch, one update per transition) on the standard objects, reported per environment step and per wall-clock hour (`--algo rlpd`, `pnr/rlpd.py`) | Tests whether a demonstration-driven off-policy method beats PPO where environment steps are expensive. Moved from the chef can: the scripted side grasp yields no Fetch demonstrations to learn from |

TD3+HER is dropped: same family as SAC, and on the gripper RL is not the bottleneck.

## Milestones

1. Protocol frozen (this document, `configs/study.json`).
2. Hand policies trained on the Fetch with the wrist-end mount (standard objects and the chef can).
3. Wide objects complete the full task with the hand (side-grasp carry posture).
4. Matrix filled with one seed per condition, both tasks.
5. Three seeds per learned condition.
6. Algorithm comparison on the hand (PPO vs RLPD, standard objects) and ablations (imitation only, RL from scratch).
7. Report (5–8 page NeurIPS-style) and presentation material (figures, clips from the dashboard).

**Stretch, only after milestones 1–6 give a complete set of results:** orientation-constrained
delivery (the object must arrive in a required orientation, e.g. a lying bottle delivered upright),
where the hand can turn the object in its fingers and the gripper must set it down and re-grasp.

## Reporting rules

Report every completed condition and seed, including failures; development numbers are labelled
as such. State the control boundary for every system (what is planned, scripted or learned).
Count training interactions for every learned component, including demonstration collection.
