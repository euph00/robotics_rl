# Aligned project specification

Reconciled on 2026-09-28 against upstream commit `54ab2d1faa54d1aa7e2738f114531a76c6514cd3`. This document describes planned work, not completed simulation or training results.

## Authority and objective

The current user instruction makes the [course repository](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project) the foundation. Its README supplies task requirements and rubric; its source supplies the actual simulator contract. Where these differ or leave gaps, document the gap rather than treating an implementation defect as an intended task rule. This specification and the updated handoff/guide replace conflicting assumptions in the archived drafts and original presentation.

Build a working system that approaches the table, generates antipodal grasps from the supplied point cloud and normals, physically picks and lifts a supplied object, carries it through the maze, and brings the object within **0.1 m in 3-D** of the goal sphere while retaining it. Then compare SAC+HER with TD3+HER across two gripper configurations, using independently trained complete systems and a reproducible evaluation.

The extension is the goal-conditioned learning study, staged learned control, gripper comparison, and rigorous evaluation on the original task. A new placement surface, object release, a different simulator, or wheel-dynamics task is unnecessary for this extension.

## Reconciliation decisions

| Earlier assumption | Upstream evidence | Working decision |
|---|---|---|
| Pick a block | Fifteen YCB-style object assets; default `object_idx=5` is a banana | Use supplied objects. A simple object can be a pilot curriculum, not the final task restriction. |
| Place, release, and settle on a target surface | Target is a collision-free visual sphere at z=0.9 m; no destination support surface is created | Final skill is reach while holding the object. Release/placement is outside the core. |
| Wheel velocity control | Fixed robot root with movable x/y/yaw joints; wheel joints fixed | Preserve the supplied planar base and position-control semantics. |
| Fixed maze; unseen layouts optional | README evaluates navigation/reach on five unprovided-seed mazes | Fixed maze is only an early curriculum. Held-out maze layouts are required. |
| Generic RL manipulation alone | README explicitly requires grasp generation, motion planning, and navigation modules | Implement these components and a classical reference; describe shared assistance in the learned extension. |
| Near-object initial condition | Robot root starts at [-2, 0, 0], object above the table at [0, 0, 1] | Include the table approach in full episodes; close starts are training curricula only. |
| Complete RL environment supplied | Only `step(action) -> (obs, info)` and a partial reset exist | Add a Gymnasium/goal adapter, rewards, episode handling, deterministic resets, and checks. |
| RTX 5080 and A100 both available | Only gpu09/A100 access has been verified | Plan locally around measured gpu09 capacity; do not assume access to the 5080. |

## Task and observation contract

The original robot has 15 controlled joints: planar x/y/yaw, torso lift, two head joints, seven arm joints, and two finger joints. Upstream actions are **absolute position targets**, clipped/wrapped by the environment and executed through position control. Each upstream action uses five physics steps. A policy can use a smaller normalized action interface, but its conversion into those targets must be explicit and identical between learners for the same robot.

Supplied observations are joint positions/velocities, object pose, goal position, 1,024 world-frame surface points and normals, and maze wall centers. Wall count depends on the maze; the README's `(140, 2)` is an example, not a fixed observation-space guarantee. Navigation needs a consistent geometry encoding, such as an occupancy grid derived from the given wall centers. No image perception is required.

Preserve the source maze geometry: a 12-by-16 maze plus a six-column starting room, 1 m horizontal cells, and 2 m walls. Preserve the original objects, table, goal height, and planar robot mechanism. Source repairs and training-only reset curricula must be distinguishable from changes to the evaluation task.

## Controllers and learning boundary

First implement the course-required components in our own controller module:

1. Antipodal grasp proposals from point pairs and normals, filtered for the actual gripper aperture and approach clearance.
2. IK with verified joint/link indexing, and collision-aware arm motion planning (for example, our own RRT-style search with checked interpolation) to approach and execute proposed grasps.
3. Maze representation, footprint/held-object clearance checks, and a navigation reference (for example, our own grid A* and path following).
4. Physical grasp validation, lift and hold checks, and a complete approach–pick–carry–reach reference rollout using the original action contract.

The reference establishes physical feasibility and provides rubric-visible functions. It is evaluated separately from the learned systems. Bullet's kinematics/collision primitives can support our algorithms; calling a keyboard controller or importing an end-to-end planner does not replace their implementation.

Use a fixed supervisor and independently trained goal-conditioned skills for the extension:

| Stage | Learned behavior | Shared assistance and constraints |
|---|---|---|
| Approach, grasp, lift | Local end-effector refinement, closure, and lift control | Antipodal candidates, verified IK, collision checking, and potentially a planned pre-grasp approach; all assistance declared and measured |
| Carry | Planar base motion conditioned on maze geometry and a fixed episode goal | Arm carry posture and gripper closure; final learned navigation must not quietly receive an A* route oracle |
| Reach | Fine base/arm positioning of the held object at the goal | Verified IK, collision checks, closed-gripper retention; no release or support surface |

During feasibility work, freeze exactly which approach segments and arm motions are planned versus learned. The default direction is a **hybrid system with common geometric assistance**, not a claim of learning every low-level motion from scratch. Count autonomous controller steps in full-task duration and report their contribution. Learned manipulation should make a substantive decision, not merely select an algorithm label while a script performs the whole task. Do not introduce planner-generated navigation waypoints without explicitly revising the learned-navigation claim.

Train skills on easier valid starts, then broaden them to the states produced by earlier skills and to the actual object/maze distribution. Final evaluation starts at the original task start and has no inter-skill resets or teacher corrections. Training states with an object already held must be physically valid, not welded attachment shortcuts.

## Success, rubric, and evaluation

Keep two named result families:

- **Upstream proximity:** preserve the source `success = norm(object_pos - goal_pos) < 0.1` exactly as a compatibility measurement. This alone does not establish successful grasping and transport.
- **Complete physical task:** document pickup/lift/hold history, retained maze transport, and a final held-object reach satisfying the same position tolerance. Pushing, throwing through the sphere, or dropping and accidentally passing the proximity check do not establish this metric. No final release requirement is added.

Record every explicit rubric item: grasp-generation function, motion-planning function, lift by 3,000 simulation steps, lift height at least 10 cm above the table, at least 50 stable hold steps, pose error relative to the target grasp, navigation function, reach within 10 cm, reach within 15,000 simulation steps, and carry stability/drop/re-grasp events.

Several scoring details are unspecified or inconsistent upstream:

- “Simulation steps” could mean environment calls or physics ticks; one source call advances five ticks. Record policy decisions, source action calls, and physics ticks separately, plus both interpretations of deadline flags. Also log cumulative and navigation-stage times. Do not claim a definitive grading interpretation until documented.
- “Final pose close to target grasp,” lift reference point, hold-stability tolerance, and collision penalties are not fully defined. Preserve continuous diagnostics and predeclare operational research thresholds before tuning/final evaluation.
- Carry stability is headed “4 pts” but its subline says “2 pts each (max 8).” Report component measurements; do not invent an authoritative corrected total score.

The README calls for five test objects from the provided set and five unprovided-seed test mazes, but supplies no exact seed list or pairing rule. Prepare a development bank and a separate held-out bank in that spirit; do not label self-selected tests as the instructor's hidden tests. Support all 15 object assets and evaluate a declared representative five-object set at minimum, with broader coverage when feasible. The proposed research bank crosses those five objects with five distinct held-out mazes, producing 25 cases per trained system; this cross-product is our protocol, not an asserted course requirement. Final object choices and seeds are frozen after feasibility and before final training/testing.

Inspect maze-content hashes as well as seed numbers: the existing generator resets an internal RNG to 5, so new environment seed values do not automatically establish distinct or unseen layouts. Preserve a course-compatible seed mode. Any repaired independent maze RNG mode is a separately versioned extension distribution, with course-mode evaluation retained.

## Experimental matrix and fair comparison

| Gripper | SAC + HER | TD3 + HER |
|---|---|---|
| Original Fetch parallel fingers | 3 independent system seeds | 3 independent system seeds |
| Same robot, meaningful finger/contact-geometry modification | 3 independent system seeds | 3 independent system seeds |

Keep the original Fetch configuration as the course-compatible track; the modified gripper is an explicitly labeled research extension, not a replacement for that track.

This is **12 independently trained complete systems**, potentially up to 36 trained skill policies if every system has three learned skills. Shared learned skills would create dependence and must not be counted as fresh system replications. A classical reference is additional to, not a cell in, this matrix.

Prefer a controlled finger-pad/shape modification because the supplied fingers already use simple collision boxes. Validate its aperture, contact, inertia, IK reach, and maze clearance before selecting exact dimensions. A cosmetic mesh change is not a morphology experiment. Keep the base, arm, joint count, closure abstraction, task bank, and controller assistance matched where feasible; document unavoidable differences.

Use the same observations, rewards, curriculum rules, and total real interaction budgets for SAC and TD3 within a robot configuration. Track all skills, curriculum/data-generation transitions, integration fine-tuning, actor/critic updates, wall time, and peak resource use. Separate pilots from final runs. Preserve algorithm-specific behavior such as TD3's delayed actor updates. Three seeds support a modest empirical comparison; neither universal algorithm/morphology claims nor HER's causal benefit follows from this matrix.

Demonstration replay and optional behavior-cloning (BC) actor initialization are permitted development aids when explicitly labeled and counted. A pilot using them is “SAC+HER with demonstration replay” or “BC-initialized SAC+HER,” not from-scratch learning. Pilot protocols may vary while diagnosing feasibility; **no demonstration/BC protocol is selected for the final comparison yet**.

If retained for the final study, freeze a common per-skill initialization protocol before training the twelve systems: demonstration collection/data and replay use, BC objective, sampling and update budget, initialization/checkpoint rules, and subsequent online budgets. Match these between SAC and TD3 within each gripper; collect physically valid data with matched declared collection rules and budgets across grippers. Count demonstration collection, failed collection attempts, physical setup and online interactions, and report BC updates/examples and compute separately. Each of the twelve systems must independently initialize and train its own skill actors, including any BC stage; shared learned actor weights must not stand in for independent system replications.

BC, if used, ends before unchanged online SAC/TD3+HER updates begin. Evaluate the initialized actor at zero online steps and subsequent learning separately; do not attribute imitation-only competence to online RL. BC action labels use their original desired goals; HER remains part of the RL replay process. Evaluation still uses the original physical task with no teacher corrections or inter-skill resets.

Primary reporting: uninterrupted complete-task success by training seed, success versus cumulative interactions, and upstream/rubric diagnostics. Supporting results: stage success, drops, collisions, target position error, grasp pose error, path/time efficiency, inference loadability, and compute. Fix budgets, checkpoint rule, evaluation cases, and failure/recovery policy before final experiments; retain valid unsuccessful seeds. No hindsight goals are used during evaluation.

## Implementation gates and deliverables

1. **Environment and reference:** create a project-local Python 3.10 environment; pin compatible dependencies; integrate headless PyBullet; repair/reset/seed/isolation issues; implement explicit baseline algorithms; verify contact grasping and complete reference rollouts. Measure table height, controllable frame transforms, simulator time, physical feasibility, and end-to-end throughput.
2. **Goal-conditioned adapter:** define observation/achieved/desired goals and replayable vectorized rewards; check HER behavior, episode boundaries, truncation and physical failures, observation consistency, fresh-process load, and resume. Add Gymnasium, SB3, and PyTorch after compatibility checks.
3. **First learned complete system:** run bounded SAC+HER pilots on the original gripper; train and join skills; inspect real contact and handovers; establish repeated uninterrupted successes and progression toward held-out mazes/objects.
4. **Comparison readiness:** physically validate the second gripper, integrate TD3, estimate affordable budgets/concurrency from profiling, and freeze the exact assistance/evaluation/experiment manifest.
5. **Replicated results:** train the 12 independent final systems, evaluate the same held-out cases, preserve trajectories/checkpoints/raw results, analyze failures and variability, and verify reproduction from a fresh process.
6. **Delivery and understanding:** runnable code/configs, asset provenance and patches, dependency record, train/evaluate commands, policies and preprocessing, manifests/recovery records, plots, selected raw clips, and an evidence-based 5–8 page NeurIPS-style report as requested in the local brief. The upstream README does not verify that report format; check course submission instructions later. Slides, final-video production, external publication, and submission remain deferred.

The available machine has an A100 80 GB, 48 logical CPUs, and about 251 GiB RAM. PyBullet simulation is expected to need CPU profiling; GPU availability alone is not a training-rate estimate. Keep dependencies, caches, temporary meshes, replay storage, and outputs on the workspace volume because the system disk is nearly full. No extra GPU machine or paid resource is assumed.

## Current limits of knowledge

The original reconciliation above was based on source/document inspection. Subsequent environment execution, physical-reference results, and skill pilots are recorded in [project_state.md](project_state.md) and [implementation_log.md](implementation_log.md). Final comparative budgets, held-out results, and gripper dimensions remain to be established.
