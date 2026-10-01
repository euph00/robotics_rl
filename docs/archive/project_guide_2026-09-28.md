# Project guide: Goal-conditioned RL for Pick–Nav–Reach

## Purpose and source of truth

Build and evaluate an extension of the [NUS-LinS-Lab Pick-Nav-Reach default project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project), using its PyBullet Fetch environment, supplied objects, maze generator, initial conditions, and course task. The research extension compares learning algorithms and a feasible gripper modification while retaining the required grasp-generation, motion-planning, and navigation components.

The current user request takes precedence, followed by the upstream task specification and implementation, then these aligned project documents. Record and resolve contradictions between upstream prose and code explicitly. The [canonical project specification](docs/project_spec.md) records the agreed interpretation; the [upstream audit](docs/upstream_audit.md) records implementation details and defects. Earlier drafts in [docs/archive](docs/archive/) and `Project_Proposal.pptx` are historical context, not the active task definition.

The core is **pick, carry, and reach while retaining the object**. The earlier place-and-release requirement is superseded: the upstream goal is an elevated sphere at z = 0.9 m, with no required placement surface, release phase, or release-only skill. A fixed maze is an early curriculum aid; generalization to the course test mazes is part of the required task.

The immediate priority is a working, correctly evaluated policy system. Setup, implementation, debugging, training management, and experiment execution may proceed autonomously within the agreed scope. Correct reward semantics, trustworthy evaluation, and reproducible records must exist from the beginning. The current request is documentation alignment and synchronization only; implementation remains pending in [docs/project_state.md](docs/project_state.md). Inspection does not establish runtime feasibility or training success.

## Task, robot, and control boundary

Use a fixed supervisor to coordinate the complete sequence:

1. **Approach, pick, and lift:** preserve the original approach from the Fetch base start `[-2, 0, 0]`; generate antipodal grasp proposals from the object point cloud and surface normals; use IK and collision-aware motion planning to reach a feasible pre-grasp; learn local grasp execution/refinement and lift.
2. **Carry through the maze:** learn base navigation with maze geometry observations while retaining the object and respecting collision rules. A carrying posture controller and commanded gripper closure are allowed and must be disclosed.
3. **Reach while holding:** learn final arm/object positioning at the requested 3D goal, maintaining the grasp and carry history. Keep the object physically retained at the elevated target.

The supervisor may select the active skill, check transition conditions, and handle failures. Describe the system as hybrid control with a fixed task decomposition. Implement explicit grasp-generation, motion-planning, and navigation functions to satisfy the upstream rubric; a generic RL wrapper alone does not meet those requirements.

Use numerical simulator observations and available low-level position control. The supplied action is a `(15,)` array of **absolute joint position targets**. The mobile base uses planar x/y/yaw joints; it is not a differential-drive wheel-velocity interface. A higher-level policy interface may produce bounded base/Cartesian increments, provided a documented adapter converts them into valid upstream position targets. Do not silently change the robot's kinematics or action semantics.

Use one shared geometry front end across learners. Establish a complete classical reference to verify feasibility. Learned manipulation may use common planned pre-grasp assistance and IK/collision checking, then learn local execution/refinement. Finalize the exact controller/planner boundary during feasibility and record it before comparative runs. Keep assistance matched between algorithms and account for every executed controller step. Learned navigation must receive useful maze geometry; introducing planned waypoints changes the navigation claim and must be explicit in the experiment definition. Do not supply an undisclosed A* waypoint oracle.

The supplied assets contain 15 YCB objects, not a single block. The rubric uses five test objects from these assets and five test mazes with unprovided seeds. Training may start with an easy object and fixed maze, but the final system must support this required variation. Do not claim unseen-maze evaluation is optional or invent the unavailable official test seeds.

Report two success measures: the unchanged upstream distance predicate (`dist_to_goal < 0.1 m`), and a separately named strict staged predicate that requires valid pickup/lift, retained transport, and reaching while still grasping. Raw upstream success does not establish the task history. Include relevant history in observations when it affects transitions or rewards. Ground lift, hold, collision, drop, and goal tolerances in the actual simulator and course rubric; record any unresolved rubric ambiguity in the audit.

## Learning algorithms and robot designs

Establish the complete task with **SAC + Hindsight Experience Replay (HER)** on the supplied Fetch first. Then add **TD3 + HER** through the same environment and replay interface.

| Condition | Learner | Robot |
|---|---|---|
| A | SAC + HER | Supplied Fetch arm and parallel gripper |
| B | TD3 + HER | Supplied Fetch arm and parallel gripper |
| C | SAC + HER | Same base and arm with one validated gripper geometry modification |
| D | TD3 + HER | Same modified robot as C |

Prefer a substantive finger-pad/contact and collision-geometry modification that preserves the original gripper's coupled control interface. Choose the specific change after inspecting the assets and validate its physical effect, reach, grasping, carrying clearance, and goal reach before training. A new hand is not necessary. Train separate policies for each configuration; cross-robot transfer is outside the core.

These conditions compare learning algorithms and robot systems. They do not isolate HER's effect because every condition uses it. A no-HER control and a third robot are optional after the core comparison is credible. Raw joint-torque control, independently controlled dexterous manipulation, legged locomotion, learned task sequencing, image-based perception, and simulator migration are outside the core. Document differences in geometry, contact behavior, control, and learnability without attributing every difference to one feature.

## Setup and getting the full task to work

Start from upstream's Python 3.10 setup with PyBullet, NumPy, Matplotlib, and Trimesh. Add Gymnasium, Stable-Baselines3, and PyTorch for the learning adapter after compatibility checks, then pin actual working versions. Audit and fix the imported-environment `time.sleep` issue, reset behavior, headless execution, observations, contacts, and action semantics before meaningful training; see the detailed upstream audit. A successful import is not evidence that the physical task works.

Demonstrate a complete feasible sequence with the geometry/planning reference. Inspect valid resets, table approach, grasp stability, robot/maze clearance, carry behavior, and final reach. The reference is a diagnostic and comparative baseline; it is not evidence of learned-policy success.

Train the skills in manageable settings before composing them. Use reachable goal distributions, valid intermediate resets, simpler approach configurations, short carrying routes, and target-near reach starts. These are training aids; final evaluation starts from the original full-task initial conditions. Curriculum resets must create valid physical states. Artificially welding the object to the gripper is not evidence of physical grasp retention.

Connect learned skills early enough to reveal handover problems. Navigation and final reach must tolerate grasps, carrying configurations, and arrival poses produced by earlier skills. Evaluate uninterrupted episodes without teacher intervention or intermediate resets.

HER reuses achieved outcomes; it does not create successful pickup or transport experience when the object never moves. Monitor actual object motion and genuine requested-goal success. Keep relabeling within the appropriate episode and skill. Rewards must be recomputable from stored transition data, including any required contact or history information. Handle goal-dependent completion, physical failure, and time-limit truncation deliberately. See the handoff for the detailed goal/replay contract.

## Evidence gates

| Gate | Evidence needed before expanding |
|---|---|
| Environment ready | Audited interfaces and rubric, valid resets, correct goal/reward handling, explicit geometry/planning components, and a physically feasible complete reference |
| Skills learning | Genuine requested-goal improvement and meaningful grasp, transport, or reach behavior |
| Full baseline working | Repeated uninterrupted pick–carry–reach episodes with real skill handovers and physical object retention |
| Comparison ready | TD3 integrated, modified gripper validated, task variation exercised, assistance frozen, and an affordable measured configuration |
| Results ready | Completed independent seed runs, held-out object/maze evaluation, interpretable failure analysis, and loadable policies |

Do not multiply robot variants to compensate for a baseline that does not learn. Diagnose physical feasibility, implementation, exploration, optimization, and composition separately. Simplify training and control within the agreed task before adding scope. Further changes to the required final task need a concrete user decision; corrections already authorized by the upstream-first instruction do not need repeated approval.

## Experimental protocol

Use **three independent training seeds per condition**, for twelve complete systems by default. Each seed includes all of its learned skills. Three skills from one system are not three independent replications. Increase seed count only when variability and measured capacity justify it.

Use shared task definitions, evaluation cases, observation information, controller assistance, curriculum rules, and bounded tuning effort. Count interactions across all skills, any training-data generation, and planner/controller execution. Distinguish learned decisions, `env.step` calls, and physics substeps. One upstream action advances five physics ticks; the rubric does not specify which count its simulation-step limits mean. Report both deadline interpretations, plus cumulative and per-stage time, without asserting an official interpretation. Preserve algorithm-defining differences such as TD3's delayed actor updates, and report actual update counts.

Separate development cases from held-out evaluation cases spanning object and maze variation. The five official unprovided seeds remain unavailable until supplied; use a declared independent local bank to estimate generalization without calling it the official test. Freeze task distribution, raw and strict success predicates, checkpoint rule, seeds, interaction budget and allocation across skills, and interruption policy before final runs. The agent may freeze this manifest autonomously when the pipeline is ready and report the decision and rationale.

Preserve a course-compatible seed mode. The upstream maze generator reseeds an internal RNG, so different environment seed numbers do not prove different mazes. Verify held-out distinctness using scene/maze-content hashes. Any independently seeded maze-generator repair defines a separately versioned extension distribution; retain course-mode evaluation. The proposed local research bank crosses five supplied objects with five distinct held-out mazes; that 25-case cross-product is our protocol, not a stated course requirement.

Report strict complete-task success centrally, alongside raw upstream success and rubric-aligned outcomes: lift within 3000 simulation steps, at least 10 cm above the table, hold for at least 50 steps, grasp-pose error, goal distance, reach within 15000 simulation steps, and carry/drop behavior. Resolve the rubric's inconsistent carry-stability point totals without inventing an authoritative score. Also report genuine-goal learning curves, stage failures, collisions, and computational cost. Keep isolated-skill tests distinct from uninterrupted full-task results.

Retain correctly executed non-learning seeds. Distinguish infrastructure interruptions and implementation defects from poor algorithmic outcomes. Do not secretly retune unsuccessful final runs, select only favorable checkpoints, or report pilot results as final. Three seeds support a modest course comparison, not sweeping statistical claims.

## Autonomous execution and resource use

The confirmed host is `gpu09`: one A100 with 80 GB VRAM, 48 accessible CPU threads, about 251 GiB RAM, and approximately 726 GB free on the workspace filesystem at inspection. These are observed resources, not an unlimited or exclusive allocation. The earlier RTX 5080 is unverified. System/root storage is limited; keep large environments, caches, datasets, logs, and checkpoints on `/mnt/sdb` where practical and recheck capacity before installation or training.

Measure simulation, reset, replay, training, evaluation, and resource throughput before choosing batch sizes and concurrency. Small state policies do not automatically benefit from the A100 more than from additional simulator CPU capacity. Use headless training and render selected evaluation rollouts.

The agent may install project-local dependencies, choose implementation details, run bounded pilots, diagnose failures, adjust development settings, select the validated gripper modification, and launch successive batches within actual allocations. Preserve user work and shared-machine constraints. Use durable supervision, logs, checkpoints, and a concise project-state record. Record code version, configuration, seeds, hardware, interactions, updates, status, and recovery history. Test clean inference and training continuation, including replay state and cumulative counters where required.

Routine tuning and individual runs do not need separate approval. Notify the user at meaningful findings and decisions. Pause for an additional substantive change to the promised task or comparison, paid resources, actual allocation exhaustion, or a consequential ambiguity that cannot be resolved from the environment and existing instructions.

## Deliverables and understanding

Deliver the working integration, explicit geometry/planning/navigation components, robot definitions and provenance, configurations, dependency record, trained policies, raw results, experiment manifest, useful rollout clips, and commands for reproducing evaluation in a clean process. Preserve enough provenance to regenerate every reported result.

Support a **5–8 page NeurIPS-style final report**, with references and permitted appendices outside the main-content limit, subject to confirmed course requirements. Explain the actual task, hybrid controller boundary, training method, comparative evidence, failures, and limitations. Never invent successful runs or omit inconvenient results.

After a working system exists, use its actual code and recorded episodes to explain SAC, TD3, HER, geometry/planning assistance, curricula, and experimental interpretation. Slide production is deferred. Preserve the uploaded presentation as the original proposal; do not submit assessed work, publish results, or assume responsibility for final-video production independently.

## Technical references

- [Default project specification and code](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project)
- [Hindsight Experience Replay, Andrychowicz et al.](https://arxiv.org/abs/1707.01495)
- [Stable-Baselines3 HER interface and replay documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) and [TD3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [Official Bullet/PyBullet repository](https://github.com/bulletphysics/bullet3)

These are implementation starting points, not evidence that this custom task will learn. Verify documentation against the pinned installed dependencies.
