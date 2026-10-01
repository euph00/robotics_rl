# Codex handoff: Autonomous RL development for Pick–Nav–Reach

## 1. Assignment and source precedence

Act as an autonomous research engineer and implementation partner for the Master's Intelligent Robots project. Build working simulated robot policies, iterate on evidence, and produce a defensible comparison of learning methods and robot configurations. Own routine setup, implementation, debugging, training, monitoring, evaluation, and recovery within the agreed scope.

Use the [NUS-LinS-Lab Pick-Nav-Reach default project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project) as the foundation. The current user request takes precedence, then the upstream task specification and code, then these aligned documents. Resolve contradictions between upstream prose and implementation explicitly. Read [docs/project_spec.md](docs/project_spec.md) for the canonical project interpretation and [docs/upstream_audit.md](docs/upstream_audit.md) for code findings. The originals in [docs/archive](docs/archive/) and the uploaded `Project_Proposal.pptx` are historical drafts.

The task is **approach and physically pick/lift a supplied object, carry it through a maze, and bring it to the elevated goal while retaining the grasp**. The original place-and-release assumption is superseded: upstream uses a goal sphere at z = 0.9 m and does not require a placement surface, release phase, or release-only skill. Retain the supplied Fetch environment, its task geometry, original start conditions, objects, and maze variation.

The student has an AI/LLM background and introductory RL knowledge. Prioritize working policies and reliable evidence; theory explanations can follow. This document neither selects the agent model nor grants access to unavailable machines. Priority order: **correct task and training pipeline → working complete-task baseline → replicated comparison → analysis and understanding → optional breadth**. Use evidence gates rather than a prescribed calendar. The user has now authorized first-milestone setup, implementation, training, visualization, and a concise live dashboard, with CPU/GPU resources as necessary. [docs/project_state.md](docs/project_state.md) records execution state. Do not represent unrun setup or training as complete.

## 2. Intended result and research scope

The research question is: **How do SAC+HER and TD3+HER compare in learning efficiency and complete-task performance on the upstream Pick–Nav–Reach task, across two feasible Fetch gripper configurations using a documented common hybrid controller?**

| Configuration | SAC + HER | TD3 + HER |
|---|---|---|
| Supplied Fetch base, arm, and parallel gripper | Required | Required |
| Same base and arm with one meaningful validated gripper geometry modification | Required | Required |

Prefer a substantive change to finger-pad/contact and collision geometry while retaining the original low-dimensional closure interface. Select the exact modification after asset inspection and feasibility checks. Validate effective contact behavior, gripper opening/reach, object retention, maze clearance, and final goal reach before training. A new hand is not necessary. Train separate policies; cross-robot transfer is not required.

Use three independent training replications per cell by default: **twelve complete trained systems**. A system includes all of its learned skills. Multiple evaluations, skill policies, or recombinations do not create additional independent training replications. More seeds are useful only when they resolve uncertainty and fit measured capacity.

The course environment supplies 15 YCB object assets. Its evaluation uses five test objects from the supplied set and five test mazes with unprovided seeds. A fixed object and maze may simplify early learning, but held-out object/maze evaluation and appropriate geometry observations are required in the final design. Keep official unknown seeds distinct from the independently selected local development and held-out banks.

An optional third configuration may follow a working core. Independently controlled dexterous hands, legged locomotion, raw torque learning, image-based perception, learned task sequencing, simulator migration, and a separate release/placement task are outside the core. This is an algorithm-and-configuration comparison using HER, not an experiment isolating HER's contribution. Do not claim HER caused an improvement without a matched control. Robot comparisons describe these particular systems rather than universal morphology rankings.

A successful project needs credible complete-task performance from at least the original-robot baseline, a valid comparison, and explanations of weaker conditions. Every learner and seed need not succeed. An implementation defect is not an informative negative learning result.

## 3. Required components and controller structure

Implement the upstream rubric's explicit components: **antipodal grasp proposal generation from object point clouds and surface normals; IK and collision-aware motion planning to execute grasps; and navigation using maze geometry**. Replacing the keyboard controller with a generic RL call alone does not satisfy this requirement.

Establish a complete classical geometry/planning reference first. It validates the physical task and supplies a transparent reference for the learned extension. Use a shared geometry front end across the learning conditions. Start with numerical simulator observations and separate goal-conditioned skills connected by a fixed supervisor:

| Skill | Learned decisions | Proposed common assistance |
|---|---|---|
| Approach, grasp, and lift | Local end-effector execution/refinement and gripper closure | Preserve the real table approach; generate antipodal grasps; use a planned feasible pre-grasp and verified IK/collision checks |
| Carry through maze | Base motion while retaining the object | Arm held in a carrying posture; gripper commanded closed; maze geometry observations |
| Reach while holding | Final end-effector/object positioning at the elevated goal | Verified IK/collision checks; valid retained grasp and actual arrival configuration |

Finalize the precise planner/controller boundary from physical feasibility, then record and match it across learners. The full system is hybrid learned control with supplied geometry/planning assistance and a fixed task decomposition. Count all executed controller steps, including planned approach or pre-grasp motions, and explain which decisions the policy actually makes. Do not hide oracle waypoints in the learned-navigation condition. If planner-generated navigation waypoints become necessary, explicitly revise the stated learned-navigation claim and comparison protocol.

The upstream action is a `(15,)` NumPy array of absolute joint position targets. Fetch planar base x/y/yaw joints implement mobile motion; this is not wheel-velocity control. Define any bounded base/Cartesian policy actions through an explicit adapter to the supplied position-target interface. Preserve the base start `[-2, 0, 0]` and table approach in complete-task evaluation. Do not remove required segments by evaluating from a training-only near-object reset.

Use valid pre-grasped training starts, short routes, restricted ranges, and other reset curricula where useful. Broaden these toward the required object and maze distribution. Later skills must tolerate states actually produced by earlier skills. Test the complete chain without inter-stage resetting. Do not use artificial attachments as evidence of contact-based physical grasp retention.

## 4. Success, rubric, and HER correctness

Maintain two explicit success measures:

- **Raw upstream success:** the existing object-to-goal distance condition, strictly `dist_to_goal < 0.1 m`. Preserve this result for comparability with the supplied code.
- **Strict staged success:** valid physical pickup/lift, transport while retaining the object through the required route, and reaching within the target tolerance with the grasp retained. Enforce the necessary task history rather than accepting a coincidental final object position.

The goal sphere is at z = 0.9 m. Core completion does not require release or supported placement. Ground lift, hold, grasp retention, drop/collision, and final-reach checks in the actual simulator. Report rubric-aligned outcomes: lift within 3000 simulation steps, lift at least 10 cm above the table, hold stability for at least 50 steps, pose error relative to the target grasp, reaching within 10 cm and 15000 simulation steps, and carry/drop behavior. One upstream action advances five physics ticks; the rubric leaves its meaning of simulation steps unspecified. Log action calls and physics ticks, both interpretations of deadline flags, and cumulative/per-stage time without asserting an official grading interpretation. Record the rubric's inconsistent carry-stability scoring without inventing an authoritative score. See the audit for details.

Record the evaluation definition before tuning. Training rewards and curricula may change during development; keep genuine final evaluation fixed. Include sufficient state/history to make task progress observable. Object XYZ alone does not establish pickup and retained carrying. Do not claim full observability or a Markov state without checking the observation content.

Prefer a maintained SAC, TD3, and HER implementation. Stable-Baselines3 is the default candidate, subject to compatibility checks. Start with focused environment, reward, replay, boundary, and loading checks; avoid an exhaustive test platform. Preserve these implementation requirements:

- **Goal contract:** use consistent `observation`, `achieved_goal`, and `desired_goal` fields and batch reward recomputation. Goals must describe the active skill. Do not leave stale desired-goal coordinates or goal-relative features elsewhere in observations after relabeling.
- **Stock HER behavior:** SB3 replaces the whole desired-goal vector with a sampled achieved goal. Fixed obligations such as retained grasp or completed pickup must not become arbitrary relabeled Boolean targets. Choose a compatible representation; do not assume selective component relabeling is supported.
- **Replayable rewards:** score relabeled transitions from stored next achieved goals and required stored transition information. Contacts, stability, history, action costs, and shaping must be reconstructible. Never query current simulator state to score old replay. Check `info` storage settings when rewards use it.
- **Skill and goal boundaries:** keep goals fixed within independently trained skill episodes and make replay boundaries explicit. Changing waypoints, stages, or goal-dependent progress flags can invalidate naive relabeling. Prefer simple skill environments to a new replay/hierarchy research project.
- **Termination:** verify the chosen SB3 version's handling; standard HER recomputes rewards while retaining stored terminal flags. Prefer fixed training horizons with correct time-limit truncation and goal-independent physical failures. Goal-dependent success termination needs deliberate handling. Preserve final observations across automatic resets.
- **Physical validity and inference:** inspect contact-based pickup, retained transport, and final reach. Confirm that saved policies run with matching preprocessing and interfaces in a fresh process.

Sparse skill rewards with HER are the starting point. Introduce justified, replay-correct shaping during pilots if needed, apply it consistently in the final comparison, and do not then claim exclusively sparse-reward learning. HER cannot create pickup or carrying experience when exploration never moves the object. Monitor actual object motion and genuine requested-goal success, not just relabeled replay rewards.

## 5. Environment setup and development gates

Use upstream's Python 3.10 foundation with PyBullet, NumPy, Matplotlib, and Trimesh. Add Gymnasium, Stable-Baselines3, and PyTorch after verifying compatibility, then pin working dependencies. The upstream code requires an audit of imported use of `time.sleep`, reset behavior, headless execution, observations, and the control/step contract. Track fixes separately from the pinned upstream source and record them in the audit. Do not treat inspection as proof of runtime feasibility.

Advance through these evidence gates and work on independent useful tasks while bounded experiments run:

1. **Environment understood and feasible.** Audit assets, controls, geometry, resets, observations, contacts, reward, success, and termination. Implement the required geometry/planning/navigation reference and demonstrate the full physical task. Build the learning adapter and measure the actual loop.
2. **One learner makes real progress.** Establish SAC+HER on the supplied Fetch using short diagnostic pilots and suitable training starts. Seek meaningful object interaction and genuine requested-goal improvement.
3. **The complete sequence works.** Connect skills, retain the original approach, inspect handovers, and broaden later-skill reset distributions. Establish repeated uninterrupted pick–carry–reach successes on development cases, then expand object/maze coverage.
4. **The comparison is ready.** Validate the modified gripper and integrate TD3+HER. Resolve the tuning needed for a credible matched study. Freeze planner assistance, geometry observations, curricula, budgets, seed lists, evaluation, and a final manifest; report the decision without a routine approval round.
5. **Replicated results and delivery.** Run the frozen batch, evaluate held-out object/maze cases, retain failures, analyze results, verify clean inference, and prepare reproducibility and report materials. Add optional scope only after the core is secure.

For each iteration maintain: observation → plausible cause → discriminating check/change → bounded run → result → next decision. Do not repeatedly restart non-learning runs with arbitrary seeds. Distinguish physical impossibility, environment/reward defects, exploration, interfaces, and optimization before scaling compute. Train longer only with evidence of progress or a concrete hypothesis. If access is missing, state precisely what is needed and continue independent useful work.

## 6. Authority and boundaries

The user has authorized aligning earlier setup/task assumptions to the default repository; that reconciliation does not require renewed permission. Within the aligned scope, independently choose implementation details, the validated gripper modification, pilot parameters, curricula, training rewards, logging, dependencies, process placement, and experiment ordering. Launch, stop, diagnose, resume, and replace development runs as needed. Freeze and execute the final comparison without asking about each routine run or manifest.

Inspect actual CPU, RAM, disk, GPU, access, and shared-node constraints. Select bounded pilot/batch budgets from measured throughput, publish a provisional resource envelope, and continue only within actual allocations and a defensible next step. An available GPU is not unlimited or exclusive compute permission. Do not block inexpensive profiling to demand an arbitrary quota.

Bring the user a concrete decision if further progress requires changing the promised task/evaluation beyond the already authorized reconciliation, abandoning a core comparison dimension, exceeding an actual allocation, or using paid/external resources. Explain what was tried and recommend an alternative. Continue unaffected work. Do not contact others, submit assessed work, or publish externally without authorization.

Fix correctness defects promptly. If a fix affects final results, version the manifest, preserve the audit trail, and rerun affected conditions consistently. Notify the user; do not silently combine incompatible runs. Outcome-driven retuning returns to development with a newly declared comparison.

## 7. Compute, monitoring, and continuity

The inspected machine is `gpu09`, with an NVIDIA A100 80 GB, 48 accessible CPU threads, about 251 GiB RAM, and approximately 726 GB free on `/mnt/sdb` at inspection. Recheck availability before expensive work. The earlier-mentioned RTX 5080 is not verified as accessible. Limited system/root storage means large environments, caches, datasets, logs, and checkpoints should use workspace storage where practical. Match a compatible Python/PyTorch setup to the actual driver instead of assuming the installed system CUDA compiler is suitable.

Use headless simulation for training. Profile simulation, resets, replay, updates, and evaluation together. Benchmark CPU/GPU placement and process counts; do not assume the A100 accelerates every part. Avoid distributed-training machinery unless measurements justify it.

Count actual environment interactions across skills, curricula, extra training-data generation, and integration tuning. Include controller execution in complete-system cost. Distinguish policy decisions, vectorized calls, `env.step` calls, and physics substeps, and map rubric simulation limits explicitly. Separate development expenditure from final per-condition budgets. Record actor/critic updates and wall time, preserving TD3's native delayed actor updates.

Use persistent processes and a lightweight queue/supervisor. Monitor exceptions, stalls, storage, and numerical failures. Maintain checkpoints, bounded retries, and recorded continuation/restart rules. Avoid tight polling, duplicate jobs, and endless retries. Keep durable project state with the current task, known-good commands, job identifiers, resolved configs, runs, blockers, and next actions. Commit meaningful implementation milestones when a repository author identity is configured; preserve changes locally if identity is unavailable rather than inventing one. Preserve logs and checkpoints appropriately.

Ordinary policy saving may omit replay state. Test the actual recovery procedure, including cumulative counters and needed replay state. After a restart, inspect existing processes and records before launching anything. Never report a run started, completed, or successful without evidence. Communicate meaningful findings without requiring the user to supervise routine jobs.

## 8. Final experiment and interpretation

Freeze the task version, configurations, controller/planner assistance, geometry inputs, curriculum rules, seeds, interaction budgets, local evaluation bank, checkpoint rule, and failure handling after development. Use fresh independent training replications and held-out cases that did not determine tuning. Do not use hindsight goals in evaluation. Evaluate the official five unprovided maze seeds and selected test objects when available; do not label the local bank as the official test. The proposed research bank crosses five supplied objects with five distinct held-out mazes; the resulting 25 cases are our protocol, not an asserted course requirement.

Preserve the course-compatible seed mode. The upstream maze generator reseeds an internal RNG, so distinct environment seed numbers do not establish distinct or unseen mazes. Record scene/maze-content hashes to check held-out distinctness. An independently seeded maze-RNG repair is a separately versioned extension distribution and must retain course-mode evaluation rather than silently changing the seed mapping.

Within each robot, compare learners using the same observations, actions, assistance, reward, curriculum protocol, and real interaction budget, with comparable bounded tuning. Across robots, preserve a common feasible task distribution and document physical/interface differences. Disclose shared learned resources or policies and their dependence. Shared deterministic geometry code is common assistance, not another independent learned replication.

Predeclare the total interaction budget and its allocation or allocation rule across skills. Use the final system at that budget as the default checkpoint. If a development-selected checkpoint is preferable, state its selection rule before final tests and apply it uniformly. Record tuning effort and retain all correctly executed final seeds, including non-learning or unstable outcomes.

Report strict uninterrupted complete-task success centrally, raw upstream distance success alongside it, and rubric-aligned component outcomes. Include per-training-seed results and a clearly labeled aggregate. Support them with genuine requested-goal curves, grasp/transport/reach failures, drops/collisions, final goal error, and compute accounting. Distinguish isolated skills from full episodes. More evaluation episodes do not replace independent training runs; three seeds support a modest course study.

Evaluate from the actual upstream initial conditions with no inter-stage resets or training-only help. Keep preprocessing frozen and disclose all agreed planner assistance. Inspect a reproducibly selected sample of successes and failures. Attractive demonstrations illustrate behavior rather than establish reliability. Separate measured results from hypotheses; add targeted follow-ups only when they resolve a relevant uncertainty.

## 9. Deliverables and later learning

Deliver a lean runnable repository with environment adapters, explicit grasp/planning/navigation components, original and modified robot assets with provenance, configs, dependency versions, training/evaluation commands, and focused correctness checks. Preserve checkpoints, preprocessing, resolved manifests, raw results, and enough recovery state for the chosen continuation policy.

Provide a concise results summary, reproducible figures, failure analysis, raw simulation clips, and fresh-process inference. Support the eventual **5–8 page NeurIPS-style course report**, with references and permitted appendices outside the main-content limit; verify actual course formatting/submission requirements when preparing it. Do not edit the uploaded slides during this work. Presentation/final-video work and submission require separate instructions.

Once a working system exists, explain its task/goal formulation, SAC/TD3 updates, HER relabeling, geometric front end, planners/controllers, curricula, and evidence. Use actual configurations, trajectories, and failures. Keep a short decision/source record so later understanding does not depend on reconstructing forgotten experiments.

## 10. Starting actions and reference anchors

For the next implementation phase, use the reconciled specification and pinned upstream audit to establish Python 3.10, repair confirmed runtime issues, validate reset/headless execution, implement the required physical reference, add the learning adapter and focused correctness checks, then launch the first bounded SAC+HER pilot. Advance only with observed evidence and retain the full approach–pick–carry–reach task.

Use primary sources to resolve concrete questions, and check installed versions rather than treating a documentation example as a proven recipe:

- [Default task, rubric, assets, and simulator](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project)
- [Hindsight Experience Replay — original paper](https://arxiv.org/pdf/1707.01495)
- [SB3 HER documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
- [SB3 HER replay source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/her/her_replay_buffer.html)
- [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) and [SB3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [Bullet/PyBullet repository](https://github.com/bulletphysics/bullet3)
