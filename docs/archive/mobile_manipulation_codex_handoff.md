# Codex handoff: Autonomous RL development for mobile manipulation

## 1. Your assignment

Act as my autonomous research engineer and implementation partner for a Master's Intelligent Robots project. Build working simulated robot policies, iterate on evidence, and produce a defensible comparison of learning methods and robot configurations. Take ownership of routine setup, implementation, debugging, training, monitoring, evaluation, and recovery. Continue useful work without waiting for me to approve each experiment.

The task is to make a wheeled robot arm **grasp and lift a block, carry it through a maze, then place and release it at a specified target**. The course will supply PyBullet code; it has not yet been inspected. A fixed task sequence is acceptable. Treat the supplied code as an asset to audit, not an assurance of a complete RL environment or reliable grasp physics.

I have an AI/LLM background and some RL knowledge. I am comfortable delegating most implementation to Codex running GPT 6. Prioritize working policies and evidence now; deeper theory, explanations, and polished analysis can follow. Do not make tutoring, derivations, or teach-back exercises prerequisites for technical progress. This document does not configure the agent model or grant access to unavailable machines.

The priority order is: **correct task and usable training pipeline → working complete-task baseline → replicated comparison → analysis and understanding → optional breadth**. Work as quickly as the evidence and available resources allow. Use completion gates, not a prescribed calendar.

## 2. Intended result and research scope

The main question is: **How do SAC+HER and TD3+HER compare in learning efficiency and complete-task performance on a staged mobile-manipulation task, across two feasible robot configurations?**

The core matrix is:

| Configuration | SAC + HER | TD3 + HER |
|---|---|---|
| Supplied wheeled arm and gripper | Required | Required |
| Same base and arm with one meaningful, feasible gripper modification | Required | Required |

Select the second gripper after inspecting available assets. A change in finger geometry or a coupled multi-finger gripper is suitable if it changes grasp behaviour meaningfully and can perform the physical task. Coupled fingers can share a low-dimensional closure command. Train separate policies for the configurations; cross-robot transfer is not required.

Use three independent training replications per cell as the default: twelve complete trained systems. More seeds are useful only when they resolve an actual uncertainty and fit measured capacity. A system replication includes all its learned skills; multiple evaluations or recombinations of the same trained skills do not create additional independent replications.

An optional third configuration may follow a working core. Independently controlled dexterous hands, legged locomotion, perception from images, learned high-level task sequencing, and simulator migration are outside the core. Do not add them to make the project look broader.

This is an algorithm-and-configuration comparison using HER, not an experiment isolating HER's contribution. Do not claim HER caused an improvement without a matched no-HER control. Such a control is optional and should only be added if a specific question warrants the cost. Differences across robot designs are results for the documented systems, not universal rankings of morphology.

A successful project needs credible complete-task performance from at least the baseline, a valid comparison, and clear explanations of weaker conditions. It does not require every learner and seed to succeed. An unresolved implementation defect is not an informative negative learning result.

## 3. Recommended controller structure

Start with numerical simulator observations, established low-level position/velocity controllers, and separate goal-conditioned skills connected by a fixed supervisor:

| Skill | Learned decisions | Initial assistance |
|---|---|---|
| Grasp and lift | End-effector motion and gripper closure | Base positioned near the object; IK or another verified arm controller |
| Carry through maze | Base motion while retaining the block | Arm held in a carrying posture; gripper commanded closed |
| Place and release | End-effector motion and gripper opening | Training starts near the target with a valid grasp |

This is a recommended starting architecture, not a requirement to implement a general hierarchy framework. Keep the stage supervisor and assistance consistent between learners. Document what is learned and what the controller supplies.

Keep the initial maze fixed, with feasible variation in starts and goals. Unseen-layout generalization is optional and requires suitable geometry observations and a separate evaluation definition. If the task requires approaching the block from farther away, extend the navigation skill or include that approach explicitly; do not silently remove a required segment of the task.

Training may use valid pre-grasped starts, shortened routes, restricted position ranges, and other reset curricula. Broaden these as learning improves. Later skills must tolerate states produced by earlier learned skills, including imperfect grasps and arrival poses. Test the complete chain without resetting between skills.

A scripted controller is useful to validate physical feasibility. If object attachment constraints, route planners, or scripted manipulation exist in the supplied code, distinguish those abstractions from learned physical performance. The core target is physical grasp/carry/release with learned skills. Adopting an attachment shortcut or replacing learned maze navigation with a planner is a scope decision, not a hidden training fix.

## 4. Success and correctness

Before meaningful training, define the real completion predicate from the assignment and simulator: genuine pickup/lift, permitted transport through the maze, and release onto the target surface with acceptable position and stability. Define drop/collision rules, tolerances, and settling criteria from the actual environment. Object XYZ alone must not count pushing, throwing, or indefinitely holding the block at the target as successful completion.

Record this definition before tuning. You may adapt training rewards and curricula, but keep the genuine evaluation task fixed. Include the state/history needed to make task progress observable. Do not claim full observability or a valid Markov state without checking what the observation contains.

Prefer a maintained implementation of SAC, TD3, and HER over reimplementing them. SB3 is the default candidate, subject to environment compatibility. Verify and pin the actual dependency versions. Start with focused checks of the environment, reward, replay, episode boundaries, and model loading; avoid building an exhaustive test platform.

The following issues need implementation checks, not a theoretical report before coding:

- **Goal contract:** provide consistent `observation`, `achieved_goal`, and `desired_goal` fields and batch reward recomputation. Goals and rewards must describe the intended skill. Do not leave stale desired-goal coordinates or goal-relative features elsewhere in observations after relabelling.
- **Stock HER behaviour:** SB3 samples an achieved-goal vector and replaces the whole desired-goal vector. Fixed obligations such as retaining or releasing the object must not accidentally become arbitrary relabelled booleans. Choose a compatible representation; do not assume stock replay supports selective relabelling of goal components.
- **Replayable rewards:** compute relabelled rewards from the stored next achieved goal and any necessary stored transition information. Contact, stability, history, action costs, or shaping terms must be reconstructible. Never query the simulator's current state to score an old replay transition. Verify information-storage settings if using `info`.
- **Skill and goal boundaries:** keep goals fixed within independently trained skill episodes, and replay boundaries explicit. Changing waypoints, stages, or goal-dependent progress flags can invalidate naive relabelling. Prefer simple skill environments to a new replay or hierarchy research project.
- **Termination:** current SB3 HER recomputes rewards while retaining stored terminal flags. Prefer fixed training horizons with correct truncation handling and goal-independent physical failures. Goal-dependent early success termination needs deliberate handling. Preserve the true final observation across automatic resets.
- **Physical validity and inference:** inspect contact-based pickup, transport, and release in rollouts. Confirm that saved policies load and run with the same preprocessing and interfaces in a fresh process.

Sparse skill rewards with HER are the starting point. You may introduce justified shaping during pilots if needed for learning, applying it consistently across the final comparison and documenting the resulting claim. HER can be used with shaped rewards, but the project must then not claim learning exclusively from sparse rewards. Reward shaping must remain correct under goal relabelling.

## 5. Autonomous development loop

Advance through these evidence gates. Work on independent useful tasks while experiments run; avoid multiplying unvalidated conditions.

1. **Environment understood and feasible.** Inspect robot assets, controls, observations, reset options, physics, maze geometry, and existing reward/termination code. Demonstrate physical feasibility with a scripted or supplied controller. Wrap the environment as needed and measure the actual training loop.
2. **One learner makes real progress.** Establish SAC+HER on the supplied robot. Use short, diagnostic pilots and suitable training starts. Look for meaningful object interaction and genuine-goal improvement, not merely rising replay reward.
3. **The complete sequence works.** Connect the skills on a manageable route. Inspect handover failures and broaden later-skill training starts. Establish repeatable uninterrupted successes on development cases before expanding the comparison.
4. **The comparison is ready.** Integrate a physically validated second gripper and TD3+HER. Resolve only the tuning needed for a credible, matched study. Freeze a final experiment manifest and report it to me; routine experiment lock does not require a separate approval round.
5. **Replicated results and delivery.** Run the frozen batch, evaluate genuine held-out cases, retain failures, analyse results, verify clean inference, and prepare the reproducibility package and report material. Add optional scope only after the core is secure.

For each iteration, maintain a short chain: observation → plausible cause → discriminating check or change → bounded run → result → next decision. Prefer interventions that resolve a specific uncertainty. Do not repeatedly restart a non-learning run with arbitrary new seeds.

When a pilot fails, distinguish environment or reward defects, physical impossibility, missing exploration, interface problems, and learner instability before scaling compute. Training longer is justified by evidence of progress or a concrete hypothesis, not by inertia.

If code or hardware access is missing, identify exactly what is needed and continue useful independent work. Avoid building an elaborate substitute simulator before inspecting the supplied code.

## 6. Authority and boundaries

You may independently choose and change implementation details, the feasible second gripper, pilot hyperparameters, skill/reset curricula, training reward design, logging, dependency choices, process placement, and experiment ordering within this scope. You may launch, stop, diagnose, resume, and replace development runs. You may freeze and execute the final comparison without asking me to approve each run or routine manifest.

Use the local RTX 5080 and the node with one A100 within the access and allocations actually granted. Inspect available CPU cores, RAM, disk, GPU compatibility, and shared-node constraints. Select bounded pilot and batch budgets from measurements, record them, and continue while the work remains within the resource envelope and has a defensible next step. Do not infer unlimited compute permission from an available GPU. If no hard cap is supplied, make the initial batch small and publish a provisional envelope; do not block inexpensive profiling to request an arbitrary quota.

Bring me a concrete decision when progress requires changing the promised task or evaluation, abandoning a core comparison dimension, exceeding an actual resource allocation, or adding paid/external resources. Explain what was tried and recommend a specific alternative. Continue unaffected work when possible. Do not contact others, submit assessed work, or publish externally without authorization.

Fix correctness defects promptly. If a fix affects final runs, version the manifest, retain the audit trail, and rerun the affected conditions consistently. Notify me; do not ask about every repair or silently combine incompatible results. Outcome-driven retuning belongs back in development with a newly declared final comparison.

## 7. Compute, monitoring, and continuity

Use headless simulation for training. Profile simulation, resets, replay, updates, and evaluation together. Benchmark practical CPU/GPU placement and simulator process counts; do not assume an A100 accelerates every part of this workload. Avoid distributed-training machinery unless measurements justify it.

Count actual environment transitions, including all skills, curriculum training, and any integration fine-tuning. Distinguish transitions from vectorized calls and internal physics substeps. Record development expenditure separately from the final per-condition budget. Record actor/critic update counts and wall time; preserve TD3's native delayed actor updates rather than forcing artificial equality with SAC.

Use persistent training processes and a lightweight queue or supervisor appropriate to the host. Training should survive an agent interruption where practical. Monitor progress, exceptions, stalled jobs, storage, and numerical failures. Use checkpoints, bounded retries, and recorded resume/restart rules; avoid tight polling, duplicate jobs, or endless automatic restart loops.

Keep durable project state in the working repository: current task, known-good commands, active job identifiers, resolved configs, completed runs, blocking issues, and next actions. Commit meaningful changes and preserve logs and checkpoints appropriately. Ordinary model saving may omit replay state; test the chosen continuation procedure and maintain cumulative counters.

After a restart, inspect existing processes and records before launching anything. Never report a run as started, completed, or successful without evidence. Send concise progress updates at meaningful results or decisions; routine jobs should not require my attention.

## 8. Final experiment and interpretation

Freeze the final task version, configurations, curriculum rules, training seeds, interaction budgets, evaluation cases, checkpoint rule, and failure handling after development. Use fresh independent training replications and a held-out evaluation bank that did not determine tuning. Do not use hindsight goals during evaluation.

Within each robot, compare learners using the same observations, actions, controller assistance, reward, curriculum protocol, and real interaction budget, with comparable bounded tuning effort. Across robots, preserve a common feasible task distribution and document necessary interface or physical differences. If a skill or resource is shared between systems, disclose the dependence and do not overcount independent training seeds.

Predeclare the total interaction budget and its allocation or allocation rule across skills. Use the final system at that budget as the default reporting checkpoint. If a development-selected checkpoint is more appropriate, state the selection rule before final tests and apply it uniformly. Record tuning effort and preserve all correctly executed final seeds, including non-learning and unstable ones.

Report complete-task success as the primary metric, with individual training-seed results and a clearly labelled aggregate. Support it with genuine-goal learning curves, stage outcomes, drops/collisions, placement error, and compute accounting where useful. Distinguish isolated-skill tests from uninterrupted task results. More evaluation episodes do not replace independent training runs. Three seeds support a modest course study, not sweeping statistical claims.

Evaluate from the real task start, with no inter-stage resets or training-only assistance. Keep preprocessing frozen. Inspect a reproducibly selected sample of successes and failures; label attractive demonstration clips as illustrations rather than evidence of reliability.

Post-hoc analysis can explain observed behaviours and suggest mechanisms. Separate measured results from hypotheses. A causal explanation may need a targeted follow-up experiment; do not add a large ablation suite automatically.

## 9. Deliverables and later learning

Keep the implementation lean. Deliver a runnable repository with environment adapters, robot assets and provenance, explicit configs, training/evaluation commands, a dependency record, and focused correctness checks. Preserve policy checkpoints, required preprocessing, resolved run manifests, raw results, and enough recovery state to support the chosen resume policy.

Provide a concise results summary, reproducible figures, failure analysis, raw simulation clips, and clean inference from a fresh process. Support the eventual 5–8 page NeurIPS-style course report, with references and permitted appendices outside the main-content limit; check formatting and submission requirements when preparing it. Do not create or edit slides now. Presentation work will follow separate specifications; do not assume responsibility for final-video production or submission.

After a working system exists, explain the actual implementation: task/goal formulation, SAC and TD3 updates, HER relabelling, supplied controllers, curricula, and why the evidence supports the conclusions. Use recorded configurations, trajectories, and failures to guide that explanation. Keep a small decision/source record from the beginning so later understanding does not depend on reconstructing forgotten experiments.

## 10. Starting actions and reference anchors

Begin by inspecting the actual repository, supplied environment code, and available compute. Summarize the task interface and the smallest credible route to a first complete baseline. Implement the necessary adapter and correctness checks, demonstrate physical feasibility, and launch the first bounded SAC+HER pilot. Record decisions and continue iterating autonomously.

Use primary sources to resolve concrete implementation questions; verify the installed versions rather than treating a documentation example as a proven recipe:

- [Hindsight Experience Replay — original paper](https://arxiv.org/pdf/1707.01495): goal relabelling, off-policy learning, and the original manipulation setup, including exploration assistance.
- [SB3 HER documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html): supported learners, goal interface, replay options, and implementation caveats.
- [SB3 HER replay source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/her/her_replay_buffer.html): actual reward, sampling, information-storage, and terminal-mask behaviour.
- [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) and [SB3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html): learner interfaces and algorithm-specific update rules.
- [Bullet repository](https://github.com/bulletphysics/bullet3): simulator resources and upstream documentation; supplied course code remains the immediate implementation reference.
