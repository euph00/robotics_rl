# Project guide: Goal-conditioned RL for mobile manipulation

## Purpose and working approach

Build and evaluate a robot that physically picks up a block, carries it through a maze, and places and releases it at a target. Use the course-provided PyBullet environment as the starting point. The project investigates how a learning algorithm and a manageable robot modification affect performance on this complete task.

The immediate priority is a working, correctly evaluated policy system. A coding agent should perform most setup, implementation, debugging, training management, and experiment execution autonomously. Detailed theory lessons and presentation preparation can follow once there is a concrete system to inspect. Correct reward semantics, trustworthy evaluation, and reproducible records must exist from the beginning; they cannot be postponed with the theory.

The student is a Master's student in AI, comfortable with general ML and introductory RL, with more experience in LLMs and agents than robotic control. Available resources are a local RTX 5080 and a node with one A100. Inspect the actual environments, CPU allocations, memory, storage, and access before choosing execution settings.

## Core task and control boundary

Use a fixed supervisor to coordinate three learned skills:

1. **Pick and lift:** approach the block, establish a physical grasp, and lift it into a suitable carrying configuration.
2. **Carry through the maze:** navigate to the placement area while retaining the block and respecting the environment's collision rules.
3. **Place and release:** position the block at its requested target, release it, and leave it stably supported.

The supervisor may select the active skill, check transition conditions, and handle failures. It does not need to be learned. Describe the resulting system accurately as learned control with a fixed task decomposition.

Start with state observations supplied by the simulator. Use available low-level controllers, such as wheel velocity control and Cartesian end-effector commands converted through inverse kinematics. Learning raw joint torques, perception, task sequencing, and manipulation simultaneously is outside the core.

Inspect the supplied code before fixing the action interface. Clearly distinguish what the agent learns from what the controller and supervisor provide. Introducing planner-generated waypoints would change the learned navigation claim; bring that scope decision to the user and describe any agreed assistance explicitly. Prefer a simple interface that can remain consistent across the experimental conditions.

Begin with a fixed maze and feasible variation in starts and goals. Unseen-maze generalization is optional and needs suitable geometry observations and a separate evaluation design. Preserve any required approach to the block rather than silently starting beyond part of the assignment.

The final task must genuinely require the specified pick–carry–place sequence. Reaching the target by pushing or dragging the block is not equivalent. Success should require the necessary task history as well as final placement, release, and stability, using tolerances grounded in the supplied environment. Include relevant history flags in the state when they affect the task's transitions or rewards.

## Learning algorithms and robot designs

Establish the complete task with **SAC + Hindsight Experience Replay (HER)** on the supplied robot first. Then add **TD3 + HER** through the same environment and replay interface.

The core comparison is:

| Condition | Learner | Robot |
|---|---|---|
| A | SAC + HER | Supplied wheeled arm and gripper |
| B | TD3 + HER | Supplied wheeled arm and gripper |
| C | SAC + HER | Same platform with one substantive, feasible gripper modification |
| D | TD3 + HER | Same modified robot as C |

Select the modification after inspecting assets and control interfaces. Candidates include a coupled three-finger gripper or a meaningful change in finger/contact geometry. Preserve the base and arm where practical to keep integration manageable and interpretation clear. Validate reach, grasping, carrying clearance, and placement feasibility before training the modified robot.

A coupled multi-finger gripper is acceptable; independently controlled dexterous manipulation is not required. Learned legged locomotion is outside the core. A third robot is optional only after the core system and comparisons are credible, and only if it adds a useful question.

These conditions compare learning algorithms and robot systems. They do not isolate the effect of HER because all use it. A no-HER control is not required unless a later research question specifically warrants one. Robot comparisons may combine changes in contact behavior, control, and learnability; describe those changes rather than attributing every difference to one physical feature.

## Getting the full task to work

First demonstrate that the supplied task is physically feasible with a scripted or manually controlled reference. This is a diagnostic baseline, not evidence that the learned system works. Inspect grasp stability, robot–maze clearance, release behavior, and reset validity.

Train the skills in manageable settings before composing them. The agent may choose curricula, reachable goal distributions, valid intermediate resets, and per-skill reward designs. Examples include simpler approach configurations, short carrying routes, and placement resets near the target. These are training aids; the final evaluation still starts from the original full-task initial conditions.

Curriculum resets must create valid physical states. Artificially welding the object to the gripper is not evidence of physical grasp retention. If the supplied environment already abstracts grasping through an attachment mechanism, make that explicit and resolve its compatibility with the intended task before committing to it.

Connect the learned skills early enough to reveal handover problems. A navigation policy that succeeds only from a perfect carrying pose may fail after the learned grasp policy. Adapt training to states actually produced by preceding skills, and evaluate uninterrupted episodes without teacher interventions or intermediate resets.

HER reuses outcomes that occurred; it does not create successful grasp or transport experience from trajectories where the block never moves. Monitor real object motion and genuine-goal success. High relabelled success alone is not evidence of useful behavior.

Use consistent achieved and desired goals, and recompute rewards from stored transition data. Keep relabelling within the appropriate episode and skill. Handle goal-dependent completion, failure, and episode truncation deliberately. Reward shaping and supervisor logic must preserve the intended physical task and remain common between learners within a comparison.

## Evidence gates

Advance based on observed evidence:

| Gate | Evidence needed before expanding |
|---|---|
| Environment ready | Valid resets, understood controls and success rules, physically feasible reference behavior, and correct goal/reward handling |
| Skills learning | Genuine-goal improvement and meaningful grasp, transport, or placement behavior in the relevant skill |
| Full baseline working | Repeated uninterrupted pick–carry–place episodes, including real skill handovers and release |
| Comparison ready | Second learner integrated, modified robot physically validated, and an affordable measured experiment configuration |
| Results ready | Completed seed runs, held-out full-task evaluation, interpretable failure analysis, and loadable policies |

Do not multiply robot variants to compensate for a baseline that does not learn. Diagnose whether the failure is physical feasibility, implementation, exploration, optimization, or composition. Simplify training and control within the agreed task before adding scope. Changing the required final task itself requires a user decision.

## Experimental protocol

Use **three independent training seeds per condition** by default. A seed identifies an independently trained complete system, including its constituent skills. Three skills from one system are not three independent replications. Increase to five only when variability, interpretability, and available resources justify it.

Use shared task definitions, evaluation cases, observation information, controller assistance, curriculum rules, and bounded tuning effort. Count training interactions across every constituent skill and any additional training-data generation. Keep algorithm-defining differences, including TD3's delayed actor updates, and report actual update counts rather than forcing artificial equality.

Separate development cases from a held-out final evaluation bank. Fix the final task distribution, checkpoint selection rule, success predicate, seed list, resource budget, and interruption policy before final runs. The agent may freeze this manifest autonomously when the pipeline is ready and notify the user of the decision and rationale.

Primary results should include complete-task success and learning progress against real requested goals. Also report stage failures, dropped objects, collisions, placement error, and computational cost where informative. Evaluate constituent skills to diagnose failures, but keep uninterrupted full-task success central.

Retain correctly executed non-learning seeds. Distinguish infrastructure interruptions and implementation defects from poor algorithmic outcomes. Do not secretly retune unsuccessful final runs or select only favorable checkpoints. Treat pilot results as pilot results.

## Autonomous execution and resource use

Measure simulation throughput, training throughput, and resource use before choosing batch sizes and concurrency. Small state-based policies do not automatically benefit from the A100 more than from additional simulator CPU capacity. Use headless simulation for training and render selected evaluation rollouts.

The agent may independently install project-local dependencies, choose implementation details, run bounded pilots, diagnose failures, adjust development settings, select the feasible gripper modification, and launch successive batches within actual resource allocations. It should preserve user work and shared-machine constraints, and avoid assuming administrative access.

Use durable job supervision, logs, checkpoints, and a concise project-state record. Training management should survive a disconnected chat or agent context reset. Record configuration, code version, seed, hardware, interactions, updates, run status, and recovery history. Test both clean policy inference and the chosen training-resume procedure.

Routine tuning and individual training runs do not need separate approval. Notify the user at meaningful results and decisions. Pause for a substantive change to the required task, paid resources, exhaustion of the actual allocation, or a consequential ambiguity that cannot be resolved from the environment and current instructions. Do not create repeated permission gates for ordinary engineering choices.

## Deliverables and understanding

Deliver the working environment integration, robot definitions, explicit configurations, trained policies, raw results, experiment manifest, useful rollout clips, and instructions for reproducing evaluation in a clean process. Preserve enough provenance to regenerate every reported result.

Prepare a **5–8 page NeurIPS-style final report**, with references and permitted appendices outside the main-content limit. Its core should explain the task, control boundary, training method, comparative evidence, failure modes, and limitations. Avoid turning it into a generic RL tutorial. Never invent successful runs or omit inconvenient results.

Once a working system exists, use its actual code and recorded episodes to explain SAC, TD3, HER, curriculum choices, and experimental interpretation to the student. The aim is eventual ownership of the complete system, without making a lengthy theory sequence a prerequisite for the first policy.

Slide production is deferred. Raw clips and figures may support later presentations, but producing or submitting presentation materials is not part of the current work. Do not submit assessed work or publish results independently.

## Technical references

- [Hindsight Experience Replay, Andrychowicz et al.](https://arxiv.org/abs/1707.01495)
- [Stable-Baselines3 HER interface and replay documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) and [TD3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [Official Bullet/PyBullet repository](https://github.com/bulletphysics/bullet3)

These establish implementation starting points, not evidence that this custom task will learn. Verify and pin the actual installed dependencies.
