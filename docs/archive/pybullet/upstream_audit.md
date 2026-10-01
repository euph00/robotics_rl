# Upstream environment audit

Audited upstream: [`NUS-LinS-Lab/Pick-Nav-Reach-Default-Project`](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project), commit **`54ab2d1faa54d1aa7e2738f114531a76c6514cd3`**.

This document records a **static source and XML audit**. No simulator was run, no dependencies were installed, and no physical grasp, navigation, timing, or learning result has been established. Source-backed behavior is distinguished below from inferences and required runtime checks. The vendored checkout was clean when audited. Paths below refer to that pinned checkout.

## Course requirements and rubric

The [upstream README](../upstream/Pick-Nav-Reach-Default-Project/readme.md) specifies this sequence (lines 4–8):

1. Generate antipodal grasp proposals, solve IK, and pick an object from the table.
2. Carry the object while navigating the robot through the maze.
3. Bring the object to the green target sphere.

The customized controller must implement grasp generation, motion planning, and navigation (README lines 63–65). Grasp generation must use the provided object point cloud and surface normals (lines 79–87). These are explicit deliverables; introducing RL does not remove them.

| Rubric component | Requirement | Points / weighting stated upstream |
| --- | --- | --- |
| Grasp generation | Function using object point cloud and normals | 20 points |
| Motion planning | Function executing generated grasps | 20 points |
| Pick execution | Lift within 3000 simulation steps | 6 points |
| Pick execution | Lift at least 10 cm above the table | 6 points |
| Pick execution | Hold stably for at least 50 steps | 6 points |
| Pick execution | Final pose close to target grasp | 6 points |
| Navigation | Navigation function using the maze configuration | 20 points |
| Reach | Object within 10 cm of goal sphere | 6 points |
| Reach | Reach within 15000 simulation steps | 6 points |
| Carry | No drop requiring a new grasp | Carry stability heading says 4 points |

Upstream labels Pick **64%** and Navigation and Reach **36%**, and describes evaluation with **five unprovided seeds**, **five test objects selected from the provided objects**, and **five test mazes** (README lines 69–75). It does not fully specify a cross-product, trial allocation, exact seeds, or the target-grasp pose tolerance. The carry bullet additionally says “2 pts each (max 8),” which is inconsistent with its 4-point heading; retain this as a rubric ambiguity rather than silently inventing a scoring scheme.

The README uses “sim steps” without defining whether that means internal physics ticks or calls to `env.step()`. The code advances five physics ticks per environment call. Record both counters and make the evaluation interpretation explicit; do not silently equate them.

## Task semantics that supersede the earlier local drafts

[Environment source](../upstream/Pick-Nav-Reach-Default-Project/pick_nav_reach_env.py), lines 126–144 and 295–306:

- The goal sphere is a visual marker with no collision shape, at **z = 0.9 m**.
- There is no support surface at the goal, no release phase, and no placement verification.
- `evaluate()` reports only object-to-goal Euclidean distance and whether it is **strictly less than 0.1 m**.
- A robot reaching the goal while leaving the object behind is not successful.
- Releasing an unsupported object at this elevated target would make it fall. The core task is therefore **pick → carry → reach while retaining the object**, not place and release.

The asset set contains **15 YCB objects**. The default `object_idx=5` is `011_banana` because object directories are sorted before indexing (environment lines 17 and 115–120). A single generic block would not represent the supplied task.

## Robot and control contract

[Fetch URDF](../upstream/Pick-Nav-Reach-Default-Project/assets/fetch/fetch.urdf), lines 21–40 and 84–118, and environment lines 58–65:

- The robot has a fixed URDF root, with virtual x/y prismatic joints and a yaw revolute joint that move its body.
- The wheel joints are fixed. This is the supplied planar, holonomic base abstraction; a differential-drive wheel controller would not match it.
- Base x/y joint limits are ±20 m relative to the loaded root, which begins at `[-2, 0, 0]`.
- The source/controller expect 15 controllable DOFs: base x/y/yaw; torso lift; head pan/tilt; seven arm joints; and two gripper fingers. Verify the actual loaded joint indices and names at runtime instead of assuming indices from XML declaration order.

Environment lines 212–286 define `step(action)` as **15 absolute joint-position targets**, not normalized increments, Cartesian displacements, wheel velocities, or torques. Revolute targets with the ±314-radian sentinel are wrapped to a nearby equivalent and all targets are clipped to joint limits. Position motors execute them with a hardcoded gain vector and a shared maximum force. `action_scale=0.05` is currently unused; incremental control is only commented out.

One call performs **five `p.stepSimulation()` calls**, with a requested wall-clock sleep of `1/240` second after each. There is no explicit `setTimeStep()` in the environment. Confirm the actual physics timestep and expose it in run metadata.

Native returns:

- `step(action) -> (observation, info)`
- `reset() -> observation`
- `info = {"dist_to_goal": scalar, "success": boolean}`

There is no Gymnasium base class, action/observation space, reward, terminated/truncated signal, time limit, `compute_reward()`, or `close()`.

## Observation contract

[Environment source](../upstream/Pick-Nav-Reach-Default-Project/pick_nav_reach_env.py), lines 308–343:

| Key | Shape | Meaning / caveat |
| --- | --- | --- |
| `qpos` | `(15,)` | Controllable joint positions; generated as float32 |
| `qvel` | `(15,)` | Controllable joint velocities; generated as float32 |
| `object_pos` | `(3,)` | Exact simulator object position |
| `object_xyzw` | `(4,)` | Exact simulator quaternion in xyzw order |
| `goal_pos` | `(3,)` | Target object position |
| `object_pc` | `(1024, 3)` | Full object mesh surface sampled at scene construction, transformed by exact object pose |
| `object_normals` | `(1024, 3)` | Corresponding transformed outward mesh normals |
| `cube_positions` | `(N, 2)` | Wall-cell centers in world x/y; N is not a guaranteed constant |

The README's `(140, 2)` is an example, not a fixed map-observation contract. The point cloud is privileged synthetic geometry, not an occluded depth-camera reconstruction. Return dtypes are mixed; the adapter must normalize types and define fixed dimensions explicitly. The unused `_get_maze_obs()` is a stub. `self.obs` is captured at construction and subsequently used for `obs_size`, but step/reset do not update that cached member.

## Scene and grasp physics

Environment lines 148–183 and [maze utilities](../upstream/Pick-Nav-Reach-Default-Project/maze_utils.py), lines 152–169 and 249–331:

- Maze: **12×16 cells**, with a **six-column left room** appended, yielding a 12×22 map.
- Each occupied cell is a **1 m × 1 m × 2 m** wall box.
- Derived world footprint is approximately x ∈ `[-4, 18]`, y ∈ `[-6, 6]`; the exit goal has x = **17.5 m**.
- Map connectivity alone does not establish clearance for the robot, arm posture, and carried object. Collision-aware inflation/planning and runtime traversability checks are required.
- The maze generator calls its path guaranteed, but its random walk has a finite step cap and adds the exit afterward even if that walk did not reach it (maze utilities lines 75–103). **Static inference:** pathological cases could lack the promised connection. Validate resulting maps rather than relying solely on the description.

The Fetch gripper uses two opposed prismatic fingers, each with range `[0, 0.05] m` and box collisions (`fetch.urdf`, lines 428–471). In its local frame, approach is along x and closure is along opposing y axes. The collision offsets imply an approximate inner gap of **2.85–102.85 mm** over the finger range. This geometry calculation needs confirmation against loaded collision shapes and the chosen end-effector frame.

There is no object attachment/weld mechanism, grasp constraint, contact-based grasp detector, drop detector, or retention metric in the Python environment. Physical contact must support the object. Each YCB URDF supplies mass/inertia and one collision mesh named `coacd_allinone.obj`; explicit contact/friction parameters were not found in these object URDFs. The effective collision geometry and contact behavior require runtime inspection; the collision filename is not proof that Bullet uses an intended decomposition.

All position motors receive maximum force **2,000,000** (environment lines 34 and 265–275), despite much lower effort values for many joints in the URDF; fingers, for example, specify 60. Self-collision flags are commented out at environment line 65. These observations justify runtime checks for tracking, excessive contact force, penetration, self-collision, and physically credible lift/retention. They do **not** establish that a grasp succeeds or fails.

## Practical repair checklist

### Compatibility and execution repairs

These repairs should preserve the course assets, world geometry, action meaning, original motor/physics settings, and goal criterion. Record their implementation/version even when their purpose is compatibility rather than a new benchmark.

- [ ] **Fix imported stepping:** line 1 imports the function with `from time import time`, while line 280 calls `time.sleep`. Only the `__main__` block later imports the module. From source, importing and stepping the environment will fail at that sleep after advancing one physics tick. Fix the import; this has not been runtime-tested here.
- [ ] **Add headless operation:** replace hardcoded `p.GUI` at line 21 with configurable GUI/DIRECT behavior.
- [ ] **Own the Bullet client:** bind every simulator call to its client, including controller/debug calls, and add `close()`/disconnect. Almost all existing calls omit `physicsClientId`; only one object-pose getter explicitly supplies the saved client.
- [ ] **Remove training pacing and per-step prints:** retain optional interactive pacing/logging without sleeping during headless training.
- [ ] **Resolve assets from the package/repository path**, not the process working directory.
- [ ] **Eliminate the shared maze-file race:** every environment currently writes then loads `./assets/maze.urdf` (environment lines 175–185; maze utilities lines 326–328). Use isolated temporary output or equivalent in-memory construction while preserving geometry.
- [ ] **Repair reset to the documented table start:** reset currently restores joints then puts the object at `[0, 0, 5]` (lines 288–293), with no settling loop, episode counter reset, or scene reload. Determine the table surface and collision-safe object pose; initialize controls; explicitly verify/reset joint and object velocities; settle the scene; reset counters and stage bookkeeping. Do not infer whether `resetBasePositionAndOrientation` already clears velocity without checking the installed Bullet behavior.
- [ ] **Record initialization and settling separately** from scored task interactions. Define how held states used by a curriculum were physically obtained.
- [ ] **Add the Gymnasium/goal adapter:** fixed spaces and dtypes, valid reset signature, step five-tuple, truncation/time limits, `is_success`, and a pure batch-compatible `compute_reward()`.
- [ ] **Add physical task metrics:** lift height, hold duration, grasp target error, retention/drop events, reach error, and action/physics counters. Keep the upstream `success` metric available separately.
- [ ] **Check save/load and deterministic replay** of scene manifests before committing to long runs.

### Changes that alter benchmark semantics

Keep a named **course compatibility mode** that preserves the pinned course geometry, native action contract, supplied force/gain/collision settings, original goal criterion, and a documented legacy seed mapping. Infrastructure fixes and corrected table reset can be shared, but their behavior must be recorded. Preserve the clean upstream checkout as the reference.

The following require an explicit, versioned experiment mode/configuration and must not be silently described as identical to upstream:

- **Randomness redesign.** `set_seed()` modifies process-global NumPy/Python state. Object mesh sampling happens before maze gate sampling, then `_load_maze()` passes hardcoded `seed=5` to a generator that resets both global RNGs (environment lines 46–48, 121–124, 155–170; maze utilities lines 30–32). Per-environment independent RNG streams are desirable, but change seed-to-scene mapping. Persist actual object IDs, maze maps/gates, sample seeds, dependency versions, and split membership. If retaining exact legacy scenes, materialize them independently before vectorized training rather than allowing worker construction to interfere through globals.
- **Changing motor force, gains, velocity limits, timestep, solver parameters, friction, object collision treatment, or self-collision flags.** Validate and document any change and apply it consistently across learner conditions. Report course-reference results separately if physics semantics differ.
- **Changing object distributions, maze density/size, start-state distributions or success criteria.** Curriculum changes are permitted training interventions but need versioned schedules; final evaluation should use the frozen full-task distribution.
- **Adding a second gripper.** This is an intentional research condition, not a compatibility fix. Preserve the original morphology as a baseline.
- **Adding target support or a release phase.** This would introduce an additional task beyond the reconciled core, not implement the existing elevated reach target.

## Learning integration and the second robot configuration

A staged controller can combine grasp proposals, IK, collision-aware motion planning, and maze navigation with learned control. Declare which decisions are learned and which are supplied by planners; evaluate the same architecture and assistance across SAC+HER and TD3+HER conditions.

The goal adapter must recompute rewards from recorded transition data and relabelled goals, without consulting the simulator's current state. Fixed obligations such as physical retention must not become arbitrary relabelled targets. Reward, replay information, episode boundaries, and terminal observations need focused checks before training. Stock upstream `evaluate()` is insufficient to implement this learning contract or certify the course pick rubric.

For the second robot configuration, a meaningful **finger/contact geometry change** is the most manageable initial candidate: keep the base, arm, wrist mounting, two finger DOFs, and shared closure command while changing documented contact geometry. A coupled three-finger design is possible in principle, but no such asset is supplied. It requires a new URDF, joint mapping, controller adaptation, collision checks, and gain-vector construction; the current motor gains are hardcoded to 15 entries (environment lines 257–262).

Before training either morphology, demonstrate a collision-aware table approach, genuine contact grasp, ≥10 cm lift, ≥50-step retention, carry posture clearance, maze traversal, and retained-object reach. No feasibility demonstration or policy training has been performed as part of this audit.
