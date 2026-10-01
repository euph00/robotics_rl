# Frozen best-policy reproduction diagnostic

The three predeclared attempts in `phase2_best_reproduction01` all succeeded and
matched the original successful evaluation. They did **not** reproduce the earlier
failed recording. Its cause remains unresolved; the failed recording remains a
valid result, and these three diagnostics do not change the **51/60** headline.
No fourth attempt, tuning, or held-out evaluation was performed.

## Frozen comparison

The case is `validation_000_object_14` (foam), environment seed `2912253947`,
maze seed `141130023`, independent-maze-v1. The original comparison is
`phase2_system_learned_balanced_040k01` versus its preselected recording
`phase2_system_learned_balanced_040k_video01`.

The original saved audit was reviewed before declaring any new attempts. Case,
execution settings, checkpoints, scene manifest and selected grasp matched. Its
five-call trajectory samples first differed at native call 470 during grasp;
there were no historical native-target or policy-input logs to establish why.

The diagnostic reused the exact frozen checkpoint bytes:

| Policy | SHA-256 |
|---|---|
| Grasp | `fa7b8d54cdc916d6e68753721993b66b7e571bf09af84e70cf4e4030618dfbe2` |
| Carry, object-balanced 40k | `a07f79281a48066ef3281d1d0b261f85ede542915b81873f7eb13cefc3a7838e` |
| Milestone-1 reach | `cd0c3f20ad0f938028533a8635e95f119056e0fd3abff185d7c7679f9629c258` |

The original plan digest is
`080fc139e84da0b8e2404190ecf3a063b72587452d3a07aa195f19bfe8680f09`.
Settings remain clearance-v2 proposals, 24cm lift goal, 0.075m carry scale,
planned-controller speed 1, internal 8cm reach margin, native 10cm final test
and 60-call hold. Original physics, gripper, step counts and policy inputs were
unchanged. Critical inference/controller/physics source hashes were checked
against the frozen evaluation before execution.

## All attempts and costs

| Attempt | Context | Result | Native calls | Action ticks | Ticks including settling |
|---|---|---|---:|---:|---:|
| Historical evaluation | Six-worker full evaluation | Success | 2,625 | 13,125 | 13,605 |
| Historical preselected video | Single process, rendering | Reach drop | 2,556 | 12,780 | 13,260 |
| New 1 | Six-worker executor, no rendering | Success | 2,625 | 13,125 | 13,605 |
| New 2 | Fresh single process, no rendering | Success | 2,625 | 13,125 | 13,605 |
| New 3 | Fresh single process, original rendering callbacks | Success | 2,625 | 13,125 | 13,605 |

Each new attempt used 42 grasp, 345 carry and 7 reach decisions, had zero drops,
and ended 6.887cm from the goal. All costs include the original-start prefix;
480 initialization ticks are counted separately per attempt. Total new cost:
**7,875 native calls, 39,375 action ticks, 40,815 ticks including settling**.
The old failed video ended 20.757cm from the goal with one drop.

Six spawned executor workers initialized successfully and each ran one two-second
no-op task. Only worker PID 3021113 then ran the selected physical case; the other
five performed no physics. This tests the executor initialization context,
not simultaneous six-case simulation or prior episode history in a reused worker.
The single-process PIDs were 3023027 and 3024218.

## What the traces establish

Across all 2,625 calls in each new attempt, the requested and executed absolute
joint targets, before/after float64 joint and object states, contact manifolds,
and every actual policy input/output were exactly equal. There is therefore no
new action-first or state-first divergence to localize. Initial scene, measured
state, physics parameters, motor targets, observation hashes and RNG digests
also matched.

Every new run matched the original successful saved float32 trajectory through
call 2,625. Each first differed from the old failed recording at saved call 470,
after equality through call 465. At that sample the maximum absolute differences
were 2.38e-7 in joint positions, 5.96e-8m in object position and 6.62e-7 in object
quaternion components. By call 815 the old audit measured 14.4mm object-position
displacement. Historical logs cannot tell whether a differing action or a
preceding physical difference initiated it.

The rendered attempt made 657 render calls: 656 recorded frames and one preview.
No render call changed the queried physical state or NumPy/Python/Torch CPU RNG
state. No policy prediction consumed those RNGs. All contexts used one Torch
intra-op thread, 24 inter-op threads, OMP/OpenBLAS/MKL thread limits of 1, enabled
MKLDNN, and the same backend configuration. Torch deterministic-algorithm mode
was false in all three. Rounding mode and MXCSR control bits stayed equal;
opaque saved floating-point instruction-pointer bytes differed across processes.

The standalone logging wrappers add read-only getters, copies and wall time.
They cannot rule out an observer effect, unspecified backend/process history,
or rare nondeterminism. One attempt per context is not a statistical reliability
estimate. Successful rendering here does not prove that rendering caused or
fixed the historical failure.

## Artifacts and verification

All artifacts live under `runs/phase2_best_reproduction01/artifacts/`:

- `predeclared_plan.json`: exactly three attempts, frozen sources/checkpoints,
  original settings, no retry/replacement.
- `summary.json`, `executor_behavior.json`, `artifact_integrity.json`: all costs,
  pairwise/historical comparisons, worker behavior and recorded file hashes.
- Each attempt directory: outcome, every native action/state, policy predictions,
  initial physics, backend metadata and before/after RNG snapshots.
- `single_render/validation_000_object_14.mp4` and `.webm`, plus
  `validation_000_object_14_preview.png`.
- `video_decode_audit.json`: both videos fully decoded without errors, 656 frames,
  640×360 at 12fps, duration 54.67s. Preview visually inspected.

The diagnostic run has its own completed dashboard status and registered
artifacts, explicitly excluded from the headline. The original failed video run
is retained and referenced. A successful diagnostic video must be labeled as
this predeclared third attempt; it does not replace the historical failed result.

The script was syntax checked and its logging helpers smoke checked without
physics before execution. Recorded call counts and outcome counters agree for
all three attempts; `git diff --check` passed. No production physics/policy
module or shared dashboard configuration was changed for this diagnostic.
