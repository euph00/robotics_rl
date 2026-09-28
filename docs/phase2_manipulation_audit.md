# Phase 2 manipulation feasibility audit

2026-09-28. Original Fetch gripper, untouched object assets, original physics, legacy development scene 42. No hidden maze bank was read or evaluated. This is a physical feasibility/development audit, not a generalization result.

## Declared five-object bank

The selected bank is **foam brick 14, banana 5, tomato soup can 2, apple 7, mug 10**. It covers a cuboid, long curved object, cylinder, rounded irregular object, and handled object. Selection uses shape diversity and physical reference pick feasibility; four of the five fail the frozen grasp policy. The parent accepted this bank before expanded learned training; later tomato-can transport failure is retained and does not justify silently replacing the object.

## All supplied objects

Every object started from the original settled pose and original robot start. The reference and frozen grasp branch from the same physically reached pregrasp snapshot; actual preparation is counted once, then both executed branches are counted. Frozen grasp uses the unchanged milestone1 checkpoint with deterministic inference and100decisions maximum, followed by 50 native hold calls when its goal is reached.

| Index | Supplied object | Original candidate count | Original reference pick | Low-contact fallback pick | Frozen grasp with original proposal |
| --- | --- | ---: | --- | --- | --- |
| 0 | 002_master_chef_can | 24 | Pass | — | Fail |
| 1 | 004_sugar_box | 23 | Pass | — | Fail |
| 2 | 005_tomato_soup_can | 24 | Pass | — | Fail |
| 3 | 006_mustard_bottle | 24 | Pass | — | Fail |
| 4 | 010_potted_meat_can | 24 | Pass | — | Fail |
| 5 | 011_banana | 0 | No proposal | Pass | Not reached |
| 6 | 012_strawberry | 0 | No proposal | Pass | Not reached |
| 7 | 013_apple | 24 | Pass | — | Fail |
| 8 | 021_bleach_cleanser | 24 | Pass | — | Fail |
| 9 | 024_bowl | 0 | No proposal | — | Not reached |
| 10 | 025_mug | 24 | Pass | — | Fail |
| 11 | 029_plate | 0 | No proposal | — | Not reached |
| 12 | 040_large_marker | 0 | No proposal | Pass | Not reached |
| 13 | 048_hammer | 0 | No proposal | Fail | Not reached |
| 14 | 061_foam_brick | 24 | Pass | — | Pass |

The original proposal succeeds physically on 9/15 objects; the frozen learned grasp succeeds only on foam brick(33 decisions). The fallback additionally picks banana, strawberry and marker, giving 12 objects with demonstrated reference pickup. Hammer has candidates but fails the stable pick criterion; bowl/plate have no opposing pair satisfying the current horizontal aperture/normal tests. This does not prove those assets physically impossible; different grasp orientations or contact strategies may be needed.

The first audit run (`phase2_manipulation_audit01`) accidentally attempted to load HER checkpoints without the required environment. Those policy branches performed zero actions; its physical reference data/costs are retained. `phase2_manipulation_audit02` is the corrected frozen-policy audit. The loader issue is not counted as learned-policy failure.

## Minimal geometric repair

The old filter required the midpoint of both contacts to sit at least 37 mm above the table. Original finger pads extend 31.5 mm below their grasp/TCP frame, so low opposed contacts can remain on the pads while the frame itself is raised to clear the table. `pnr/planning.py` now preserves the exact old candidate list whenever nonempty, and otherwise considers a raised frame with unchanged surface contact points. The maximum contact-to-frame offset is 25 mm, leaving 6.5 mm pad-end margin. Full IK, collision checking and real contact execution still follow. No gripper mesh, friction, attachment, motor parameter or simulator timestep changed.

Six focused tests pass, including preserved high-contact candidates, low-contact geometry and a real original-start banana lift/hold with zero wall/table contacts (`runs/setup/phase2_grasp_clearance_tests.log`).

## Carry and reach handovers

`phase2_manipulation_handover01` physically executes reference pickup and full-route reference transport, then evaluates the original frozen reach policy. Banana, apple, mug and foam brick retain the object through transport and pass learned reach with final errors 3.18, 3.35, 3.43 and 3.86 cm respectively. All four have zero observed wall/table contact calls and zero drops.

Tomato can picks and survives carry-posture preparation but drops late during maze transport, at 4,262 native calls. Its initial carry posture had bilateral finger contact and zero drops. This requires further grasp/transport investigation. It is not scored as a reach-policy failure because reach was never reached.

These reference-assisted handover cases are slower than the 15,000 physics-tick target and do not establish a full learned system. The frozen reach policy has useful preliminary transfer evidence; grasp clearly needs broader training.

## Multi-object training work

`pnr/general_manipulation.py` adds `GeneralGraspEnv`: original 50 state values plus live XYZ difference from end effector to the selected antipodal grasp frame, and proposal width. That frame is stored relative to the observed object and transformed using current measured object pose. No object identity, teacher action, route or hidden stage is an input. The shared planner already computes this proposal from supplied points/normals. The new feature extractor recomputes desired-minus-achieved goals after HER relabeling.

`phase2_grasp_demos.py` records actual transitions through the native action interface, four physically prepared start heights, optional physically executed centimeter-scale start variations, separate preparation/perturbation/demo costs, and all failed attempts. First preflight uses only scene 42 and the five declared objects; expanded point-cloud seeds and variations follow physical teacher validation.

The initial 5 mm teacher closure gate was too strict for taller/handled objects: can/mug developed palm contact while still open and kept pressing toward the moving object-relative target. The corrected 18 mm closure gate passed 19/20 physical preflight episodes, including all five full-pregrasp cases; its only failure was a mug start 9 cm above the target. This threshold lies within the original 31.5 mm pad span. Failures and their actions remain stored; this is training-teacher diagnosis, not an evaluation fallback.

## Artifacts

- Original/frozen all-object audit: `runs/phase2_manipulation_audit02/artifacts/summary.json`
- Low-contact physical probes: `runs/phase2_manipulation_raised01/artifacts/summary.json`
- Selected-bank transport/reach audit: `runs/phase2_manipulation_handover01/artifacts/summary.json`
- Initial physical demos: `runs/phase2_grasp_demos01/artifacts/manifest.json`
- Corrected closure preflight: `runs/phase2_grasp_demos02/`

This audit does not use the held-out maze bank. Final full-task integration must bind the 54-state grasp adapter for new grasp checkpoints; original 50-state frozen models remain unchanged.

The expanded collection uses six declared training point-cloud seeds (42 and 42001–42005), four start heights, two repeats, and physically executed XY shifts up to 1 cm. It retains 240 attempted episodes when all preparations succeed. Separate grasp-validation point-cloud seeds are 43001 and 43002; they are not hidden maze evaluations. The new trainer uses balanced object/phase BC, calibrated initial actor standard deviation, fresh-critic Bellman warmup, and ordinary SAC with 50% HER afterward. BC is never continued as an online loss in this first pilot. The online budget is at most 40,000 real decisions, with early physical checks at 256 and 1,024 decisions. Severe validation collapse is an explicit stop-and-diagnose condition, not a success claim.

Expanded collection completed: **216/240** physical successes (can 46/48, banana 43/48, apple 41/48, mug 42/48, foam brick 44/48). There were 24,000 measured decisions, 72,000 native demonstration calls, 12,666 preparation calls and 2,629 perturbation calls, plus separately recorded initial settling. `phase2_sac_grasp01` completed the bounded ordinary-SAC pilot described below.


## Ordinary SAC outcome and bounded IK correction

The 54-state actor after8,000 balanced BC updates passed9/10 full-pregrasp validation cases and4/5 seed42 training cases. Ordinary SAC+HER degraded to8/10 at256 decisions, then0/10 at1,024 and6,024 decisions. The run stopped at6,024 actual online decisions/752 full SAC updates. The first action drifted from descending near−.64 to rising near+.96 while keeping the fingers open. Fresh-critic warmup and calibrated exploration did not prevent critic-driven actor drift. The BC actor is preserved in `runs/phase2_sac_grasp01/checkpoints/bc_initialized.zip`; collapsed checkpoints are retained as failed experiments.

A separate controller diagnosis found damped local IK stalling at the torso lower limit. The existing least-squares planner found nearby exact solutions, e.g. maximum joint change.066rad within the original.35rad/.04m neighborhood. `BoundedGraspIK` now tries that solver only after local IK fails, rejects any answer outside the original neighborhood, and retains normal collision validation. All queries restore simulator/motor/contact state. Four focused tests pass, including read-only24cm binding, branch-jump rejection and the sampled-goal reset/resume contract (`runs/setup/phase2_general_grasp_integrated_tests.log`).

The three exposed full-system grasp failures (two banana and one apple) were re-executed with the unchanged BC actor and integrated fallback. All three pass at both20cm and24cm goals: **6/6**, requiring39–47 policy decisions and5–11 fallback solutions. Every native action was monitored; all six have zero robot/table and robot-or-object/wall contact calls. Artifact: `runs/phase2_grasp_failure_recheck01/artifacts/summary.json`. This is a controller repair result, not evidence of improved learning or hidden-test performance.

## Next labeled learning branch

`phase2_grasp_demos04` collects the same declared five objects and six training point-cloud seeds, with common lift goals sampled uniformly20–26cm, four physical heights, two repeats, and1cm physical XY perturbations. The shared18mm teacher closure gate remains: a separate carry diagnostic found tighter10/12/14mm gates failed all nine can pickups and offered no shared improvement. Can transport retention remains unresolved and is not hidden by successful pickup scores.

The next branch resets the actor from preserved BC, initializes fresh critics, refines on the new measured data, and trains explicitly labeled **SAC+HER+demonstration regularization**. Original-goal teacher labels are sampled independently of HER; no teacher supplies actions at inference. An offline initialized-grasp gradient probe found RL actor norm17.58 versus BC norm.1129, cosine−.675, so weight10 would be weak. The declared first branch uses weight200 with actual gradient diagnostics, fixed24cm full-pregrasp evaluation, a bounded12k-decision pilot, and the existing collapse stop rule. This is a methodological pivot after documented ordinary-SAC failure.


A paired closure diagnostic also compared the physical teacher, the frozen BC actor, and a diagnostic binary gripper mapping on identical TRAIN pregrasp states (can42/can42001/mug42/foam42). All12 pickups pass. BC first partial and negative finger commands occur together near the teacher threshold, already almost−1; binary mapping produces effectively identical contact geometry. This does not support changing the production action mapping. Traces and contact centroids are in `runs/phase2_grasp_closure_probe01/artifacts/summary.json`.


The variable-goal collection completed **238/240** successful measured lift/hold episodes: can, banana, apple and mug48/48 each; foam brick46/48. Its24,000 decisions generated72,000 native demonstration calls,12,666 setup calls and2,645 physically executed perturbation calls, plus14,400 initialization-settling ticks. All240 attempts remain in the dataset/replay;238 successful episodes provide supervised action labels. Dataset SHA256:`953257e3017397b002f86fd5fd9c370593ae72419b955901cb2374860b3902e4`. The separate `phase2_sac_grasp_regularized01` run preserves the old BC actor as a zero-refinement comparator and adds8,000 declared BC-refinement updates,2,000 fresh-critic warmup updates and at most12,000 online decisions.
