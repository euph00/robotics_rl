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


## Completed regularized legacy pilot

`phase2_sac_grasp_regularized01` completed12,000 actual online decisions and2,000 full SAC updates with weight200. The zero-online initializer, all seven scheduled online checks, and the final checkpoint pass the same10/10 fixed grasp-validation cases and5/5 training checks. This establishes stability on this small repeated bank; it does not show an increase in success rate or establish full-task generalization. The earlier ordinary-SAC pilot had collapsed, but data/controller/goal changes also separate the runs, so they are not a controlled regularization ablation.

The declared success-then-distance selection chose the3,030-decision/504-update checkpoint. Its immutable ZIP is `runs/phase2_sac_grasp_regularized01/candidate/grasp.zip`, SHA256`779a7eefd15ae06bd41e63eef1b91802943d0fa1b8f5005e706f85b341aa099a`. The12k final actor and zero-online actor are separately frozen beside it, together with evaluation records, exact saved update counters, configuration and source hashes. Inference requires the54-state adapter, bounded IK,24cm subgoal and **legacy** proposals. No teacher acts at inference.

The online SB3 CSV records the regularizer actively opposing critic drift: at update75, RL-gradient norm31.12 and weighted-BC norm30.86 had cosine−.988. This explains why the result must be labeled SAC+HER+demonstration regularization. A separate opt-in clearance-v2 dataset will address the can's contact geometry; it does not overwrite these legacy results.


## Opt-in clearance curriculum and corrected teacher data

The separately validated `clearance-v2` controller requires approach distances along its selected grasp axis. The shared curriculum now calls `controller.pregrasp_position(distance)`; legacy mode returns exactly the old vertical offsets. Builder, cached training environment, collector and trainer record the explicit proposal mode. Reset/resume rejects a mode change, and the trainer rejects mismatched dataset geometry. Eight focused tests pass, including an actual tilted can pickup from the four physically prepared approach offsets.

`phase2_grasp_demos05` produced236/240 successful lifts. Two can42001 full-pregrasp attempts stalled with98–99 rejected actions: clipping each XYZ action component separately left the checked tilted approach direction. The paired TRAIN diagnostic `phase2_tilt_approach_probe01` reproduced both failures and fixed both with vector-norm saturation before bilateral contact, with zero rejections. This is a training-teacher change only. Learned inference remains the original continuous action mapping with the same IK/collision checks. Top-down and held-goal teacher actions are exactly unchanged; a focused regression checks this.

All48 can episodes were recollected under the declared `ray-tilted` teacher option and passed. `phase2_grasp_demos06` combines them with the192 unchanged top-down records from05, giving238/240 successes. It executes zero new physics during composition; every source dataset, episode index and reused/new collection cost is explicit. An integrity audit verifies all24,000 transitions against their declared original arrays/actions/rewards/terminal flags. The old05 attempts remain intact. Dataset SHA256:`08615ac324a8e8bc774e4d2eb9945f66d4e1f4579985408f18ed7f08a3405ded`.

`phase2_sac_grasp_clearance01` starts from the frozen legacy zero-online actor and uses fresh critics,8k declared BC refinement,2k critic calibration and a12k-decision regularized SAC pilot. Its ZIP/config/checkpoint metadata embeds `_pnr_grasp_protocol_signature`, including proposal mode, goals, demonstration teacher control, dataset hash and semantic source hashes. One of six training can point-cloud seeds uses the controller's explicit legacy fallback; the mode does not guarantee every can proposal is tilted.

The clearance branch improves from8/10 validation and4/5 TRAIN before refinement to9/10 and5/5 after2k BC updates;4k,6k and8k checks retain those scores. Initialized,258-,1,026- and3,030-decision online checks also retain9/10. The sole miss is can43001, an already exposed development-validation case. A diagnostic-only physical teacher succeeds from its pregrasp in48 decisions plus50 hold calls. The BC8k actor starts with XYZ[.782,.097,-.465] versus teacher[.567,.053,-.314], leaves the intended approach direction, and closes with about60mm remaining while the teacher remains open. The first actor action reproduces exactly and the first15 replayed object positions differ by at most0.34mm; later contact dynamics diverge, so the complete replay is not claimed exact. Artifact: `runs/phase2_grasp_trace_diagnosis02/artifacts/diagnosis.json`.

All five tilted TRAIN proposals have negative yaw; the failed validation proposal has a small positive yaw, with the nearest TRAIN orientation10.9degrees away. Quaternion signs are consistent, so there is no evidence of a double-cover representation bug. The next collection fixes18 additional TRAIN seeds42006–42023 in advance, keeps every attempt, and uses unchanged clearance geometry, ray-tilted teacher, original gripper/physics,1cm perturbations and20–26cm goals. Validation states never enter learning. Dataset append checks identical protocols and disjoint object/point-seed cases, preserves nested source provenance and failures, and verifies every composed transition against its original before writing the result. The duplicate-case rejection was exercised successfully.

The initial diagnostic run01 failed during checkpoint loading because HER requires an environment binding. This occurred after physical setup and before any episode action; run02 fixes the binding. Run01's573 setup calls/2,865 physics ticks plus480 settling ticks are explicitly marked as reconstructed from the identical deterministic run02 setup, because the initial error handler had not captured them. Run02 separately records its actual573 setup calls,554 branch calls and480 settling ticks. No failed run is removed.

The clearance01 pilot completed12,000 online decisions/2,000 SAC updates, retaining9/10 fixed validation and5/5 TRAIN throughout. Its selected checkpoint is the final actor, frozen at `runs/phase2_sac_grasp_clearance01/candidate/grasp.zip` (SHA256`fa7b8d54cdc916d6e68753721993b66b7e571bf09af84e70cf4e4030618dfbe2`); the zero-online comparator is frozen beside it. The new exporter verifies the ZIP's actual counters, matching evaluation counters and exact embedded protocol signature before making files read-only.

The18 additional TRAIN can cases produced144/144 successful lift/hold episodes. Dataset07 appends all144 to the240 prior episodes, giving382/384 successes over48 object/point-seed cases. The38,400 transitions and source success labels pass exact value comparisons against their physical source records. Dataset SHA256:`95ed04d020f176e87d4dca65017ccacb871e48e2fce5359336402e36e6240fa8`. The expansion still contains only negative-yaw tilted proposals. A further fixed32-seed TRAIN pregrasp survey (42024–42055) found25 tilted proposals between-22.87 and-7.45degrees and7 legacy fallbacks; all32 preparations succeeded, but none covered positive yaw. The bounded survey ended there, without filtering by learned-policy outcomes.

The first survey's JSON writer rejected a NumPy boolean after physical execution, losing direct per-case counters. Its failed run/logs remain. The corrected survey persists a pickle backup and each JSON case immediately, and measured12,268 native calls/76,700 total ticks including initialization. Applying those costs to the original failed run is explicitly an **estimate**, separate from conservative directly established lower bounds (`phase2_grasp_proposal_survey01/artifacts/failure_accounting.json`). No inferred count is reported as exact.

`phase2_sac_grasp_clearance02` transfers only the frozen clearance01 zero-online actor and repeats the declared8k BC/2k fresh-critic/12k regularized-SAC protocol on dataset07. An explicit seeded case-order shuffle broadens actual physical coverage within the short pilot; default ordering and previous runs remain unchanged. The exact shuffled order is saved in the run configuration. Teacher and inference action semantics are unchanged.

## TRAIN contact-settling action equivalence

On apple TRAIN seeds42/42001/42002, the frozen legacy actor was paused at its first measured bilateral contact. Three continuations per physically prepared snapshot compared immediate lifting,30 native calls holding measured arm joints with closed fingers, and10 ordinary zero-XYZ/closed-finger policy decisions. All9 retained the object through the planned carry posture; navigation was not tested. Zero-XYZ actions and the native hold differed by0.11–0.41mm in final object position and at most0.012mm finger opening. Initial finger speeds6–26mm/s decayed to at most1.12mm/s. This supports testing an observable finger-speed settling teacher using the original action interface, not injecting a pause into learned inference. Artifact: `runs/phase2_grasp_settling_probe01/artifacts/summary.json`.

The next diagnostic fixes TRAIN seed42 across all five objects and compares ordinary versus2mm/s finger-speed-gated teachers after clean-teacher and frozen-actor first contact. It records every executed transition, settling trigger, trigger after2cm lift, and hold/lift switch. This is a bounded teacher/data investigation; production teacher semantics remain unchanged pending its result.

The finger-speed gate passed20/20 TRAIN lift/hold branches, but the learned-mug seating produced29 gated decisions after2cm rise and20 late hold/lift switches. Adding an absolute vertical-speed limit20mm/s retained20/20 success while only reducing those counts to15/12; the can still had2 late holds/4 switches. Native action repetition lets measured object velocity decay between Cartesian increments, so sampled velocity alone does not identify ongoing lift reliably. Neither teacher gate is adopted. Both diagnostic cohorts and executed transitions remain in `phase2_grasp_velocity_gate_probe01` and02.

The expanded full-feature clearance02 run completed the declared8k BC checks without improving the9/10 validation result, then stopped before online training. Its selected actor came from the2k BC check; initialized ZIP retains2k fresh-critic warmup updates and zero online updates. Exact measured costs were16,940 evaluation native calls/84,700ticks,6,550 setup calls/32,750ticks, and7,200 initialization ticks. The final initialized check was interrupted after13/15 cases: all10 validation cases completed at9/10, while only3/5 TRAIN checks ran. The original report is unchanged, with a separate annotation clarifying the partial bank. The runner now marks incomplete/empty evaluations undefined rather than interpreting them as complete0%-success banks; a focused regression exercises the empty-stop case.

## Offline actor-feature ablation

The original actor consumes70 features including absolute base-relative poses, absolute orientations, arm state and absolute goals, as well as relative geometry. A separate opt-in compact actor receives28 measured features: world-axis TCP/object/goal differences, object-to-hand relative quaternion, gripper positions/velocities, finger contacts, object velocities, proposal width, and explicit TCP/goal distances. Velocity/geometry scales are explicit. This changes multiple aspects of the actor representation, so improvements cannot be attributed solely to removing absolute pose. The critics retain their full70-feature state, the raw observation remains54 values, and world-coordinate actions are unchanged; no action projection or new controller is introduced.

`phase2_grasp_actor_ablation01` trained fresh full70 and compact28 actors for exactly4,000 updates×400 examples, with seeds8101/8102 and identical object/phase sampling. Every third sorted point seed per object was withheld from this **TRAIN-only** fit; development validation43001/43002 and hidden mazes were never used. Both compact seeds improved withheld TRAIN action MSE:0.00611→0.00277 and0.00466→0.00220,54% lower on average. Premature closure near contact fell4.1–4.9%→0.75–1.50%, and balanced nominal XYZ target error improved0.72–0.76mm→0.57–0.58mm. These are offline action metrics, not physical success. All four fits used zero simulator interactions and6.4million total sampled supervised examples. Three focused tests verify feature/goal arithmetic, actor/critic checkpoint roundtrip, and interrupted evaluation reporting.

Following review, `phase2_bc_grasp_compact01` trains a fresh compact28 actor on the complete dataset07 for the same4k BC budget (seed8101), then performs the fixed15-case physical check. Explicit BC-only mode returns before critic calibration or creation of online environments. Actor feature mode and its source hash are embedded in the saved protocol; critics/world actions remain unchanged. No online continuation is authorized by this run itself.

The compact BC candidate passes the complete fixed bank: **10/10 validation and5/5 TRAIN**,41–48 policy decisions each. The previously persistent can43001 succeeds in48 decisions. This is a4k-update BC result with zero critic or online updates. Immutable checkpoint: `runs/phase2_bc_grasp_compact01/candidate/grasp.zip`, SHA256`9225fa5029c36103b1282575b33ad054ad0edf986d6b707df4c5c5b1a06e7b6d`. Evaluation used2,715 native calls/13,575ticks,6,550 setup calls/32,750ticks, and7,200 initialization ticks. Its manifest records compact28 actor/full70 critic,54 raw observations,4 unchanged world-coordinate actions, clearance-v2 proposals and24cm lift target. No carry or full-task success is claimed by this skill check.

Following review, `phase2_sac_grasp_compact01` transfers that immutable actor with zero further BC updates, then uses fresh full-state critics,2k calibration updates, beta200 and a bounded12k online-decision pilot. All actor tensors must match immediately after transfer. Exact deterministic actions on1,024 recorded TRAIN states must also remain unchanged after transfer, log-standard-deviation calibration and critic-only calibration. The verification is stored in the checkpoint and an artifact before online updates. Full TRAIN case shuffling, physical evaluation cadence and collapse safeguards are retained. The pure BC control remains immutable for full-system comparison.


## One-time first-contact settling and complete TRAIN transport

The compact BC actor's perfect small pickup bank did not establish downstream retention. A proposed history feature would expose the measured number of native calls since the first policy observation reporting bilateral contact: a pre-contact sentinel, then a counter capped at 30 calls. This would let a stateless actor distinguish a one-time settling interval from subsequent lifting without the velocity-gate chatter. It would be an explicit observation change, not a hardware, physics, or action-interface change. No such feature has been integrated.

Before changing the contract, `phase2_grasp_history_transport01` compared the first two frozen TRAIN layouts across all five objects, with two independently replayed original-start branches per case. Both use the immutable compact BC policy. The intervention executes ten ordinary zero-XYZ/closed-finger actions immediately after first observed bilateral contact, then resumes the same actor. Both branches proceed through the same 50-call pickup hold, planned carry posture, complete 0.075 m teacher carry route, and 50-call transport hold. Carry is explicitly a data teacher; these are transport diagnostics, not full-task or learned-navigation scores.

All ten pairs have exactly matching first-contact qpos, qvel, object poses, hand-relative positions, action prefixes, and native-call indices. Full retained transport is **8/10 in both branches**. Settling rescues can TRAIN000 from a carry grip loss, but turns apple TRAIN000 from a complete transport success into a planned-posture grip loss. Mug TRAIN000 fails pickup in both; the other seven pairs pass both. There are no wall/table contacts. Thus the bounded test does not support a universal settling teacher or a new history-based training branch, despite the earlier improvement on two development apple cases.

All 20 attempts and native contact traces are preserved, with completed branch pickle backups written before JSON serialization. Exact cost is 49,623 native calls / 248,115 physics ticks, plus 9,600 initialization ticks, totaling 257,715 ticks. The checkpoint and protocol are frozen in the run artifacts. Production inference, observations, teacher labels, datasets, and original physics remain unchanged.


The compact SAC pilot completed **12,000 online decisions / 2,000 full SAC updates**, following 2,000 fresh-critic initialization updates and no further BC refinement. The transferred actor, initialized actor, all scheduled online checks, and final actor retained **10/10 development validation and 5/5 TRAIN pickup/hold**. This demonstrates stability on the repeated skill bank; it does not establish improved transport or full-task success. At update 2,000, weighted BC and RL actor-gradient norms were 14.51 and 10.18, with cosine -0.886, showing that demonstration regularization remained an active constraint.

The success-then-distance rule selected 7,038 decisions / 1,172 SAC updates. Immutable checkpoint: `runs/phase2_sac_grasp_compact01/candidate/grasp.zip`, SHA256 `9c22b7f232bcbd9851ca378e110a817c4b28595862d1d56221a2842d7c826912`. Final and zero-online controls, matching evaluations, exact counters, compact28/full70 metadata and source hashes are frozen alongside it. Its BC predecessor remains unchanged. No new full-system evaluation was run for the SAC candidate at this point.

The pilot executed 36,000 training native calls, 15,540 training-setup calls, 27,084 evaluation calls, and 6,550 evaluation-setup calls, plus 24,480 initialization ticks: **450,350 new physics ticks** total. Previously collected demonstrations are reused, with their collection costs reported separately. The bounded online phase visited 30 of 48 shuffled TRAIN cases, spanning all five objects; six further cases were physically prepared by final VecEnv auto-resets but received no policy actions. All 48 cases remained available through measured demonstration replay and actor regularization. This distinction is documented in `artifacts/online_case_coverage.json`; shuffled case availability is not claimed as full online visitation.


## Bounded TRAIN final-approach speed grid

`phase2_grasp_approach_speed01` fixes the first four TRAIN layouts across all five objects and independently executes three common teacher settings per case. The ordinary 0.65 action cap changes to 0.40 or 0.20 only within 3 cm of the live proposal target before bilateral contact. The 18 mm closure gate, clearance-v2 proposals, ray-tilted approach rule, 24 cm goal, original physics, planned handover, full 0.075 m carry teacher, and final 50-call hold remain fixed. These 60 branches are diagnostic teacher executions; no new learning dataset or inference behavior is introduced.

| Near-contact cap | Full retained transport | Measured mean TCP speed | Pickup success |
| --- | --- | --- | --- |
| 0.65 | 13/20 | 0.122 m/s | 20/20 |
| 0.40 | 15/20 | 0.079 m/s | 19/20 |
| 0.20 | 17/20 | 0.039 m/s | 20/20 |

All twenty proposal triples match exactly. The slowest setting adds a mean 39 grasp physics ticks versus the ordinary teacher (range 15–60). On the ten cases retained under every setting, complete transport costs a mean 47.5 additional ticks. Comparing successful-only averages across different survivor sets would confound this tradeoff, so the paired costs are reported separately.

The 0.20 setting rescues six ordinary-teacher failures but introduces two regressions. By object, transport changes from foam 4/4 to 3/4, banana 3/4 to 2/4, can 1/4 to 4/4, apple 2/4 to 4/4, and mug 3/4 to 4/4. This supports investigating a single slower shared teacher on additional fixed TRAIN cases; it does not justify an object-specific winner lookup or claim uniformly improved seating. No production teacher, reward, observation contract, or dataset has changed.

All 60 outcomes, first-contact seating, every-native contact centroids/margins, and measured action/approach-speed records are preserved. Every completed branch is backed up before JSON serialization. Exact cost is 141,504 native calls / 707,520 physics ticks plus 28,800 initialization ticks, totaling 736,320 ticks. Summary and paired comparisons are in `runs/phase2_grasp_approach_speed01/artifacts/summary.json` and `review_metrics.json`.


## Independent TRAIN confirmation rejects the slower shared teacher

`phase2_grasp_approach_speed02` fixes the next four TRAIN layouts, 004–007, all five objects, and only the ordinary 0.65 versus proposed 0.20 cap: 40 additional branches. Every original start, proposal, closure gate, lift goal, route, hold, and physical setting is retained. The relevant environment, grasp, carry, planner, controller, and system source hashes match the first cohort.

| Cohort | Ordinary 0.65 | Slow 0.20 |
| --- | --- | --- |
| Initial TRAIN000–003 | 13/20 | 17/20 |
| Confirmation TRAIN004–007 | 18/20 | 13/20 |
| Pooled matched cases | 31/40 | 30/40 |

The confirmation reverses the apparent initial benefit. Across all 40 paired cases, slower approach rescues seven ordinary failures but introduces eight regressions; both settings pass 23 cases and fail two. Measured mean approach speed falls from 0.118 to 0.039 m/s, so this is not a null intervention. The 23 shared successes cost a mean 42 additional physics ticks under the slower teacher. These results do not support changing the shared teacher, replacing its dataset, choosing settings by object identity, or starting a new BC branch from this hypothesis.

All confirmation outcomes and costs remain in `runs/phase2_grasp_approach_speed02/artifacts`; `pooled_review_metrics.json` reports each cohort and the pooled pairs separately. Exact confirmation cost is 98,079 native calls / 490,395 physics ticks plus 19,200 initialization ticks, totaling 509,595 ticks. No production policy, demonstration dataset, observation contract, or reward changed.
