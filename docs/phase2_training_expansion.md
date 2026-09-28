# Training diversity expansion

The carry actor remains at10/12 development navigation successes after explicit
branch-recovery demonstrations. Strong demonstration regularization (weight100)
preserves this rate through12,000 SAC+HER decisions; weight10 varies between9/12
and10/12. Validation007 remains a dead-end failure. Added recovery coverage fixes
validation011 in the initialized/strongly anchored actor, while validation010
becomes a looping failure. No held-out policy or physical evaluation has run.

To test broader map generalization, `configs/phase2_scene_bank_expanded.json`
appends128 TRAIN layouts to the original64. It preserves every old record,
partition RNG seed, object point-cloud case seed, validation layout, milestone
test layout and final reserve layout. The original manifest remains untouched.
New maps use the original generator and geometry-only admission: connectivity
and content-disjointness from all frozen partitions and course legacy maps.
No policy outcomes or route-length thresholds select these maps.

- Parent SHA256: `331d74f055327afbfc59394fc01249c3826ff17761303af1201b468c01b8d539`
- Expanded SHA256: `d6f130f4dae03b9ee4a27635bb5146cc5db48661240dd0d1da59e58e18424553`
- Extension RNG seed:20260929; counts192/12/20/20.
- Physical evaluation cost for creating the bank:0.

Two physical collectors use only TRAIN064–191: four full/segment noisy teacher
routes per map, and geometric off-route/entry-room recoveries. Preparation,
failed attempts and actual control costs are retained. Collection labels may
use the declared training teacher; learned evaluation receives no teacher route.
These runs are data collection, not evidence of policy generalization.

Dataset merging now requires explicit parent-bank compatibility when hashes
differ. It verifies immutable prior records, case RNG and non-TRAIN partitions,
then checks each source episode against its declared bank's TRAIN maps. An old
source cannot claim newly added layouts under its old hash. Dataset bytes and
failed-attempt accounting remain unchanged.

Validation:16 scene/merge tests passed (six physical-native tests excluded as
unaffected); tests cover deterministic append, immutable original and held-out
cases, disjoint map content, changed-lineage rejection, and merge admission.


Both collectors completed all128 new maps:512 standard and374 geometric
recovery episodes, each successful. The verified union contains1,721 episodes,
478,363 transitions and all six earlier failed replay episodes. Its SHA256 is
`3c180f1759c8e99ab69e631953913ac281400025694b8d213e4921712c689ed7`.
Its cumulative source lineage costs19,021,415 physics ticks including setup,
preparation and settling; the merge itself performs zero physics.

`phase2_carry_expanded_init01` transfers the prior geometric BC actor, refines
for16,000 batches of256, and calibrates fresh critics for2,000 updates with
warmup tau0.1. It passes all12 isolated development navigation mazes. The explicit
`--steps 0` mode saves a resumable checkpoint and performs no online training or
TRAIN scene priming: all training interaction/setup counters are zero. Its
physical validation costs remain recorded separately. This is a promising
initializer, not an online-SAC or whole-task result.

The next union may explicitly include all five declared original objects.
Default merging still rejects differing object sets; multi-object use requires
an explicit complete object union, verified source membership and unchanged
reward/action contracts. Optional `--balance-objects` gives each object equal
BC sampling mass before applying its existing direction/recovery strata. Object
identity is sampling metadata only, never added to the policy observation.
Resume validation checks this sampling mode; old runs default to unbalanced.
Twenty-four merge/checkpoint tests pass, including unchanged default behavior,
failed-label exclusion, explicit object admission and sampling-mode mismatch.


The five-object learned-handover union now contains503,901 transitions from1,795
real episodes. `phase2_carry_learned_init01` refines the expanded actor with8,000
object-balanced BC batches and no online decisions; isolated reference-prepared
navigation remains12/12. Its full uninterrupted validation is evaluated
separately. The archive retains all18 physical failure endpoints.

The next declared branch uses this frozen actor, fresh critics with2,000
warmup updates (tau0.1), and weight100 continuing demonstration regularization.
Both BC and original HER transition sampling give each of the five objects
20% probability. This corrects the merged buffer's96.1% foam exposure without
changing goal relabeling, rewards, physics or observations. The online worlds
use the first eight fixed TRAIN layouts crossed with all five objects and the
frozen clearance01 grasp. Every preparation failure and physical cost remains
recorded; a worker or object with no valid handover stops training. Restored
states receive the declared, counted five-call physical contact refresh.
The initial budget is40,000 online decisions, with checkpoints every8,000.
Reference-prepared navigation remains a diagnostic; full-task validation is
needed to assess object retention and select a system candidate.

Preparation checkpoint SHA, settings, exact ordered cases, replay class and
sampling mode are resume-checked.41 focused replay, seeding, checkpoint and
learned-cache tests cover this integration. This remains SAC+HER with continuing
demonstration-action regularization, not ordinary unregularized SAC.


A post-run accounting check found that the initial balanced BC run still used
the older fixed-floor group batch allocation: each256-row batch contained49
examples each for can, banana, apple and mug, and60 for foam, because all integer
rounding remainder went to the final stratum. This is approximately object
balanced, not exactly20% in expectation. Its archived run/source remain intact.
Future balanced BC initialization uses multinomial group counts to honor the
declared probabilities in expectation. Online demonstration regularization and
object-balanced HER already used categorical draws and need no correction.
The active balanced online branch has zero additional BC updates and is
unaffected; its parent initializer's actual allocation is disclosed here.


The object-balanced branch reaches51/60 uninterrupted validation successes at
40,000 decisions (5,000 SAC updates), versus46/60 for its BC initializer. Timely
completion including settling is47/60. Its trajectory is not monotonic:8k and
24k checkpoints score50/60 and46/60. The compact SAC grasp combination scores
46/60 with the8k carry, so clearance01 remains the stronger complete system.
No held-out policy or physical evaluation has been used.

A remaining replay mismatch is explicit: learned foam initially receives only
0.195% of all object-balanced draws because its4,698 rows share a stratum with
478,363 reference-prepared foam rows. The next controlled branch changes only
replay preparation-family sampling:10% reference foam,10% learned foam and20%
each of the other four learned-prepared objects. The actor initializer, physical
40-case TRAIN cache, dataset, BC regularization, fresh2k critic warmup and40k
online budget match the object-only branch. It imports fresh source labels from
a declared collection-mode mapping; online labels are added by a metadata-only
wrapper. No old replay is relabeled or migrated, and no teacher acts at inference.
Every required object/family stratum must remain eligible; missing provenance
or a vanished stratum stops sampling. Source labels do not enter observations.
This is a preparation-distribution ablation, not prioritized failure replay.
