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
