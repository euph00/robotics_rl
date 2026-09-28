# Expanded-map carry in the full task

The best original-start development result is **51/60 complete tasks**, with
**47/60 finishing the final retained hold within 15,000 physics ticks including
settling** (78.3%). Excluding settling, 50/60 meet the deadline. This is the final
40k SAC+HER carry checkpoint with object-balanced replay and continuing
demonstration-action regularization. All results below are validation; held-out
evaluation remains untouched.

| Candidate | Complete | Within 15k including settling | Action ticks only |
|---|---:|---:|---:|
| Previous recovery BC carry, original grasp |42/60|38/60|41/60|
| Expanded BC carry, original grasp |44/60|42/60|43/60|
| Expanded BC carry, clearance grasp |46/60|45/60|45/60|
| Expanded BC carry, compact BC grasp |43/60|41/60|42/60|
| Learned-handover BC carry, clearance grasp |46/60|41/60|45/60|
| Balanced SAC carry at 8k, clearance grasp |50/60|47/60|48/60|
| Balanced SAC carry at 24k, clearance grasp |46/60|42/60|45/60|
| **Balanced SAC carry final 40k, clearance grasp** |**51/60**|**47/60**|**50/60**|
| Balanced SAC carry at 8k, compact SAC grasp |46/60|41/60|44/60|

BC-labeled carry candidates have zero online SAC updates after initialization.
The balanced carry branch keeps the spatial encoder over 580 measured values,
uses uniform original-object sampling before standard HER future-goal
relabeling, and retains a demonstration-action loss weighted by β=100 during
online SAC. Its 8k, 24k and final 40k checkpoints contain 999, 2,999 and 5,000 online
updates respectively. The final bundle was selected from the immutable latest
pointer only after training completed, including its final gradient update.
This is explicitly a modified SAC+HER method, not ordinary SAC alone.

The selected clearance grasp is regularized SAC+HER with 12,000 decisions,
2,000 updates and β=200; reach is the frozen milestone SAC policy. The compact
SAC substitution has a compact28 actor, 7,038 decisions, 1,172 updates and β=200;
it is distinct from the earlier zero-online compact BC grasp. All use the
original gripper. Full-system execution uses a 24 cm grasp lift goal,
clearance-v2 proposals, 8 cm reach handover, native 10 cm final acceptance and the
mandatory 60-call retained hold.

| Object | 8k complete|Final 40k complete|Final within 15k including settling|Final action ticks only|
|---|---:|---:|---:|---:|
| Foam14 |10/12|11/12|10/12|11/12|
| Banana5 |12/12|12/12|12/12|12/12|
| Tomato can2 |8/12|9/12|6/12|8/12|
| Apple7 |9/12|7/12|7/12|7/12|
| Mug10 |11/12|12/12|12/12|12/12|

Final 40k improves on its learned-handover BC control from 46/60 to 51/60
complete, and from 41/60 to 47/60 within the deadline including settling.
Six previously failing case IDs pass and one regresses. Compared with 8k,
four recover and three regress; apple performance falls from 9/12 to 7/12.
Including settling, timely success remains 47/60. The 24k dip also shows that
continuation does not improve every checkpoint or object monotonically.

The final nine failures are seven carry grasp losses, one grasp horizon and one
reach drop; no navigation horizon or final-hold failures occurred. The 8k
checkpoint had six carry grasp losses, two failed final holds, one grasp
horizon and one reach drop. At 24k, thirteen carries lose grasp and one grasp
hits its horizon. The compact-SAC combination has thirteen carry grasp losses
and one reach drop. All failures remain in each 60-case denominator.

These comparisons use the same declared validation cases and execution
settings. Carry checkpoint comparisons preserve exact frozen grasp/reach bytes;
the compact-SAC combination deliberately replaces grasp. Exact physical prefix
equivalence has not been audited, so no gain is attributed to replay balancing
alone. These are development checkpoints, not independent training replications.

The final carry SHA-256 is
`a07f79281a48066ef3281d1d0b261f85ede542915b81873f7eb13cefc3a7838e`,
from generation `step_000040000_1790592316309877827`. Immutable plans, checkpoint
hashes, trajectories, failures and paired case comparisons are preserved:

| Evaluation run | Total physics ticks including settling |
|---|---:|
|`phase2_system_learned_balanced_008k01`|794,910|
|`phase2_system_learned_balanced_024k01`|775,865|
|`phase2_system_learned_balanced_040k01`|787,405|
|`phase2_system_learned_balanced_008k_compact01`|770,410|

The clearance run's14 failures are10 carry grasp losses, three navigation
horizons and one grasp horizon. All completed carries pass reach and the final
hold. The original-grasp run has13 carry grasp losses, one navigation horizon
and two planned-posture losses. Learned handovers therefore remain the primary
training gap, despite successful isolated reference-grasp navigation.

The two grasp candidates differ in checkpoint and proposal protocol, so their
difference cannot be attributed solely to tilt geometry. Cross-run contact
sensitivity is also visible: original-grasp mug010 has identical grasp decision
counts and sub-nanometre pick-pose-error variation versus its earlier successful
control, but now loses contact during planned posture. These results are not
presented as exact paired physical prefixes or statistically independent trials.

The two new60-case runs cost773,900 and841,640 physics ticks including settling.
Their frozen plans, checkpoints, all per-case trajectories/failures and matrices
are in `runs/phase2_system_expanded_legacy01` and
`runs/phase2_system_expanded_clearance01`.

The compact BC grasp passes 15/15 isolated pickup checks, but completes only
43/60 full tasks with the same frozen carry and reach checkpoints: 42/60 within
the action deadline and 41/60 including settling. Its 17 failures comprise
13 carry grasp losses, one planned-posture grasp loss, two navigation horizons
and one reach drop. All 60 attempts and 806,055 total physics ticks including
settling are retained in `runs/phase2_system_expanded_compact01`. Isolated pickup
success is therefore insufficient for full-task candidate selection. These
aggregate results do not establish why the compact candidate differs from the
46/60 clearance candidate; no paired trajectory/contact audit has been done.

Refining the expanded carry actor with 8,000 object-balanced BC updates on the
merged five-object learned-handover dataset gives 46/60 complete tasks, 45/60
within the action deadline and 41/60 including settling. Grasp and reach use the
same frozen clearance-run checkpoints. This ties complete success but falls
short of that run's 45/60 timely completions including settling. The 14 failures
are 12 carry grasp losses, one grasp horizon and one reach horizon; all attempts
and 784,035 total ticks are retained in `runs/phase2_system_learned_init01`.
Carry received zero online updates and retains the spatial encoder over the
580-value observation. The frozen plan's descriptive algorithm text incorrectly
says “Compact” carry and “BC reach”; the actual reach policy is the same SAC
checkpoint. The immutable plan and its hash are unchanged.

The featured recording separately preselected the first declared validation
case, foam14/validation000, using the exact frozen final 40k carry, clearance
grasp and SAC reach bytes. It completes grasp and 345 carry decisions but drops
the object during reach. This failed reproduction remains visible in
`runs/phase2_system_learned_balanced_040k_video01`; it was not rerolled or added
to the full 60-case denominator. The earlier 60-case instance succeeded, so 51/60
is one validation run, not a guarantee of repeatability.

Recording cost is 2,556 native calls, 12,780 action physics ticks and 480 settling
ticks: 13,260 total. MP4 and WebM both decode cleanly: 639 frames, 640×360, 12 fps,
53.25 seconds. The saved preview was inspected; no rendering error was reported.

A read-only trace comparison finds identical saved states through native
call 465; the first recorded difference is during grasp at 470 (maximum joint
position difference 2.38×10⁻⁷, object-position distance 6.67×10⁻⁸ m). Grasp selection
and all approach/grasp/hold/posture/carry action counts agree. The object-state
difference grows to 1.44 cm near the carry handover; first sampled contact
disagreement occurs at 2550 during reach. Policy actions and full solver state
were not logged, so the first action divergence cannot be located. Evaluation,
physics and inference source hashes agree; only training helpers changed.
Both use CPU inference with one Torch thread, but the earlier run uses six
spawned workers while the video uses one process and adds rendering callbacks.
Code inspection finds no explicit added physics step, motor command, reset or
wall-clock control budget in rendering. Its causal role remains unresolved;
no new physics was run for this audit. Details are in
`artifacts/reproduction_trace_audit.json` under the recording run.
