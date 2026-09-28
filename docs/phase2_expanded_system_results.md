# Expanded-map carry in the full task

The192-map BC carry initializer improves isolated navigation to12/12 development
maps. In unconditional original-start evaluation, the best tested combination
now completes46/60 cases;45/60 also finish the final retained hold within15,000
physics ticks including settling. No held-out evaluation has occurred.

| Candidate | Complete | Within15k action ticks | Including settling |
|---|---:|---:|---:|
| Previous recovery BC carry, original grasp |42/60|41/60|38/60|
| Expanded BC carry, original grasp |44/60|43/60|42/60|
| Expanded BC carry, clearance grasp |46/60|45/60|45/60|
| Expanded BC carry, compact BC grasp |43/60|42/60|41/60|
| Learned-handover BC carry, clearance grasp |46/60|45/60|41/60|

The expanded carry actor has zero online SAC updates after its BC initialization.
The original and clearance grasp candidates use SAC+HER with continuing
demonstration-action regularization; compact grasp uses BC with zero online
updates. Reach is the frozen milestone SAC policy. This is not a
claim that all three new skills have successfully improved through online SAC.

| Object | Original grasp | Clearance grasp | Compact BC grasp | Learned-handover BC carry |
|---|---:|---:|---:|---:|
| Foam14 |11/12|9/12|7/12|8/12|
| Banana5 |11/12|11/12|10/12|12/12|
| Tomato can2 |6/12|8/12|8/12|8/12|
| Apple7 |7/12|7/12|8/12|7/12|
| Mug10 |9/12|11/12|10/12|11/12|

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

The first declared clearance case, foam14/validation000, was recorded rather
than selected after observing success. It completes at7.86cm final goal error,
with zero drops and14,165 total ticks including settling. Its MP4/WebM is featured
on the dashboard; the original first-milestone recording remains in its own run.
