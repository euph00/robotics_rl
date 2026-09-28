# Expanded-map carry in the full task

The best original-start development result is now **50/60 complete tasks**,
with **47/60 finishing the final retained hold within 15,000 physics ticks
including settling**. Excluding settling, 48/60 meet the deadline. This uses
SAC+HER carry with object-balanced replay and continuing demonstration-action
regularization. All results below are validation; held-out evaluation remains
untouched.

| Candidate | Complete | Within15k action ticks | Including settling |
|---|---:|---:|---:|
| Previous recovery BC carry, original grasp |42/60|41/60|38/60|
| Expanded BC carry, original grasp |44/60|43/60|42/60|
| Expanded BC carry, clearance grasp |46/60|45/60|45/60|
| Expanded BC carry, compact BC grasp |43/60|42/60|41/60|
| Learned-handover BC carry, clearance grasp |46/60|45/60|41/60|
| Balanced SAC carry at 8k, clearance grasp |50/60|48/60|47/60|

Carry candidates labeled BC have zero online SAC updates after initialization.
The new balanced SAC candidate has 8,000 online decisions and 999 SAC updates,
with object-balanced HER replay and continuing demonstration-action loss weighted
by β=100. This is explicitly a modified SAC+HER method. Original and clearance
grasp use regularized SAC+HER; compact grasp uses BC. Reach remains the frozen
milestone SAC policy.

| Object | Original grasp | Clearance grasp | Compact BC grasp | Learned-handover BC carry | Balanced SAC carry |
|---|---:|---:|---:|---:|---:|
| Foam14 |11/12|9/12|7/12|8/12|10/12|
| Banana5 |11/12|11/12|10/12|12/12|12/12|
| Tomato can2 |6/12|8/12|8/12|8/12|8/12|
| Apple7 |7/12|7/12|8/12|7/12|9/12|
| Mug10 |9/12|11/12|10/12|11/12|11/12|

The new 8k SAC run improves on its learned-handover BC control from 46/60 to
50/60 complete, and from 41/60 to 47/60 within the deadline including settling.
Five previously failing case IDs pass and one previously successful case
regresses. The same frozen grasp/reach checkpoints and declared execution
settings are used, but exact physical prefix equivalence has not been audited;
this is not a causal attribution to replay balancing alone. Its ten failures
are six carry grasp losses, two failed final holds, one grasp horizon and one
reach drop. All 60 attempts and 794,910 total physics ticks including settling
are retained in `runs/phase2_system_learned_balanced_008k01`.

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
on the dashboard as a **prior BC-carry result**, pending a recording of the new
50/60 candidate; the original first-milestone recording remains in its own run.
