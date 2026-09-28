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

The carry actor in both new runs has zero online SAC updates after its expanded
BC initialization. The grasp is SAC+HER with continuing demonstration-action
regularization, and reach is the frozen milestone SAC policy. This is not a
claim that all three new skills have successfully improved through online SAC.

| Object | Original grasp | Clearance grasp |
|---|---:|---:|
| Foam14 |11/12|9/12|
| Banana5 |11/12|11/12|
| Tomato can2 |6/12|8/12|
| Apple7 |7/12|7/12|
| Mug10 |9/12|11/12|

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

The first declared clearance case, foam14/validation000, was recorded rather
than selected after observing success. It completes at7.86cm final goal error,
with zero drops and14,165 total ticks including settling. Its MP4/WebM is featured
on the dashboard; the original first-milestone recording remains in its own run.
