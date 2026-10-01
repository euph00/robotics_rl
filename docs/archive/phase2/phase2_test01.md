# First unseen-maze test

**78/100 complete tasks; the 80/100 milestone was not met.**
63/100 finished within 15,000 physics ticks including settling and the final hold
(67/100 when settling is excluded). These are the first frozen-candidate results,
with one original-start attempt per case and every failure retained.

| Original object | Complete /20 | Within 15k including settling /20 |
|---|---:|---:|
| Foam brick 14 |17|14|
| Banana 5 |18|14|
| Tomato can 2 |16|14|
| Apple 7 |11|8|
| Mug 10 |16|13|
| Total |78/100|63/100|

The 100 cases cross five supplied objects with 20 genuinely unseen maze layouts,
disjoint from 192 TRAIN and 12 validation layouts. The final reserve remains
untouched. The candidate was selected for its 51/60 complete validation tasks;
all selection and checkpoint bytes were frozen before the first test action.

The system has three learned skills. Grasp and carry use BC-initialized SAC+HER
with continuing demonstration regularization; reach uses ordinary SAC+HER.
Planning supplies approach, carry posture and IK. No navigation route oracle,
inter-skill reset, teacher correction, object attachment or changed gripper
physics is used.

| Selected skill | Online decisions | SAC gradient updates | Continuing BC weight |
|---|---:|---:|---:|
| Grasp |12,000|2,000|200|
| Carry |40,000|5,000|100|
| Reach |15,000|13,996|0|

There were 14 carry grasp losses, six carry horizons, one planned-posture loss
and one reach drop. All 78 successful cases passed native 10 cm goal acceptance
and the mandatory 60-call retained hold. All 100 records, source/checkpoint hashes,
stage/controller counters and timing ledgers passed independent auditing.
Total cost: 291,238 native calls, 1,456,190 action ticks and 48,000 settling ticks,
or **1,504,190 physics ticks**. Successful final holds contribute 4,680 of those
native calls; they are already included, not added twice.

The longer same-data control and broader-grasp branch did not beat the selected
candidate on validation. [Comparisons](phase2_diverse_carry_preparation.md).
An existing [TRAIN reward audit](phase2_carry_return_audit.md) identifies a possible
speed–reliability incentive mismatch, without establishing the cause of failures.
The subsequent [controlled discount ablation](phase2_discount_trial.md) did not
improve validation. A matched navigation-network investigation is ongoing;
physics and evaluation criteria remain fixed.

This first test is preserved as a historical result. Because its outcome now
informs further development, it is retired from future held-out claims and its
trajectories will not enter training. A fresh confirmation bank has been frozen
before revised training/selection; the final-study reserve stays sealed. No
repeated attempt on this bank will be presented as a new independent test.

[Selected candidate](../configs/phase2_candidate.json) ·
[Frozen evaluation plan](../runs/phase2_milestone_test_plan01/artifacts/evaluation_plan.json) ·
[All outcomes](../runs/phase2_system_milestone_test01/artifacts/summary.json) ·
[Independent audit](../runs/phase2_system_milestone_test01/artifacts/independent_result_audit.json) ·
[Outcome matrix](../runs/phase2_system_milestone_test01/artifacts/validation_matrix.png).
