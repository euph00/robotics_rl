# Grasp retention feedback

**The direct-feedback pilot improved its matched control but missed its promotion
gate: 35/40 retained transports versus 31/40, with 36 required.** The trial is
closed. No full-system validation, new held-out test or reserve evaluation followed.
The selected system remains unchanged at 69/100 complete and 53/100 within the
inclusive 15,000-tick deadline on its last unseen-maze test.

| Fixed TRAIN comparison | Control | Transport feedback |
|---|---:|---:|
| Retained transports |31/40|35/40|
| Prepared handovers |40/40|40/40|
| Carry grasp losses |9|5|

There were seven candidate-only successes and three control-only successes.
Foam, banana, can, apple and mug scored 5/8, 8/8, 5/8, 5/8 and 8/8 in the
control; feedback scored 6/8, 8/8, 7/8, 6/8 and 8/8. These are planned-transport
TRAIN diagnostics after learned pickup, not full-task or unseen-maze results.

The improvement does not establish an advance over the starting grasp policy.
On the 20 cases overlapping the earlier qualification trial, the original `fa7`
actor retained 18/20, the new continuation control 15/20, and feedback 18/20.
Feedback exchanged two original successes for two wins. Shared source hashes,
settings, map seeds and planned approaches match; recording harnesses differ.
This historical comparison suggests recovery of continuation damage, rather
than a demonstrated net gain over the original actor.

Both branches began from the same original actor and received four fixed rounds
of 3,000 native pickup decisions plus 120 fresh stochastic original-start
transport attempts. Both used unchanged SAC+HER with demonstration regularization
and the same extra BC minibatch; only one included the actual transport outcome
in its extra policy-gradient step. Each frozen collection was consumed once.
The two first-round collections matched exactly before treatment. All four
single-step proposals per arm passed the fixed change limits; critics,
temperature, SAC optimizer states and native update counters were untouched by
these auxiliary steps. No old trajectories entered the score update or HER.

Per arm, new training totaled 12,000 pickup decisions, 2,000 SAC updates, 2,000
separate critic-warmup updates and four auxiliary updates. An independent audit
reconciled all 960 training suffixes and 80 endpoint attempts, original rewards
and ending flags, checkpoint identities and **14,013,305 new physics ticks**
including setup and settling. The pilot took 43.0 minutes. No retries, checkpoint
selection or threshold changes followed its result.

![Matched TRAIN retention comparison](../runs/phase2_grasp_retention_pg_compare01/artifacts/retention_comparison.png)

[Frozen protocol](../runs/phase2_grasp_retention_pg_plan01/artifacts/proposal.md) ·
[Execution](../runs/phase2_grasp_retention_pg_plan01/artifacts/execution_v1.json) ·
[All paired outcomes](../runs/phase2_grasp_retention_pg_compare01/artifacts/summary.json) ·
[Independent result audit](../runs/phase2_grasp_retention_pg_compare01/artifacts/independent_result_audit_scene_bank.json).

## Earlier prediction screen

The preceding fixed logistic predictor also failed its feasibility gate. Across
118 real handovers from eight TRAIN layouts, its case-weighted Brier score was
0.23460 versus 0.19458 for object prevalence, and within-case ranking was 0.52
(required 0.65). All eight leave-one-layout-out fits converged and were independently
reproduced. Fitting took 0.62 seconds, excluding preparation and auditing; there
were no policy updates or physics interactions. No feature or hyperparameter
search followed. This negative small linear screen does not establish fundamental
unobservability. The proposed auxiliary action-value critic did not proceed.

[Prediction screen](../runs/phase2_grasp_retention_predictor01/artifacts/report.md) ·
[Prediction audit](../runs/phase2_grasp_retention_predictor01/artifacts/independent_software_audit.json).
