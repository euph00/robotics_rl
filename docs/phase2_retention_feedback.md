# Retention feedback investigation

**The fixed prediction screen failed; an auxiliary retention critic is not proceeding.**
A small classifier could not reliably distinguish grasps that would survive
subsequent transport from those that would fail.

| Case-weighted measure | Logistic model | Comparison / gate |
|---|---:|---:|
| Brier score (lower is better) |0.23460|Object prevalence: 0.19458|
| Brier improvement |−20.57%|Required ≥10%|
| Within-case ranking |0.52|Required ≥0.65|

The cohort contains 118 actual successful grasp handovers: 88 subsequent retained
transports and 30 downstream failures. Two grasp-horizon failures remain recorded
as exclusions. The 40 canonical cases span eight TRAIN layouts; variants from a
case always remain together. One fixed L2 logistic configuration used the
existing 70 measured/derived grasp features, with scaling learned inside each
training fold. Eight leave-one-layout-out fits were compared with global and
object-specific training-fold prevalence. Within-case ranking is averaged over
25 discordant cases; its 50 dependent pairs are not treated as independent cases.

All fits converged, but the classifier lost to object prevalence on seven of
eight excluded layouts. Independent checks reproduced input identities, scaling,
case weights, coefficients and predictions. The eight fits took 0.62 seconds, excluding preparation and auditing;
there were no policy updates, policy inference or physics interactions. No
hyperparameter, feature or cohort search followed. This small negative linear
screen does not establish fundamental unobservability; it does not justify
optimizing an action-value critic from these data either.

The next experiment adds direct feedback from actual transport outcomes alongside
unchanged SAC+HER. Synthetic checks and independent implementation review passed. The two matched
branches launched at 22:31:52 UTC on 28 September 2026; their fixed endpoint
comparison follows only if both finish all four rounds. No improved performance
has been established yet.

Two matched arms start from the original grasp checkpoint. Each receives four
rounds of 3,000 native pickup decisions followed by 120 fresh stochastic
original-start transport attempts on TRAIN layouts. Both get the same extra
imitation step; only one also uses the measured transport outcome to update its
grasp policy. Each collection uses a frozen actor and is consumed once. A strict
change limit accepts or restores the single proposed update. Native HER rewards
and ending flags remain unchanged; all preparation and transport costs count.

The fixed endpoints will face 40 paired TRAIN cases with deterministic actions.
The candidate must retain at least 36/40 objects, improve on its matched control,
and avoid new preparation failures. A passing result can authorize one full
60-case development evaluation, requiring at least 57 complete tasks and 38
within the inclusive tick limit. The physical pilot is capped at two hours.
These are development checks; neither retired test bank nor the final-study
reserve will be used. High variance and limited useful action variation remain
reasons this experiment could fail.

[Exact prospective protocol](../runs/phase2_grasp_retention_pg_plan01/artifacts/proposal.md) ·
[Independent implementation review](../runs/phase2_grasp_retention_pg_plan01/artifacts/independent_implementation_review_scene_bank.json) ·
[Frozen execution](../runs/phase2_grasp_retention_pg_plan01/artifacts/execution_v1.json).

[Frozen screen](../runs/phase2_grasp_retention_predictor_plan01/artifacts/plan_v2.json) ·
[Results and all folds](../runs/phase2_grasp_retention_predictor01/artifacts/report.md) ·
[Software audit](../runs/phase2_grasp_retention_predictor01/artifacts/independent_software_audit.json).
