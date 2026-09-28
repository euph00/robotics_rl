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
case weights, coefficients and predictions. Computation took 0.62 CPU seconds;
there were no policy updates, policy inference or physics interactions. No
hyperparameter, feature or cohort search followed. This small negative linear
screen does not establish fundamental unobservability; it does not justify
optimizing an action-value critic from these data either.

The next proposal considers direct on-policy credit from actually measured
transport outcomes, without a learned retention predictor. It would remain
separate from native SAC+HER and retain actual observations, native ending flags
and all delayed physical costs. Its variance, exploration/deployment mismatch
and feasible development gate require review before any policy training.

[Frozen screen](../runs/phase2_grasp_retention_predictor_plan01/artifacts/plan_v2.json) ·
[Results and all folds](../runs/phase2_grasp_retention_predictor01/artifacts/report.md) ·
[Software audit](../runs/phase2_grasp_retention_predictor01/artifacts/independent_software_audit.json).
