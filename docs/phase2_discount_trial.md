# Controlled carry discount trial

The predeclared gamma 0.999 branch was checked at 40k and final 80k online decisions on the same 60 validation cases as the historical gamma 0.995 control. Neither checkpoint improved the control’s complete-task or inclusive timed result. No promotion is proposed.

| Carry checkpoint | Complete task /60 | Within 15k including settling /60 | Within 15k action-only /60 |
|---|---:|---:|---:|
| Historical gamma .995, 40k | 51 | 47 | 50 |
| Gamma .999, 40k | 45 | 37 | 40 |
| Gamma .999, 80k | 44 | 39 | 42 |

Exact saved pre-carry traces and stage records match the historical control in **60/60** cases at 40k and **60/60** at 80k. The two gamma checkpoints match each other in **60/60**. All recorded historical prefixes match, within the logging limits below.
The logs sample every five native calls and omit action commands, velocities, contact forces and solver state; equality of saved prefixes does not establish hidden-state or unsampled-handover equality.

The initializer, 503,901-row TRAIN dataset, first eight TRAIN layouts × five objects, single deterministic learned preparation, eight workers, seed 7401, object-balanced HER, beta100 continuing demonstration regularization, actions and physics are unchanged. New critics receive 2,000 warmup updates with tau .1; BC performs zero updates. The disclosed scheduling differences are an 80k budget and isolated reference validation every 40k instead of 8k. The 40k callback checkpoint has 4,999 updates versus the historical final 40k control’s 5,000; this one-update difference is disclosed. Evaluator differences are reporting and the evaluation-bank loader only; physical execution functions/source hashes match the control.

The 40k planned source inventory omitted the already-existing shared `scripts/evaluate_system.py` helper. The evaluator archived it correctly; its bytes match both the historical control and current helper. All explicitly planned sources match. The post-evaluation bookkeeping assertion and recovery are retained; the completed 40k evaluation was not rerun and its frozen plan/results were not rewritten. The 80k plan includes the helper before execution. Accordingly, the 40k claim is verified matching planned sources plus an explicitly audited unchanged extra helper, not an unexplained exact-inventory claim.

Checkpoint ZIPs verify gamma .999 and update counts 4,999 /10,000. All 503,901 source rewards were verified at gamma .995; 502,432 imported real rewards were recomputed at .999 (502,391 numerically changed), with failure cost 1,000 and shaping 2; native target-environment HER recomputation and terminal/timeout semantics are retained. The original dataset is unchanged. Reward relabeling and this audit add zero physical interactions.

40k: 9 carry (lost_grasp); 1 grasp (decision_horizon); 5 carry (decision_horizon). Evaluation cost: 179,533 native calls, 897,665 action ticks + 28,800 settling = **926,465 ticks**. Carry SHA256 `78761f554944d8ad33527f8406275aa9d1acdaadda3970b6769797cc4f246a1f`.
80k: 9 carry (lost_grasp); 1 grasp (decision_horizon); 6 carry (decision_horizon). Evaluation cost: 178,940 native calls, 894,700 action ticks + 28,800 settling = **923,500 ticks**. Carry SHA256 `5fec5e907950c2bd8b174d7024a346067b6345d7d331e64b14ce41402e9ee2a6`.

The historical control had seven carry grip losses, one grasp horizon, and one reach drop. At 40k the gamma branch rescues one control failure but introduces seven regressions; at 80k it rescues two and introduces nine. Its nine carry grip losses at both checkpoints and five/six carry horizons provide no measured retention or completion benefit in this bounded trial. This is one training seed, not a general claim about higher discounts.

Per-case outcomes, per-object totals, exact costs, source/plan/checkpoint hashes and reward provenance are stored in each run’s `artifacts/independent_discount_audit.json`; detailed historical prefix comparisons are in `prefix_reproducibility_audit.json`. The 80k run also contains `prefix_vs_gamma040k_audit.json`. No held-out tests or rerolls were performed.
