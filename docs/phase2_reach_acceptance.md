# Full-system reach acceptance audit

The source task requires held-object distance **below 0.10 m** in 3-D. This is the contract already recorded in `docs/project_spec.md` and implemented by `PickNavReachEnv.evaluate()`. The full-system runner accidentally used the reach training wrapper's stricter **0.04 m** target for its handover. Training, rewards, policy observations, actions, and checkpoints remain unchanged.

The saved validation evidence identifies the same can case, `validation_003_object_02`, in both runs:

| Saved run | Full-task result, unchanged | Native completion first observed | Final distance | Reach decisions |
| --- | --- | --- | --- | --- |
| `phase2_system_bc_ik24_01` | 28/60 | decision 6; physics tick 13,740 | 0.049844 m | 150 |
| `phase2_system_carryreg01` | 26/60 | decision 8; physics tick 13,815 | 0.049675 m | 150 |

Each run has exactly one failed case whose final native `complete_task_success` is true. Both have zero rejected actions, zero drops, zero wall contacts, and bilateral contact at every saved reach point. The final native distance and fixed-bind-frame goal distance agree within 0.00000014 m. Successful can reaches in the same runs took 8–10 decisions. The failed policy instead settled about 5 cm from the target: motion extent over the last 60 calls was 0.000566 m and 0.0000116 m respectively.

This is an overly strict full-system acceptance check combined with a residual reach-policy error. There is no evidence of a goal-frame bug. Reach directly commands base XY and torso increments through the existing collision check; its action path does not call IK. The zero rejection counts also rule out endpoint rejection as the cause in these cases.

`LearnedSystem` now defaults to native 10 cm acceptance. Handover requires current native complete-task success, distance below 10 cm, bilateral contact, no drop, and no physical termination. The existing final **60-call retained hold** and subsequent native success check remain mandatory. A hold failure or movement outside the goal still fails the full task. This is a task-contract correction, not a newly trained policy.

Phase-2 evaluation plan **v3** introduced a frozen `reach_tolerance_m` and archived execution source. Explicit `0.04` retains the old wrapper `is_success` predicate exactly. Verified v1/v2 plans expand to `0.04` when executed, with v1's original 0.20 m grasp lift preserved; their existing digests and historical files are not rewritten. New plans default to `0.10`, and frozen-plan overrides are rejected. Held-out source-hash checks remain in force. The generic milestone-1 entry point, `scripts/evaluate_system.py`, explicitly passes `0.04` so baseline reproduction also preserves its original acceptance.

The historical results are **not relabeled as successes**. Although native completion was observed, a final hold at the earlier handover was not executed. Its outcome and corrected deadline performance require a new physical evaluation. The minimal next validation is a new validation-only can case with the same policies and explicit 0.10 m acceptance, including the final hold; this audit ran no physics.

Reproducible read-only audit:

```sh
.venv/bin/python scripts/phase2_reach_acceptance_audit.py \
  --output runs/phase2_reach_acceptance_audit01/artifacts/result.json
```

The existing artifact includes source result/trajectory SHA-256 values, exact native completion timing, frame-error comparisons, and successful can examples. The script requires a fresh output path. Traces are sampled every five native calls; each reach decision repeats three calls. Exact per-call first-success fields are used for timing, so a saved trace point is not assumed to be a decision boundary.

Validation: `tests/test_reach_completion.py` and `tests/test_phase2_system_evaluation.py`: **28 passed**, using mocked physics and geometric scene-bank metadata only. Tests cover threshold boundaries, current native history, contact/drop/termination checks, final hold failure, final native recheck, legacy predicate preservation, and versioned plan validation. No physical rollouts or training updates were performed.

Plan **v4** additionally freezes `grasp_proposal_mode` (`legacy` or opt-in `clearance-v2`). Verified v1/v2/v3 plans retain legacy grasp proposals. The optional measured-geometry proposal does not change the native physics or the reach/retention acceptance rule.
