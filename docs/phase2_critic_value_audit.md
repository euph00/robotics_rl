# Carry critic value audit

This is a TRAIN-only diagnostic, with no physical interactions or changes to active experiments. Its purpose is to determine whether the initialized critics provide a useful learning signal before further actor updates.

The probe uses one successful deterministic, original-exit-goal demonstration from each of the 64 training mazes. Its 1,152 sampled states include all route stages and final goal holds. Actual executed actions equal the expert labels. Source datasets and checkpoint hashes are recorded in `runs/phase2_critic_return_audit01/artifacts/result.json`.

## What the saved models show

| Measurement | Initialized, 2,000 critic updates | After 4,000 online decisions / 499 SAC updates |
|---|---:|---:|
| Median expert-action Q, states with >200 decisions remaining | −16.51 | −19.95 |
| Median discounted demonstrated return, same states | −117.10 | −117.10 |
| Median paired Q minus demonstrated return | +100.58 | +97.18 |
| Actor action ranked above expert, all sampled states | 32.3% | 59.7% |
| Mean physical actor-action error from expert | 0.0161 | 0.0303 |
| Median disagreement between twin critics | 0.0154 | 0.0196 |

The critics agree closely while assigning much less negative values than the demonstrated long-route returns. Online SAC makes only a small correction to that scale, while its action rankings increasingly favor the moving actor. These measurements support a concern about incomplete value propagation. They do **not** establish that this mechanism alone causes failures on unseen mazes.

Local action probes also reveal imperfect interpolation. The best of four ±0.1 action perturbations gains about 0.024 Q on average. In 35 nontrivial cases, scaling the raw expert action produces the same unit-normalized physical control within 1e-5, yet all scaled versions receive higher Q (median gain 0.036 initially, 0.034 after 4k). The critic consumes raw actions; the environment subsequently projects them onto the unit disk. These are measured inconsistencies, not proof that a probed alternative action would fail physically.

## Return and entropy qualifications

A demonstrated hard return is a **reference from a different continuation policy**, not the exact soft-Q target of the checkpoint's stochastic actor. The hard reference includes every recorded reward and a conditional stationary goal-hold tail. Success ends recording through a timeout, not an absorbing terminal state; HER replay therefore bootstraps at success. Assuming continued retention and an unchanged goal distance `d`, the shaped hard tail is `2d`, discounted back to the sampled state. Here that correction is at most 0.000019.

This continuing convention is intentional. At final goal states the initialized median stop Q is −3.27. A conditional stationary-state entropy calculation gives approximately −3.14, explaining most of that negative value. Calling negative goal Q a terminal-handling bug would be incorrect.

For constant alpha, the ideal squashed policy's total **positive** future entropy contribution is bounded by `alpha * gamma * log(4) / (1-gamma)`, approximately 1.38. SB3's numerical Jacobian epsilon slightly changes its implemented log density. A more conservative implemented bound uses the two-dimensional Gaussian entropy with `log_std <= 2`, plus `2 log(1+1e-6)`; its total allowance is 6.80 initially and 6.89 after 4k. There is no corresponding global finite negative-entropy bound. Neither positive allowance explains a gap near 100, but differing future actions still prevent treating the demonstration reference as exact checkpoint-policy Q.

The recorded Bellman residual uses next-state deterministic mean-action Q plus sampled entropy. It is explicitly approximate because `E[Q(s',a')]` need not equal `Q(s',E[a'])`.

The SAC objective and soft policy evaluation follow [Haarnoja et al. (2018)](https://proceedings.mlr.press/v80/haarnoja18b.html). HER reward relabeling and timeout bootstrapping were checked against the installed implementation and the [SB3 HER documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

## Bounded warmup comparison

The actual original warmup tau is **0.01**, inherited from the saved model, and the online runner also uses 0.01. An earlier informal suggestion of 0.005 was incorrect. The helper does not override tau.

A simple constant-reward, exact-fitting scalar illustration has target contraction factor `1 - tau*(1-gamma)`. With gamma 0.995, 2,000 updates propagate about 9.5%, 63.2%, or 100.0% of the fixed-point value for tau 0.01, 0.1, or 1.0 respectively. Neural approximation, changing replay goals, and changing state values make this only an illustration.

The authorized experiment uses disposable copies of the same fresh pre-warmup BC checkpoint, the same saved HER replay, identical RNG and sampled tensor hashes, and 2,000 critic-only updates per arm. Only warmup tau differs (0.01 / 0.1 / 1.0). Batch size is 256 and critic learning rate is 0.0003. Actor and temperature remain frozen and their hashes are checked. Checkpoints at 500 / 1,000 / 2,000 updates preserve online tau 0.01. The physics-free schema only recomputes rewards; its reset and step methods raise errors.

The completed CUDA comparison is recorded in `runs/phase2_critic_warmup_probe02/artifacts/result.json` (source snapshots in the adjacent `source` directory). All 6,000 critic updates completed in 204 seconds, with no actor updates, temperature updates, or physical interactions. The replay contains 174,824 valid transitions, with 50% HER in each sampled batch. Every checkpoint's sampled-tensor hash matches across arms; actor/temperature hashes remain unchanged, and original checkpoint/replay file hashes are unchanged. CUDA used cuDNN benchmarking and high float32 matmul precision; numerical execution is not guaranteed bitwise reproducible.

| Warmup tau | Median long-route Q | Median Q minus hard reference | Actor ranked above expert | Stop ranked above expert | Median goal-hold stop Q |
|---|---:|---:|---:|---:|---:|
| 0.01 | −16.26 | +100.79 | 28.0% | 9.1% | −3.55 |
| 0.1 | −113.02 | +4.36 | 14.0% | 6.2% | −5.20 |
| 1.0 | −135.63 | −18.18 | 18.9% | 10.9% | −46.23 |

All values above are after 2,000 critic updates. “Long route” means >200 demonstrated decisions remaining; ranking percentages cover all 1,152 sampled states. For tau 0.1, the 10th–90th percentile long-route reference gap is −10.36 to +18.24; the small median does not imply every state is calibrated.

Tau 1.0 remained finite but was unstable. Goal-hold median Q changed from −130.71 at 500 updates to −42.98 at 1,000 and −46.23 at 2,000. At 1,000 updates it ranked stopping above the expert on 88.0% of states and the actor above the expert on 79.5%. Its final low critic loss alone would conceal this behavior. Tau 0.1 gives a substantially better balance between route-value propagation and stable action ranking in this bounded comparison.

**Recommended next intervention:** use 2,000 fixed-actor critic initialization updates with **warmup tau 0.1**, then restore **online tau 0.01**. Compare with the existing branch using the same chosen actor, replay/data, and online method. This is an initialization hyperparameter change, not evidence that SAC generalization is solved. The held-out navigation test is still necessary. Do not use tau 1.0 based solely on its faster propagation.

The earlier CPU throughput pilot is retained as `runs/phase2_critic_warmup_probe01/artifacts/result.json`, explicitly stopped and incomplete after its saved 500-update checkpoint. Its exact interrupted update count is unknown; at least 500 critic updates were spent. Its outputs were not combined with the CUDA pairing. The existing geometric-data training runs were not changed.

Additional read-only checks passed: the stationary shaped-goal tail reproduces its analytic continuing value; diagnostic probes preserve CPU and CUDA RNG streams; and vectorized Q evaluation matches SB3's ordinary critic forward calculation. No new physics was used for these checks.

![Offline critic comparison](../runs/phase2_critic_warmup_probe02/artifacts/critic_comparison.png)


## Matched physical continuation

`phase2_carry_geometric10_fastwarm01` starts from the exact `phase2_carry_geometric100/checkpoints/bc_initialized.zip` actor and the same geometric demonstration dataset as `phase2_carry_geometric10`. It retains beta 10, seed 7401, 64 training layouts, eight workers, 2,000 fresh-critic initialization updates, 12,000 online decisions, and the same 4,000-decision evaluation/checkpoint cadence. The only training hyperparameter changed is initialization tau 0.01 → 0.1. The actor receives no additional BC updates. The frozen 64-layout bank remains unchanged; the separate new 192-layout bank is not used here.

`critic_only_warmup(..., warmup_tau=...)` uses a local target-update rate and never mutates `model.tau`. The carry CLI exposes `--critic-warmup-tau` (default 0.01), records it in config/logs/initialization metadata, and retains online tau 0.01. Resume inherits and checks this historical initialization setting rather than silently changing its recorded provenance. All 17 initialization tests pass, including exact actor/temperature preservation, the actual target update fraction, unchanged online tau on normal completion and interruption, and invalid-rate rejection before any update.

The run's `artifacts/controlled_comparison.json` records matching zero-online actor, fresh critic, demonstration, and scene-bank hashes. Compare exact saved callback checkpoints and `_n_updates` at 4k: a final callback and a post-training checkpoint at the same decision count may differ by one actor/critic update.
