# Phase 2: why the BC-refined carry policy degraded under SAC

Audit date: 2026-09-28. This is a diagnosis of the **selected 14,000-update BC carry actor** and its failed `sac_carry_bc_finetune02` continuation. It supersedes speculation based only on the earlier 7,000-update actor. No simulator interaction or new navigation success was measured during this audit.

## Findings

**The effective learning rate and saved HER rewards are correct. The stochastic actor initialization and training distribution are poorly matched to the demonstrated full-route behavior, and inherited critic gradients move that behavior away from its demonstrations.** These are measured protocol weaknesses, not evidence that SAC is fundamentally unsuitable.

### 1. No effective learning-rate loading bug

The continuation requested `1e-6`. Immediately after loading, its actor optimizer has that rate while saved critic/temperature optimizer metadata still says `3e-4`. This looks suspicious but is harmless: installed SB3 2.7.0 calls `_update_learning_rate` for all three optimizers **before** the first gradient. An executable scheduling probe yields `1e-6` for all three; both the 90,128 and 92,000 checkpoints retain that rate. The current `learning_rate` override rebuilds the schedule during `SAC.load`.

### 2. BC preserved the std head's weights, not the stochastic policy

The BC stage changes the actor's shared latent network and masks arm/payload inputs while leaving the Gaussian `log_std` head weights unchanged. That head reads the changed latent features. On 2,048 evenly spaced records from the selected actor's eight full-route demonstrations:

| Quantity before online updates | Measured value |
|---|---:|
| Median pre-tanh sigma | 1.465 |
| 95th-percentile sigma | 7.389 = exp(2), the SB3 clamp ceiling |
| Sigma components at ceiling | 25.6% |
| Median stochastic target displacement from deterministic action, after the carry adapter's norm bound | 30.1 mm |
| 95th-percentile target displacement | 46.8 mm |

These are command differences, not measured physical displacement. The old adapter permits 45 mm per decision, so stochastic sampling can request a substantially different direction from the demonstrated one. A deterministic successful video did not establish a suitable stochastic behavior policy for collection or critic targets.

The earlier 7k-BC entropy audit cannot be used to dismiss this problem in the final 14k policy. Conversely, high pre-tanh sigma is not itself a measurement of differential entropy after tanh; saturation and state-dependent means also matter.

### 3. Small SAC updates measurably change the successful deterministic behavior

| Saved checkpoint | Full-route teacher-action MSE | Mean deterministic action change from BC |
|---|---:|---:|
| BC, zero new online decisions | 0.00688 | 0 |
| +128 decisions | 0.01089 | 0.0547 |
| +2,000 decisions | 0.16791 | 0.4601 |

Actions here are the raw normalized two-dimensional outputs, before the adapter's norm bound. The 128-decision checkpoint contains **120** completed SAC gradient iterations: the evaluation callback runs before the final vector rollout's eight updates. This is a counting detail, not a training error.

On full-route records, the inherited critic contribution to the actor parameter-gradient norm is 178.1 versus 4.82 for the alpha-weighted entropy contribution. The SAC gradient has cosine similarity **−0.243** with the teacher-MSE gradient; a small ordinary gradient step therefore tends locally to worsen imitation error. On the actual mixed HER replay sample, the corresponding cosine is **−0.533**, with Q/entropy gradient norms 36.63/4.52. Only 40.1% of full-route Q action gradients point toward the recorded teacher correction.

Teacher actions are not proved optimal, so disagreement alone does not prove an erroneous Q function. Combined with the measured physical failures and rapid action drift, however, this directly supports a critic/policy/distribution mismatch. Reducing alpha alone cannot remove a Q-dominated update.

### 4. The BC goal distribution and resumed SAC distribution differ sharply

BC used only eight correlated full-route episodes, 5,259 rows; its MSE on the dataset's 4,034 other-goal rows is **0.450**. The resumed training restores all eight workers at curriculum level 4: **5 m route segments**, not full journeys.

The initial replay has 99,240 eligible transition slots. In a sample of 20,000 real records, only **4.97%** have the full-route desired goal. HER replaces goals in 80% of the batch; in a 2,048-row mixed batch, **0.73%** still have that exact exit goal. The median requested distance is 0.293 m before HER and 0.118 m afterward. Thus resumed SAC immediately trains mainly on goals outside the narrow distribution on which BC was successful. Shared weights couple these changes back into full-route behavior.

This is not a defect in HER: the selected initialization simply is not a broadly goal-conditioned actor. New multi-maze/full-goal and segment-goal data need adequate coverage before the same transfer is attempted again.

### 5. No sampled reward or physical-terminal inconsistency found

All 20,000 sampled real rewards exactly recompute from saved achieved/desired goals and saved contact/failure information. All sampled physical-failure flags agree with replay's non-timeout terminal flags. The replay preserves `copy_info_dict=True`. Goal-dependent success does not cause training termination; physical failures do, and horizons truncate. Derived desired-goal, relative-goal and periodic features are recomputed after HER relabeling. Demonstration replay uses executed actions rather than offline teacher-action labels.

These checks narrow the diagnosis; they are not an exhaustive proof of all reward or state semantics.

## Bounded offline probes

Each probe performs 128 ordinary SAC actor/critic iterations on **unchanged saved replay**, without generating physical transitions. Small-alpha variants use fixed alpha 0.001. Std-reset variants zero the log_std weights and set bias −3 (sigma 0.0498). Warmup variants perform 512 prior critic-only iterations at `3e-4`, then joint updates at `1e-6`. Other settings remain inherited.

| Probe | Full-route MSE after 128 updates | Mean action drift |
|---|---:|---:|
| Ordinary saved initialization | 0.01162 | 0.0599 |
| Small alpha | 0.01193 | 0.0625 |
| Small alpha + inherited-critic warmup | 0.00985 | 0.0420 |
| Small alpha + std reset | 0.01299 | 0.0717 |
| Small alpha + std reset + inherited-critic warmup | 0.01075 | 0.0516 |
| Small alpha + std reset + fresh-critic warmup | 0.01769 | 0.1054 |

**None of these probes proves a fix.** Std reset repairs the initial noise distribution but does not by itself preserve the deterministic mean. A fresh critic with 512 updates is not sufficiently justified by this evidence; it can still provide harmful gradients. Do not replace a diagnosed inherited critic with an unvalidated random critic and assume transfer is solved.

## Recommended next experiments

1. Train the new spatial actor on diverse maze/start/goal data and actual corrective states. Include both final-task goals and the segment-goal distribution that online SAC+HER will query. Preserve a separate development validation bank and untouched final test bank.
2. After BC, initialize a controlled Gaussian std explicitly. Inspect sampled actions on real task states, including turns and near-obstacle states. If auto alpha is used, choose its target entropy based on measured **squashed-action** noise, not the unchanged-head claim or just the unsquashed Gaussian formula. Fixed small alpha is also an ordinary SAC option to ablate.
3. Fit or align critics while the initialized actor is frozen, preferably with fresh, representative data. Compare critic-value scale, Bellman errors, Q action rankings, and early actor drift before releasing ordinary SAC updates. No auxiliary online imitation loss is required by this initialization.
4. Gate continuation at zero, 128, 512 and 2,000 updates on fixed validation cases, including genuine physical success and action-noise/critic metrics. A learning-rate reduction slows bad gradients; it does not make their direction useful.
5. If replay coverage remains weak, explicitly balance demonstration and online data while retaining executed actions and valid HER boundaries. Layer normalization in fresh critics is a reasonable bounded architecture experiment if value extrapolation is observed. Do not import a large new algorithm solely because transfer is difficult.

The supporting literature is consistent with these priorities, but does not establish that a particular change will solve this environment. [RLPD](https://arxiv.org/html/2302.02948v3) studies ordinary off-policy learning with prior data, demonstrating the importance of balanced sampling and critic normalization and the sensitivity of entropy-related choices. Its full method includes additional changes; we are not claiming to implement it. [Cal-QL](https://arxiv.org/abs/2303.05479) motivates calibrated value initialization for avoiding early fine-tuning degradation. [AWAC](https://arxiv.org/abs/2006.09359) studies this broader transfer difficulty, but uses a different policy update and is not the selected SAC baseline. The installed [SB3 SAC source](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/sac/sac.html) and [HER documentation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/her.html) determine the implementation semantics checked here.

## Deliverables and reproduction

- `scripts/phase2_sac_audit.py`: CPU-only saved-checkpoint/replay diagnosis and six bounded in-memory update probes.
- `runs/phase2_sac_audit01/artifacts/result.json`: checkpoint hashes, sample sizes, scalar results, optimizer probe, gradients, replay checks, and offline probes.
- `runs/phase2_sac_audit01/artifacts/accounting.json`: zero physical interactions, two CPU threads, 83.14 seconds across the preliminary and expanded audits; all modified probe weights discarded.
- `pnr/sac_initialization.py`: `calibrate_actor_std` and `critic_only_warmup`. Helpers modify only the intended initialization components; they are **not yet evidence of improved navigation**.
- `tests/test_sac_initialization.py`: 9 passing compute-only tests using real SB3 SAC+HER. Verify constant std across changed backbone inputs, preservation of deterministic means/critics during calibration, actor/feature-buffer/temperature preservation during warmup, critic/target updates, separate accounting, LR/mode restoration on interruption, and rejection of shared tensors or invalid std.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/phase2_sac_audit.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_sac_initialization.py
```

Warmup usage, after populating an appropriate replay and before online learning:

```python
from pnr.sac_initialization import calibrate_actor_std, critic_only_warmup
std_report = calibrate_actor_std(model, log_std=-3.0)
warmup_report = critic_only_warmup(
    model, gradient_steps=2000, learning_rate=3e-4,
    log_every=100, on_log=record_initialization_metric,
)
# Save both reports, then evaluate at zero online steps before model.learn(...).
```

The helpers do not choose a target entropy, replace critic weights, rebalance replay, or silently enable auxiliary imitation. Those choices belong to the declared training protocol. Critic warmup preserves the actor in evaluation mode, freezes temperature, uses the current SAC Bellman target including alpha/gamma, rejects shared learned actor/critic tensors, and records critic-only updates separately from complete online SAC iterations.
