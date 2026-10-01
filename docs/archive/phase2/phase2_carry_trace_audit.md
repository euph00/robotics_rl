# Carry hesitation, demonstration coverage and SAC gradient audit

The BC policy's long validation_002 episode is a route-choice problem. It succeeds after887 decisions but spends548 of them (decisions69–616) moving up and down the narrow strip x=1.500–1.539m, y=2.03–4.37m, before the maze entrance at y=1.5m. It reverses vertical direction25 times and travels34.33m in that strip. It first enters the gate cell at decision634, then finishes in253 further decisions over18.14m.

Only10 of886 measured moves are below1cm, and only one is below1mm. Median movement is7.11cm/decision, compared with7.18cm in the other five BC cases. The full trace travels58.18m. Simply increasing adapter speed would not address the repeated wrong turn. Saved traces contain post-decision XY only, so the first displacement is unavailable.

The reproducible `scripts/phase2_carry_trace_audit.py` produces `runs/phase2_carry_trace_audit01/artifacts/trace_summary.json` and `validation_002_trace.png`. These diagnostics read existing validation traces; they introduce no validation transitions into training.

## Training coverage

The current dataset contains352 episodes,77,238 transitions and64 training layouts.128 episodes were requested as full routes, while171 actually use the full exit goal because curriculum resets sometimes choose full routes. Actual full-exit-goal transitions total64,159: **83.1% of the dataset**. The central issue is not a low overall fraction of full-goal examples.

Only1,379 transitions (1.8%) are near the entrance turn with the original exit goal. More specifically, there are **zero** full-goal states beyond an outer entrance by>0.5m while within0.2m of x=1.5m. This is the class of states visited by the validation oscillation. The124 overshoot examples elsewhere all concern entrances at y=±0.5m, where table-avoidance paths can legitimately go past the entrance before returning. The training layout `train_052` has the same entrance/exit rows as the problematic validation case and672 full-goal transitions, but zero overshoot-recovery examples.

Definitions and counts are in `scripts/phase2_carry_coverage_audit.py` and `runs/phase2_carry_trace_audit01/artifacts/demo_coverage.json`. This supports a training-data coverage hypothesis; it does not prove the network's internal cause. The current CNN already has a31-cell receptive field covering the12×22 map, so a lack of nominal receptive field is not established.

The next bounded intervention is physical recovery data on TRAIN layouts: drive the retained payload to valid off-route positions before the entrance, then record teacher recovery toward the original fixed exit goal. Include starts on both sides of the gate where geometry permits, and count preparation and failed attempts. Preserve the original goal throughout each recorded episode. Refine on a mixture of existing route/hold examples and recovery examples, so corrections do not displace the useful route policy. An inference-time A* route or hidden-test collection is unnecessary.

If corrected data remains insufficient, a learned iterative map module is a separate architecture ablation. Value Iteration Networks demonstrate differentiable planning structure learned from data, but adopting one changes the architecture and requires matched baseline treatment; it does not justify inserting a ground-truth route at inference. [Primary paper](https://arxiv.org/abs/1602.02867)

## Reward, timeout and HER interpretation

`GeneralCarryEnv` truncates after five consecutive goal-success decisions. SB3's timeout handling then bootstraps, including for HER samples. This is coherent with a continuing goal-reaching-and-holding training objective; it is not a verified terminal bug. Actual physical failure remains absorbing and goal independent. Relabeling a goal does not retain an original-goal success terminal. `previous_achieved_goal` is stored so both potential terms are recomputed under the relabeled goal.

The distinction matters: the evaluation stops at first retained arrival, while the training critic estimates continuing value after arrival. Five held steps give limited support for that continuation. Replacing truncation with an original-goal success terminal without recomputing terminals for HER would create a different error. Finite-horizon tasks would additionally need remaining time as input. [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378), [SB3 HER implementation](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/her/her_replay_buffer.html)

Future HER selects goals from states actually reached later in the same episode. Failed or oscillating collection therefore supplies many nearby reachable goals, not evidence of an unseen exit route. With50%HER and a short-segment curriculum, monitor original exit-goal support separately from aggregate training success. A fall in curriculum success alone is not a controlled fixed-case performance result. Continuing to collect goal-holding states would be a narrowly motivated future ablation, after the ongoing actor-regularization comparison.

## Measured gradient scale

Eight independent batches from each saved zero-online initializer were analyzed with freshly seeded physical demonstration replay. No optimizer steps or physical interactions occurred. The carry probe used256 examples/batch and the intended50/35/15 demo strata. Grasp used400 examples/batch and balanced object×phase strata from the prior dataset.

| Initializer | RL gradient norm | Raw BC gradient norm | Gradient cosine | Weighted BC/RL, beta10 | Weighted BC/RL, beta100 |
|---|---:|---:|---:|---:|---:|
| Carry after2k critic warmup |0.629|0.182|−0.602|2.94|29.4|
| Grasp after2k critic warmup |17.58|0.113|−0.674|0.064|0.643|

The carry evidence supports trying beta10 before assuming100 is needed. Grasp's old initializer suggests100–200 is more appropriate for a bounded pilot; the new refined actor/fresh critic may change that balance. These are raw parameter gradients before Adam, not a predicted success rate. At exact imitation agreement the BC restoring gradient is zero, so even a large fixed coefficient permits some deviation.

Code: `scripts/phase2_regularization_gradient_probe.py`. Artifacts: `gradient_balance.json`, `grasp_gradient_balance.json` in the same audit directory. The first unsuccessful grasp probe exited on a missing teacher-action key before computing any gradients; grasp correctly uses executed demonstration actions, matching its existing BC labels. The rerun completed in3.85s on two CPU threads; carry completed in14.96s. The regularized method and fair SAC/TD3 comparison contract are documented in `docs/phase2_regularized_sac.md`.
