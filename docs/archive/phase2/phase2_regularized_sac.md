# Explicit SAC+HER+demonstration regularization branch

`pnr/regularized_sac.py` implements a modified algorithm. It must be reported as **SAC+HER+demonstration regularization**, separately from the earlier BC-initialized ordinary SAC+HER runs. It does not establish a training success result by itself.

The actor minimizes

```
mean[alpha * log pi(a|s,g) - min(Q1,Q2)(s,g,a)]
  + beta * mean[(tanh(mu(s_demo,g_demo)) - a_demo)^2]
```

The first expectation uses the existing SAC/HER replay batch. The second uses an independent demonstration batch with its original goals and original normalized teacher action labels. MSE is averaged over examples **and action coordinates**. `beta=10` is a transparent initial setting, not a theoretically optimal weight. There is no Q-filter, adaptive weight, residual controller, extra critic objective, or new inference-time planner. Critic targets, HER sampling/rewards/dones, entropy-temperature updates, target-network updates, actor exploration and learning-rate schedules retain SB3 2.7 SAC behavior. The deterministic-action term does not directly constrain the Gaussian standard deviation; ordinary entropy optimization can still increase exploration.

Local motivation: in the grasp pilot, the preserved BC policy passed 9/10 fixed validation cases, then passed 8/10 after 256 decisions and 0/10 after 1,024 ordinary SAC decisions. This happened after constant initial sigma and fresh-critic warmup. The earlier carry audit also measured an RL actor gradient opposing the imitation gradient. An ongoing demonstration term directly tests whether preserving useful actions prevents this failure. It does not repair unsupported states, reward errors or an inaccurate critic, and excessive weight can prevent improvement beyond demonstrations.

Nair et al. combine a demonstration buffer, auxiliary actor cloning loss and HER; they also describe a Q-filter and demonstration-state resets. We use the auxiliary-loss idea, without treating their complete DDPG method as our implementation. Their critique of initialization followed by unconstrained RL is directly relevant to our observed regression. [Primary paper, sections IV and VII](https://arxiv.org/html/1709.10089v2)

Fujimoto and Gu show that an actor BC term can limit exploitation of inaccurate action values in offline TD3. This supports the mechanism, but their offline results do not establish that our online SAC branch will succeed. We do not implement TD3+BC's value-dependent scaling. [Primary paper](https://arxiv.org/html/2106.06860v2)

The unchanged SAC update equations and order are checked against the locally installed SB3 2.7.0 implementation. [Official source](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/sac/sac.html)

## Integration

```python
from pnr.regularized_sac import RegularizedSAC

model = RegularizedSAC(
    'MultiInputPolicy', env,
    # The existing SAC/HER architecture and configuration go here.
    bc_weight=10.0,
    demo_batch_size=256,
    gradient_diagnostics_interval=100,
)
manifest = model.set_demonstrations(
    observations=demo_observations,   # dict: raw original-goal batched arrays
    actions=teacher_actions,         # normalized [-1, 1], not native joint targets
    groups=object_phase_groups,      # dict: group name -> integer row indices
    seed=seed,
)
```

Groups are sampled equally, then rows uniformly within each group. `group_probabilities` optionally supplies explicit probabilities in dictionary insertion order. A group can deliberately omit or overlap rows; the persisted manifest reports unique coverage. For carry, supply the intended turn/straight/hold probabilities explicitly if preserving the existing 50/35/15 sampler. The helper never calls a teacher or the physical environment.

The class accepts NumPy or Torch arrays and copies them into owned CPU storage. `VecNormalize`, if used, transforms sampled raw demo observations with the current statistics. Actor and critic must have separate feature extractors. Std calibration and critic-only warmup remain separate explicit initialization operations; this class does not repeat or conceal them.

To start a fresh branch from a BC checkpoint, construct a fresh regularized model with the matching architecture and transfer only the saved actor weights. Record the ancestor checkpoint and demonstration costs. Do not silently inherit failed online critics or optimizer state. Archive this helper with the runner's source snapshot.

Save/load must use **`RegularizedSAC.load`**, not `SAC.load`. The ZIP saves the hyperparameters, demonstration/group digest, sampler RNG, optimizers and counters, but excludes the demonstration arrays. Reattach the same arrays/groups with `set_demonstrations(..., seed=None)` before continuing; that preserves the sampler RNG. Changed labels, ordering, groups or group probabilities fail digest validation. Inference works without the arrays. A change of data belongs to an explicitly recorded new branch.

`last_regularization_metrics` and `train/*` logging expose RL loss, raw and weighted BC loss, total actor loss, coefficient, cumulative regularized updates and demonstration examples sampled. Every configured interval additionally logs actor parameter-gradient norms, weighted BC/RL norm ratio and gradient cosine. These are raw parameter gradients before Adam; they are diagnostic, not an automatic weight controller. Diagnostics appear only on sampled update calls. They add two actor backward traversals at those intervals. Demonstration samples are compute/data reuse and are not new physical interactions.

## Verification and experiment boundary

Five CPU-only tests in `tests/test_regularized_sac.py` pass with actual SB3 SAC and HER: weight zero is bitwise ordinary SAC over five updates; a positive weight changes actor behavior while the first critic/target/temperature/std update remains identical; group sampling balances a rare class and keeps original goals; save/load preserves the demonstration sampler RNG and requires matching data; invalid/missing data fail before updates. The tests generate no simulator interactions.

The first physical pilot should retain the BC checkpoint and evaluate fixed full-pregrasp/full-carry cases before online training and after short checkpoints. Track deterministic success separately from noisy collection success. If the BC term is weak relative to the opposing RL gradient, change its coefficient only in a recorded branch; compare that branch against the same BC-only reference. A successful modified branch remains distinct from a successful ordinary SAC result.

For the later SAC-versus-TD3 comparison, keep original-gripper physics, train/test cases, observations, adapters, demonstrations, reset curriculum and physical budgets matched. Compare ordinary SAC+HER with ordinary TD3+HER, and, if studying this pivot, compare each with the same independently reported auxiliary imitation protocol and tuning budget. Do not compare assisted SAC against unassisted TD3 and attribute the difference solely to the base algorithm.
