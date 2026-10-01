# Conditional next encoder experiment

If the expanded 192-maze CNN still loops, test a **GPPN-style gated map encoder** before a plain value-iteration network. This is a proposal only; no encoder or training implementation has been changed.

The current `CarrySpatialExtractor` already has a 31-cell receptive field covering the entire 12×22 map. Its limitation is therefore not simply an inability to see distant walls. Repeated, shared learned updates could supply a more useful bias for reasoning about detours and returning from dead ends. That is an architectural hypothesis, not an established diagnosis of our remaining failures.

The primary evidence is narrow. [Gated Path Planning Networks, Lee et al. (ICML 2018)](https://proceedings.mlr.press/v80/lee18c/lee18c.pdf), especially sections3 and5, replaces VIN's recurrent max-pooling operation with gated updates. It reports better optimization stability and generalization across its maze experiments. Those experiments largely used discrete actions and substantially more training mazes than our project, so they do not establish SAC performance here.

[Value Iteration Networks, Tamar et al. (NeurIPS 2016)](https://papers.nips.cc/paper_files/paper/2016/file/c21002f464c5fc5bee3b98ced83963b8-Paper.pdf), section4.3, does demonstrate a learned coarse-map representation feeding a continuous two-action controller. However, it used guided policy search and transition-weight pretraining, not SAC. This supports the interface idea, not a direct algorithm transfer or a promised gain.

Proposed integration: keep the exact measured occupancy/geometry planes, goal construction, local physical inputs, masks, 128-dimensional feature interface, and continuous action adapter. Replace only the learned map-processing core with shared convolution-plus-gated recurrent updates. Sample its learned map features at the current base/goal as before, then retain the existing continuous actor head. A bounded starting specification is 32 hidden channels, a5×5 spatial kernel, and20 internal iterations; treat it as one prespecified candidate, not a validation-driven sweep. Measure parameters, memory, and forward/backward latency before selecting the final width for a roughly comparable capacity budget.

The recurrent state resets for **every observation**, including every relabeled HER sample. This is recurrence inside one policy evaluation, not temporal memory across environment steps; ordinary SAC/HER replay therefore remains applicable. Recompute goal-dependent features from the current `desired_goal` each time. Actor and critic encoders remain separate, following the current SB3 arrangement.

The smallest defensible experiment is:

1. Compare one fresh CNN and one fresh gated encoder on the same frozen192-maze physical demonstrations, same sampler, action labels, BC update/example budget, and seed. Do not compare a fresh gated model against a CNN with several uncounted previous BC phases. Preserve the current low-level inputs, reward, and action scaling.
2. Use initialization-only runs and the existing12 validation maps before priming192 TRAIN worlds. Compare route completion, decision-horizon failures, collision/retention failures, and decisions among successes. An offline action-loss improvement alone is insufficient. Report one seed as a screening result; replicate a promising gain before making a generalization claim.
3. Only if that screen improves physical navigation, run the same bounded online SAC+HER+demonstration-regularization continuation with unchanged critic warmup, beta, and evaluation cadence. Keep the final held-out20-maze evaluation sealed.

At inference this proposal uses **learned map computation**, not a supplied solution: no A*, Dijkstra, shortest-path distance, route/waypoint input, or hand-coded search fallback. Existing physical expert action labels remain training-only. Gated hidden maps are not automatically valid value functions or certified paths.

For the eventual SAC/TD3 comparison, freeze the chosen encoder, input masks, physical demonstrations, initialization stages, and evaluation cases for both methods. Count BC/critic initialization and online interactions separately, report parameter/compute costs, and retain the explicit demonstration-regularization label. Neither architecture should receive privileged inference information or a larger demonstration budget.
