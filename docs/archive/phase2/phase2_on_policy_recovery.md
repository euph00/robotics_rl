# TRAIN-only learner-induced carry recovery

`scripts/phase2_carry_demos.py --mode on-policy-recovery` queries expert labels at states physically visited by frozen learner checkpoints. Only declared TRAIN layouts are accepted. The checkpoint ZIPs and source are copied and hashed before collection; inference is CPU-only and no policy parameters are updated.

The original carry goal remains fixed. The actor runs for at most 450 decisions. Before each action, the collector queries A* from measured current XY to the original exit and stores a separate expert action. A single expert takeover occurs when the best geometric remaining-path distance improves by less than 0.5 m over 60 decisions, or when the learner budget is reached. Geometric distance is used so legitimate detours that increase Euclidean distance do not trigger takeover. The total physical episode is capped at 900 decisions. There is no reset, snapshot restoration, goal change, or physics-counter reset at takeover.

Replay records the **actual executed action**, its physical reward and next observation. `teacher_action` is a separate BC label. Rows also record `control_source` and training-only geometric remaining distance. Physical failures stay terminal; success/horizon/collector interruptions close as timeouts under the existing HER contract. No unavailable teacher label is fabricated. All executed calls and setup/settling costs remain counted.

Optional correlated perturbations are confined to learner prefixes. The actor's deterministic mean, actual post-clipping perturbation, and executed action are stored separately. Expert recovery has no added action noise. The first noisy preflight prespecifies stationary Gaussian AR(1) std0.35, rho0.9, component clip±0.5, collection seed203000 with `1000*layout_index+actor_index` offsets. These values are frozen in the run configuration. This is training data collection, not an evaluation result or a change to deployed policy behavior.

Mock-only tests cover exact replay continuity, action/label separation, no reset at takeover, true physical termination, unavailable expert routes, geometric progress during a Euclidean detour, seeded perturbation reproducibility, and cost accounting: **9 passed** across `tests/test_phase2_on_policy_recovery.py` and `tests/test_phase2_carry_recovery.py`.

Deterministic preflight `phase2_carry_onpolicy_preflight01` used TRAIN000/016/032/048 and both frozen `phase2_carry_recovery01`/`phase2_carry_navigation01` BC checkpoints. All **8/8** episodes succeeded in313–387 decisions. There were **zero takeovers and zero interior westward teacher labels**: the deterministic actors fit these TRAIN routes, so this run did not fill the diagnosed coverage gap. It took62.1 seconds and18,769 native calls /95,765 physics ticks including setup and settling. The 2,785 recorded transitions pass exact observation continuity, original-goal consistency, action/label checks, checkpoint hashes, and exact reward recomputation. Dataset SHA-256: `5552c0936a3b0883fef4b14d6092f26ae0fb413209c9b70ee091a75a8d018e21`.

`phase2_carry_onpolicy_noise_preflight01` repeated only those eight cases with the prespecified correlated perturbations. It produced **3/8 successes, two true collision terminations, three unavailable expert endpoint queries near the starting table, zero takeovers, and only one interior westward teacher label** (no actual westward recovery). Its1,444 transitions pass the same exact continuity/goal/reward checks. Cost:12,064 native calls /62,240 physics ticks including setup,48.7 seconds. Dataset SHA-256: `0e748d147e47fd4d4574be0f3a565fa1d22ca8b3256fb4fe17f5d1f927e824f6`. The query failures were retained as collector timeouts, not mislabeled as physical collisions. The planner rejects a start when its rounded grid endpoint lacks clearance; no clearance was relaxed to invent a label.

The noisy prefix mostly induced collision or unlabelable states before triggering recovery. Do not scale this mode to64 layouts based on this result. The next recommended collection is systematic TRAIN-only interior branch coverage: select reachable off-solution cell centers geometrically, reach them with real teacher controls, then continuously recover to the fixed exit while counting all preparation and failed-attempt costs. This directly supplies missing backtracking without choosing starts from validation trajectories. No 64-layout batch has been authorized or launched. Data remains separate from production training mixtures. Before integrating useful interior corrective examples, the BC sampler needs an explicit on-policy corrective group; the existing entrance-only prefix rule does not cover arbitrary interior recovery.

Read-only verification:

```sh
.venv/bin/python scripts/phase2_on_policy_dataset_check.py \
  --dataset runs/phase2_carry_onpolicy_preflight01/artifacts/demonstrations.pkl \
  --output /tmp/on_policy_validation.json
```

The checker verifies TRAIN geometry, all goals, exact adjoining observations, closed terminal boundaries, source hashes, action/label separation, optional applied perturbations, reward recomputation, and total transition costs. It also reports expert takeovers, physical failures, collision transitions, and interior westward coverage. Use a fresh output path.
