# TRAIN branch and entry-room recovery data

The deterministic learner-induced preflight added no backtracking, while correlated noise mainly caused collisions or unavailable teacher queries. This collector instead chooses physical preparation targets systematically from **TRAIN geometry only**. It does not use validation trajectories as starts or labels.

For each TRAIN map, `geometric_recovery_targets` computes free-cell distances from the entrance and exit. Interior candidates must lie off every shortest entrance-to-exit grid path, remain reachable, and have a shortest return path that includes westward travel. It selects up to two candidates, ranked by westward return distance then extra route length, with distinct attachment points to the solution path. Candidate X is limited to cell centers4.5–15.5 m. Grid selection is only a proposal: normal inflated-occupancy planning and real physical carrying must reach each start.

Each map also receives entry-room targets `(1.5, −4.25)` and `(1.5, +4.25)` m, independent of the gate position. This covers the opposite side of low and high entrance gates without fitting offsets to individual validation failures. Across the64 TRAIN maps, the selector produces44 interior branch targets on37 maps plus128 room targets.

The collector restores only an existing physically prepared grasp/start snapshot, then physically executes preparation to each target. Preparation allows at most600 carry decisions, and its calls/ticks and failed attempts are counted separately. Recovery begins from the measured live endpoint with the original exit goal, retained contact history, and no object or joint teleportation. The recording horizon is900 decisions. Current geometric batches use deterministic expert controls (`--recovery-noise-std 0`). Every stored transition keeps its actual action, teacher label, physical reward, next observation, and true termination/timeout.

The BC sampler reserves **20%** for successful geometric interior-recovery rows with measured base x>3.5 m and expert x-action<−0.5. These indices are excluded from all other groups. The remaining80% preserves the prior entry-prefix/turn/straight/hold proportions: **24%,28%,19.6%,8.4%** overall when all five groups exist. Entry-room geometric episodes use the existing prefix end rule, first observed x>3.5. Interior rows outside the explicit westward group enter the ordinary groups. Failed episodes remain in the dataset/replay and supply no BC labels. Datasets without geometric westward examples retain their previous sampler behavior.

The four-map preflight (`phase2_carry_geometric_preflight01`, TRAIN000/016/032/048) produced11 episodes with10 successes: **3/3 interior recoveries and7/8 room recoveries**. One TRAIN048 room recovery lost grasp; it remains recorded and excluded from BC. All11 physical preparations succeeded. The run added **69 successful interior westward rows** and674 entry-prefix rows. Its3,043 transitions passed exact adjoining-observation checks, fixed original goals, physical preparation endpoint matching, and exact reward recomputation. Total cost, including setup/settling: **28,634 native calls /145,090 physics ticks**,67.9 seconds. Dataset SHA-256: `a9920a58f7933c4af351699c48aa6e070a8841fae362e245636c410e268b0081`.

After those targeted physical and replay checks, the authorized64-map batch was launched as `phase2_carry_geometric64` with six CPU workers and the same frozen settings. Its outcome is stored separately; source datasets are never overwritten. This is physical expert dataset expansion followed by the declared supervised/regularized learning method, not ordinary SAC and not an inference-time route oracle.

Validation: **20 focused tests passed** across geometric target selection, disjoint supervision groups and preserved old mixture ratios, preparation accounting, continuous replay boundaries, learner/expert labels, and seeded-noise reproducibility. The standalone checker `scripts/phase2_geometric_dataset_check.py` validates the resulting physical dataset without simulation:

```sh
.venv/bin/python scripts/phase2_geometric_dataset_check.py \
  --dataset runs/phase2_carry_geometric_preflight01/artifacts/demonstrations.pkl \
  --output /tmp/geometric_recovery_validation.json
```

Use a fresh output path. The checker includes all attempted preparations and failed episodes when reporting costs and separately measures successful interior westward coverage. No held-out test or reserve scenes are used in selection, collection, or supervision.
