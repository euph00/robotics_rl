# Physical training-only recovery collection

`scripts/phase2_carry_demos.py --mode recovery` collects a separate original-exit-goal dataset. It is corrective demonstration data, not an online SAC result. The actor receives no new input or route oracle.

For each TRAIN layout, the collector considers y offsets +0.75, −0.75, +1.5, −1.5, +2.5 and −2.5m from the entrance, at x=1.5m on the room side of the wall. Targets beyond the conservative room range y∈[−4.35,4.35] are skipped before physical motion. A* additionally checks the inflated supplied walls/table. Real carry actions reach each accepted target from a saved, physically prepared original-gripper grasp. The original exit goal is unchanged. The recorded recovery then follows a physical teacher route toward that exit, with0.12 action noise and reduced noise near the goal.

No joint/object teleportation constructs a start. The initial saved physical grasp may be restored between attempts. At the reached off-route start, only the wrapper's collection horizon resets; the simulator state, native counters and retention history continue. Preparation paths and costs are stored separately because their actions deliberately aim at a temporary preparation target and would be incorrect original-goal imitation labels. Recovery actions and teacher labels are stored separately; only executed actions belong in RL replay.

Every run archives config and source hashes. `collection.json` separates rollout, preparation, setup and settling costs, including failed preparation attempts. Geometry skips have no preparation cost. SIGTERM requests a cooperative stop, closes any partially recorded trajectory as a timeout and writes completed worker shards. `full_route` now follows actual reset scope in standard collection. Recovery episodes explicitly have `full_route=false`, `original_exit_goal=true` and `collection_mode=recovery`; they should not be reported as full-start policy evaluations.

The four-layout preflight (`phase2_carry_recovery_preflight01`) used TRAIN000/001/007/052 only. All21 geometrically valid attempts succeeded;3 targets were skipped. Its7,105 recorded transitions passed exact goal-continuity, transition-boundary, reward-recomputation, retention and accounting checks. Total costs were47,829 native calls and241,065 physics ticks, including preparation/setup/settling. It fills121 states in the previously empty outer-gate overshoot region. This dataset is retained separately; its collection costs remain part of project cost even when using the subsequent64-layout dataset.

The unchanged collector completed all64 TRAIN layouts as `phase2_carry_recovery64` with six workers in652s. It produced311 episodes/98,218 transitions:307 successful recoveries and4 retained lost-grasp failures, with zero preparation failures and73 geometry skips. Full verification passed. Total collection cost was677,572 native calls and3,418,580 physics ticks, including preparation/setup/settling. It adds986 states to the previously empty outer-gate overshoot region (1,674 overshoot states including central gates). The immutable original-plus-recovery mixture is `runs/phase2_carry_recovery_mix01/artifacts/demonstrations.pkl`; its source/count/cost manifest is adjacent. The separate preflight cost is not duplicated inside this mixture.

```bash
.venv/bin/python scripts/phase2_carry_demos.py \
  --run-id phase2_carry_recovery64 --mode recovery \
  --layouts 64 --workers 6 --episodes 6

.venv/bin/python scripts/phase2_verify_recovery_data.py \
  runs/phase2_carry_recovery64/artifacts/demonstrations.pkl
```

For actor refinement, sample recovery prefixes/overshoot states as an explicit additional demonstration stratum. Most recovery trajectories subsequently traverse a long familiar route, so uniform row sampling would dilute the few corrective states. Keep the original route/hold strata as well. Any mixture must be a new dataset/branch with explicit source ancestry and costs; preserve the original352-episode dataset and all failed attempts.

`pnr.demonstrations.merge_carry_datasets(paths, destination)` validates and creates the immutable mixture, returning a small path/hash/count/cost report. It verifies the active carry reward/action contract, frozen TRAIN geometry, closed/continuous trajectories and all original goal labels. The original dataset's absent bank hash is reconstructed from every stored TRAIN layout/map and explicitly recorded. Input bytes are unchanged; complete merged data are published atomically without overwriting an existing destination. Nested source metadata retains failed-attempt details and preparation costs; merging generates zero new physical interactions.

`carry_supervision(dataset)` returns `(rows, groups, probabilities, stats)`. A recovery prefix ends at the first stored observation with base x>3.5m and never restarts if the robot later returns. The groups partition successful demonstration rows. With prefixes, sampling probabilities are30% prefix,35% turn/vertical,24.5% straight and10.5% hold; without prefixes they remain50/35/15%. Failed episodes stay in the merged dataset and HER replay, but are excluded from BC. Tests cover sampling membership, probability mass, immutable provenance, failed-attempt costs and rejection of bad action/reward/partition/cost inputs.
