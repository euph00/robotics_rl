# Phase 2 checkpoint and continuation contract

The carry runner now publishes immutable checkpoint generations. A generation contains `model.zip`, `replay.pkl`, `rng.pt`, `state.json`, and `manifest.json`; every data file is flushed and hashed before `checkpoints/latest.json` atomically points to the completed generation. An interrupted write leaves the previous pointer usable. `best_online.zip` remains a separate inference checkpoint, selected only by a complete declared validation sweep.

Resume into a new run ID:

```bash
.venv/bin/python scripts/phase2_train_carry.py \
  --run-id phase2_carry_continued01 \
  --resume-run phase2_carry_spatial02 \
  --steps 20000 --device cuda
```

`--steps` means **additional** online decisions. The parent online/update counters, replay, RNG states, callback costs, initialization records, and worker curriculum/rotation metadata are retained. BC, std calibration and critic warmup are skipped. Open replay tails become truncations before fresh physical episodes start; this does not promise bitwise simulator continuation.

Method parameters default to their parent values. Explicit changes to worker count, layouts, seed, horizon, action scale, batch size, learning rate, reward/entropy/HER settings, dataset contents, or scene-bank contents are rejected. Physics, control, geometry, reward and feature source hashes must match. Logging and persistence code may change. One deliberately narrow resource exception permits adding `MultiSceneCarry.prime_cache` while the remaining scene-wrapper AST stays identical; the priming implementation must preserve scene seeds, reset order and physical state construction. Raw source archives still record the difference.

New records identify `cost_scope: cumulative_lineage`, inherited and incremental online decisions, per-field `incremental_accounting`, and parent checkpoint ancestry. Initialization and demonstration costs are inherited references rather than newly repeated work. A child's inherited best checkpoint may remain in the parent run; consult `best_checkpoint` in the result or checkpoint callback state.

Stopped/completed legacy phase-2 runs are supported using their old `latest.json`, model, `latest_replay.pkl`, RNG, config and source archive. Active legacy writers must first stop because those files were mutable. Legacy missing callback fields are explicitly labeled. The old stop callback could omit the last executed vector transition from replay; the loader accepts and flags exactly one such vector omission, preserving physical-interaction counters, and rejects larger position inconsistencies.

Evaluation setup failures remain failed cases in the denominator, with their actual setup costs. Their unknown physical goal distance is `null`; a declared 30 m **selection penalty** is separate from measured distance. Interrupted partial sweeps are recorded but cannot replace the best checkpoint.

Validation: `tests/test_phase2_checkpoints.py` exercises actual tiny SB3 SAC+HER continuation, exact-once interrupted transition flushing, legacy compatibility, corrupt/partial bundle rejection, RNG/counter restoration, complete validation ranking, and the full resume CLI branch without repeating initialization. These are compute-only persistence tests, not robotics performance evidence.

Regularized branches also pin `bc_weight` and the active `demo_batch_size`, and hash `pnr/regularized_sac.py`. Old ordinary-SAC checkpoints are interpreted as weight zero, with demo batch size inactive. Ordinary/regularized changes, positive-weight changes and changed regularization code require a new branch. Positive-weight resume validates the actual `RegularizedSAC` class and stored demonstration manifest, guarding against accidentally loading that ZIP with ordinary `SAC.load`. Demo arrays are reattached separately and checked against their persisted content/group digest.
