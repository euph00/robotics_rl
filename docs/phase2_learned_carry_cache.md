# Carry training after a frozen learned grasp

`pnr.learned_carry.LearnedMultiCaseCarry` is an opt-in TRAIN environment. Existing `MultiSceneCarry` and unconditional full-task evaluation are unchanged. It uses the same `LearnedCarryPreparation` as the learned-handover demonstration collector: original start, planned approach, frozen 54-state grasp with clearance-v2 proposals and a 24 cm lift goal, physical 50-call hold, and the existing planned carry posture. No navigation teacher or route bank runs inside this cache.

The constructor receives the exact case list, immutable grasp ZIP and expected SHA-256. It checks the ZIP hash, observation/action shapes, and embedded grasp protocol metadata (observation version, IK control version, proposal mode and lift). Each cache identity also includes the case's object, point seed, maze hash, full native configuration, preparation/restore protocol, action scale, reward parameters and horizon. The runner selects cases before seeing physical outcomes: first eight TRAIN layouts × five objects, divided among workers. This selection is separate from the 192-layout demonstration replay.

## Physical restoration and costs

PyBullet snapshot restoration can leave contact-force queries stale. After each restore, the cache executes exactly five native hold calls using the restored measured joint positions and zero finger targets, then requires measured lifted bilateral contact and no drops. It also checks for collision on every hold call. These 25 physics ticks are preparation work, including the initial cache check. They are never replay transitions. No state/contact bit is fabricated. Pickup and drop history is preserved with `reset_counters=False`.

A separate monotone ledger counts newly executed calls. Native counters rewind with a snapshot, so their absolute values cannot represent total training work. `setup_action_calls/setup_physics_ticks` alias `preparation_source_action_calls/preparation_physics_ticks`; **do not add both names**. Initialization settling is counted separately. Every prefix attempt and restore attempt retains its outcome and costs; carry steps count actual calls even if the wrapper raises after partial execution.

## Rotation, failure and resume

The finite cache must fit its entire worker slice. There is no eviction, fabricated transition, automatic retry or replacement case. A failed prefix/restore remains in metadata and is excluded from conditional carry replay. `prime_cache()` returns all declared/valid/failed IDs and string-keyed `valid_object_counts`; the runner checks five-object coverage globally. `reset()` errors if a worker has no valid handover. Full-task evaluation still includes every declared original-start case.

Valid cases rotate deterministically, one episode per case by default. Every episode uses the original fixed exit goal; callback curriculum requests are recorded but do not create segment starts or goals. Per-case resets, completed decisions, native calls and physics ticks show actual coverage. Demonstration replay and HER can still provide shorter goal coverage.

Call `set_training_state()` **before** `prime_cache()` on resume. It verifies the case/checkpoint/settings signature and restores rotation, RNG, cumulative per-case counts and attempt history. Failed cases remain failed without automatic retry. Successful physical caches must be rebuilt; their new costs start at zero in the worker and are added to inherited runner totals once. Physics snapshots are not serialized, and an interrupted episode is not resumed.

## Verification

Focused tests cover identity differences (object, point seed, checkpoint and reward), finite capacity, TRAIN admission, protocol rejection, idempotent priming, rotation, history preservation, missing-contact refresh, partial preparation/transport failures, disabled cases and resume accounting. Together with the shared collector tests, 18 tests passed.

The bounded physical check is recorded in `runs/phase2_learned_cache_preflight01`: prespecified TRAIN layout000 with foam, can and apple; three reset/transport cycles per case; at most20 carry decisions per cycle; zero training updates. The can case was also among the frozen collector's zero-call restored-contact failures. All outcomes, costs, exact checkpoints, source hashes and measured trajectories are preserved. This is a cache check, not a full-task success estimate.
