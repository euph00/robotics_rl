# Minimum demonstration exposure in carry replay

The gated carry continuation loses online TRAIN success without a sustained increase in logged critic or imitation loss. Its replay also changes composition substantially: balancing objects does not balance imported demonstrations against new online experience. This is a measured exposure shift, not evidence that critic error caused the performance regression.

The immutable 40k and 80k replay snapshots retain all **502,432 imported demonstration transitions** from the 503,901-row source dataset. No demonstration has been overwritten at either checkpoint. The existing sampler chooses each object with probability 20%, then uniformly samples that object's eligible completed transitions. Therefore its effective demonstration fractions are:

| Object | Eligible demo rows | Eligible online rows at 40k / 80k | Demo fraction at 40k | Demo fraction at 80k |
|---|---:|---:|---:|---:|
| Foambrick | 483,061 | 6,733 / 12,308 | 98.63% | 97.52% |
| Banana | 5,903 | 8,214 / 16,521 | 41.81% | 26.32% |
| Can | 4,706 | 8,936 / 17,734 | 34.50% | 20.97% |
| Apple | 4,013 | 7,169 / 15,046 | 35.89% | 21.06% |
| Mug | 4,749 | 7,221 / 15,843 | 39.67% | 23.06% |

The expected demonstration fraction across a replay batch falls from **50.10% to 37.79%**, despite demonstrations comprising most stored rows. These are exact probabilities from saved eligible pools, not logged source-draw frequencies. All five objects retain their eight declared TRAIN cases. The 80k snapshot has 307–3,684 online decisions per case; online experience still covers the first eight layouts, while broader demonstration coverage is predominantly foam. The 600,000-row ring does not start replacing imported slots until 97,568 online decisions; native HER may then invalidate the rest of partially overwritten episodes as well.

Through the saved 96k telemetry, recent-100 TRAIN success falls from 79% at 40k to 30%, while critic-loss medians remain approximately 2.6–2.9, BC action MSE medians remain 0.00036–0.00049, and entropy coefficient decreases from approximately 0.005 to 0.00427. Finite losses do not establish accurate Q estimates. Reference-prepared validation remains strong and does not resolve the learned-handover failure mechanism. The earlier preparation-family sampler separates reference from learned grasps; it does not separate offline from online learned-handover transitions.

The controlled change is `--min-demo-fraction .5`. After choosing an object uniformly, let D and O be its eligible demonstration and online counts. When both are nonempty, sample demonstrations with `max(.5, D/(D+O))`, then sample uniformly within the selected origin. If only one origin has eligible episodes, use it and count those draws explicitly. If neither exists, fail. At 80k this would preserve foam's 97.52% demonstration share and raise each other object's share to 50%, without an object-specific exception. This changes critic and ordinary SAC actor replay-state exposure; the independent BC term remains β=100. Increasing β alone does not restore demonstration transitions to critic updates.

The new `DemonstrationFloorHerReplayBuffer` inherits native HER sample construction, future-goal selection, rewards, terminal handling, and episode/ring validity. Whenever the floor is nonbinding, it preserves the parent sampler’s exact selected indices and RNG consumption, including all-demo critic warmup. Permanent origin labels come only from an explicit fresh-buffer import context; all other additions are online. Labels are replaced on overwrite. Serialization validates labels, per-episode consistency, protocol and counters. Old replay cannot migrate into this mode. Default runner/checkpoint signatures remain unchanged; enabled runs add the fraction, sampler protocol and source hash. Checkpoints record eligible counts, intended probabilities, actual origin draws and available-partition fallback draws. The seeder, physical environments and inference remain unchanged.

The motivation has precedent in RLPD's separate offline/online sampling experiments, but this ablation is not full RLPD: that method also studies critic normalization and ensembles. Its results do not establish that this project's regression has the same cause. We change only the replay exposure rule here and retain the existing SAC+HER and BC settings. [Ball et al., *Efficient Online Reinforcement Learning with Offline Data*, §§4.1 and 5.1](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf).

Audit artifacts and the reproducible read-only script are in `runs/phase2_gated_replay_audit01/artifacts/`. Replay SHA-256: 40k `93244f1de0ac7845ab1fac3066d92af3fa87be8514c590be60be119a24f1362f`; 80k `78949c7884d0e3fa36a8c3b038aa5f8994a6f6a34111f3b490a6737d4345ae46`. The audit used zero replay samples, policy predictions, training updates or physical interactions. Focused tests cover distributions, empty/incomplete partitions, exact native real/HER rewards and timeouts, same-object episode isolation, ring overwrite, immutable data import, serialization, CLI rejection and conditional resume signatures. A separate diversity run remains a separate experiment.

Validation: all 72 affected replay, seeding, reward-relabel and checkpoint tests pass. Independent review found no blocking defects and passed its 65 selected checks. Actual historical gated, ordinary object-balanced, and discount-trial signatures remain compatible. The completed, separately authorized experiment is reported below.


## Completed 80k trial

The predeclared run `phase2_carry_encoder_gated192_demofloor01` completed 80,000 online decisions / 10,000 updates. It transferred the exact zero-online 16k-BC gated actor, added no BC updates, seeded the original 503,901-row dataset, warmed fresh critics for 2,000 updates, and retained γ=.995, β=100, eight TRAIN layouts × five deterministic learned handovers. All 40 preparations succeeded; all subsequent failures remain recorded. The plan SHA-256 is `de35284dcd3d52e3ad1b9ef95f450bc1f0cf075756668fa45e3e95ea42e652e4`.

| Validation policy | Full / 60 | ≤15k ticks, including initialization | ≤15k action ticks |
|---|---:|---:|---:|
| Historical CNN candidate | 51 | 47 | 50 |
| Gated end-of-BC initializer | 50 | 43 | 44 |
| Gated ordinary replay, 40k | 51 | 39 | 42 |
| Minimum demo fraction, 40k / 4,999 updates | 47 | 35 | 39 |
| Minimum demo fraction, 80k / 10,000 updates | 46 | 38 | 40 |

The floor checkpoints have **60/60 identical recorded pre-carry prefixes with each other and the historical candidate**, but only **11/60 with the gated initializer or ordinary-replay gated40k**. The other 49 first differ during grasp (46), pickup hold (2), or planned carry posture (1), before this sampler or carry policy executes. Thus the fresh gated controls cannot establish an isolated sampler effect; the historical candidate matches the observed preparation but differs in actor architecture/initialization. These are every-five-native-call traces plus timing-stripped stage records, not proof of equality of unlogged velocities, forces or solver state.

Within the floor run, 40k→80k rescues three cases and loses four. Carry losses decrease 11→9, but three carry horizons appear; the same one grasp horizon and one reach drop persist. Neither checkpoint improves on the historical candidate. **Retain this as a negative, single-seed diagnostic; do not promote the sampler or extend this branch.** The already declared initialization and frozen-map-core trials address separate hypotheses.

Actual saved replay and ZIP audits pass at both checkpoints. All **502,432 imported rows remain eligible**, zero are overwritten, and all 40 TRAIN cases have online rows. Model counters, explicit origin labels, native episode boundaries and saved draw reports agree. Final eligible online counts for foam/banana/can/apple/mug are **15,381 / 16,206 / 15,614 / 14,142 / 16,555**. Intended demo fractions are **96.914% / 50% / 50% / 50% / 50%**; the object-averaged fraction is 59.383%. Across online updates, measured demo draws are **98.434% / 57.378% / 56.433% / 56.939% / 57.049%**. Values above the floor include the earlier nonbinding period. Exactly **512,000 critic-warmup + 2,560,000 online replay draws** reconcile; independent BC batches are separate.

Costs: training plus preparation/initialization **2,192,135 physics ticks**; in-run reference validation **1,666,330**; the two full-system validations **812,310 + 878,350**. Total new physical cost is **5,549,125 ticks**, excluding inherited demonstration generation. There were no heldout tests, rerolls or substituted checkpoints. The read-only audit added no predictions, replay samples or physical interactions.

Model SHA-256: 40k `2d0644500eeef614d74758faff789844562f3cf70c1de3ca5991f20a18e165c2`; 80k `083072ee2ac69c6104fdbc0ad9d0ff68959de43ad801688ad5bfd0a4d5a1fe9d`. Final replay SHA-256: `dcfc4bdc750bb7a479ec636b1e2a3bff8df2e50277ef12d5e28d8ff69058fbc2`. Evidence is in both `phase2_system_encoder_gated192_demofloor_{040k,080k}01/artifacts/` directories: `checkpoint_audit.json`, `prefix_reproducibility_audit.json`, and `replay_origin_audit.json`; the 40k initializer comparison is `initializer_prefix_audit.json`. Reproduce the final read-only checks with `runs/setup/audit_demofloor_checkpoint.py --tag 080k`.

After both evaluations completed, the watcher caught its own successful `SystemExit(0)` in a broad exception handler and incorrectly wrote `watcher_error`. The original status/log remain intact; `runs/setup/demofloor01_watcher_exit_annotation.json` records the verified bookkeeping correction. This did not interrupt either evaluation or trigger a retry.
