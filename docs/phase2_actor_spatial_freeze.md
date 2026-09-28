# Actor map-core freeze ablation

This opt-in branch tests whether preserving the gated map representation learned by BC reduces navigation degradation during SAC adaptation on eight online TRAIN layouts. It is a single-seed methodological diagnostic, with no assumed benefit or new held-out claim.

`--freeze-actor-spatial` requires `gated48-k20-f5` and positive demonstration regularization. After BC and the existing standard-deviation calibration, it freezes only `actor.features_extractor.spatial`: 54,769 of 127,605 actor parameters. The remaining 72,836 fusion, latent-action, mean and log-standard-deviation parameters continue learning. The nine compact grip measurements feed the trainable fusion layer. Independent critic encoders, critic heads, target updates, temperature, native HER sampling and rewards remain unchanged. Each forward call reconstructs its goal-conditioned maps and internal recurrent state.

The new `FrozenSpatialRegularizedSAC` checks the core's canonical tensor hash and parameter flags before and after each SAC training call and before saves. Its loader verifies saved bytes and explicitly restores `requires_grad=False`; ordinary PyTorch state dictionaries do not save that flag. Frozen gradients are cleared, and existing optimizer groups remain intact. Immutable checkpoint state records the same freeze manifest. Legacy checkpoints cannot silently resume as frozen models; an explicit fresh branch can transfer their zero-online BC actor. Absent flags retain existing configuration/signature vocabulary and semantic source dependencies.

The proposed bounded branch starts from the exact gated 192-map, 16,000-update BC archive `25876665f15eb83563b68ea8aebf84ebeafd16567224440de8fa5e676d07946a`, with zero extra BC, fresh critics and 2,000 warmup updates at target tau 0.1, then 40,000 SAC decisions. Data remain 503,901 real rows, online preparations remain first eight TRAIN layouts × five objects, and gamma 0.995, beta 100, object-balanced HER, seed 7401 and physical settings remain fixed. There is no demonstration floor or diversity change. One final 60-case original-start validation uses the existing clearance grasp/reach bytes, 24 cm lift and 8 cm internal reach margin; native 10 cm acceptance and final hold remain unchanged. All failures and full/action-deadline/including-initialization metrics are retained.

The immutable review plan is `runs/phase2_actor_spatial_freeze_plan01/artifacts/experiment_plan.json`. The root independently reviewed the implementation, passed 44 additional focused checks, and approved this concrete 40k plan. A second read-only review may finish alongside initialization; any concrete issue stops further work.

Focused compute-only verification passed 84 tests across freeze, checkpoint, encoder, regularized SAC and replay-floor suites. Synthetic SAC/HER updates changed the fusion/action heads, independent critics/target and temperature while preserving the map core exactly. Save/resume restored the flags; goal relabeling remained stateless. Actual stored signatures for the original CNN 40k, gated 40k and active demonstration-floor run remain compatible. The completed trial and independent result audit are summarized below.

A frozen core does not freeze navigation behavior: fusion and action heads can reinterpret its features. Critic error, limited physical coverage and retention may still dominate. Gradient diagnostics now cover only the remaining trainable actor parameters and must not be compared as though their parameter scope were unchanged.


## Completed result

The predeclared final 40,000-decision checkpoint completed 5,000 SAC updates after 2,000 separate critic warmup updates. Its actor map core stayed byte-identical to the BC initializer; fusion/action heads, independent critics/targets and temperature changed. All 40 declared physical preparations succeeded, and all 60 validation attempts and costs were retained.

| Candidate | Full task | Within 15k action ticks | Within 15k including settling |
| --- | ---: | ---: | ---: |
| Gated BC initializer | 50/60 | 44/60 | 43/60 |
| Gated 40k, map core trainable | 51/60 | 42/60 | 39/60 |
| Gated 40k, actor map core frozen | **52/60** | 41/60 | **38/60** |
| Prior selected CNN 40k | 51/60 | 50/60 | 47/60 |

Frozen-core full successes were foam 11/12, banana 12/12, can 9/12, apple 8/12 and mug 12/12. Seven failures lost the grasp during carry; one dropped the can during reach. There were three new successes and two regressions against gated 40k, and five new successes versus three regressions against gated BC. Recorded grasp/handover traces and stage records matched both gated controls in 60/60 cases. They matched the historical CNN control in only 11/60; the latter comparison cannot isolate an algorithmic cause. Traces sample every five native calls and omit velocities, actions, contact forces and solver state, so even matching records do not prove exact hidden-state equality.

The full validation consumed 160,098 native action calls / 800,490 action physics ticks plus 28,800 settling ticks, totaling 829,290. Training and its preparation consumed 1,189,990 ticks including settling; isolated training/validation checks consumed 1,002,525; total new physical cost including the full 60 evaluation was 3,021,805 ticks. Existing demonstration collection was not recounted as new physics.

Root accepted this candidate under the predeclared full-completion-first selection rule. The timing regression is explicit: it improves full completion by one case over the previous selected candidate while losing nine within-deadline successes. The conditional transport-qualified grasp collector remains closed because the frozen-core candidate beat the selected control under its declared gate. No collection, rerolls or additional training were launched.

Model SHA256: `268ff93c6fbb1d8021cb415113eefeb17500e2f31d2e15828c3020457b3bf6bd`. Frozen core SHA256: `ba33b048812a34eed3264c45813ec158c11ecad7a1f64f538d05c179bcecc07b`. Exact generation, checkpoint files, per-case ledgers and prefix comparisons are in `runs/phase2_system_encoder_gated192_frozen_040k01/artifacts/independent_result_audit.json`; the freeze manifest is also recorded in the final training checkpoint's `state.json` and `runs/phase2_actor_spatial_freeze_plan01/artifacts/final_checkpoint_audit.json`.

No held-out policy evaluation occurred during this trial. Root will separately freeze the revised candidate for the fresh confirmation bank; this development result does not establish held-out success.
