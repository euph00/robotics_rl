# Carry BC curriculum and exposure audit

The original selected CNN carry policy inherited **60,000 BC updates / 15.36M sampled labels** before its 40k SAC run. The fresh gated initializer received **16,000 /4.096M**. Their comparison therefore confounds architecture with training budget and curriculum; the matched fresh CNN/gated 16k comparison remains valid on its own.

| Original CNN actor ancestry | Added BC updates | TRAIN maps | Dataset rows | Preparation |
|---|---:|---:|---:|---|
| `phase2_carry_spatial02` | 12k | 64 | 77,238 | Reference foam |
| `phase2_carry_recovery01` | 12k | 64 | 175,456 | Reference foam, entrance recovery |
| `phase2_carry_geometric100` | 12k | 64 | 228,059 | Adds geometric backtracking |
| `phase2_carry_expanded_init01` | 16k | 192 | 478,363 | Expanded reference-map coverage |
| `phase2_carry_learned_init01` | 8k | 192 | 503,901 | Adds five-object learned grasps on first 8 maps |

Every inherited archive was the final declared BC checkpoint, saved before critic warmup or online SAC. Ancestors' subsequent SAC updates were not inherited. Each stage reset Adam at 3e-4 and the NumPy sampler to seed 7401, used batch 256 / gradient clipping 10, and calibrated log standard deviation to −3 afterward. The selected 40k CNN model is `a07f7928…`, with 40,000 decisions / 5,000 SAC updates.

Sampler indices were reconstructed from the archived trainer rules, original immutable datasets and seed; these counts are reconstructed draws, not saved batch logs. No optimization, policy inference or physics was performed.

| Exposure | Original CNN BC ancestry | Fresh gated 16k BC |
|---|---:|---:|
| Total label draws | 15,360,000 | 4,096,000 |
| Draws outside TRAIN000–007 | 12,495,853 | 771,037 |
| Draws on TRAIN064–191 | 2,586,676 | 462,596 |

Fresh object balancing assigned roughly 80% of draws to four non-foam objects available only on the first 8 maps; those maps actually received 3,324,963 draws (81.18%). Only 11.29% reached the 128 newly added maps. The original expanded 16k foam stage alone sampled 2,319,646 labels on those maps; the full ancestry supplied 5.59× as many as fresh gated. Counting 192 admitted maps does not describe the actual training distribution.

There is also a smaller sampler difference: the original 8k mixed stage used fixed-floor group sizes, giving each non-foam object 49/256 labels and foam 60/256. Current multinomial sampling gives 20% per object in expectation. The historical refinement therefore used 392,000 labels per non-foam object and 480,000 foam labels. Raw source/sampler hashes and per-map counts are retained under `runs/phase2_carry_bc_exposure_audit01/artifacts/`.

## Approved bounded next initializer

`phase2_carry_gated_staged_bc01` uses the existing datasets and unchanged gated48/K20/F5 encoder, 580-state interface and masks:

1. Fresh actor, 52k BC updates on the 478,363-row reference dataset (`3c180f17…`), using 477,257 successful labels across 192 TRAIN maps. Each batch uses west/prefix/turn/straight/hold counts 51/61/71/50/23.
2. Exact stage-one actor, fresh Adam/sampler, 8k BC updates on the 503,901-row mixed dataset (`42eb7f46…`), with 500,395 successful labels and current multinomial object balancing.

All actor parameters remain trainable. Both immutable stage checkpoints are retained. Total budget is60k updates / 15.36M labels; critics, target critics and temperature are verified unchanged, with zero replay sampling, SAC updates or physical interactions during computation. Expected compute time is about 29 minutes plus loading/checkpoint overhead, extrapolated from the previous gated 16k run.

This matches the old total BC budget while changing historical stage order and map exposure; it is not an isolated architecture experiment. Map-core freezing is deliberately absent from this initializer. A single fixed 60-case BC validation follows, using the same frozen grasp/reach, original physics, .24m lift, .08m reach handover and final hold. At least 50 full successes is the predeclared screen for considering further SAC; timing remains reported, and online training requires a separate result review. No rerolls or held-out evaluation are included.

Frozen plan/source: `runs/phase2_gated_staged_bc_plan01/artifacts/experiment_plan.json`, digest `e2f4ccd2…`, approved proposal `5d625237…`. Full-system run ID: `phase2_system_gated_staged_bc01`. The staged result is pending; no success claim is made.
