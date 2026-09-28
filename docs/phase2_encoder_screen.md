# Matched carry encoder screen

The gated encoder completed **45/60 full validation cases, 42/60 within 15,000 physics ticks including settling and final hold**. The matched fresh CNN completed **40/60 and 33/60**, respectively. This supports a further matched experiment; it does not improve the historical best or establish milestone completion.

| Carry actor | Full completion | Timed, including settling | Timed, action only | Carry drops | Carry timeouts |
|---|---:|---:|---:|---:|---:|
| Fresh CNN32 | 40/60 | 33/60 | 33/60 | 14 | 6 |
| Fresh gated48/K20/F5 | 45/60 | 42/60 | 44/60 | 12 | 3 |

Both carry actors received exactly 16,000 BC updates, with identical batches and seed 19041. Neither received online SAC updates. The measured 580-state inputs, compact grip features, raw-pose mask, continuous action head and physical execution were matched. Each full-system evaluation retained the frozen clearance SAC grasp (`fa7b8d…`, 12k online decisions) and SAC reach (`cd0c3f…`, 15k). Original-start validation comprised the same 12 layouts × five objects, .24m lift, .08m reach handover, .075 carry scale and mandatory native success plus final hold. No test or reserve cases were evaluated.

The gated network is inspired by GPPN, not an exact reproduction or an inference-time route oracle. Its recurrence resets within each forward call and recomputes from the supplied desired goal, including HER goals.

## Offline screen

Training used the immutable 503,901-row physical dataset (`42eb7f…`), fitting only TRAIN000–159. Successful episodes supplied 437,808 fitting rows and 62,587 development rows from TRAIN160–191; 18 failed episodes remained in source data but supplied no BC labels. The development fold contained reference-prepared foam states only. Both actors used the existing object-balanced direction/recovery sampler and exactly 4,096,000 sampled examples.

| Equal-map development metric | CNN32 | Gated48/K20/F5 |
|---|---:|---:|
| Overall normalized action MSE | .004190 | .003167 |
| Turn/vertical MSE | .006248 | .004974 |
| Backtracking MSE | .004433 | .002571 |
| Overall main-direction accuracy | 99.21% | 99.43% |
| Turn/vertical main-direction accuracy | 98.44% | 98.79% |
| Backtracking main-direction accuracy | 98.31% | 99.06% |
| Hold accuracy | 99.82% | 99.07% |

Gated action MSE improved by 24.4% overall, 20.4% on turns and 42.0% on backtracking; holding worsened. Backtracking covered only 526 states on 17 maps. Paired map bootstrap intervals favored gated MSE, while the backtracking direction-accuracy interval included zero. These intervals do not capture training-seed uncertainty or rollout error accumulation.

CNN/gated BC computation took 87.7/404.3s (4.61×), with 88,064/94,065 encoder parameters. CUDA peaks were 1,363.7/4,865.3MiB including resident input arrays. Total offline wall time was 539.2s; no new physics was used. Both final archive counters confirm zero environment decisions and zero SAC updates.

## Physical and reproducibility audit

The two 60-case runs used exactly the frozen final-16k actor bytes (`62c54b…` CNN, `40ef78…` gated), with no refinement, retries or favorable-case selection. Gated succeeded in ten cases where CNN failed; CNN succeeded in five where gated failed. All failures occurred during carry. Every unique case, checkpoint/source hash, native success/hold predicate, action-to-tick conversion and cost ledger passed the saved-record audit.

| Object | CNN full / inclusive timed | Gated full / inclusive timed |
|---|---:|---:|
| Foam, ID14 | 7 / 5 | 7 / 7 |
| Banana, ID5 | 10 / 9 | 10 / 10 |
| Can, ID2 | 8 / 6 | 9 / 6 |
| Apple, ID7 | 6 / 5 | 7 / 7 |
| Mug, ID10 | 9 / 8 | 12 / 12 |

Each object denominator is 12. CNN used 186,243 native calls and 960,015 ticks including settling; gated used 163,428 calls and 845,940 ticks. Combined physical cost was 1,805,955 ticks. All failed attempts are retained.

The matched pair's sampled pre-carry traces and stage fields were exactly equal in all 60 cases. This supports their paired comparison. Only 11 prefixes exactly matched the historical 40k control, despite matching checkpoint bytes, settings and protected physical source hashes. First differences appeared during grasp in 46 cases, pick-hold in two and posture in one. Initially small numeric differences later amplified; prefix action counts matched in 34/60, and one historical posture failure became successful in both new runs. The saved traces omit requested actions and complete simulator state, so the numerical/runtime cause is unresolved. No reruns were performed. The historical 51/60 remains the nominal best, but this is not an isolated carry-only contrast against that older run.

## Next controlled branch

The approved narrow opt-in setting is `--encoder gated48-k20-f5`; default `cnn32` remains unchanged. Fresh matched branches are proposed on all 192 TRAIN maps and the same dataset, common seed7401, 16k BC updates, fresh 2k critic warmup at tau .1, then 40k SAC+HER decisions with beta100 demonstration regularization and object-balanced replay. Both retain the same first-eight-layout × five-object learned handovers, gamma .995 and online tau .01. No actor transfer is used. This document does not claim those branches have run.

Evidence: `runs/phase2_encoder_offline_screen01/artifacts/` contains the frozen offline config, metrics, plot, final checkpoint audit, physical comparison and historical prefix audit. Physical run IDs are `phase2_system_encoder_cnn160_01` and `phase2_system_encoder_gated160_01`. Their immutable parent plans are `phase2_system_encoder_cnn160_plan01` and `phase2_system_encoder_gated160_plan01`. The proposed training plan is `runs/phase2_encoder_matched_192_plan01/artifacts/experiment_plan.json` (digest `7f50b2495f3cccdee340cc1c5cd50d42cd4f81a002fe2f6d8dd36a5ab8622a86`), pending review and launch.
