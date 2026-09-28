# Preparation-source replay audit

If the current continuation plateaus, balancing preparation source within each
object is a justified, bounded ablation. It addresses a measured replay imbalance;
it is not evidence that the change will improve physical success. No sampler,
training run, physics or validation data was changed or used for these statistics.

The exact eight-worker seed contains 502,432 eligible real transitions. Sampling
an object uniformly, then an available preparation family uniformly, gives:

| Object / preparation | Eligible transitions | Physical failure endpoints | Current exposure | Proposed exposure |
|---|---:|---:|---:|---:|
| Foam / reference | 478,363 | 6 | 19.805% | 10% |
| Foam / learned | 4,698 | 3 | 0.195% | 10% |
| Banana / learned | 5,903 | 0 | 20% | 20% |
| Can / learned | 4,706 | 4 | 20% | 20% |
| Apple / learned | 4,013 | 4 | 20% | 20% |
| Mug / learned | 4,749 | 1 | 20% | 20% |

Learned-foam exposure increases 51.4-fold, and total learned-handover exposure
increases from 80.195% to 90%. Existing physical loss endpoints remain unchanged.
Their expected draws per 256-sample batch rise from 0.1063 to 0.1220, with the
probability of at least one endpoint rising from 10.085% to 11.488%. Across 2,000
such batches, expected learned-foam endpoint draws rise from 0.64 to 32.7; expected
endpoint draws across all objects rise from 212.6 to 244.0. These are analytic
projections for independent sampling with replacement, not recorded draws or
failure prioritization. HER keeps the same physical endpoint/reward semantics.

The immutable 32k TRAIN replay confirms that online additions have not removed
the imbalance: 533,560 completed-episode transitions are eligible, including
10,992 learned-foam states, whose current exposure is only 0.449%. The proposed
exposure is still 10%, a 22.3-fold increase. Its 31 physical endpoints comprise
the original 18 losses plus 11 online losses and two online collisions. Another
872 online transitions are in unfinished episodes and are excluded. No buffer
wrap has occurred, so the original seeded population is intact.

## Source labels and compatibility

Exact canonical preparation labels are **not currently stored in replay info**.
All 502,432 seeded entries have `replay_source`, copied from episode collection
mode: `demonstration` (77,238), `standard` (129,883), `recovery` (98,218),
`geometric-recovery` (173,024), or `learned-handover` (24,069). The first four are
reference preparation only because this frozen merged dataset's nested source
manifests and collectors establish that provenance. `demonstration` by itself is
not a universally reliable preparation label.

All 31,128 eligible online entries lack `replay_source`. The archived learned
cache implementation supplies case ID, object ID, point seed and
`conditional_on_prepared_handover=True`. For this checkpoint, the immutable
learned-preparation configuration, frozen grasp hash, declared 40 TRAIN cases,
online insertion range and matching case/object/seed establish their learned
origin. Missing source labels must not generally be interpreted as learned.

The smallest explicit option is a versioned extension of the current
object-balanced index sampler: object → nonempty preparation family → uniform
eligible real transition. Add canonical `info['preparation_source']` values
`reference`/`learned` at offline import and online step creation, independently
of collection mode. Store immutable per-transition source labels and enforce
one object/family per episode. Keep stock SB3 2.7.0 future-goal selection, HER
ratio, reward recomputation and timeout handling unchanged; old defaults remain
object-only. Source identity stays outside the policy observation.

An existing replay needs a separately recorded migration into a new checkpoint:
verify the dataset/checkpoint/source hashes, map only the proven source modes
and online cases, and reject unknown provenance. Preserve every numerical
transition array, episode boundary and cost. Never modify the old checkpoint.
If an eligible family disappears through ring eviction, the declared rule uses
the remaining family for that object and reports its changed probability. An
object with no eligible transitions remains an error. Telemetry should expose
counts, probabilities and actual draws for each object/family.

## Risks and required checks

This shifts reference foam from 19.805% to 10% of draws, reducing exposure to
192-map reference coverage while emphasizing only eight learned-handover TRAIN
layouts and a small set of grasps. It also changes SAC actor-state sampling,
not just critic sampling. Continuing BC regularization remains a separate
sampling distribution. These tradeoffs warrant a separate declared comparison,
not a silent change to the current continuation.

Before use, test exact hierarchical probabilities and absent-family behavior;
unknown/missing/changing labels before buffer mutation; incomplete episodes and
ring eviction; save/load and migration preserving arrays, boundaries, RNG and
counters; and exact stock-HER rewards, goals, terminal and timeout behavior for
identical sampled indices. Verify old default sampling is unchanged. No failure
stratum, duplicate episode, synthetic observation or extra physical cost is
needed.

Verified replay digest, complete counts/projections, provenance assumptions and
the reproducible read-only script are in
`runs/phase2_replay_source_audit01/artifacts/result.json` and `audit.py`.
