# First milestone: working development baseline

The original Fetch completes pickup, retained maze transport and held-object reach in the pinned course environment. **Three fresh original-start repetitions passed**, followed by a separately recorded successful rollout. The case is supplied foam brick 14 / scene 42; these are repeatability checks of one development case.

[Watch in the dashboard](http://127.0.0.1:8766/?run=learned_system_demo_bc01) · [WebM](../runs/learned_system_demo_bc01/artifacts/episode_000.webm) · [MP4](../runs/learned_system_demo_bc01/artifacts/episode_000.mp4) · [Measured route](../runs/learned_system_demo_bc01/artifacts/episode_000_route.png).

## What is frozen

This is a **demonstration-assisted SAC + HER pilot**, with the training order disclosed below. It is not a from-scratch RL result or the final algorithm comparison.

| Skill | Selected learning history |
| --- | --- |
| Grasp | Physical demonstration replay and SAC + HER to 40k decisions; 500 supervised actor updates; ordinary SAC continuation, selected at 41,008 decisions. |
| Carry | SAC + HER checkpoint at 90k decisions; 14k supervised actor updates using recorded full-route demonstrations and consistent lookahead labels. |
| Reach | SAC + HER, selected at 15k decisions from a completed 20k run. |

Carry observes the wall/table map, base position, fixed goal and contact state. Its declared fixed arm posture permits masking nuisance arm/payload pose features in the actor; critics retain full state. Measured spatial features are recomputed after HER. The selected model used **no synthetic lateral training examples**.

At inference, the shared supervisor provides planned antipodal pregrasp approach and carry-posture preparation. Grasp, transport and reach use saved neural policies. There are no inter-skill resets, planner waypoints supplied to the carry policy, teacher corrections, object attachments or final release.

Exact frozen ZIPs, hashes, training provenance and protocols are in [the milestone manifest](../configs/milestone1_candidate.json). Validation also archives implementation sources.

## Measured results

| Measurement | Three validation repetitions | Recorded rollout |
| --- | --- | --- |
| Complete physical successes | 3/3 | 1/1 |
| Final object–goal distance | 3.94 cm each | 3.08 cm |
| Drops / wall or table contact calls | 0 / 0 | 0 / 0 |
| Learned grasp / carry / reach decisions | 33 / 627 / 9 each | 33 / 627 / 9 |
| Final stable hold | 60 native calls each | 60 native calls |
| Native action calls | 4,066 each | 4,069 |
| Physics ticks including initial settling | 20,810 each | 20,825 |

The video contains 1,017 frames at 12 fps, lasting 84.75 seconds. Both WebM and MP4 decode successfully. [Validation details](../runs/milestone1_validation01/artifacts/milestone_validation.json) and [recorded physical result](../runs/learned_system_demo_bc01/artifacts/episode_000.json) retain all stage metrics and assistance counts.

Lift occurs within 3,000 native calls and physics ticks. Reach satisfies the 15,000 **native-call** interpretation, but exceeds 15,000 **physics ticks**. The upstream wording is ambiguous; both flags are retained rather than claiming a resolved grading interpretation.

## Reproduce

Use the configured project environment and a new output directory:

```bash
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_system.py \
  --grasp runs/milestone1_validation01/artifacts/checkpoints/grasp.zip \
  --carry runs/milestone1_validation01/artifacts/checkpoints/carry.zip \
  --reach runs/milestone1_validation01/artifacts/checkpoints/reach.zip \
  --episodes 1 --scene-seeds 42 --object-idx 14 \
  --grasp-horizon 200 --carry-horizon 1500 --video \
  --algorithm-label 'Demonstration-refined SAC+HER pilot' \
  --output runs/milestone1_reproduction01
```

## Next milestone

Further ordinary SAC updates did **not** preserve the refined carry route at learning rates 1e-5 or 1e-6; both failed continuations are archived. An offline audit found weak or incorrect sideways correction outside the demonstrated path. Stabilizing subsequent RL and broadening physical training coverage are next-stage work, not claimed results here.

The independent SAC/TD3 × original/modified gripper × three-seed study and held-out object/maze bank remain pending. Freeze a matched demonstration protocol before that comparison. Per-run cost records describe known lineage costs; they are not a deduplicated total for all development searches.

Validation: 64 tests passed in the final production suite. Dashboard layout, links, plots and actual video playback are checked separately in a browser.
