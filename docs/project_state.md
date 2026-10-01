# Project state

Updated 2026-09-29. Goal for this phase: a strong SAC+HER baseline on the
original gripper that completes the full task on unseen mazes. Earlier history
and every experiment of this phase: [experiment_log.md](experiment_log.md).

## Current best system: candidate 3

[configs/candidate03.json](../configs/candidate03.json). Models are the candidate 1
checkpoints (`runs/phase2_milestone_test_plan01/artifacts/checkpoints/`):
BC-initialized, demonstration-regularized SAC+HER grasp and carry (CNN map
encoder), ordinary SAC+HER reach. Planned approach and carry posture are declared
assistance; they now run faster (3× without the object, 2× while holding it).

| Fresh hold-out (20 unseen mazes × 5 objects, never used for selection) | Complete | Within 15k ticks |
|---|---:|---:|
| **Candidate 3** | **80/100** | **79/100** |
| Candidate 1, original execution settings | 70/100 | 58/100 |

On the 260-case development set candidate 3 scores 211 (81.2%) complete and
200 (76.9%) timed, versus 203 and 167 for candidate 1's original settings.

Per object on the hold-out: banana 19/20, foam 17/20, mug 17/20, apple 14/20,
can 13/20. Failures: 12 carry grasp losses, 4 carry timeouts, 2 reach timeouts,
1 reach drop, 1 grasp timeout.

## Course rubric estimate

The upstream README rubric: 60 points for implementing grasp generation (point
cloud + normals), motion planning and navigation functions, plus per-trial points
averaged over 5 graded seeds. `scripts/rubric.py` estimates the per-trial part.
Candidate 3 under the unmodified course seed mapping (`eval.py --set course`,
seeds 1000–1019, 19 distinct mazes, none in training) completes 83/100, 81/100
within 15k ticks:

| Per-trial item | Points | Candidate 3 average |
|---|---:|---:|
| Lift within 3000 sim steps | 6 | 5.94 |
| Lift ≥ 10 cm above table | 6 | 5.94 |
| Hold ≥ 50 steps | 6 | 5.94 |
| Final pose close to target grasp (no threshold given; scored at 2 cm / 0.1 rad; median 10.9 mm) | 6 | 5.70 |
| Reach within 10 cm | 6 | 4.98 |
| Reach within 15,000 sim steps | 6 | 4.86 (4.98 if steps = env calls) |
| Carry stability (heading says 4 pts; sub-line says 2 each, max 8) | 4 | 3.36 |
| **Per-trial total** | **40** | **36.7** |

With the 60 function points, the expected score is about 97/100; nearly all lost
points come from objects dropped during carry.

## How we evaluate

`scripts/eval.py` runs full original-start episodes in parallel (~5 minutes for
260 cases on 44 CPU workers). Two numbers per run: **complete** (retained object
within 10 cm of the goal after the 60-call final hold) and **timed** (complete
within 15,000 physics ticks, including initial settling and the final hold).

| Set | Cases | Role |
|---|---:|---|
| `dev` | 260 | Development: 12 validation + 20 first-test + 20 confirmation mazes × 5 objects; ±7 cases noise |
| `holdout` | 100 | Fresh bank built 2026-09-29 (`configs/holdout_bank_20260929.json`); used once for candidate 3 |
| `reserve` | 100 | Sealed for the final SAC/TD3 study |

Objects: foam brick 14, banana 5, tomato can 2, apple 7, mug 10.

## What limits performance

- **Grasp creep during carry** is most remaining failures. The object slides down
  between the finger pads under base-motion pulses; it is partly random from run
  to run and not predicted by route length. Slower carry, settle holds and a
  tilted carry posture did not help (see the experiment log). A modified finger
  geometry, planned for the gripper study, is the natural remedy.
- **Online SAC adds no measurable performance** for grasp or carry: replacing both
  with their imitation-only initializers gives 205/260 vs 211/260 (within noise).
  Competence comes from planned assistance plus imitation of a planner (with
  noise-injected demonstrations over 192 mazes); only reach is learned by RL.
  The demonstration loss (weight 100) keeps the actor near the teacher. Removing it where the critic
  disagrees (Q-filter) collapsed the policy, so the critic, not the regulariser,
  is the bottleneck for making online RL contribute more.

## Reproducibility

Retraining the candidate 1 carry recipe with a new seed (7402) gives 214/260
complete and 200/260 timed on dev, matching the original model (211/200). The
recipe is stable across at least two training seeds.

## Agreed study design (2026-10-01)

Imitation-only (BC) is a baseline arm of its own. Per gripper, compare:
**BC only**, **BC + SAC+HER**, **BC + TD3+HER**, with matched data and budgets.
Next work gives RL a measurable job the teacher does not do (grip-safe, fast
carrying) and makes online RL strong enough to move beyond the BC policy.

## Operations

- Dashboard: `python dashboard/server.py --port 8766` → http://127.0.0.1:8766.
- Replay buffers, superseded datasets and caches were deleted on 2026-10-01
  (`runs/` 340 GB → 17 GB). Superseded run folders (small) remain in `runs/`.
- The Codex agent that built most of this repository is paused.
