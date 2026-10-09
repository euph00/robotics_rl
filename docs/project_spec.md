# Project spec

Redesigned 2026-10-04 for fair comparisons, a physical value-add and a sealed final evaluation.
Earlier plans: [archive/project_spec_2026-10-03.md](archive/project_spec_2026-10-03.md) (first study
version, whose results are now development evidence), [archive/project_spec_2026-09-28.md](archive/project_spec_2026-09-28.md)
and the PyBullet-era guide in [archive/](archive/). Machine-readable plan (drives the dashboard):
[`configs/study.json`](../configs/study.json).

## Question and claims

**Where does RL add value in mobile manipulation, and how does that depend on the end effector's
complexity?**

| | Claim | Evidence the study must produce |
|---|---|---|
| C1 | RL's gain over the best hand-written grasp grows with end-effector complexity. For a parallel gripper a planner is already near the ceiling; for a 16-joint hand no planner exists, a script reaches about half, and RL closes the gap | The same learning recipe and budget for both end effectors: hand-written → imitation → imitation + RL → RL from scratch |
| C2 | The RL hand does something the gripper physically cannot: it picks up and delivers objects wider than the gripper's 9.8 cm opening | Delivered on the wide task, the gripper's best system (planner, BC → PPO or PPO from scratch, chosen on development data) vs the RL hand, with a margin outside both confidence intervals |
| C3 | Which RL algorithm suits dexterous mobile manipulation: on-policy PPO vs demonstration-driven off-policy RLPD | Equal starts, workers, hardware and step budget; reported per step and per hour |

C1 is expected by construction: imitation is capped by its demonstrator, and the gripper's demonstrator
(the planner) is near the ceiling while the hand's (our script) is not. The study quantifies it; C2 is
the capability contribution.

## Task

The course task: a Fetch drives from the start room to a table, picks up an object, carries it through
an unseen maze and holds it within 10 cm of a goal 0.9 m high. MuJoCo (`pnr/env.py`), **realistic
physics** for both end effectors (joint velocity, acceleration and effort limits; rubber grip μ 1.0;
Fetch jaws ≤ 245 N; Allegro joints ≤ 0.7 N m). Allegro mounted on the wrist flange by its wrist end.

Object sets:

- **Standard**: the course's five objects (foam brick, banana, soup can, apple, mug).
- **Wide**: vertical-walled cylinders wider than the gripper's opening. Rule, from the development survey
  (2026-10-04, 52 dev cases per object, gripper planner, random starts): course objects and same-density
  enlargements with diameter > 9.8 cm; gripper pick rate ≤ 50%. Members: master chef can (10.2 cm, 50%),
  soup can ×1.6 (10.6 cm, 12%). Additional widths may be added on development evidence before the freeze.
  Excluded with reason: bowl, plate, large marker (the gripper fails for lack of jaw clearance in our
  planner, not for physical reasons; a real parallel gripper picks a marker up).

## Study matrix: end effector × grasp controller

| Grasp controller | Parallel gripper | Allegro hand |
|---|---|---|
| Hand-written | Planner: antipodal grasp from the point cloud (the default project) | Scripted top-down or side power grasp |
| Imitation | BC of the planner's grasp | BC of the script |
| Imitation + RL | BC → PPO | BC → PPO |
| RL from scratch | PPO | PPO |

Only the grasp (and for the hand the hold through the carry, where chosen) is learned; approach,
carry posture, A* maze carry and reach are planned for every system. The carry posture and the grip
after the grasp are set per end effector and task on development data and declared.

## Fairness controls

| Control | Setting |
|---|---|
| Learning recipe | Both end effectors through one trainer and environment interface (`scripts/train_hand.py`, `--robot fetch` / `fetch-gripper`): demonstrations from the end effector's hand-written grasp (100 attempted episodes per object) → BC → PPO, same reward structure, network, PPO settings, 22 environment workers, 300-decision grasp horizon (the deployed budget), 10M environment steps (`scripts/campaign_v2.sh`) |
| Training distribution | The protocol's object-start range (±8 cm, ±12 cm, random yaw) for every learned grasp |
| Wide-task recipe | Each end effector starts from its best hand-written prior. Gripper: no hand-written grasp handles the wide cylinders (teacher 0–1/12, planner 12–50%), so it clones its standard-object demonstrations; PPO found a tilted rim pinch (~0.5 lift and carry). Hand: cloning the top-down standard demonstrations never lifted a wide can in 9M steps (0.00), so the hand clones its own side power-grasp script, which works only on the idealised hand rig (chef can 69/300 carried, soup can ×1.6 11/300; same 300-per-object budget), and PPO trains on the Fetch from the side pregrasp. Adopted for the campaign only if a development seed clearly beats the gripper's ~0.5 (training-time evaluations); otherwise the uniform-prior result is reported (2026-10-05). Met: the development seed averaged 0.66 at 2.5–4M vs the gripper's 0.19–0.32 at the same steps (0.36–0.55 at 10M). The hand's wide runs from scratch start from the same side pregrasp |
| RL from scratch | Same trainer and budget, no demonstrations, both end effectors, both tasks |
| PPO vs RLPD | Standard objects, both end effectors, 5 seeds (`scripts/algorithms_v2.sh`). Both start from the same imitation start (the same demonstrations, 40,000 cloning updates, log-std −0.7) and the same 1M-step value warm-up with the actor frozen; same 22 workers, reward, horizon and start range; 4M environment steps (the tuning budget). PPO: the tuned campaign recipe. RLPD: layer-normalised critics, demonstrations in half of every batch, published defaults (learning rate 3e-4) except one update per transition and two critics (wall-clock). Reported per step (5 seeds) and per training hour (evaluations excluded, load logged) from seed 0, which runs in turn with the machine to itself; seeds 1–4 run four at a time (an isolated RLPD run takes ~14 h). Sensitivity: RLPD without the warm-up, one seed per end effector |
| Tuning | Equal budget per end effector on development data (`scripts/tune_v2.sh`): a 2^(4-1) fractional factorial over learning rate, exploration noise after cloning, demonstrations per object (100 / 300) and interleaved cloning during RL; 8 configurations × 4M steps each; per factor the level with the higher mean score (last three training-time evaluations), per end effector (`scripts/tune_v2_report.py`). Prompted by the untuned recipe: the gripper stayed at its cloned level (~50% lifted) while its teacher lifts ~88%. Result (2026-10-05, `runs/tune_v2_report.json`): the same levels won for both end effectors — learning rate 1e-4, exploration log-std −0.7, 300 demonstrations per object, no interleaved cloning; scores of the best configuration 0.75 gripper / 0.77 hand vs 0.44 / 0.42 untuned |
| Checkpoints | The best training-time evaluation (lift and carry) within the 10M budget, the later one on ties; imitation = step 0 of the BC → PPO run; never the hold-out or reserve banks |
| Grip after the grasp | Per end effector and task, frozen targets vs held by the policy, on development mazes with the BC → PPO cells (5 seeds): the grip with more pooled deliveries, frozen on a tie; applies to every system of that end effector and task (`scripts/dev_v2.sh`, `runs/dev_v2/grip_choice.json`) |
| Seeds | 5 per learned cell |

## Evaluation protocol

| Item | Setting |
|---|---|
| Development | All decisions (carry posture, grip, checkpoints, object set) on the development mazes (`--set dev`, 52 layouts). The hold-out bank (20 layouts) was used during the first study and is development evidence from now on |
| Final | One run of every frozen system on the sealed **reserve** bank (20 unseen layouts), after the frozen system and this protocol are committed. `--allow-reserve` only for that run |
| Cases | Standard: 20 layouts × 5 objects = 100. Wide: 20 layouts × 2 objects × 3 seeded object starts = 120 |
| Object start | Seeded random offset (±8 cm, ±12 cm) and yaw per case |
| Physics and speeds | Realistic; planned motions at speed 3 (holding 2); hand grasp ≤ 300 decisions; one settle rule for all systems |
| Primary metric | Delivered (`complete_task_success`) |
| Diagnostics | Picked; delivered given picked; delivered within 15,000 physics ticks; failure stage |
| Statistics | Per seed and pooled; Wilson 95% intervals; differences called only when intervals separate |

## Development findings carried over (2026-10-03/04)

- Hand carry posture `plan_tuck`: object at the goal height, hand turned about the vertical, arm compact,
  object clear of the robot (scripted hand 23 → 44 delivered).
- Grip after the grasp: standard objects frozen, wide objects held by the policy (dev comparisons).
- The task-carry fine-tune did not help (PPO 10M checkpoints 84/46/88 vs fine-tuned 79/48/84).
- A palm-up cradle carry is implemented but the object falls out mid-roll with current policies; a policy
  trained to roll the object into the palm is a candidate for the wide-object hold.
- Neither end effector drops standard objects at faster carries; the hand finishes about 1,000 ticks
  sooner (dev speed sweep).
- PPO vs RLPD (unequal setup): PPO ~70% in an hour; RLPD 3–12% after 6 h.

## Plan

1. **Unified trainer for the gripper**: a gripper rig behind the hand environment's interface (wrist
   pose + jaw actions), a scripted planner-grasp teacher for demonstrations, BC → PPO.
2. **Retrain every learned cell** with the fairness controls: both end effectors, both tasks where
   applicable, imitation, imitation + PPO, PPO from scratch, 5 seeds.
3. **Wide-object hold** (C2): the hand's hold through the carry, learned end to end in the real
   post-grasp sequence, or the learned palm-up cradle; chosen on development data.
4. **Fair PPO vs RLPD** on both end effectors.
5. **Freeze** systems and protocol (commit), then the single reserve-bank evaluation.
6. Report, figures, clips. Stretch, only after 1–5: orientation-constrained delivery.

## Reporting rules

Report every completed condition and seed, including failures; development numbers are labelled as such.
State the control boundary for every system (planned, scripted or learned). Count training interactions
for every learned component, including demonstration collection. Report deviations from this plan with
their reason and the development evidence behind them.
