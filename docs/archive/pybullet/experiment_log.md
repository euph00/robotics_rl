# Experiment log

Condensed record of development up to the 2026-09-29 cleanup. The original
per-experiment reports are in [archive/phase2/](archive/phase2/) and git history.

Unless stated otherwise, "validation" means the full original-start task on 60
cases (12 validation mazes × 5 objects). One standard deviation of a 60-case
success count is about ±3 cases, so most differences below are noise-level.

Objects: foam brick 14, banana 5, tomato can 2, apple 7, mug 10.

## Milestone 1: one maze, one object

Foam brick on course scene 42. After a physical reference controller
(antipodal grasp, IK, A* navigation) and three SAC+HER skills were built, three
fresh original-start repetitions passed. Pure SAC+HER did not learn grasping or
full-route carrying; demonstrations (BC initialization plus demonstration
replay) were needed. Report: [archive/phase2/milestone1.md](archive/phase2/milestone1.md).

## Phase 2: five objects, unseen mazes (target ≥80/100 on unseen mazes)

| Experiment | Change | Result | Verdict |
|---|---|---|---|
| Object bank | Chose five diverse supplied objects | Reference picks feasible for all | Adopted |
| Reach acceptance | Full-system reach handed over at 4 cm instead of native 10 cm | Fixed to an 8 cm internal margin | Adopted |
| Carry recovery demos | Off-route / branch / entry-room recovery demonstrations | BC carry 42/60; SAC continuations 31/60 and 35/60 | BC kept; SAC degraded |
| Training expansion | 192 TRAIN mazes of carry demonstrations | BC carry 44/60 (46/60 with clearance grasp) | Adopted |
| Clearance-v2 grasp | Tilted 60° approach for tall objects (can) | Makes can transport feasible | Adopted |
| Learned-handover carry | Carry demos and online SAC started from the learned grasp; object-balanced HER; demonstration-regularized SAC (β=100), 40k decisions | **51/60, 47 timed** | **Candidate 1** |
| First unseen test | Candidate 1 on 20 new mazes | **78/100, 63 timed**; 14 carry drops, 6 carry timeouts | Target missed; bank retired |
| Longer SAC / varied grasps | 200k decisions; 240 varied grasp preparations | 48/60; 49/60 | Rejected |
| Discount | γ 0.995 → 0.999 | 45/60, 44/60 | Rejected |
| Gated map encoder | GPPN-style recurrent map core vs CNN, matched 16k BC | Gated BC 50/60; gated SAC 40k 51/60 (39 timed); CNN SAC 42/60; gated SAC 80k 39/60, 120k 44/60 | Gated > CNN; longer SAC degrades |
| Demo replay floor | Minimum 50% demonstration share in replay | 47/60, 46/60 | Rejected |
| Staged BC pretraining | 52k map + 8k object BC updates | 49/60 | Rejected |
| Preparation-source replay | Balance replay by grasp-preparation family | Below object-only balancing | Rejected |
| Actor map-core freeze | Freeze gated map core during SAC | **52/60, 38 timed** | **Candidate 2** (picked on +1 completion) |
| Confirmation test | Candidate 2 on 20 fresh mazes | **69/100, 53 timed**; 24 carry drops | Target missed; bank retired |
| Retention physics | Teacher carry speed 0.075 vs 0.020 m/decision | 12/15 vs 15/15 retained (0.020 is ~3× slower) | Speed is the main drop lever |
| Retention physics | Tighter closure targets; centred carry posture | All worse | Rejected |
| Apple settling | 30-call closed hold after first finger contact | Rescued 2/2 apple failures (diagnostic only) | Not yet integrated |
| Qualified grasp BC | Imitate only grasps that survived teacher transport | 18/20 → 15/20 retained | Rejected |
| Retention predictor | Logistic model of which grasps survive transport | Worse than object prevalence | Rejected |
| Retention feedback | Policy-gradient step on the grasp from transport outcomes | 35/40 vs 31/40 control; equals original actor | Closed |
| Command slew limit | Cap change of carry command between decisions | Implemented, never run | Dropped |

## 2026-09-29: execution tuning on the 260-case dev set

After the cleanup, evaluation moved to `scripts/eval.py --set dev`: 260 cases
(12 validation + 20 first-test + 20 confirmation mazes × 5 objects), about
5 minutes on 44 CPU workers. One standard deviation of the success count is
about ±7 cases. No training was run; only execution settings changed.

| Setting (candidate 1 models unless stated) | Complete /260 | Timed /260 | Verdict |
|---|---:|---:|---|
| Candidate 1, original settings | 203 | 167 | Baseline |
| Candidate 2, original settings | 186 | 144 | Worse on every subset; reverted to candidate 1 |
| Carry 0.05 m/decision (horizon 1800) | 173 | 0 | Rejected: more drops and timeouts |
| Carry 0.035 m/decision (horizon 2600) | 122 | 0 | Rejected |
| Grasp settle hold, 30 / 60 calls at first contact | 204 / 202 | 166 / 156 | No effect; removed |
| Tilted carry posture (finger under object) | — | — | Infeasible: arm exceeds navigation envelope; reach fails; removed |
| Planned motion speed 1.5 / 2.0 / 2.5 / 3.0 | 203 / 212 / 199 / 214 | 187 / 197 / 191 / 203 | Faster planned motions help timing; posture drops rise at ≥2.5 |
| Empty-hand 3.0, holding 1.0 | 211 | 195 | |
| **Empty-hand 3.0, holding 2.0** | **211** | **200** | **Candidate 3** |
| Same + carry 0.09 m/decision | 206 | 205 | More drops; not adopted |

Fresh hold-out (20 new mazes × 5 objects, built after the tuning above and used
once): **candidate 3 completed 80/100, 79/100 within 15k ticks**; the same models
with candidate 1's original settings completed 70/100, 58/100.

Carry retraining from candidate 1's BC initializer (same recipe: 40k decisions,
5k SAC updates, β=100), evaluated with candidate 3 settings:

| Carry policy | Complete /260 | Timed /260 | Verdict |
|---|---:|---:|---|
| Candidate 1 carry (seed 7401) | 211 | 200 | Reference |
| Same recipe, seed 7402 | 214 | 200 | Recipe reproduces across training seeds |
| Seed 7402 + Q-filtered demonstration loss (Nair et al. 2018) | 0/60 validation | 0 | Collapsed: critic preferred the actor in ~90% of demo states, so the demo anchor vanished; removed |

SAC ablation (candidate 3 settings, SAC reach throughout; grasp/carry swapped for
the imitation-only checkpoints SAC started from):

| Grasp | Carry | Complete /260 | Timed /260 |
|---|---|---:|---:|
| SAC fine-tuned | SAC fine-tuned (candidate 3) | 211 | 200 |
| SAC fine-tuned | Imitation only | 211 | 205 |
| Imitation only | SAC fine-tuned | 212 | 201 |
| Imitation only | Imitation only | 205 | 199 |

All four are within noise (±7): online SAC fine-tuning adds no measurable
performance for grasp or carry. Reach (pure SAC+HER, no demonstrations) is the
only skill whose competence comes from RL.

Planner-only baseline (classical reference: antipodal grasp, IK and planned
descent/lift, A* carry, base/torso reach; the reach now clamps the torso to its
joint range instead of aborting):

| Planner only | Complete | Timed | Rubric / trial | Carry drops |
|---|---:|---:|---:|---:|
| Course seeds, original speed | 63/100 | 0/100 | 24.2/40 | 33 |
| Course seeds, 3× empty-hand / 2× holding | 76/100 | 76/100 | 36.2/40 | 21 |
| Dev, 3× / 2× | 190/260 | 183/260 | 35.5/40 | 62 |

The learned system (candidate 3: 83/100 course, 211/260 dev) wins mainly by
dropping the object about half as often during carry; A* never fails to reach
the exit.

Imitation-only arm (`configs/imitation_only.json`) versus candidate 3 on all
evaluated sets: dev 205 vs 211, hold-out 81 vs 80, course 76 vs 83 — 362 vs 374
of 460 cases, about one standard error apart. SAC fine-tuning gives at most a
small gain.

RL plan step 1 — can carry motion reduce drops? Hand-written rules on the
imitation-only carry policy (`pnr/carry_rules.py`), 260 dev cases:

| Rule | Complete | Carry drops | Wall collisions | Carry timeouts |
|---|---:|---:|---:|---:|
| None | 205 | 41 | 0 | 8 |
| Turn limit 30°/decision | 193 | 38 | 23 | 1 |
| Turn limit 15°/decision | 80 | 20 | 156 | 0 |
| Half speed after 2 mm slip | 187 | 40 | 1 | 30 |
| Turn 15° + slip slowdown | 99 | 31 | 125 | 2 |

Neither smoother turning nor slowing once the object slips reduces drops (the
lower counts with a 15° limit only reflect episodes ending earlier in wall
collisions). With the earlier speed sweep, settle hold and tilted-posture
results, carry motion has little leverage on retention.

Which stage decides drops? A 2×2 swap on dev (`eval.py --grasp-mode/--carry-mode
planned`), imitation-only policies vs the classical controller's stages:

| Complete (carry drops) | Learned carry | A* carry |
|---|---:|---:|
| Learned grasp | 205 (41) | 177 (71) |
| Planned grasp | 209 (35) | 190 (62) |

The carry controller decides drops; the grasp barely matters. The A* carry moves
waypoint to waypoint (stopping at each, sharp corners, 18 mm per native call at
2× holding speed) and drops nearly twice as often as the learned policy's
continuous motion. Yet the step-1 rules (gentler turns, slowing on slip) did not
reduce the learned carry's drops further.

Drop analysis (saved traces and repeated runs):

- Lost objects slide down 10–35 mm relative to the base before contact is lost;
  successful carries drift under 1 mm. A stationary grasp holds, so the slip
  is triggered by base motion and pulled down by gravity.
- Route length does not predict drops. The same grasp proposal is chosen for an
  object in every maze, and repeated runs show drops are partly random: across
  five candidate-1 runs, 139 cases never dropped and only 8 dropped four or more
  times.
- Slip measured during the planned retreat weakly flags later drops (12/24 drops
  vs 17/69 successes at −1 mm), too weak for a useful re-grasp rule.

## 2026-10-01: dexterous-hand feasibility spike

Allegro hand vs Fetch gripper on a gantry that reproduces the carry motion
([hand_feasibility.md](hand_feasibility.md)). Scripted Allegro power grasps retain
20/50 vs Fetch 40/50 over 10 recorded routes; after a 5,400-trial robustness search,
14/25 vs 18/25 on unseen routes. No-go: the hand does not hold better in PyBullet.

Still-hold diagnosis (same rig, object held motionless for 2,000 commands after the lift, 20 trials per hand):
Fetch 19/20 held, median penetration 1.5 mm, drift 2.0 mm, jitter 0.42 mm/s, 489 N total normal force;
Allegro 15/19 held, 3.1 mm, 7.5 mm, 1.20 mm/s, 374 N. The grasp fails without any carry motion, so the
contacts are unstable on their own. Likely contributors on our side: the course's finger control reused for the
hand (targets deep inside the object, 2e6 N force cap, no damping: four fingers pushing against each other give
the solver conflicting constraints) and default contact parameters. Tuning them was skipped in favour of the
MuJoCo port, which tested the simulator with the gripper first.

## 2026-10-01: MuJoCo backend and planner-only replication

`pnr/mj_env.py`, same interface as the PyBullet environment. Steps that mattered:
- Base servo saturating at a force cap overshot (14 mm in one tick vs 4.5 mm);
  uncapped stiff velocity servos then hit MuJoCo's instability reset on the light
  upper-arm roll joint (scene silently reset to its spawn state).
- Inertia-scaled implicit velocity servos tracked perfectly in free motion but the
  constraint solver ignores implicit damping: fingers closed through the objects
  (contact forces 10–100 kN). Explicit computed torque, τ = M(v_cmd − v)/dt + bias,
  fixed it; finger armature 4.09 calibrates the blocked grip to PyBullet's.
- The object's MjSpec inertial origin defaults to unset (NaN) and compiled 1 m
  above the mesh; the banana rolled off the table during the first motions.
- Friction: default pyramidal soft friction 4/20 planner-only cases; noslip 14/20;
  elliptic + impratio 10 17/20; both 19/20 (PyBullet 15/20 on the same cases).
- Full sets: course 96/100 (PyBullet 76), dev 250/260 (190); rubric 39.2/40 (36.2).
  Grip slips at the same external load in both engines (~110 N ≈ μN).

## Main conclusions carried forward

1. Navigation generalises to unseen mazes: only 3–6% of unseen cases time out.
2. Most failures (24/31 on the confirmation test) are the object slipping during
   carry. The native position controller produces a large acceleration pulse at
   every command; slower carry retains objects better but costs time.
3. Online SAC adds little over the BC initializer for carry and often degrades it
   with more training (learning rate 3e-5, one update per 8 decisions, β=100).
