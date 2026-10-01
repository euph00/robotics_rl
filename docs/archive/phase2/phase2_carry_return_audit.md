# Carry reward versus reliable completion within 15,000 ticks

The recorded returns do **not** reward an actual drop over a successful carry.
The objective does, however, trade reliability against speed even when both
options could satisfy the deadline. This is a plausible objective mismatch, not
evidence that it caused the learned policy's failures. No physics, training,
reward settings, or success criteria were changed for this audit.

The implementation uses gamma 0.995, ordinary cost −1 per carry decision until
retained contact is within 15 cm of the goal, and terminal physical-failure cost
−200 = −1/(1−gamma). A normal decision executes 25 physics ticks; failures may
interrupt it. The reward also includes `2*(distance_before − gamma*distance_after)`,
with the after-distance set to zero on an absorbing physical failure.
For an N-step recorded trajectory, its exact discounted reward sum is therefore:

`G = discounted ordinary/failure costs + 2*d_start − 2*gamma^N*d_end`,

where `d_end=0` for a physical terminal. The identity agrees with all 409 audited
TRAIN episodes within 0.0000053, consistent with float32 stored rewards. The
potential term is a known start/end adjustment; it does not create a preference
for early dropping. A drop before any goal visit has adjusted return **−200
regardless of when it occurs**.

| Recorded TRAIN source | Carry successes / physical failures | Successful decisions | Successful adjusted return | Failed adjusted return |
|---|---:|---:|---:|---:|
| Diverse teacher collection | 175 / 55 | 295–441 | −177.515 to −153.256 | −200 |
| Earlier clean/noisy teacher collection | 62 / 12 | 295–454 | −178.933 to −153.256 | −200 |
| Completed online episodes, original balanced run through 40k | 91 / 14 | 299–631 | −191.325 to −154.184 | −200 |

All 328 recorded successes also exceed all 81 failures in their *unadjusted*
reward sums in this particular sample. On the 42 canonical TRAIN cases containing
both outcomes, the smallest adjusted success advantage is 8.675. These are
recorded episodes, not independent estimates of policy reliability; different
preparations can have different grasp seating. Online reconstruction excludes
1,795 seeded demonstration episodes and 2,028 unfinished online transitions,
leaving 37,972 completed online decisions. Failed preparations have no carry
trajectory and are not assigned a fictitious return.

For a simplified strategy that succeeds with probability p after m costly
non-goal decisions and otherwise drops before reaching the goal, its adjusted
expected return is `−200 + 200*p*gamma^m`. A faster strategy wins when
`p_fast/p_slow > gamma^(m_slow−m_fast)`. Saving 20 decisions (500 ticks) can thus
outweigh reliability falling from 100% to 90.46%; saving 100 decisions (2,500 ticks)
can outweigh a fall to 60.58%. The discount half-life is only 138.28 decisions
(3,457 ticks), while 600 decisions receive weight 0.0494.

A **mathematical example, not a measured intervention**: 80% success after 300
costly decisions scores −164.433, better than certain success after 400 decisions
at −173.068. Including five zero-cost goal-hold decisions and an assumed equal
3,000 ticks for the other stages gives total times 10,625 versus 13,125 ticks;
both satisfy 15,000. The metric would prefer the reliable option, while this
carry reward prefers the faster risky option.

There is also a measured deadline mismatch: online episode `train_002_object_02`,
worker 4, replay start 67,172, completes the carry goal in 631 decisions / **15,775
carry ticks**, and receives success reward despite already exceeding the whole
task deadline before adding grasp, reach, and final hold. Training's carry horizon
is 900 decisions; its state/reward does not contain the remaining full-task time.
Carry success also excludes downstream reach/hold failures. Successful collection
ends as a timeout for HER, so critics bootstrap; finite trajectory sums above
must not be mistaken for complete soft critic targets. SAC entropy and ongoing
imitation regularization additionally affect the trained actor.

If the current diversity/control work still leaves a speed–retention concern,
one controlled option is a paired **gamma 0.999** carry branch with the matching
failure cost **1,000**, keeping physics, actions, starts, actor initialization,
regularization, and training budget fixed. For the 100-decision example its
reliability threshold becomes 90.48%; the reliable hypothetical strategy wins.
This is a discount/risk ablation, not a full deadline solution: longer horizons
can encourage slower task completion, and the critic value scale increases.
Select by frozen full-task success *within 15,000 ticks*, not carry return alone.

Such a branch needs fresh critics and replay reseeding with recomputed rewards
for **every** stored physical transition. Both real-goal and HER rewards must use
the new gamma, failure cost, and potential term; old saved rewards/replay are
incompatible. The same genuine observations/actions, failures, terminal masks,
and timeout boundaries remain usable, with explicit new reward metadata and
checks. Merely raising gamma while retaining failure cost 200 can make dropping
cheaper than prolonged −1-per-step motion and is not the proposed option.

Audit artifact: `runs/phase2_carry_return_audit01/artifacts/audit.json` contains every
episode return, exact ranges, source hashes, exclusions, and counterfactuals;
`audit_source.py` reproduces the read-only calculation. Input SHA256 values:

- Diverse teacher dataset: `2a8d2b9cca53fcbbb22334357f36d85d2ae2fa61df02f654b69749469c36b8a4`.
- Earlier teacher dataset: `f5831619dd05b17a88bb7b1f41d628c8a16b9c4b1f0622f1642507288210ae7f`.
- Online replay at `phase2_carry_learned_balanced01/checkpoints/step_000040000_1790592316309877827/replay.pkl`: `fb410ab3da41e2b0afd15a86bbbb4aad3ef1399034b2dea5dbe7924ae91bc026`.

No held-out data were accessed. These observations support checking objective
alignment; they do not establish why any particular policy dropped its object.
