# Carry retention: measured-state audit

Omitted object motion and orientation contain a plausible warning signal, but
these data do not establish that actor masking causes grasp loss. They support
a small controlled feature ablation, not promotion of a new encoder.

The current spatial carry actor retains nine grip values: two finger positions,
three hand-to-object offsets, two base velocities and two contact flags. It masks
the other 48 physical values; the critics retain them. The audit verified the
503,901-transition merged TRAIN dataset, then analyzed its 74 learned-handover
episodes (25,538 transitions). Twelve end in physical grasp loss: three foam,
four can, four apple and one mug. Six zero-transition restore failures remain in
the collection accounting but cannot supply predictor observations.

All fitting uses eight leave-one-TRAIN-layout-out folds. Both attempts and all
objects from a layout stay together. A fixed regularized logistic model uses
training-only normalization and equal episode/class weights. Its label is loss
within the next ten decisions. Features use only the stored current observation;
actions, future states and post-action info measurements are excluded. The final
nine successful-episode states are censored because their ten-step outcome is
unobserved. There is no simulator, policy training or validation-state access.

Added features are measured object linear velocity minus planar base velocity
(three values), object angular velocity (three), and hand-relative object rotation
(six continuous matrix values). Base-relative velocity is an approximation to
hand-relative motion: exact hand velocity is absent from the observation.

| Predictor features | All-state average precision | Bilateral, 3–10 decisions before loss |
|---|---:|---:|
| Current compact nine | 8.30% | 1.44% |
| + linear/angular velocity | 9.05% | 3.15% |
| + relative orientation | 14.01% | 6.42% |
| + both | 12.96% | 3.93% |

The two columns contain 24,980/24,956 states and 120/96 positive labels; positive
prevalence is only 0.48%/0.38%. These correlated states represent 12 events, not
120 independent failures. Adding both feature groups improves early ranking on
six of seven layouts with failures, but early ROC-AUC barely changes (0.703 to
0.710). At thresholds fixed from each training fold's negative 99th percentile,
early event alerts increase from 1/12 to 4/12 while held-layout false alarms rise
from 0.20% to 2.71%. The descriptive layout-resampling interval for the all-state
average-precision difference spans zero (−0.009 to +0.144). These results do not
establish a reliable operating threshold.

Nine failed attempts have a successful alternate attempt from the same original
case. Comparing the same decision indices 3–10 steps before failure, while both
attempts still have bilateral contact, yields:

| Median across paired case means | Failed attempt | Successful attempt |
|---|---:|---:|
| Base-relative object speed | 0.043 m/s | 0.0010 m/s |
| Object angular speed | 0.91 rad/s | 0.063 rad/s |
| Already-visible hand/object offset | 37.5 mm | 21.6 mm |

Speed and visible offset are larger in all nine failed attempts; angular speed
is larger in eight. Thus motion can reveal developing slip before contact loss,
but the existing actor also has a useful displacement signal. Alternate attempts
use different action noise and restored contact histories; they are descriptive
controls, not paired causal interventions. No banana failures and only one mug
failure further limit generalization. A linear predictor also cannot represent
the nonlinear policy encoder's full capacity.

The smallest supported next test, if a new actor branch is warranted, is an
explicit ablation adding these 12 measured values while keeping spatial features,
controls and training data fixed. Preserve the current production encoder until
that comparison demonstrates improved physical retention. Prediction alone
cannot show which action prevents loss, and it does not justify a speed rule.

Reproducible script, frozen audit protocol, fold predictions, source hashes and
paired descriptions are in `runs/phase2_carry_observability_audit01/artifacts/`.
No new physical calls or actor updates were made.
