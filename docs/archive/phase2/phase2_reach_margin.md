# Internal reach handover margin

`LearnedSystem(..., reach_tolerance_m=.08)` keeps the reach policy active until
the measured object-to-goal distance is strictly below 8 cm and native task
success, current bilateral contact, and zero drops all hold. This gives the
following stationary hold some room for drift. The final criterion remains the
original native 10 cm goal test after all 60 native hold calls. The 4 cm legacy
and 10 cm native handover modes retain their previous behavior and the default
remains 10 cm. Training rewards and all physics are unchanged.

The evaluator accepts `--reach-tolerance-m .08` only in a frozen v4 evaluation
plan. Earlier plan versions retain their original allowed settings. Changing
the margin invalidates the plan digest until a new plan is explicitly frozen.

## Bounded validation comparison

The four cases are `validation_000` through `validation_003`, object 2 (tomato
can), from the frozen phase-2 scene bank. Cases 000 and 001 are successful
controls; cases 002 and 003 previously completed navigation and reaching, then
drifted beyond 10 cm during the final hold. This is a targeted development
comparison, not an estimate of unseen-scene performance.

The comparator is `runs/phase2_system_recovery_bc01`; the margin run is
`runs/phase2_system_reach_margin08_01`. Both use the exact same frozen checkpoint
bytes: selected regularized SAC grasp, grip-aware recovery BC navigation, and
the original SAC+HER reach policy. The original-start execution uses legacy
grasp proposals, a 24 cm lift subgoal, navigation delta 0.075, planned speed
scale 1, and grasp/carry/reach horizons 600/1200/150. There are no resets between
stages. Every attempt and its costs remain in the four-case denominator.


| Validation case | Prior 10 cm handover | New 8 cm handover | Final error, prior → new | New ticks: actions / including initialization |
| --- | --- | --- | --- | --- |
| 000 | Pass | Pass | 9.767 → 7.950 cm | 13,995 / 14,475 |
| 001 | Pass | Pass | 8.561 → 6.870 cm | 12,140 / 12,620 |
| 002 | Final distance failure | Pass | 10.033 → 6.502 cm | 12,835 / 13,315 |
| 003 | Final distance failure | Pass | 10.380 → 7.882 cm | 15,410 / 15,890 |

All four new attempts completed the mandatory 60-call hold with bilateral
contact and zero drops. Full success improved from 2/4 to 4/4; success within
15,000 ticks, including initialization, improved from 2/4 to 3/4. Case 003
already exceeded the deadline in the comparator and remains late. The four new
attempts cost 10,876 native action calls / 54,380 action physics ticks, plus
1,920 initialization ticks: **56,300 total physics ticks**. Detailed results,
source hashes, and both runs' stage costs are in
`runs/phase2_system_reach_margin08_01/artifacts/margin_comparison.json`.

There is a reproducibility limitation in the paired comparison. Case 003's
physical trace matches through the original reach handover; three extra reach
decisions (45 ticks) directly change its final error from 10.38 to 7.88 cm.
The other cases show small numerical trace differences during grasp despite
identical checkpoint bytes, case specifications, and native physics source.
Case 000 uses two additional planned carry-posture calls; case 002 uses 45
fewer grasp calls and three fewer carry-posture calls. The cause of that
numerical divergence is not established. Their total-tick changes therefore
cannot be attributed solely to the margin. Navigation call counts match in all
four pairs. Each run archives its exact source files, including intervening
proposal-mode plumbing that leaves the selected legacy mode unchanged.

This bounded result supports using the declared 8 cm internal margin in the
next full development evaluation. It does not replace the previous 60-case
score, establish a new held-out score, or address navigation/retention failures.

## Verification

The 45 targeted tests in `tests/test_reach_completion.py` and
`tests/test_phase2_system_evaluation.py` pass. They cover strict 8 cm handover,
continued control between 8 and 10 cm, native pickup/contact/drop checks, all
60 final hold calls, the final native recheck, unchanged legacy behavior, and
the frozen v4 setting. A retained object at 9 cm after the hold still passes
the unchanged final 10 cm criterion.
