# Transport-qualified grasp refinement

**Rejected: retained transport fell from 18/20 to 15/20.**
The fixed imitation refinement produced one improvement and four regressions.
Both policies completed all 20 grasp preparations. No full-task validation or
held-out test followed; this hypothesis is closed without checkpoint retuning.

| Object | Existing grasp /4 | Refined grasp /4 |
|---|---:|---:|
| Foam brick |4|3|
| Banana |4|4|
| Tomato can |3|3|
| Apple |3|2|
| Mug |4|3|
| Total |18/20|15/20|

The lift reward does not measure later retention. We therefore collected 120
original-start TRAIN preparations: first 8 layouts × five objects × three
predeclared deterministic/stochastic grasp variants. Each used the same planned
carry posture, geometric transport teacher at 0.075 m, existing goal hold and 50
additional native hold calls. This was training data, not learned navigation
or full-task evaluation. Every attempt and failure was retained.

Of 120 attempts, 116 reached the carry handover and 88 survived transport and hold:
foam 16/24, banana 22/24, can 15/24, apple 15/24, mug 20/24. Failures were 28 carry losses,
two posture losses and two grasp horizons. The collection retained 6,425 actual
grasp transitions; 3,881 qualified for imitation. Actual observations, actions,
native rewards and ending flags were unchanged. Caller handover prefixes can
remain natively nonterminal and were never inserted into HER replay.

The actor received exactly 2,000 supervised updates of 400 examples, half from
existing grasp demonstrations and half from qualified prefixes, sampling objects
uniformly. Fresh Adam used 3e−4, seed 10101 and gradient clipping 10. No additional
SAC updates or physics occurred during fitting. Critics, target critics,
log-standard-deviation head, temperature and historical 12,000/2,000 SAC counters
remained unchanged. Shared actor layers may change conditional standard deviation;
deployment was deterministic. Only the fixed endpoint was evaluated.

The comparison used 20 TRAIN008–011 cases, each with one original start per policy.
These cases may appear in ancestor training and are not unseen-test evidence.
All 40 outcomes and costs passed independent audit; all 20 initial recorded grasp
inputs matched, without establishing complete hidden-state identity.
Collection cost 1,502,725 ticks; comparison cost 522,190; total **2,024,915 physics
ticks**, including initialization. Dataset SHA256:
`998437ab00a43c6d6835d2c32bc2a971f4ca924eefaff5bac8171c7bab9181d1`.

The prospective gate required three net wins and no preparation regressions.
Afterward, the observed 18/20 control revealed that at most two wins were possible.
The actual −3 result fails any improvement gate, so this design flaw does not
change the rejection. Future hypotheses need a feasible gate fixed before fitting.
The earlier conditional Plan02 remains skipped; this completed Plan03 was a
separate development phase after the failed 69/100 confirmation.

[Frozen Plan03](../runs/phase2_grasp_qualified_plan03/artifacts/plan.json) ·
[Collection audit](../runs/phase2_grasp_qualified_demos03/artifacts/independent_data_audit_scene_bank.json) ·
[Fixed endpoint](../runs/phase2_grasp_qualified_bc03/artifacts/result.json) ·
[Paired outcomes and audit](../runs/phase2_grasp_qualified_compare03/artifacts/independent_endpoint_pair_audit_scene_bank.json).
