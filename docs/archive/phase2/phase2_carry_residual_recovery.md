# Remaining carry failures after entrance recovery data

Both zero-online policies (`phase2_carry_recovery01` and `phase2_carry_navigation01`) complete 10/12 validation carry cases. The two residual failures have different causes. They were inspected only to diagnose the failure class; no validation states or labels enter training.

| Case | Recovery actor | Navigation-only actor | Interpretation |
| --- | --- | --- | --- |
| `validation_007` | Enters maze by decision 84; 805 decisions near x=4.5; 28 vertical reversals | Enters by 87; 594 decisions near x=4.5; 28 reversals; loses grasp at 691 | Wrong first interior branch; needs backtracking |
| `validation_011` | Never enters; 849 decisions near x=1.5, y=0.33–4.33 | Never enters; 849 decisions near x=1.5, y=0.37–4.32 | Wrong side of the entrance room; gate is y=−3.5 |

Median movement remains 0.0717 m per decision. The policies repeatedly choose the wrong route rather than suffering slow physical actuation. Navigation-only features did not remove these two route failures.

A scan of the actual 663 merged TRAIN episodes found:

- 174,865 successful BC rows, including 133,868 inside x>3.5. **None** of these interior rows has expert x-action below −0.5: strong westward backtracking has no supervised example.
- All 307 successful recovery episodes start outside the maze; the collector only varies entrance-room starts.
- Six TRAIN layouts share entrance y=−3.5. Their 1,001 entry-strip rows near x=1.5 cover y=−4.25 to−1.00034, with **zero** rows at y>0. The residual learner visits lie beyond this support.

Offline geometry confirms that the trapped branch in validation007 requires returning west from x=4.5 to x=3.5 before heading north. This geometric diagnosis is not an inference-time controller or new training example. Reusing a forward reference route and projecting onto its nearest segment is not sufficient to label arbitrary learner-induced branches safely.

The proposed next data collection is an explicitly labeled **DAgger-style dataset aggregation** step: query a training-only expert at states visited by a frozen learner, aggregate the labels, refine BC, then continue the declared SAC+HER+demonstration-regularization method. It does not become ordinary SAC, and later TD3 comparison should receive the same dataset and collection budget.

Minimal extension, before any physical execution:

1. Add an explicit on-policy recovery collector mode with a hashed immutable learner checkpoint. Use only declared TRAIN layouts, the normal physically prepared original carry start, and the original exit goal throughout.
2. Execute learner actions for at most 450 decisions. For each visited state, query `maze_astar` from measured current XY to the fixed goal, and use `route_action` for a separate expert label. The current planner supports these endpoints without advancing physics. Refuse and record states for which a valid expert route is unavailable; do not fabricate labels or relax collision clearance.
3. Use a generic, declared takeover rule: switch once to expert execution after a 60-decision window without at least 0.5 m improvement in training-only geometric path distance, or at the actor budget. The detector and geometric distance exist only in data collection. A fixed-prefix baseline is also possible, but a stalled-progress trigger focuses cost on learner-induced errors without using validation coordinates.
4. Continue physically from the exact live state, with no reset at takeover. Keep at most 900 total decisions. Every transition contains the actual executed `action`, separately computed `teacher_action`, reward, following observation, and original termination/timeout. Preserve failed episodes and all costs. Initially keep the existing conservative exclusion of failed episodes from BC while retaining them in replay.
5. Mark the actor prefix and the expert recovery phase explicitly. The existing `carry_supervision` prefix criterion only recognizes entrance-room recovery until x>3.5; it needs a separate supervision group for learner-induced corrective states anywhere in the maze. Otherwise rare interior backtracking may be diluted by 175k existing examples. Do not relabel interior DAgger data as the old entrance-recovery mode.

Preflight: first four TRAIN layouts, one rollout from each of the two frozen zero-online actors, at most eight episodes. Using two distinct checkpoint hashes avoids duplicating deterministic same-start trajectories. Check transition continuity, reward recomputation, action/label separation, physical retention, source lineage, and failure accounting. Proceed to the 64 TRAIN layouts only after that result is reviewed. If neither frozen learner visits informative failures on the small preflight, expand TRAIN coverage; do not select starts from validation traces.

Cost estimate: historical recovery collection achieved about 180 native calls per worker-second. A geometric-only benchmark on the first 16 TRAIN layouts measured A* queries at 30 ms median and 90 ms maximum. With six CPU workers and CPU inference, budget roughly **2–5 minutes for the four-layout preflight** and **15–30 minutes for 64 layouts × two actors**, depending on takeover timing and route lengths; roughly 0.3–0.6 million native calls for the larger batch. These are estimates, not completed runs. No GPU is needed.

Evidence: `runs/phase2_carry_residual_audit01/artifacts/result.json`, generated by `scripts/phase2_carry_residual_audit.py`. It records immutable zero-online evaluation hashes and the TRAIN shard list. All analysis was offline: **zero new physics ticks and zero gradient updates**. No collector, environment, reward, or production training code was changed for this audit.
