# Revised-policy confirmation test

**69/100 complete tasks; the 80/100 milestone remains unmet.**
53/100 completed within 15,000 physics ticks including settling and the final
hold; the action-only timed count was also 53/100. Every case ran once, with no
rerolls or replacement failures.

| Original object | Complete /20 | Within 15k including settling /20 |
|---|---:|---:|
| Foam brick 14 |15|11|
| Banana 5 |19|15|
| Tomato can 2 |10|8|
| Apple 7 |13|9|
| Mug 10 |12|10|
| Total |69/100|53/100|

The five supplied objects were crossed with 20 new maze layouts. The bank was
fixed before revised training and excludes all 244 parent-bank maps and 100
legacy maps. All checkpoint bytes, execution settings, 100 cases and 41 source
files were frozen and independently checked before the first action.

The revised carry policy keeps its imitation-trained map core fixed while SAC
updates the remaining actor weights, independent critics and temperature. It
inherited 16,000 BC updates, then ran 40,000 online decisions and 5,000 SAC updates
with HER and continuing demonstration regularization. The grasp and reach
policies remain unchanged. Planned approach, carry posture and IK are declared
assistance; original gripper/physics and native final success criteria remain
unchanged. There is no navigation teacher or inter-skill reset at evaluation.

The candidate's 52/60 validation completions narrowly beat the previous 51/60
under the completion-first selection rule, despite a worse timed count of 38 versus 47.
That validation improvement did not establish better unseen-maze performance.
The earlier candidate scored 78/100 on a different test bank; these two test
scores are not a paired comparison or a causal estimate of the map-core change.
Repeated reuse of the small validation bank also limits confidence in small gains.

There were 24 carry grasp losses, three carry horizons, one planned-posture loss,
one unstable final hold, one reach drop and one grasp horizon. All 69 successes
passed native 10 cm acceptance and the mandatory 60-call retained hold.
Independent auditing found no missing cases or inconsistent costs. Total cost:
271,882 native calls, 1,359,410 action ticks and 48,000 settling ticks, or
**1,407,410 physics ticks**. The 4,200 final-hold calls include a failed hold and
are already included in those totals.

This confirmation is preserved as a historical result and retired from further
held-out claims because it now informs development. Neither test bank's
trajectories enter training. The final-study reserve stays sealed. The next
bounded investigation targets grasps that retain objects through actual TRAIN
transport; it does not count teacher-assisted transport as full-system success.
The earlier conditional grasp collection was skipped as planned; any successor
requires a new recorded protocol.

[Revised candidate](../configs/phase2_candidate02.json) ·
[Frozen evaluation plan](../runs/phase2_confirmation_test_plan01/artifacts/evaluation_plan.json) ·
[All outcomes](../runs/phase2_system_confirmation_test01/artifacts/summary.json) ·
[Independent audit](../runs/phase2_system_confirmation_test01/artifacts/independent_result_audit.json) ·
[Outcome matrix](../runs/phase2_system_confirmation_test01/artifacts/confirmation_matrix.png).

Five predeclared validation recordings show the revised policy on the first
validation maze: banana, can and mug complete; foam and apple drop during carry.
All five match their original validation traces and costs exactly; all ten
video files decode successfully. These are diagnostics, not additional unseen
cases. [Recording set](../runs/phase2_candidate02_recordings01/artifacts/summary.json).
