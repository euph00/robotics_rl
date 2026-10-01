# Matched BC and SAC navigation in the full task

The selected SAC navigation continuation does **not** improve the full system:
31/60 validation successes versus 42/60 for its BC initialization. Every SAC
success also succeeds with BC. Keep the BC policy as the stronger development
control while refining SAC training; retain the SAC result as the measured
all-learned-skill candidate, with the disclosed planned stage transitions.

## Controlled protocol

Both runs cover all 12 validation layouts × the same five objects from the
frozen scene bank. They use the same selected regularized SAC grasp checkpoint,
the same SAC+HER reach checkpoint, legacy grasp proposals, a 24 cm lift subgoal,
0.075 navigation action scale, planned speed scale 1, and the new 8 cm internal
reach handover margin. The final native 10 cm check follows all 60 hold calls.
Each attempt starts at the original scene, with no inter-skill reset, attachment,
or waypoint teacher during policy navigation. All setup and physical failures
remain in the 60-case denominator. These are development validation results;
sealed milestone-test and final-reserve layouts were not evaluated.

- BC run: `runs/phase2_system_recovery_bc08_01`.
- SAC run: `runs/phase2_system_recovery_sac08_01`, navigation selected after
  4,000 online decisions by the existing isolated carry validation protocol.
- Exact source hashes, case specifications, execution settings, grasp bytes and
  reach bytes match between frozen plans. All 60 measured physical traces and
  stage call counts match through the start of learned navigation.
- Carry SHA256: BC `ddcb965a03b8329520b7adfe4f20f136c2c011c87ed0364a380661f558e6b3f5`;
  SAC `9463ef90e4c91a48cff8ffea210740af70037c3c095f00c51b1c458ec18b72dc`.

## Results

| Metric | BC navigation | SAC continuation |
| --- | ---: | ---: |
| Full physical success | 42/60 (70.0%) | 31/60 (51.7%) |
| Success within 15,000 action physics ticks | 41/60 | 31/60 |
| Success within 15,000 ticks including initialization | 38/60 | 29/60 |
| Successful grasp and 50-call hold | 60/60 | 60/60 |
| Successful planned carry posture | 59/60 | 59/60 |
| Successful navigation | 42/60 | 31/60 |
| Successful reach and final hold | 42/60 | 31/60 |
| Carry grip loss | 10 | 17 |
| Carry decision limit | 5 | 10 |
| Carry collision | 2 | 1 |
| Planned carry posture grip loss | 1 | 1 |
| Median action ticks, successful cases only | 13,452.5 | 13,400 |
| Median final error, successful cases only | 7.52 cm | 7.08 cm |

| Object, 12 cases each | BC full success | SAC full success |
| --- | ---: | ---: |
| Foam | 10 | 5 |
| Banana | 9 | 8 |
| Tomato can | 7 | 5 |
| Apple | 6 | 5 |
| Mug | 10 | 8 |

Paired outcomes: **31 both pass, 18 both fail, 11 BC-only, zero SAC-only**.
The 11 losses are foam on layouts 002/004/005/008/009; banana on 004; can on
004/009; apple on 009; mug on 004/010. Of these, six lose grip, four reach the
navigation decision limit and one collides. No case that completes navigation
fails the reach or final hold in either run.

Both policies still fail every object on layouts 007 and 011. SAC additionally
gets stuck on layout 004 for banana and on layout 009 for foam/can/apple. In the
last 1,000 recorded native navigation calls, those layout-009 cases remain near
world XY (4.55, −4.43), while their paired BC cases reach the exit. These are
observed policy traces, not planner-based inference.

## Timing and complete costs

BC has four successful cases exceeding 15,000 ticks including initialization,
all on layout 003: foam 15,420; can 15,890; apple 15,410; mug 15,440. Only the can
also exceeds 15,000 action ticks (15,410). SAC has two successful cases exceeding
the limit including initialization: layout-003 can 15,230 and apple 15,425.
Every scene contributes the original 480 initialization physics ticks.

| Cost, including every attempt | BC | SAC | Both |
| --- | ---: | ---: | ---: |
| Native action calls | 173,423 | 193,922 | 367,345 |
| Action physics ticks | 867,115 | 969,610 | 1,836,725 |
| Initialization ticks | 28,800 | 28,800 | 57,600 |
| Total physics ticks | 895,915 | 998,410 | 1,894,325 |

All per-case failures, stage costs, paired changes and timing details are in
`runs/phase2_system_recovery_sac08_01/artifacts/matched_bc_comparison.json`.
The accompanying simple object comparison figure is registered on that run's
dashboard entry. Each run also has its complete case matrix and frozen plan.

## Next step

Keep the declared 8 cm margin: it yields no downstream reach/hold failures in
these 73 successful transports. Prioritize improved grasp retention and TRAIN
recovery examples for turns and missed branches, then assess the next SAC
continuation against the same BC control. The present isolated carry selection
score was insufficient to predict full-task quality across these five payloads;
full-system validation must remain a promotion criterion. The 42/60 result is a
BC-navigation control, not evidence that SAC navigation has reached that score.
