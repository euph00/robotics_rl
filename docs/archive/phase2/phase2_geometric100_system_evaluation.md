# Geometric-recovery SAC: full-task validation

The stronger demonstration-regularized SAC candidate reaches35/60 full-task
successes, versus42/60 for the recovery BC control. It improves on the earlier
SAC candidate's31/60, but does not displace BC as the best development control.

`runs/phase2_system_geometric100_08_01` evaluates all12 validation layouts × five
objects from the original start. The frozen navigation checkpoint is
`phase2_carry_geometric100/checkpoints/best_online.zip`, selected at12,000 online
decisions after10/12 isolated carry validation successes. SHA256:
`67781361c0055ede19b91b0270db9ffa45e559a241fb3d649b8337e0bfad10db`.

The grasp/reach checkpoints and execution settings exactly match
`phase2_system_recovery_bc08_01`: selected legacy-proposal regularized SAC grasp,
24cm lift,0.075 navigation scale, planned speed1,8cm reach handover, and native
10cm final goal after60 hold calls. Six CPU workers ran all60 attempts. No held-out
partition was evaluated. Exact measured traces match in all60 pairs through the
start of learned navigation. Only archived training/scene utilities changed
(`demonstrations.py`, `sac_initialization.py`, `scenes.py`); physics, controller,
and feature-extractor inference sources match.

| Metric | Recovery BC control | Geometric-recovery SAC |
| --- | ---: | ---: |
| Full physical success | 42/60 (70.0%) | 35/60 (58.3%) |
| Within15,000 action physics ticks | 41/60 | 32/60 |
| Within15,000 ticks including initialization | 38/60 | 32/60 |
| Successful grasp/pick hold | 60/60 | 60/60 |
| Successful carry posture | 59/60 | 59/60 |
| Navigation completed | 42/60 | 35/60 |
| Carry grip loss | 10 | 16 |
| Carry decision limit | 5 | 8 |
| Carry collision | 2 | 0 |
| Planned carry-posture grip loss | 1 | 1 |

All35 completed SAC transports also pass reach and the final hold. Per-object
full successes, each out of12:

| Object | BC | SAC |
| --- | ---: | ---: |
| Foam | 10 | 8 |
| Banana | 9 | 9 |
| Tomato can | 7 | 5 |
| Apple | 6 | 5 |
| Mug | 10 | 8 |

Paired outcomes:31 both succeed,14 both fail,4 SAC-only,11 BC-only. SAC-only wins
are banana003 plus foam/banana/mug011. The three layout011 wins exceed the
15,000-tick limit:16,830/17,065/16,970 ticks including initialization. In contrast,
SAC now completes all five layout003 objects on time, including four BC cases
that were late once initialization was counted.

The11 BC-only cases are can/mug004,foam005,can006,foam/banana/apple/mug008, and
foam/banana/mug010. Five now lose grip; six reach the decision limit. Both
policies still fail all five objects on007. The new SAC candidate also fails
all five on008 and010, from a mixture of retention loss and stalled navigation.

The new60-case batch costs184,695 native action calls /923,475 action physics
ticks plus28,800 initialization ticks: **952,275 total physics ticks**. Every
failed preparation and physical attempt stays in the60-case denominator.
Successful cases have median13,195 action ticks and median final error7.32cm;
these conditional medians do not include failures.

The simple validation matrix is registered on the run's dashboard entry.
`artifacts/matched_bc_comparison.json` contains all paired changes, per-case
failure reasons, timing and complete costs; `carry_selection.json` archives the
selection metadata. The result supports continuing retention and training-state
distribution work alongside navigation recovery, rather than promoting this
checkpoint on its isolated carry score alone.
