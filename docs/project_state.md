# Project state

The original Fetch system completed **78/100 tasks on 20 unseen maze layouts**,
with 63/100 completed within 15,000 physics ticks including settling and final
hold. The 80/100 milestone is **not met**; refinement continues within the agreed
investigation window ending 2026-09-29 05:53:20 UTC.

[First unseen-maze result](phase2_test01.md) ·
[Selected candidate](../configs/phase2_candidate.json) ·
[Dashboard](http://127.0.0.1:8766).

## Current system

Three policies perform grasp, carry and reach. Grasp and carry use
BC-initialized SAC+HER with continuing demonstration regularization; reach uses
ordinary SAC+HER. Selected online decisions are 12k/40k/15k respectively.
Planned approach, carry posture and IK are explicit assistance. The original
gripper, native physics, goal acceptance and retained hold remain unchanged.
There is no learned-navigation route oracle, teacher correction, object weld or
inter-skill reset at evaluation.

The best validation candidate scored 51/60 complete and 47/60 within 15k ticks
including settling. Extending it to 200k decisions scored 48/60 and 47/60; training
with 240 varied grasp preparations scored 49/60 and 48/60. All outcomes and costs
remain recorded. [Training comparisons](phase2_diverse_carry_preparation.md).

## Evidence and remaining work

The first 100-case test used five supplied objects crossed with 20 distinct mazes.
Its 22 failures were 14 carry grasp losses, six navigation horizons, one posture
loss and one reach drop. An independent audit reconciled all 100 outcomes and
1,504,190 physics ticks. Full success and deadline success are separate metrics.

The first test is now a historical result and will not be reused for held-out
promotion after further development. Its trajectories are not training data.
Training and validation records remain frozen; a new confirmation bank was
fixed before revision, while the separate final-study reserve remains untouched.

The [carry-discount experiment](phase2_discount_trial.md) did not improve
validation: 45/60 and 44/60 complete at 40k and 80k decisions. It is not promoted.
The [matched navigation-network experiment](phase2_encoder_screen.md) reached
51/60 complete and 39/60 within the deadline with the gated encoder, versus
42/60 and 34/60 for its fresh CNN control. Its imitation-only initializer already
scored 50/60 and 43/60, so the contribution of SAC remains small and mixed.
Continuing the gated branch regressed to 39/60 and 29/60 at 80k decisions,
then 44/60 and 35/60 at 120k. Broader grasp experience scored 47/60 and 40/60.
A [minimum demonstration replay share](phase2_demonstration_floor_replay.md)
scored 47/60 and 35/60 at 40k, then 46/60 and 38/60 at 80k; it is not promoted.

The [pretraining audit](phase2_carry_bc_curriculum.md) found that the historical
best inherited 60k BC updates, versus 16k for the fresh encoder comparison.
A staged 52k map-pretraining + 8k object-adaptation experiment is running.
A separate [actor map-core freeze](phase2_actor_spatial_freeze.md) tests whether
retaining learned map features prevents harmful SAC drift. No improved candidate
has been selected. The eventual SAC/TD3 × original/modified gripper × three-seed study
still requires independent replications and matched assistance/data protocols.
Current development runs are not those replications.

## Operation and recordings

Dashboard: **http://127.0.0.1:8766**. From another machine, forward that port to
this host with SSH; see [instructions](../dashboard/README.md). Server PID/logs
are in `dashboard/.runtime/`.

The featured 40k-policy recording is one of three predeclared successful diagnostic
repeats. An earlier first-case recording dropped during reach and is preserved;
its discrepancy remains unresolved. The three later repeats had identical logged
actions and physical states across rendered/non-rendered execution. They are
reproduction diagnostics, not extra held-out cases.

Models, replay, data, videos and immutable source snapshots live under ignored
`runs/`; preserve those directories when transferring experiments. Source and
small manifests are versioned in Git. The upstream checkout stays pinned at
`54ab2d1`; the local Python 3.10/CUDA environment uses the available A100 80 GB,
48 logical CPUs and 251 GiB RAM.

## Historical first development milestone

The earlier single-maze demonstration remains frozen and recoverable:
[report](milestone1.md) · [candidate](../configs/milestone1_candidate.json).
It passed three repetitions on foam brick 14/scene42 but did not establish
cross-maze generalization. It must not be confused with the first 100-case result.
