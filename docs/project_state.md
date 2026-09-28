# Project state

First development milestone complete: a demonstrated original-Fetch pickup–carry–reach system, with the pinned course environment, physical reference, frozen learned policies and local dashboard.

## Active phase: broader original-gripper SAC+HER

Started 2026-09-28 05:53:20 UTC; reassess by 2026-09-29 05:53:20 UTC. Target: at least80% original-start physical task success on five supplied objects crossed with20 distinct held-out mazes; separately report15,000-physics-tick success. Objects are foambrick14, banana5, tomato can2, apple7, mug10, selected for geometry diversity and reference pickup feasibility before policy generalization tests. Tomato-can transport failures remain part of the task.

The original64/12/20/20 scene bank remains frozen. A verified extension appends128 TRAIN layouts only; all existing records, case point-cloud seeds, validation and held-out cases are unchanged. No held-out policy or physical evaluation has occurred. See [training expansion](phase2_training_expansion.md).

Ordinary SAC continuation collapsed in both learned skills. Continuing demonstration-action regularization stabilized grasp and helped carry, but is explicitly a modified SAC+HER method. The best completed uninterrupted development system is46/60 successes,45/60 within15,000 ticks including settling. This uses clearance-aware regularized SAC grasp,192-map BC carry, and SAC reach. The original grasp with that carry gives44/60; the previous BC system gives42/60. A strongly anchored SAC carry alternative with64-map data reaches35/60. Neither meets the held-out target. Full-task evaluations retain every failure and use the native10cm criterion plus mandatory60-call retained hold; the internal reach handover is8cm. See [paired system audit](phase2_recovery_system_comparison.md).

The new192-map carry initializer passes12/12 isolated navigation validation mazes, compared with10/12 before expansion. Its dataset contains1,721 episodes/478,363 real transitions; six failed episodes remain in replay. It received16,000 additional BC updates and2,000 fixed-actor critic updates, with zero online SAC decisions. Complete-task checks reached44/60 with the original grasp and46/60 with the tilted-grasp candidate. Isolated navigation is not full-task evidence.

A controlled critic audit found badly optimistic initial long-route values. Warmup tau0.1 substantially improves calibration compared with0.01, retaining online tau0.01. However the matched64-map physical branch still finishes9/12 navigation validation; calibration alone did not solve generalization. See [critic audit](phase2_critic_value_audit.md).

Carry training has a concrete handover mismatch: all previous demonstrations and online environments use one reference-prepared foam grasp, while most learned handovers lie outside its measured grip-feature range. A fixed2-TRAIN-layout×5-object preflight succeeds10/10 through learned pickup, planned posture and100 teacher carry decisions. Full-route TRAIN collection is now being added. This uses original native controls and measured states, with failed preparations retained; no inference-time route teacher or object attachment is introduced. See [distribution audit](phase2_carry_distribution_audit.md).

The clearance-aware tilted grasp candidate remains9/10 on its small manipulation validation bank through12,000 regularized SAC decisions. Adding18 TRAIN can seeds supplies144 successful physical demonstrations but leaves the same rare-yaw miss after8,000 BC updates; that branch was stopped before online training. A compact BC grasp with zero online updates passes15/15 isolated pickup checks, but its full system completes43/60 cases,41/60 within15,000 ticks including settling, versus the current best46/60 and45/60. Its failures are13 carry grasp losses, one planned-posture loss, two navigation horizons and one reach drop. Pickup success alone is insufficient for full-task selection; the cause of the difference has not been established by a paired trace audit. See [expanded system results](phase2_expanded_system_results.md). Teacher settling probes support the physical value of a pause, but simple velocity gates chatter and have not been adopted. Gripper hardware, forces, gains and physics remain unchanged.

Git snapshot `7650b3e` preserves the pre-cleanup development tree. Commit `9f2a827` removes obsolete pilot trainers/diagnostics; active source and useful tests remain in the working tree. Logs, datasets and checkpoints remain under ignored `runs/` with lineage and costs.

## Frozen first milestone result

Three fresh original-start repetitions on foam brick 14 / scene 42 passed every stage, with no drops or wall/table contacts. Each ended 3.94 cm from the goal. A separate recorded rollout passed at 3.08 cm.

[Milestone report](milestone1.md) · [Frozen checkpoints and protocol](../configs/milestone1_candidate.json) · [Validation details](../runs/milestone1_validation01/artifacts/milestone_validation.json) · [Dashboard video](http://127.0.0.1:8766/?run=learned_system_demo_bc01).

This is a **demonstration-assisted development baseline**. Grasp uses SAC + HER, physical replay demonstrations, actor refinement and subsequent SAC. Carry uses a 90k-decision SAC + HER checkpoint followed by 14k supervised actor updates. Reach uses SAC + HER without actor cloning. These are not independent final-study replications.

The shared controller supplies planned antipodal pregrasp and carry-posture preparation. Saved policies control grasp, maze transport and held-object reach. No inter-skill reset, navigation waypoint oracle, teacher correction, object weld or release is used. Actor pose masking makes navigation less sensitive to pickup differences while preserving physical collision/retention checks.

## Evidence and limits

- The physical reference passes on the same development case; [recorded reference](../runs/reference_demo_visible_seed42/artifacts/rollout.webm).
- The final production test suite passed 64 tests. Focused geometry, HER, accounting and checkpoint checks accompany individual experiments.
- Both learned video formats decode successfully: 640×360, 12 fps, 1,017 frames. Dashboard browser checks cover desktop/mobile layout, experiment history, direct run links, playback and seeking.
- The three repetitions share one object and one maze hash. They do not establish held-out generalization or statistical reliability.
- Source-call deadline flags pass. The separately reported 15,000-physics-tick reach flag does not; upstream “simulation steps” wording remains ambiguous.
- Further ordinary SAC updates at 1e-5 and 1e-6 lost the refined carry route. These failed continuations are preserved and are not the selected baseline. An unused lateral-coverage prototype is archived outside production; no selected model used synthetic state augmentation.

## Operation

Dashboard: **http://127.0.0.1:8766**. The detached server's PID and log are in `dashboard/.runtime/`. From a remote workstation, forward port 8766 to this host's loopback address; see [dashboard instructions](../dashboard/README.md).

No training job is required to view or run the frozen baseline. Reproduction commands are in the milestone report. Logs, videos, models, replay and source snapshots live under `runs/` and are ignored by Git; preserve those directories when transferring experiments.

The project Python 3.10 environment and CUDA are configured on the workspace volume. Available hardware includes an A100 80 GB, 48 logical CPUs and approximately 251 GiB RAM. The upstream checkout remains pinned at `54ab2d1`; workspace setup still requires escalated shell execution because of the host sandbox launch issue.

## Next milestone

Stabilize continuing RL and broaden training coverage before freezing the independent SAC/TD3 × original/modified gripper × three-seed study. Use a matched demonstration/assistance protocol and a separate held-out object/maze bank. Existing development branches and repeated cases must not be counted as independent research replications.
