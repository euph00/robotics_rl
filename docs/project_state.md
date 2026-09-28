# Project state

First development milestone complete: a demonstrated original-Fetch pickup–carry–reach system, with the pinned course environment, physical reference, frozen learned policies and local dashboard.

## Active phase: broader original-gripper SAC+HER

Started 2026-09-28 05:53:20 UTC; reassess by 2026-09-29 05:53:20 UTC. Target: at least80% original-start physical task success on five supplied objects crossed with20 distinct held-out mazes; separately report15,000-physics-tick success. Objects are foambrick14, banana5, tomato can2, apple7, mug10, selected for geometry diversity and reference pickup feasibility before policy generalization tests. Tomato-can transport failures remain part of the task.

Frozen scene bank:64training/12validation/20milestone-test/20final-reserve layouts, all distinct from each other and legacy maps. Only training and validation data are used during development. First64-maze physical demonstration collection succeeded, including corrective action noise. Navigation is being retrained with a learned spatial map encoder, goal-consistent potential reward shaping, calibrated exploration, fresh critic warmup, then ordinary SAC+HER. New manipulation proposals support low objects without altering existing feasible proposals; multi-object grasp retraining is in progress. The final test bank has not been evaluated.

An audit verified excessive stochastic action noise in the old final navigation actor and mismatch between its narrow BC goals and inherited HER/curriculum data. Neither std calibration nor critic warmup alone is claimed sufficient; physical validation before and after online updates will decide. See [audit](phase2_sac_audit.md), [scope](../configs/phase2.json), and live dashboard.

Current evidence: the new BC navigation actor passed4/4 development navigation checks, but ordinary SAC continuation fell to0/4. Multi-object grasp fell from9/10 after BC to0/10 after1,024 online decisions. Both failed continuations are stopped and retained as evidence. The active pivot adds an explicitly labeled demonstration-action loss during SAC; carry coefficients10/100 are being compared, and grasp uses a separately justified coefficient200. This is not ordinary SAC.

The initial uninterrupted phase-2 BC system passed3/10 development cases, all within15,000 ticks including settling. A bounded IK fallback subsequently fixed all three exposed grasp failures with the unchanged actor and zero actual wall/table contacts. A shared24cm lift subgoal addresses long-object clearance; broader full-system validation is running. Can retention remains difficult: original position control produces measured within-call acceleration pulses up to259m/s². Slower physical transport works but misses the timing target. Reduced finger closure and tighter teacher closure gates were tested and rejected.

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
