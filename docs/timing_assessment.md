# Next-phase timing assessment

Assessment date: 2026-09-28. The user accepted the proposed next-phase target: original gripper, SAC+HER, at least 80% uninterrupted complete-task success over five supplied objects crossed with 20 distinct held-out maze layouts (100 cases). Object identities may appear in training. Keep training/validation/test layouts separate and reserve a separate final-comparison evaluation bank. Preferred protocol is demonstration initialization followed by ordinary SAC+HER; user also permits further documented algorithmic/methodological improvements if this clearly fails. Preserve the eventual SAC/TD3 × gripper × independent-seed comparison.

The user requested a return after this assessment to change reasoning effort. New training has NOT started. The agreed 24-hour reassessment window should begin when the next phase actually starts, not during this clarification pause.

## Timing evidence

Native physics timestep is 1/240 second and each native action advances five ticks. Thus 15,000 ticks = 62.5 simulated seconds = 3,000 native calls. This is not a wall-clock training budget. Upstream does not clearly define whether the reach deadline starts at the original task start or navigation start, nor whether steps mean calls or ticks; retain all counts and provisionally target the conservative original-start physics-tick interpretation.

The frozen learned baseline first reaches the strict physical goal at 19,975 action ticks (83.23 s), and completes its final hold at 20,330 action ticks (84.71 s). Initial settling is a separate 480 ticks (2 s). Navigation alone uses 15,675 ticks, so faster or more efficient navigation is necessary to satisfy the conservative deadline; additional training alone does not change the current action-speed limits.

Bounded physical reference tests used the unchanged simulator, original gripper, scene42/object14, and existing reference speed_scale option. These are geometric reference diagnostics, NOT learned-policy improvements or generalization results.

| Reference speed scale | Result | Action ticks | Notes |
| --- | --- | --- | --- |
| 1.5 | Failed | 5,765 before stopping | Carry planner rejected a colliding target; no observed wall/table contact or drop before stopping. |
| 2.0 | Success | 12,890 | Complete task including final hold; zero drops/wall/table contact calls; lift first reached at 2,880 ticks. |

The 2.0 run took 53.71 simulated seconds, or 55.71 seconds including initial settling. It demonstrates feasibility below 15,000 ticks on this one case. It does not establish feasibility for all objects/layouts, and non-monotonic reference outcomes show that blind speed scaling is insufficient.

Artifacts: `runs/timing_assessment_speed1p5/artifacts/result.json` and `runs/timing_assessment_speed2/artifacts/result.json`.

Recommendation: retain 15,000 ticks as a credible stretch target; measure strict task success and success within deadline separately. Validate control speed, collisions, grasp retention and complete stage costs before freezing a protocol. Training curricula may allow longer episodes while the final evaluation deadline remains fixed. Do not change physics timestep, native substeps, contact properties or reported counters to meet the target.
