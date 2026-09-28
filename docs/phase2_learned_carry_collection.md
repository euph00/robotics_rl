# Carry data from learned grasps

The fixed TRAIN collection completed40/40 original-start learned handovers and62/80 declared full-route teacher attempts. It supplies **74 real closed episodes /25,538 transitions** across all five objects, with no case replacement or route-outcome filtering. These are teacher-data results, not learned-navigation or full-task evaluation.

| Object | Learned preparation | Clean teacher route | Noisy teacher route | Noisy restore failures |
| --- | ---: | ---: | ---: | ---: |
| Foam14 | 8/8 | 6/8 | 5/8 | 2 |
| Banana5 | 8/8 | 8/8 | 8/8 | 0 |
| Can2 | 8/8 | 6/8 | 5/8 | 1 |
| Apple7 | 8/8 | 4/8 | 6/8 | 2 |
| Mug10 | 8/8 | 7/8 | 7/8 | 1 |

The18 failed opportunities comprise12 physical `lost_grasp` episodes and6 failed restored-state contact checks before any new native call. There were no route collisions or timeouts. All failures remain in the case/attempt manifest; the12 real failure trajectories remain in the dataset. A zero-call restore failure has no fabricated transition.

Each case uses the first eight frozen TRAIN layouts crossed with the fixed five objects, including its declared point-cloud seed. Preparation follows the shared `LearnedSystem` original-start prefix: geometric pregrasp, frozen54-state clearance01 learned grasp at24cm lift,50 native hold calls, and the shared planned carry posture. The checkpoint is the selected12,000-online-decision candidate (`fa7b8d54cdc916d6e68753721993b66b7e571bf09af84e70cf4e4030618dfbe2`); proposal mode is clearance-v2. All original meshes, forces, gains, friction and five-tick native calls remain unchanged.

At the retained handover, one full-route geometric teacher episode runs continuously. The second TRAIN episode restores that exact physically prepared snapshot and applies Gaussian action noise with standard deviation0.12, reduced near the exit. Restoration preserves original pickup history and consumes no physics; a separate monotone ledger retains the real cost of both attempts. A failed first route does not suppress the declared second attempt. The original exit goal stays fixed throughout each episode. The data uses the existing580-state, two-action carry schema with delta0.075m, repeat5, gamma0.995 and potential weight2. Teacher labels and actually executed actions are stored separately.

Snapshot restoration does not reliably restore PyBullet's contact-force cache: six restored states failed the existing positive-force bilateral-contact test. They were counted and left unchanged in this frozen run. The subsequent cache protocol is being investigated separately with an explicitly counted hold/recheck; it must not retroactively repair this dataset. Among the34 pairs that executed both routes, all580 measured starting values match exactly:25 pass both,6 pass only the noisy route, and3 pass only the clean route. Unobserved solver/contact cache differences limit causal claims about noise, so this is not a controlled demonstration that noise improves retention.

The physical cost is127,662 carry calls plus33,530 original-start preparation calls:161,192 native calls,805,960 action physics ticks, and19,200 initialization ticks, **825,160 total ticks**. All40 preparation attempts and80 transport opportunities are represented. Failed preparation metadata remains supported even though this cohort had no original-start preparation failures.

The dataset is `runs/phase2_learned_carry_demos01/artifacts/demonstrations.pkl`, SHA256 `f5831619dd05b17a88bb7b1f41d628c8a16b9c4b1f0622f1642507288210ae7f`. `dataset_audit.json` verifies fixed cases, closed episode boundaries, exact adjacent observation continuity, action/label separation, true terminal failures, matching physical counters and stored reward recomputation with zero error. Four focused collector tests pass. Source/checkpoint/case hashes, every attempt, sampled measured trajectories and the audit script are preserved with the run.

Implementation lives in `scripts/phase2_learned_carry_demos.py`. It imports the shared `pnr.learned_carry.LearnedCarryPreparation` prefix rather than copying controller logic. The next use should retain the existing broad maze coverage while adding this five-object learned-handover distribution. Apple and can still need retention work: even the route teacher loses some grasps, so further maze-only training cannot by itself establish robust end-to-end transport.
