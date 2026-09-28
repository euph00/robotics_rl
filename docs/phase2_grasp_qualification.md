# Transport-qualified grasp refinement

The lift objective does not test downstream retention. On the first eight TRAIN layouts, all 40 learned handovers passed, but only 31/40 clean teacher routes retained the object. Across 80 TRAIN cases with three independent grasp preparations, 43 cases had mixed transport outcomes despite identical proposal triples; 17/20 deterministic failures had a sampled success. The 77/80 cases with any successful variant are an oracle coverage bound, not a policy success rate. Sampled failures also occurred on 26 deterministic successes.

This proposal qualifies **grasp supervision** using subsequent physical transport. Earlier diversity experiments trained carry from varied grasps without refining grasp this way. The rejected pause, velocity gate, slower approach and compact actor are not repeated.

Historical Plan02 remains **SKIPPED**: its watcher stopped when frozen-core 52/38 beat 51/47 under the declared full-success-first rule. No collection, BC refinement or 20-case comparison ran. Its plan and closed watcher status remain unchanged. Plan02 is `runs/phase2_grasp_qualified_plan02/artifacts/plan.json`, SHA256 `16267e77640fff5352c018c181b97ea649ba593a5fa7b06d234a9ef8a9103760`.

Plan03 is a separate development phase following the failed, now retired confirmation: 69/100 full completions and 53/100 within the inclusive deadline; the root audit reported 24 carry-loss failures. Only those aggregate results and their source digest motivate this phase. No test case parameters, observations or trajectories enter training. Both retired tests remain recorded, the reserve stays untouched, and any future final confirmation needs its own independent declaration.

The new plan is `runs/phase2_grasp_qualified_plan03/artifacts/plan.json`, SHA256 `d69efd920e808e55690614aed519031374e7d60c3508636748e06b4c57595f60`, targeting `phase2_grasp_qualified_demos03`. It requires an independent review and a separate root authorization tied to that exact hash. It does not reuse or relax the historical gate. At plan creation, no collection, BC, comparison or automatic launcher had run.

The schedule preserves the first eight TRAIN layouts × five objects × three independent original starts: **120 preparations**. Each case uses deterministic, first native-SAC sampled and second sampled variants, with isolated Torch seeds 9401–9520. The frozen clearance01 grasp starts with SHA `fa7b8d54`. Geometry, physics, the 24 cm lift goal, planned posture and deterministic deployment remain unchanged. Every branch uses the same geometric carry teacher at 0.075 m, its existing five-decision goal hold, then 50 additional native hold calls. This measures retained teacher transport, not learned navigation or full-task success. No failed case is retried or replaced, and no best variant is chosen per case.

`RecordingGrasp(wrapped, persist)` delegates each grasp action once. It copies the actual observation, action, next observation, native reward, ending flags and info, then persists the prefix **before returning to the system**. Later posture or transport failure cannot rewrite those tuples. A separate boundary record identifies the caller's handover; both native ending flags may correctly remain false. These are BC prefixes, explicitly not directly replay-seedable. Qualification changes only episode-level BC eligibility. Malformed tuples or uncertain costs stop collection while preserving raw shards and failures.

The checkpoint contract was verified without prediction or simulation:

| Quantity | Dimension / count |
|---|---:|
| Raw observation | 54 |
| Achieved / desired goal | 3 / 3 |
| Action | 4 |
| Engineered actor / critic features | 70 / 70 |
| Historical online decisions / SAC updates | 12,000 / 2,000 |

The proposed endpoint adds **2,000 BC updates**, each with 400 examples: exactly 200 from existing demos06 and 200 from qualified prefixes. It uses fresh Adam at 3e−4, seed 10101 and gradient clipping at 10. Objects are sampled uniformly; old data retains its phase sampling, while new data samples eligible prefixes and then steps uniformly. Missing qualified coverage for any object stops fitting. Every qualifying variant is eligible, while failures stay in the raw collection. The fixed endpoint preserves the actor's historical SAC counters. There is no representation, log-standard-deviation, reward or HER change, and no online SAC continuation in this plan.

The paired comparison uses 20 TRAIN008–011 cases, frozen clearance01 versus the fixed BC endpoint, each from an independent original start with the same teacher route and hold. These are development cases that may appear in ancestor training, withheld only from this refinement. It must produce **at least three net retention wins**, calculated as candidate-only successes minus control-only successes, with no new preparation failures: every case where the control reaches a valid handover must also do so with the candidate. Preparation includes approach, learned grasp, pickup hold and planned posture. Missing or incomplete cases fail the gate.

At most **one** subsequent full60 development evaluation is permitted after passing that comparison. Promotion requires **at least 57/60 full completions and 38/60 inclusive timed completions**, against frozen 52/38, with every paired win, loss and cost reported. A missed gate closes this refinement hypothesis without checkpoint retuning or retries. BC and evaluation still require separate decisions after collection auditing.

The implementation reuses the existing collector, physical preparation and teacher helpers. `freeze-successor-plan` creates metadata and checkpoint copies without physics. `collect --plan PLAN --review-record REVIEW --authorization-record AUTHORIZATION` requires both records to name the exact plan hash and collection-only scope. Root authorization also identifies `/root` and the declared development phase. The old `freeze-plan` and legacy gate remain unchanged. All 27 no-physics tests pass, covering historical immutability, exact TRAIN schedules, exclusion of test-case data, authorization mismatch, actual boundary recording, reward preservation and complete accounting.

Plan03 collection completed once and passed the independent data audit. All 120 attempts and 6,425 actual grasp tuples remain in `phase2_grasp_qualified_demos03`; 116 reached the planned carry handover and 88 qualified after full teacher transport and hold. The 32 unsuccessful attempts comprise 28 transport losses, two planned-posture failures and two grasp horizons. There were no schema or accounting errors. Total cost was **1,502,725 physics ticks**, including 57,600 initialization ticks; this is data collection, not full-task evaluation.

| Object | Qualified / attempts | Eligible grasp tuples |
|---|---:|---:|
| Foam brick | 16 / 24 | 704 |
| Banana | 22 / 24 | 932 |
| Tomato can | 15 / 24 | 717 |
| Apple | 15 / 24 | 653 |
| Mug | 20 / 24 | 875 |

The immutable dataset SHA256 is `998437ab00a43c6d6835d2c32bc2a971f4ca924eefaff5bac8171c7bab9181d1`. The audit verifies all 45 collection sources, raw-prefix hashes, unchanged rewards and native flags, all failures, isolated sampling and complete costs. Its record is `runs/phase2_grasp_qualified_demos03/artifacts/independent_data_audit_scene_bank.json`. All five objects have eligible prefixes.

The separate lean runner `scripts/phase2_refine_qualified_grasp.py` implements `inspect`, `fit` and `compare` as explicit stages. It reuses the existing old-data groups and unchanged physical collector, checks the audited dataset and raw-prefix identity, and never edits native tuples or seeds replay. Fitting uses a separate BC optimizer, verifies unchanged critics, target critics, log-standard-deviation head parameters, temperature and inherited SAC counters, then saves only the fixed endpoint and checks its reload. Shared actor layers can change the conditional standard deviation even though the standard-deviation head is not updated or recalibrated. Deterministic inference remains the comparison contract. The 20-case comparison freezes both checkpoints and settings before its 40 fresh starts, retains all outcomes, and implements the declared net-win and preparation guards without launching a full60 evaluation. At this implementation-review stage, no BC fit or paired comparison has run.

The reviewed implementation was committed as `4a54166`. Root then authorized exactly one fit and one paired comparison in `runs/phase2_grasp_qualified_plan03/artifacts/root_refinement_authorization.json`, binding runner SHA `bd5409f3397b2c4ecf1e9678d404825f7b0d68d8d076e44eca5cfabeba5664bd`. The fixed fit completed **2,000 updates / 800,000 examples**, exactly 400,000 from each source, with uniform-object sampling counts between 79,672 and 80,264. All protected parameter/counter checks and checkpoint reload passed. There were zero new physical interactions or SAC updates during fitting. The candidate is `runs/phase2_grasp_qualified_bc03/checkpoints/grasp_bc_002000.zip`, SHA256 `b88768e204c98cec391691f9e48a0340b033ce76e74f21704c9f72a96e52473b`.

The fixed 20-case TRAIN comparison **failed its prospective gate**: control retention was 18/20 and candidate retention 15/20. There was one candidate-only success and four control-only successes, giving **−3 net wins**, below the required +3. Both policies reached all 20 handovers, so the preparation guard passed. All 40 branches and costs remain in `runs/phase2_grasp_qualified_compare03/artifacts/summary.json`.

| Object | Control retained / 4 | Candidate retained / 4 |
|---|---:|---:|
| Foam brick | 4 | 3 |
| Banana | 4 | 4 |
| Tomato can | 3 | 3 |
| Apple | 3 | 2 |
| Mug | 4 | 3 |

The candidate-only win was `train_011_object_02`; control-only wins were `train_009_object_02`, `train_009_object_14`, `train_010_object_10` and `train_011_object_07`. Comparison cost was **522,190 inclusive physics ticks**; collection plus comparison cost was **2,024,915 ticks**. This is measured teacher-transport retention on development cases, not full-task or independent-test performance. The result does not support this fixed qualification-and-BC refinement. Under the prospective rule, the hypothesis is closed: no retuning, replacement collection, full60 evaluation or heldout rollout follows. The original grasp checkpoint remains selected.

Independent endpoint/comparison audit passed at `runs/phase2_grasp_qualified_compare03/artifacts/independent_endpoint_pair_audit_scene_bank.json`. It reproduced all 800,000 sampling draws, verified unchanged protected state and six changed actor mean-network tensors, checked every paired branch and cost, and confirmed the failed gate. All 20 pairs had identical first recorded grasp inputs; this does not establish identity of unrecorded simulator state.
