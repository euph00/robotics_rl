# Carry training from varied learned grasps

The current best development result is 51/60 full-task validation completions,
47/60 within 15,000 physics ticks including initialization (50/60 action-only).
A separately recorded first-case reproduction failed during reaching. These are
provisional validation results; the held-out partitions remain unused.

The carry learner has so far trained online from one deterministic grasp per
TRAIN scene. A fixed 30-branch TRAIN diagnostic found 10/10 valid deterministic
handovers and 8/10 teacher transports, versus 19/20 valid sampled handovers and
18/20 transports. This establishes feasibility for broader training experience,
not an inference-time performance improvement.

The new protocol declares three preparations for each canonical TRAIN case:
one deterministic grasp and two draws from the frozen grasp SAC policy's saved
action distribution. Each begins in a fresh native world. Seeds, actor digest,
variant identity, failures and physical costs are retained. No external action
noise, grasp parameter changes, object attachments or altered physics are used.
Full-system inference remains deterministic.

Collection covers the first 16 TRAIN layouts crossed with all five required
objects, yielding 240 declared independent preparations and at most one clean
teacher carry trajectory per preparation. Failed preparations remain in the
manifest; transport failures retain their actual replay transitions. All
variants belong to the existing learned-preparation replay family. No validation
or held-out trajectory is added to training.

Online caches use distinct preparation slots while returning the canonical scene
case ID and separate preparation ID in transition information. Cache restores
execute the established five native hold calls and log the resulting measured
observation. This post-refresh state is not claimed to equal an untouched
continuous handover. A failed preparation/restore is retained and disabled,
without silent replacement. Resume rebuilds successful worlds and counts the
new physical work; it does not serialize the simulator's hidden state.

The original cache implementation is unchanged. A separate variant protocol,
source fingerprint and case/seed signature reject incompatible resumes. Initial
integration verification passed 54 tests, including variant identities, failure
retention, physical ledgers and reconstruction. Both existing 40k training
branches retain matching resume signatures. The fixed TRAIN-only physical cache
preflight passed all nine slots (foam, banana and can, three variants each) and
all 27 reset/20-decision prefixes. It consumed 10,540 native calls plus 4,320
initialization ticks, or 57,020 physics ticks total; no training updates or
failed/replaced slots occurred. Artifacts: `phase2_diverse_cache_preflight01`.

A parallel control extends the current object-balanced run from 40k to 200k
online decisions with the same demonstrations, physical preparations and method.
This tests training duration separately from the broader-data branch. Full-task
validation is planned at total 80k, 120k and 200k decisions; every attempted case
and timed result will be retained. Neither run is a new independent final-study
seed.

The completed collection retained 230/240 handovers and 175/230 teacher
transports. Deterministic, first sampled and second sampled variants completed
60/80, 64/80 and 51/80 declared routes respectively. Sampling therefore supplies
diversity; it is not established as a better inference strategy. All ten failed
preparations and 55 carry-loss episodes remain recorded. Independent auditing
matched every replay row to its raw preparation shard and 13,263 measured trace
endpoints; all seeds, actor digests, native counters and 2,976,075 physics ticks
reconciled (`phase2_diverse_collection_audit01`).

The merged dataset contains 576,522 rows in 2,025 episodes (SHA256
`3bd28e114176be0c4d1a24dd33e11785a31df65f13c14587d04330a2caefd10a`).
`phase2_carry_diverse01` uses the same zero-online actor initializer, no additional
BC updates, 2,000 fresh-critic warmup updates, object-balanced HER, and continuing
BC weight100. Its fixed budget is 200,000 online decisions, with full60 validation
at40k,120k and200k. The replay capacity is800k to hold the broader data and online
experience. No inference action, observation or reward contract changes.

Preparation-source balancing was rejected: at40k it completed46/60 tasks,
43/60 within15k ticks including settling, versus51/60 and47/60 for object-only
balancing. Its8k endpoint was also worse. The diverse-data branch retains
object-only replay balancing.

The longer-control evaluations are complete:

| Online decisions | Full completion | Within15k action ticks | Within15k including settling |
|---|---:|---:|---:|
| 40k |51/60|50/60|47/60|
| 80k |47/60|47/60|46/60|
| 120k |48/60|48/60|47/60|
| 200k |48/60|48/60|47/60|

The40k checkpoint remains the best complete-task result. Final200k failures were
11 carry grip losses and the recurring can grasp timeout. All60 cases at every
milestone exactly matched40k in saved pre-carry states and stage counts. These
traces sample every five native calls and omit actions, velocities and solver
state, so they do not establish complete hidden-state equality. No held-out
case was evaluated. The three added full validations consumed2,321,875 ticks.

The diverse-data branch is also complete. Its three predeclared full-system
validations used the same 60 cases, frozen grasp/reach checkpoints, 24cm lift,
8cm internal reach margin and clearance-v2 proposals as the selected control:

| Diverse online decisions | Full completion | Within15k action ticks | Within15k including settling |
|---|---:|---:|---:|
| 40k |49/60|47/60|45/60|
| 120k |47/60|47/60|47/60|
| Final 200k |49/60|49/60|48/60|

Final200k completed 25,000 online SAC updates, following the separately counted
2,000 critic warmup updates. Its exact carry checkpoint SHA256 is
`668017d1c072827469b42f3d7302d0b77482254e36201fd4431f6a1206a50aba`.
Per-object completions were foam10/12, banana12/12, can8/12, apple7/12 and
mug12/12. The eleven failures were ten carry grip losses and one grasp decision
horizon. Artifacts are `phase2_system_diverse_040k01`,
`phase2_system_diverse_120k01` and `phase2_system_diverse_200k01`.

The original object-balanced40k checkpoint remains selected because full-task
completion is the primary criterion:51/60 versus the diverse branch's best49/60.
The final diverse checkpoint achieved one more completion within15k ticks
including settling (48 versus47), but fewer full and action-deadline completions
(49 versus51 and49 versus50). Both outcomes remain recorded; this comparison
does not establish a causal effect of preparation diversity or a statistically
reliable difference. No exact physical-prefix comparison was performed for
these diverse runs.

Independent metadata auditing verified the immutable model/state hashes and
update counters, all60 case records per milestone, and complete native-cost
ledgers. These three validations consumed448,372 native calls,2,241,860 action
ticks and86,400 initialization ticks: **2,328,260 physics ticks total**. They
remain separate from training and collection costs. All outcomes are retained;
this audit ran no new physics. At candidate selection, no held-out case had
been evaluated; these are development-validation results.
