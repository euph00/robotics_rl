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
