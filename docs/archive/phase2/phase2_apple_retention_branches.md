# Apple closure-settling diagnostic

A 30-native-call closed hold immediately after first bilateral contact rescued
both selected apple failures through the first200 navigation calls. It also
preserved the healthy control. This supports training a better closure/settling
phase; it does not establish full-task retention or authorize a production pause.

## Exact protocol

`runs/phase2_apple_retention_branches01` contains12 separate original-start
attempts: apple validation006 and000 (failures), and008 (healthy control), each
with four branches. All use the frozen selected regularized SAC grasp and BC
recovery navigation checkpoints from `phase2_system_recovery_bc08_01`, legacy
proposals, a24cm lift subgoal, original gripper, original physics and0.075
navigation action scale. Six CPU workers execute the attempts. No inter-skill
resets, new attachments or waypoint teachers are used.

The first-contact branch triggers once, after the first grasp **policy decision**
reports bilateral contact. Each grasp decision already comprises three native
calls. The branch freezes the15 measured joint positions, sets both finger
position targets to zero, executes30 native calls (150physics ticks;0.625s),
refreshes the measured54-state observation, then resumes the same learned policy.
It does not issue additional learned decisions during the pause.

The post-lift control adds the same30-call hold immediately after learned grasp
success, before the standard50-call pickup hold. The slower-transition control
uses controller speed scale0.5 only inside `prepare_carry`, reducing planned
joint-position increments; its original50-call posture hold is unchanged.
Navigation remains unchanged in every branch and stops after200 native calls or
an earlier physical failure. These outcomes are retention diagnostics, not
full-task success.

## Outcomes and counted costs

| Branch | Apple006 | Apple000 | Apple008 | Action physics ticks, all three |
| --- | --- | --- | --- | ---: |
| Original | Posture loss | Navigation loss at63calls | Retained200calls | 13,925 |
| Pause at first contact | Retained200calls | Retained200calls | Retained200calls | 16,085 |
| Pause after lift | Posture loss | Retained200calls | Retained200calls | 14,780 |
| Half transition speed | Posture loss | Posture loss | Retained200calls | 16,835 |

All12 attempts count: **12,325 native action calls,61,625 action physics ticks,
5,760 initialization ticks,67,385 total ticks**. The original branches reproduce
the previously recorded outcomes and failure locations/call counts. Pauses add
150ticks each, but changed physical handover states also change subsequent
planned path lengths; total-cost differences are not just the pause cost.

## Measured contact and settling evidence

First-contact pause triggers occur at native calls411/423/458 for006/000/008.
All three objects are still on the table and not lifted. Finger velocities at
that moment are respectively (−3.54,−8.95), (−8.99,−13.45), and
(−8.65,−4.22)mm/s. After30 pause calls they fall to
(0.318,−0.264), (0.148,−0.139), and (−0.010,−0.079)mm/s.
Measured object speeds also settle below1mm/s.

In006 the finger gap closes from75.5 to73.1mm while the object height stays near
0.6626m. Normal forces change from approximately237/235N to234/232N: improvement
is not explained by stronger force. Finger targets were already zero except
for a negligible1.26micrometer target in006 before the pause.

The force-weighted actual finger contact centroids explain the downstream risk:

| Case/branch, after posture transition | Finger-centroid X in hand frame |
| --- | --- |
| 006 original | 30.7 /26.8mm |
| 006 pause before lift | 14.8 /2.5mm |
| 000 original, after posture hold | 30.6 /19.8mm |
| 000 pause before lift, after posture hold | 12.9 /5.7mm |

The original pad extends to31.5mm along this axis. Original contacts migrate
close to that end before loss, despite substantial normal force and continued
bilateral flags. Slower transitions prolong the fragile grasp and do not rescue
it. Apple000 after the first-contact pause still reaches a27.9mm centroid after
200 navigation calls, so delayed failure remains possible on a complete route.

Actual contact positions, normals, normal/friction forces, both finger states
and targets, joint velocities, object pose/velocity, and object-in-hand pose are
recorded after **every native call** in each `*_contacts.jsonl` file. The
`contact_migration.png` figure is registered on the diagnostic dashboard entry.

## Summary-writer failure and recovery

The original harness incorrectly passed a result dictionary to a JSON fallback
converter. That raised `TypeError: dict` only after physical execution and native
environment closure. All12 physical attempts finished, but their original
per-case summary dictionaries were not written. Every per-call contact trace is
complete: consecutive call numbers start at1 and each recorded physics count
matches five times the native call count.

`recover_from_traces.py` reconstructs every outcome, stage call count, controller
call count and physics cost from those complete traces. Each reconstructed JSON
states its provenance and hashes its source trace. Initialization costs use the
frozen original constructor's480-tick settling contract. **No physical attempt
was rerun or omitted.** Wall-clock duration, the original native final evaluation
dictionary (including final goal error), wall-contact totals and the original
exception message text were lost; they are explicitly unavailable, not guessed.
The dashboard summary identifies the recovery.

The standalone harness is fixed. Four mocked tests pass, including one for
nested NumPy result serialization, exact30-call pauses, state refresh, and no
stale positive contact flag. No production module changed.

## Training follow-up

The manipulation agent will compare ten zero-XYZ/closed-finger policy decisions
with the joint-position hold on identical **TRAIN** first-contact snapshots.
The existing54-state observation includes finger and object velocities. A fixed
hidden ten-decision teacher timer risks assigning different actions to nearly
identical observations; an observable settling condition should be investigated
before replacing data. No timer or pause is silently inserted at policy
inference, and the current clearance/can training branch stays unchanged.
