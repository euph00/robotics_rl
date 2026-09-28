# Original-gripper retention diagnostics

All probes preserve the original meshes, friction, force limits, motor gains and five physics ticks per native action. No attachment is used. They use development layouts only, count failed setup/pickup/transport attempts, and keep the selected five objects unchanged. Teacher navigation results do not establish learned navigation or full-system success.

## Matched speed comparison

Frozen BC004000 grasp, legacy42 plus the first two TRAIN layouts, all five objects, three controllers:45 attempts. Matched triples have exactly identical carry-start payload coordinates.

| Carry controller | Retained transport | Successful action ticks through transport hold |
|---|---:|---:|
| Fixed0.075m increment |12/15; all three can cases fail|12,225–14,520|
| Fixed0.020m increment |15/15|34,075–40,470|
| Reactive0.075→0.020 after2mm payload shift |14/15; legacy mug fails|12,605–39,805|

Add480 initialization ticks per attempt. Final reach is excluded. The reactive rule rescues the can by slowing after its first decision, but can damage a previously reliable mug grasp; it is not adopted. Fixed0.020 supplies a physically retained untimed fallback on these cases.

Cost:248,782 native actions,1,243,910 action physics ticks, plus21,600 initialization ticks. [All45 outcomes and trigger points](../runs/phase2_carry_retention_reactive01/artifacts/comparison.md), [machine-readable comparison](../runs/phase2_carry_retention_reactive01/artifacts/comparison.json).

## Cause and rejected changes

The native position controller produces large within-action velocity pulses: the first physical tick reaches1.080m/s from rest (259.2m/s²), followed by decay across the remaining four ticks. Policy-rate smoothing does not remove this steady within-action pulse.

The upright can is pinched near its top rim, roughly49mm above its center of mass with the original learned topdown grasp. This creates a large moment arm. The same grasp remains stable during a stationary3,500-action hold. A failed learned foam handover also showed a contact concentrated at the finger-pad edge.

- Measured-opening closure targets1–3mm tighter than the initial opening lost all15 positive-margin transports, including foam and mug controls. They weaken contact force relative to the original zero target.
- Closing nearer the original topdown target at10/12/14mm failed all nine can pickups;18mm remains the teacher gate.
- A centered carry arm posture lost all three tested can handovers. Rotating the pinch direction alone did not improve transport.

[Contact geometry](../runs/phase2_carry_retention01/artifacts/grip_contact_geometry.png) and [native acceleration pulses](../runs/phase2_carry_retention01/artifacts/native_carry_impulses.png).

## Side approach feasibility

A declared matrix used can legacy42 and TRAIN000's frozen point seed, approach tilts45/60/90° from vertical, an8mm TCP setback, and closure gates5/18mm. All12 preparations failed collision checks before pickup. Horizontal90° puts the wrist into the table;45/60° put the palm into the can. Two additional counted fold/approach preparations tested read-only endpoint geometry at8/16/24mm setbacks and identified feasible60°/24mm candidates.

Four subsequent physical60°/24mm attempts, both scenes and both closure gates, passed pickup plus0.075 teacher transport and60-action hold. They used13,855–13,915 action ticks each, before final reach. The18mm gate had lower payload drift than5mm on both scenes. Contact was substantially nearer the can's center of mass. The approach direction and pregrasp were changed consistently: pregrasp=TCP−0.17×approach axis. The measured antipodal contact pair and width were preserved.

Costs: initial12 attempts plus two geometric preparations used4,200 native actions/21,000 action ticks +6,720 initialization ticks. Four successful side transports used11,109 native actions/55,545 action ticks +1,920 initialization ticks. All rejected geometric proposals remain logged.

[Initial matrix and collision diagnosis](../runs/phase2_side_grasp01/artifacts/summary.json), [successful side probes](../runs/phase2_side_grasp02/artifacts/summary.json).

The shared proposal mode remains a separate opt-in integration requiring TRAIN validation across the selected objects and a new declared demonstration dataset. Existing runs keep their original proposal mode.
