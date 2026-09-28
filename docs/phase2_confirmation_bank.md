# Fresh confirmation after the first sealed test

The first sealed milestone test completed **78/100 full tasks**, including
**63/100 within15,000 physics ticks with initialization**. Its run,
`phase2_system_milestone_test01`, and every failure remain unchanged. It remains
the original untouched-test result for that candidate. Because its failure
analysis now guides refinement, those cases are **retired from future held-out
claims** for revised candidates. They must not be rerolled or imported into
training. The original final reserve remains untouched.

A fresh confirmation bank was declared and generated before revised training,
using fixed seed **20260930**. It contains20 content-distinct layouts crossed
with the same five objects, in order `[14,5,2,7,10]`: **100 fixed cases**.
Layout IDs are `confirmation_20260930_000` through
`confirmation_20260930_019`; case IDs append `_object_14`, `_object_05`,
`_object_02`, `_object_07` or `_object_10`. Their evaluator partition remains
`milestone_test` for compatibility, but the IDs and bank identity are distinct.

Generation excludes every layout in the expanded parent:192 TRAIN,12 validation,
20 first-test and20 final-reserve layouts, plus all100 course-legacy geometries.
The first20 RNG candidates passed the existing geometry-only gate-connectivity
rule and content-hash separation, with zero rejections. There was no route-length
filter, policy outcome filter, manipulation feasibility trial or new physics.
Future candidate draws, if a declared geometry collision occurs, advance the
same fixed RNG stream; they do not tune the seed using task outcomes.

The new file is an evaluation-only clone. TRAIN, validation and final-reserve
records, the original partition seed, physics geometry and object point-seed
formula are unchanged. The parent bank files and all historical manifests remain
unaltered. The final reserve cannot be accessed through this clone; its original
bank is retained for its original purpose.

Artifacts:

- `configs/phase2_confirmation_bank_20260930.json`: raw version
  `confirmation-scene-bank-v1`, manifest SHA256
  `be4d117896c5ed641c5619abe86263a30407d6144abd26f04865a211f41e4bc1`.
- `configs/phase2_confirmation_bank_20260930.cases.json`: all100 sealed case
  specifications, case-manifest digest
  `283ff33a834209801bd4cc57e25458ec365d0dd278cb6a6e01175ec9800d27e9`.
- `.generation.json`: timestamped declaration written before generation,
  fixed seed/rules, parent-file digest, source fingerprints and retired-test
  summary digest. `.summary.json` records counts and preserved-partition hashes.
- Parent manifest SHA256:
  `d6f130f4dae03b9ee4a27635bb5146cc5db48661240dd0d1da59e58e18424553`.

`pnr/confirmation_scenes.py` builds and verifies this distinct bank version.
Existing training loaders and training-extension verification reject it;
`pnr/scenes.py` and its training-resume fingerprint were not changed. The new
evaluation loader validates geometry using a private normalized copy, then
returns the original raw version and manifest digest. It permits only the
complete confirmation partition with explicit held-out access and the declared
five-object order. Existing evaluation of ordinary banks delegates unchanged.
The evaluator archives the new module with its other inference sources.

Eight new geometry/protocol tests and thirteen existing evaluator tests passed.
They cover content separation, original-record preservation, deterministic seeds,
RNG isolation, tampering, rejected training access and raw-digest consistency
through frozen evaluation plans. No simulator clients or physical test episodes
were created for bank generation or these tests.

Select any revised candidate using TRAIN and the existing validation set only.
Then freeze its exact checkpoints and execution settings before opening these
100 cases once, preserving all failures and costs. Do not treat this new bank as
another development-validation set. At bank creation its physical-evaluation
count is zero, and the original final reserve is still sealed.
