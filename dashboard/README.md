# Experiment dashboard

Read-only, dependency-free local dashboard. It shows milestone state, registered runs, measured evaluation success, demonstrations, recent events, and host resources. Empty states stay empty until real measurements exist. Success is labeled by evaluation scope; a skill result does not imply complete-task success. The experiment table initially shows the ten most recent runs plus any active or selected run; Show all / Show recent changes only this view, and the selector always retains the full history. Data collectors show decisions without an untyped episode-budget fraction.

```bash
# From the repository root; Python 3.8+ works.
python3 dashboard/server.py
# Open http://127.0.0.1:8766
# Remote workstation: ssh -L 8766:127.0.0.1:8766 <user>@<host>
python3 -m unittest dashboard.tests.test_dashboard
# Optional, when Node is available: node dashboard/tests/test_display.js
```

Use `--port` to select another local port or `--runs /path/to/runs` to view another experiment directory. Port 8766 is the project default. The server only binds loopback. No training start/stop controls or remote write API are provided.

## Write progress from training

```python
from pnr.telemetry import RunLogger, update_project

update_project(phase="reference", summary="Verifying a contact grasp and retained transport.")
log = RunLogger("sac-grasp-seed0", kind="train", algorithm="SAC + HER",
                skill="grasp", seed=0, budget=50_000)
log.event("Started grasp curriculum")
log.update(transitions=1_000, summary="Training on the original gripper")
# Only after real evaluation; scalar values must be JSON serializable.
log.metric(transitions=1_000, success_rate=successes / episodes,
           episodes=episodes, scope="skill", object_goal_distance=mean_error_m)
# Save the actual recording first, then register its relative path.
log.artifact("artifacts/rollout.mp4", "Grasp evaluation", "video")
log.finish("complete", "Pilot evaluation complete")
```

`RunLogger.directory` is the run directory. One process owns each run. Use a unique ID, or `resume=True` and `status="running"` for explicit resume. Call `update()` periodically for a fresh heartbeat and call `finish("failed", reason)` on a caught training failure. Status is written by atomic replacement; metrics and events append complete JSONL rows. The reader tolerates a partially written last row and reports `interrupted` if the recorded local process has disappeared. `update_project` merges existing project state; replacing `milestones` requires the full list.

## On-disk contract (version 1)

All experiment state lives under `runs/` by default:

```text
runs/
  project.json
  <run_id>/
    status.json
    metrics.jsonl
    events.jsonl
    artifacts/
      rollout.mp4
```

| File | Fields |
| --- | --- |
| `project.json` | `schema_version: 1`, `updated_at` (UTC ISO), `phase`, `summary`, `milestones: [{id, label, status}]`; optional `run_labels` and `featured_run` |
| `status.json` | `schema_version: 1`, `id`, `name`, `kind`, `algorithm`, `skill`, `seed`, `status`, `started_at`, `updated_at`, `finished_at`, `transitions`, `budget`, `summary`, `metrics`, `artifacts: [{label, path, kind}]` |
| `metrics.jsonl` | One object per row: `timestamp`, `transitions`, numeric metrics, optional `scope`. `success_rate` is a fraction in `[0,1]`. |
| `events.jsonl` | One object per row: `timestamp`, `level`, `message` |

Run `kind`: `reference`, `train`, `initialization`, `evaluation`, or another explicitly labelled diagnostic kind. Run `status`: `running`, `complete`, `failed`, or `stopped`. Milestone `status`: `pending`, `active`, `complete`, or `blocked`. Default milestone IDs: `environment`, `reference`, `learning`, `evaluation`, `demo`. Evaluation `scope`: `skill`, `full_task`, `reference`, or `upstream`. Unspecified scope is explicitly labeled as such.

Optional metrics displayed: `episodes`, `full_task_success_rate`, `upstream_success_rate`, `object_goal_distance` / `goal_error_m` (metres), `drops`, `physics_ticks`, `wall_time_seconds`. `transitions` counts real environment interactions, not HER relabels. Explicitly name and log `policy_decisions`, `source_action_calls`, and `physics_ticks` when reporting the upstream rubric. For `kind="train"`, the chart uses cumulative online policy decisions, excluding demonstrations; `inherited_transitions` separates the previous run's decisions from new ones. For `kind="initialization"`, the chart uses `updates` (or `bc_updates`) for supervised progress and plots `bc_validation_loss` when present. Physical success remains in the separate evaluation card. Initialization `transitions` denotes physical evaluation calls and is never presented as online training decisions. Without loss measurements, physical success is plotted against supervised updates when recorded, otherwise against explicitly labelled evaluation calls.

For `kind="evaluation"` with `planned_cases`, progress and the chart x-axis use completed `eval_episodes` against the planned count. Logged success and deadline rates retain their fixed planned denominator, including during a partial run; the dashboard does not estimate missing outcomes. Physical calls remain separate. A permalink to a declared `run_labels` entry waits for that run to register instead of showing an older result.

Accounting uses the latest `metrics.accounting`, falling back to top-level `accounting` while a run initializes. Known total costs are displayed directly; nested `bc_initialization.incremental_accounting` is labelled as already included when `included_in_cumulative_evaluation_totals` is true and is never added a second time. Missing historical costs remain explicit.

Success charts only join points with the latest evaluation's scope. JSONL is preferred; `metrics.csv` with the same column names is accepted when JSONL is absent. Rows are bounded to a 2 MiB tail and the chart is downsampled to at most 240 measurements.

`GET /api/state` returns project state, runs, event tail, and cached host resources. The three bundled public files, registered run artifact directories, and a fixed allowlist of published project reports under `/reports/` are exposed; directory listings, arbitrary repo files, traversal, escaped symlinks, and write methods are unavailable. Media types are restricted, and video byte ranges support seeking. Resource measurements include all host jobs; they are not exclusive allocations.

`featured_run` places that run’s video after any explicitly selected run’s video. When viewing another run, its caption identifies it as a historical recording. A paired WebM is embedded once, with the original MP4 available as a download. Run labels also appear in captions.

The overview links to `/reports/phase2_test01.md`, rendered as escaped HTML with working report/artifact links and the outcome matrix. The explicit allowlist also includes `/reports/phase2_confirmation_test01.md`, its confirmation-bank and spatial-freeze supporting reports, and `phase2_candidate02.json`. An unpublished report returns 404 until its file exists; other repository paths remain inaccessible. The evaluation card distinguishes full-task success from deadline counts including settling and action-only counts when those measurements exist.
