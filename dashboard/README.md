# Dashboard

Read-only, localhost-only status page using only the Python standard library.

```bash
python dashboard/server.py --port 8766      # then open http://127.0.0.1:8766
ssh -L 8766:127.0.0.1:8766 <user>@<host>    # from a remote workstation
```

The running server's PID and log are in `dashboard/.runtime/`.

## What it shows

- **Headline, study arms, plan, documents, recordings:** curated in
  [`overview.json`](overview.json). Each study arm lists the evaluation run IDs
  for its `dev`, `holdout` and `course` results; the server fills in the numbers.
  Update this file when the project's shape changes; no restart is needed.
- **Evaluations:** every `scripts/eval.py` run under `runs/` (identified by its
  `config.json` and `results.jsonl`), newest first, with per-object, per-set,
  failure and rubric breakdowns computed from the results file.
- **Running now:** live jobs with progress: evaluation cases completed or
  training decisions against the budget.
- **Historical runs:** every other run folder, collapsed.
- **Resources:** CPU load, free memory and disk, GPU use.

## Interface

- `GET /api/overview`: everything the page renders.
- `GET /api/state`: the older per-run status and metric series.
- `GET /artifacts/<run>/<path>`: files a run declared as artifacts (with byte ranges for video).
- `GET /reports/<name>`: allow-listed project documents, Markdown rendered as HTML.

Tests: `python -m pytest -q dashboard/tests`.
