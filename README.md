# Goal-conditioned RL for Pick–Nav–Reach

Build and compare SAC + HER and TD3 + HER systems that physically pick an object, carry it through the supplied maze, and reach the goal while holding it. The eventual study compares two grippers across three independent system seeds per algorithm/gripper condition.

The foundation is [NUS-LinS-Lab/Pick-Nav-Reach-Default-Project](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project), pinned at `54ab2d1` in [upstream.lock.json](upstream.lock.json). Its checkout stays unchanged under `upstream/`.

## Current milestone

The first unseen-maze test completed **78/100 tasks**, with **63/100 within
15,000 physics ticks including settling and final hold**. The 80/100 target was
missed; refinement continues. The test covers five supplied objects and 20
unseen layouts, with every failure retained and no rerolls.

[Dashboard](http://127.0.0.1:8766) · [Test report](docs/phase2_test01.md) ·
[Current state](docs/project_state.md) · [Frozen candidate](configs/phase2_candidate.json).

This is a three-skill hybrid: BC-initialized SAC+HER grasp/carry with continuing
demonstration regularization, ordinary SAC+HER reach, and declared planned
approach/carry-posture assistance. Learned navigation has no route oracle.
The original gripper and upstream physics remain unchanged.

The first test is preserved as a historical result and retired from future
held-out claims after further development. Its trajectories are not added to
training; a fresh confirmation bank was fixed before revision, while the
final-study reserve remains sealed. The earlier single-maze
[development baseline](docs/milestone1.md) remains frozen.

## Environment

Run commands from the repository root. This host already has the Python 3.10 environment and dependencies:

```bash
source .venv/bin/activate
python -c "import pybullet, gymnasium, stable_baselines3, torch; print(torch.__version__, torch.cuda.is_available())"
```

For a fresh checkout, restore the pinned source and install using an available Python 3.10 interpreter:

```bash
git clone https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project.git upstream/Pick-Nav-Reach-Default-Project
git -C upstream/Pick-Nav-Reach-Default-Project checkout --detach 54ab2d1faa54d1aa7e2738f114531a76c6514cd3
PNR_PYTHON_BIN=python3.10 bash scripts/setup.sh
source .venv/bin/activate
```

`PNR_PYTHON_BIN` can be an absolute interpreter path. [requirements.txt](requirements.txt) pins direct dependencies; [requirements.lock.txt](requirements.lock.txt) records the installed environment, including PyTorch 2.7.1+cu118. Setup keeps package caches and temporary files on the workspace volume. A fresh machine setup has not yet been independently reproduced.

Checks covering real physics, rewards, HER, geometry, and inference are available through:

```bash
python -m pytest -q
```

## Dashboard and reference

The current dashboard is **http://127.0.0.1:8766**. To start it when absent:

```bash
python dashboard/server.py --port 8766
```

From a remote workstation, forward the server's loopback port:

```bash
ssh -L 8766:127.0.0.1:8766 <user>@<host>
```

The dashboard displays live runs, scoped evaluations, reference/learned recordings, and host resources. Runtime files for the currently detached server are `dashboard/.runtime/server.pid` and `dashboard/.runtime/server.log`. [Logging and dashboard details](dashboard/README.md).

To reproduce the validated physical-reference case, use a new output directory:

```bash
python scripts/run_reference.py --seed 42 --object-idx 14 --video --output runs/reference_reproduction01
```

This command executes the original approach, physical pickup, maze transport, and retained-object reach. It writes results, a trajectory, video, and simple figures.

## Training and evaluation

For a new reach pilot using the current handover configuration:

```bash
python -m pnr.train --skill reach --run-id reach_example01 \
  --steps 20000 --seed 0 --scene-seed 42 --object-idx 14 \
  --horizon 100 --num-envs 4 --device cpu --gradient-steps -1 \
  --goal-features --eval-every 2500 --eval-episodes 1
```

Every run needs a unique ID. `--skill grasp` and `--skill carry` select the other adapters; their curricula and evaluation scopes differ. Inspect existing jobs before launching additional workers. Training records setup interactions, metrics, source snapshots, policy/replay checkpoints, and curriculum state under `runs/<run-id>/`. Resume requires matching skill/curriculum state; use `--resume` with the existing run ID and a larger total `--steps` budget.

Once candidate checkpoints are available, the full-system evaluator accepts run folders or exact model ZIPs:

```bash
python scripts/evaluate_system.py \
  --grasp runs/milestone1_validation01/artifacts/checkpoints/grasp.zip \
  --carry runs/milestone1_validation01/artifacts/checkpoints/carry.zip \
  --reach runs/milestone1_validation01/artifacts/checkpoints/reach.zip \
  --episodes 1 --scene-seeds 42 --object-idx 14 \
  --algorithm-label 'Demonstration-refined SAC+HER pilot' \
  --video --output runs/system_candidate01
```

It freezes the exact checkpoint bytes, begins at the original start, uses no inter-skill resets or planner waypoints during learned carry, and records failures as well as successes. Planned pre-grasp approach and carry-posture transitions are declared assistance. The selected development pilot passed the reported same-case checks; this does not establish held-out performance.

Generated videos, logs, and checkpoints stay on the workspace volume and are ignored by Git. Preserve the relevant `runs/` directories when transferring an experiment.

## Project documents

- [Aligned specification](docs/project_spec.md): task authority, learning boundary, evaluation, and final comparison.
- [Current state](docs/project_state.md) and [implementation record](docs/implementation_log.md): measured progress and known limits.
- [Upstream audit](docs/upstream_audit.md): source defects and compatibility decisions.
- [Project guide](mobile_manipulation_project_guide.md) and [handoff](mobile_manipulation_codex_handoff.md).

Original drafts are in [docs/archive/](docs/archive/). The unchanged presentation is historical; release/placement assumptions are superseded by the aligned specification.
