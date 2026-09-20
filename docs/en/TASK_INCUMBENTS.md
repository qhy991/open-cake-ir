# Reusing the best verified task implementation

Task incumbents let the next engineering Run compete against the best
verified implementation for the same exact cell. They do not change scientific reference
baselines.

## Promote a result

Promotion reads an original custody-bearing engineering Run, audits it and appends a
sealed promotion to an external incumbent registry:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python tools/promote_task_incumbent.py \
  --project-root /absolute/open-cake-ir \
  --registry-root /absolute/external/task-incumbents \
  --run /absolute/task/run.json \
  --evidence-root /absolute/task/run-evidence
```

Retained artifact-optimization Campaigns use `--campaign-lock` with their original evidence
root. Study-assigned Runs cannot use the engineering entrypoint to bypass their Study
promotion policy. Both inputs use the same material-win and append-only registry writer;
existing records remain unchanged.

The selected candidate must have a correctness-passing, measurement-quality-passing,
materially faster confirmation. A copied Git archive has no Filesystem Custody and cannot
promote. After the first promotion, the Run's fixed baseline must be the current
incumbent, preventing a side experiment from overwriting a stronger result.
Historical Campaigns are audited through their own Executor-bound Python and source tree,
not through the current TaskPackage renderer.

## Use incumbents in the next task or matrix

Add the registry to the ordinary launcher:

```sh
python tools/launch_task.py ... \
  --incumbent-registry /absolute/external/task-incumbents
```

or to the matrix:

```sh
python tools/launch_task_matrix.py ... \
  --incumbent-registry /absolute/external/task-incumbents
```

Each task resolves a key containing Workload identity, case, exact target, backend and
Evaluation Protocol. An exact match becomes the Run's fixed black-box baseline. A
missing registry or key explicitly falls back to the starter reference; the first material
confirmation can then create generation zero through the promotion command. The workspace records this in
`baseline-selection.json`; `run.json` freezes the actual candidate identity and bundle
path. A registry-current Program or native kernel uses its own audited execution contract;
starter/reference baselines retain their source and launch matching checks.

## What does not happen

- Search-only, unstable, slower and close-null results do not advance the incumbent.
- A winner for one shape, target or timing protocol is not borrowed by another.
- Incumbent source is not shown to a clean-start author.
- Scientific matched studies do not silently adopt a rolling baseline.
- The registry never stores a mutable best-score table; current state is replayed from
  sealed promotion Runs.
