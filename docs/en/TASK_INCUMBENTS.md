# Reusing the best verified task implementation

Task incumbents let the next artifact-optimization Campaign compete against the best
verified implementation for the same exact cell. They do not change scientific reference
baselines.

## Promote a result

Promotion reads an original custody-bearing Campaign, audits it and writes one sealed Run
to an external incumbent registry:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python tools/promote_task_incumbent.py \
  --project-root /absolute/open-cake-ir \
  --registry-root /absolute/external/task-incumbents \
  --campaign-lock /absolute/campaign/campaign-lock.json \
  --evidence-root /absolute/campaign/campaign-evidence
```

The selected candidate must have a correctness-passing, measurement-quality-passing,
materially faster confirmation. A copied Git archive has no Filesystem Custody and cannot
promote. After the first promotion, the Campaign's fixed baseline must be the current
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
Evaluation Protocol. An exact match becomes the Campaign's fixed black-box baseline. A
missing registry or key explicitly falls back to the starter reference; the first material
confirmation can then create generation zero through the promotion command. The workspace records this in
`baseline-selection.json`; the Campaign Lock freezes the actual candidate identity and
bundle path.

## What does not happen

- Search-only, unstable, slower and close-null results do not advance the incumbent.
- A winner for one shape, target or timing protocol is not borrowed by another.
- Incumbent source is not shown to a clean-start author.
- Scientific matched studies do not silently adopt a rolling baseline.
- The registry never stores a mutable best-score table; current state is replayed from
  sealed promotion Runs.
