# First use: inspect a fused multiply-add plan

[中文原文](../GETTING_STARTED.md) · [English home](README.md)

You need a terminal, not a GPU. This tutorial checks a Schedule and generates source without calling AI or submitting an experiment.

[English guide](wiki/README.md) · [Next: reading the Schedule](wiki/schedule.md)

## 1. Prepare the project

Use an account with repository access:

```bash
git clone git@github.com:qhy991/open-cake-ir.git
cd open-cake-ir
python3 --version
```

Python 3.10 or newer is required. If the command reports an older version, use an already available newer interpreter before creating the environment. Create an environment and install the project and checking dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

An existing suitable environment is fine. Run subsequent commands from the repository root.

## 2. Understand the computation

The example computes `y=a*b+c`, so `2*3+4=10`. It processes eight rows of 128 values, combining matching positions. Inputs a, b, and c remain unchanged; y stores the output.

FP32 fused multiply-add rounds only once at the final result. A separately rounded multiplication followed by addition can differ. Use the complete [FMA Schedule](../../corpus/schedules/fma-b8-smoke.json); you do not need to author JSON yet.

## 3. Assess the plan

```bash
.venv/bin/open-cake-ir compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke.json
```

The current `--format text` display is Chinese. Look for `结构检查：通过` (structural check passed) and `生成代码：允许` (source generation allowed). `RESIDENCY_BOUND` may be an informational resource finding, not a rejection or performance result. Paths such as `operations[3]` locate a part of the JSON.

Omit `--format text` to get the complete machine-readable JSON. [Reading results](wiki/results.md) explains the separate outcomes.

## 4. Generate source

Create new output outside the checkout:

```bash
CAKE_TUTORIAL_DIR=$(mktemp -d)
.venv/bin/open-cake-ir compiler lower --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke.json \
  --output "$CAKE_TUTORIAL_DIR/fma.py"
```

The command prints the file path and entry `cake_fma_b8_smoke`. The source reads values, performs `fma.rn.f32`, and writes results. An existing output file is refused.

This is Triton source, not yet an executed GPU binary. Source generation supplies no GPU correctness result.

## 5. Inspect an intentional rejection

The repository includes a sibling missing an FMA input:

```bash
.venv/bin/open-cake-ir compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke-arity-drift.json
```

Expect a failed structural check and a nonzero exit status. That is the intended rejection, not a broken tutorial. The positive plan remains intact.

Now check the complete Compiler Corpus:

```bash
.venv/bin/open-cake-ir compiler check-corpus --format text \
  --revision compiler/revision.lock.json
```

Matching expectations includes both accepted positives and rejected negatives. Read the actual count from the command rather than copying an old report.

## 6. Continue

Read the [Schedule](wiki/schedule.md), [operator examples](wiki/operators.md), and [experiment workflow](wiki/experiments.md). The historical [Flash-KMeans teaching qualification](../../inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json) belongs to its old versions; it is not current machine setup advice.
