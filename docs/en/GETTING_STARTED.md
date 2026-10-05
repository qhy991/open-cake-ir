# First use: inspect a fused multiply-add plan

[中文原文](../GETTING_STARTED.md) · [English home](README.md)

You need a terminal, not a GPU. This tutorial checks a Schedule and generates source without calling AI or submitting an experiment.

[English guide](wiki/README.md) · [Next: reading the Schedule](wiki/schedule.md)

## 1. Prepare the project

Use an account with repository access:

```bash
git clone https://github.com/qhy991/open-cake-ir.git
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

FP32 fused multiply-add rounds only once at the final result. A separately rounded multiplication followed by addition can differ. Write the kernel in [examples/python/fma.py](../../examples/python/fma.py); its shape, target, execution groups, and FMA contract are in that Python file.

## 3. Assess the plan

```bash
.venv/bin/open-cake-ir compiler assess --format text \
  examples/python/fma.py
```

The current `--format text` display is Chinese. Look for `结构检查：通过` (structural check passed) and `生成代码：允许` (source generation allowed). `RESIDENCY_BOUND` may be an informational resource finding, not a rejection or performance result. Paths such as `operations[3]` locate canonical IR; Python authors also get source locations.

Omit `--format text` to get the complete machine-readable JSON. [Reading results](wiki/results.md) explains the separate outcomes.

## 4. Generate source

Create new output outside the checkout:

```bash
CAKE_TUTORIAL_DIR=$(mktemp -d)
.venv/bin/open-cake-ir compiler lower --format text \
  examples/python/fma.py \
  --output "$CAKE_TUTORIAL_DIR/fma.py"
```

The command prints the file path and entry `cake_fma_b8_smoke`. The source reads values, performs `fma.rn.f32`, and writes results. An existing output file is refused.

This is Triton source, not yet an executed GPU binary. Source generation supplies no GPU correctness result.

## 5. Optional: inspect canonical IR

The Compiler constructs a canonical Schedule internally from the Python file. The
[FMA JSON sample](../../corpus/schedules/fma-b8-smoke.json) shows its serialized form
for Corpus tests; it is not an input you must supply when writing or lowering this kernel.

## 6. Continue

Continue with [Python authoring](PYTHON_FRONTEND.md), [Schedule fields](wiki/schedule.md), [operator examples](wiki/operators.md), and [experiment workflow](wiki/experiments.md). The historical [Flash-KMeans teaching qualification](../../inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json) belongs to its old versions; it is not current machine setup advice.
