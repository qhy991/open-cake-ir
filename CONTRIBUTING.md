# Contributing

open-cake-ir develops GPU kernels and compiler capabilities through agent-driven work.
Small, reproducible examples of missing expressibility, incorrect diagnostics, or lowering
errors are particularly useful. Describe the expected behavior, the program involved, and
the target; attach only the evidence needed to reproduce the issue.

## Development

Use a clean Git checkout with Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m open_cake_ir.cli compiler check-corpus \
  --revision compiler/revision.lock.json
.venv/bin/python -m pytest tests/contracts
```

The [CI](.github/workflows/ci.yml) discovers every contract test on Python 3.10, 3.11,
and 3.12. Individual tests report missing optional dependencies or exact historical
environment requirements as skips; whole modules are not excluded from discovery.
The released Executor's profiler admission test requires its bound runtime and host:
set `OPEN_CAKE_RUN_BOUND_EXECUTOR_TESTS=1` only in that environment. Tests requiring
Torch, a GPU toolchain, a specific device, or historical filesystem custody have
additional requirements; see the [runbook](docs/RUNBOOK.md). A normal checkout does
not reconstruct historical custody by restoring permissions.

The AKA expressibility tools accept `--parent-validator /absolute/path/to/validate_completion.py`.
The selected validator is bound when a work root is created and cannot be replaced
when continuing it. Contract tests use an explicit protocol fixture; passing them
does not qualify a real parent kernel or establish custody.

## Propose a change

Read [AGENTS.md](AGENTS.md) and the nearest applicable design decision before editing.
Compiler changes and new primitives require the independent review and release procedure
in [ADR 0052](docs/adr/0052-independent-agent-release-review.md). Check the frozen source
closures before changing implementation, examples, or corpus inputs; historical releases
and expectations are not edited to make a new change pass.

Use a focused branch and pull request. Explain the behavior changed, its purpose, and the
relevant validation. Keep generated experiment directories outside the checkout. A new
kernel result should distinguish correctness, timing, profiling, and its workload scope.

By submitting a contribution, you confirm that you have the right to contribute it under
the project's [Apache-2.0 license](LICENSE), and that you have identified any third-party
material and preserved its required notices. See [third-party provenance](THIRD_PARTY_NOTICES.md).

中文设计与使用说明从[中文入口](docs/zh-CN/README.md)开始。提交问题时可以使用中文或英文。
