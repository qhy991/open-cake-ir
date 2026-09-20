# Contributing

open-cake-ir develops GPU kernels and compiler capabilities through agent-driven work.
Small, reproducible examples of missing expressibility, incorrect diagnostics, or lowering
errors are particularly useful. Describe the expected behavior, the program involved, and
the target; attach only the evidence needed to reproduce the issue.

## Development

Start with the [branch and worktree guide](docs/DEVELOPMENT_BRANCHES.md): platform tasks
target their platform branch, shared changes target `main`, and each task has its own
worktree. The guide owns branch names and the integration flow.

Use a clean Git checkout with Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m open_cake_ir.cli compiler check-corpus \
  --revision compiler/revision.json
```

**Run the suite in its own worktree, at a commit.** Source identity is the clean commit of
the checkout, so a tracked file changing while the suite runs takes that identity away and
every test needing a Compiler or an Executor fails from that moment on — which reads like
broken code rather than like a checkout someone edited. The suite takes tens of minutes,
and this repository is often worked on from more than one place at once.

```bash
git worktree add /tmp/oci-test HEAD
cd /tmp/oci-test && PYTHONPATH=src python -m pytest tests/contracts
```

Say which commit a reported run was at. When you are the only one touching the checkout and
it is clean, running in place is the same thing.

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
Compiler changes and new primitives follow the Corpus Gate and commit-based identity
procedure in [ADR 0065](docs/adr/0065-source-identity-is-the-commit.md), with independent
review before integration into `main` under the branch workflow. Historical records replay
at their producing commits; historical releases and expectations are not edited to make
a new change pass.

Use a focused branch and pull request. Explain the behavior changed, its purpose, and the
relevant validation. Keep generated experiment directories outside the checkout. A new
kernel result should distinguish correctness, timing, profiling, and its workload scope.

By submitting a contribution, you confirm that you have the right to contribute it under
the project's [Apache-2.0 license](LICENSE), and that you have identified any third-party
material and preserved its required notices. See [third-party provenance](THIRD_PARTY_NOTICES.md).

中文设计与使用说明从[中文入口](docs/zh-CN/README.md)开始。提交问题时可以使用中文或英文。
