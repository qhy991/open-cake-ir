# Replay an old task from its complete frozen source

[中文原文](../../wiki/replay.md) · [English home](../README.md)

Main can advance while old experiments remain bound to their original source. Placing an old lock beside new code and fixing the resulting errors does not replay history.

[Guide index](README.md) · [Results](results.md)

Separate the source commit, the runtime environment, and the retained results. Reconstructing source and a preparation bundle does not run a GPU or recreate missing historical results.

| Task | Complete source commit | Compiler | Reconstructable boundary |
| --- | --- | --- | --- |
| state-store v5 | c40bb399c7bebe026bcd30e27255dfc15ad23f5c | v41 | Bundle and static checks; GPU pending |
| FMA correctness | d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d | v41 | Ordinary/nested preparation; historical observations remain separate |

State-store v4 uses `82491bb7658a4da92be444fa029b26b02635a154` and Compiler v40. These are historical locators, not instructions to reset main.

## A no-GPU replay

After [environment setup](../GETTING_STARTED.md), run from the main checkout:

```bash
CAKE_REPLAY_DIR=$(mktemp -d)
git worktree add --detach "$CAKE_REPLAY_DIR/source" \
  c40bb399c7bebe026bcd30e27255dfc15ad23f5c
.venv/bin/python \
  "$CAKE_REPLAY_DIR/source/examples/gpu/state_store_b200_correctness/prepare_candidate.py" \
  --output-root "$CAKE_REPLAY_DIR/bundle"
```

The command keeps the current branch in place and runs the old checkout's actual Compiler, checking its lock and source closure. Expect v41, an accepted positive, and rejected owner/axis siblings. New provenance.json and preflight.json describe this preparation; deployment and GPU evaluation have not happened. Existing output roots are refused.

For FMA, create a worktree from the listed FMA commit and call its examples/gpu/fma_b200_correctness/prepare.py, also supplying `--python` for the evaluator interpreter and `--judge-cwd` for its working directory. Preparation alone can use an appropriate local interpreter; actual GPU execution must qualify the host. Read the [task notes](../../../examples/gpu/fma_b200_correctness/README.md).

A shallow clone may lack the required objects. Obtain the exact history; do not skip source verification. Mixed Python modules, changed old permissions, or a more permissive oracle cannot restore original evidence.
