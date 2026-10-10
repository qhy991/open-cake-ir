# External hmz Compiler tools

[`tools/hmz_compiler_tools.py`](../tools/hmz_compiler_tools.py) is the shared author
adapter for external hmz engineering workspaces. It exposes the selected Compiler's
transformation catalog, resolves an explicit action against the author's own source,
and replays the retained result. The implementation is shared through `main` by
`metal`, `amd`, `dcu`, `nvidia`, and `metax`.

The adapter reuses `lab.knowledge.transformation_surface` and
`lab.actions.resolve_action`. The Compiler registry owns names, parameters,
preconditions, and refusal reasons. There is no second registry in hmz. The external
launcher still owns the author loop, budget, evaluation, confirmation, and release.
Native Lab retains its existing Run, Study, and historical replay consumers.

## Workspace and version boundary

This adapter supports the existing external campaign layout. It does not convert an
arbitrary native Run or create a new workspace schema.

| Existing owner | Required input |
| --- | --- |
| Independent Bench binding | `campaign/binding.json` with `compiler_commit` |
| Development binding | `campaign/development-binding.json` with `compiler` |
| Selected Compiler | Clean checkout at `.deps/cake-ir`, matching the binding |
| Prepared source | Git-tracked `campaign/compiler-api.json` |
| Bench intake | `campaign/intake.json` with `source_commit` |
| Development intake | `campaign/deadline.json` with `source_commit` |
| Search budget | `campaign/deadline.json` with `search_stop_at_epoch` |

Exactly one binding must exist. Before intake, the catalog is checked against `HEAD`.
After intake, it is checked against the prepared source commit. An approved development
`campaign/search-close-owner.json` also closes mutation. Replay remains available.

Pin the adapter as part of the external launcher source. A launcher may retain an
exact copy of this file if it records the source commit and checks the copy when
updating it. Such a copy is a projection, not a separately maintained implementation.
Use the same adapter version in both comparison conditions. It imports Compiler code
from each workspace's `.deps/cake-ir`, not from the adapter's producing checkout.
This permits a control Compiler that predates this adapter. Use a fresh process for
each Compiler condition; an already imported different Compiler is refused.

## Connect the author loop

1. Prepare a new workspace with its binding and clean selected Compiler.
2. Run `freeze` before intake, then commit the generated catalog with prepared source.
3. At intake, call `catalog(root)` to check the frozen catalog.
4. Add `author_context(root)` to each actual hmz agent request. It contains the frozen
   API and summaries of the last 20 own actions, with an omitted count.
5. Expose `inspect`, `transform`, and `verify` to the author. Send resulting complete
   candidates to the existing platform evaluator. Retain `stage_origin` in its source
   receipt when evaluating a generated stage.

Run the CLI from the external workspace. An importing launcher can instead call
`main(root)` with its explicit workspace root.

```bash
python3 /checkouts/cake-tools/tools/hmz_compiler_tools.py freeze
python3 /checkouts/cake-tools/tools/hmz_compiler_tools.py catalog
python3 /checkouts/cake-tools/tools/hmz_compiler_tools.py inspect \
  --parent campaign/candidates/candidate.py
python3 /checkouts/cake-tools/tools/hmz_compiler_tools.py transform \
  --id trial-001 --request campaign/candidates/request.json
python3 /checkouts/cake-tools/tools/hmz_compiler_tools.py verify --id trial-001
```

The request has `action: "transform"`, an own `parent` path, a `transformation` name
from the frozen catalog, and its `parameters`. Use `inspect` for the actual stage name.
Each create-only `campaign/compiler-actions/<id>/` retains the request, parent
snapshot, result or refusal, and any complete Program and stage Schedules. Reusing an
ID fails. Interrupted actions remain unknown. Verification recomputes the result from
the retained snapshot; editing the author's next proposal does not rewrite history.

Parents are restricted to own candidates, own Schedules, declared development
inheritance, and replayed prior tool results. Paths outside the workspace are refused.
Independent Bench authors do not inherit development results through this adapter.

## Adoption and evidence

BW1100's Bench and development launchers are the first consumers. The shared tool
contains no HCU, CUDA, MACA, Metal, GPU allocation, native compilation, timing, or
model invocation. Its CPU contracts exercise own-source inspection across all five
vendor routes and transformation replay without device work.

A platform branch receiving this file establishes source availability. Enabling it
in another launcher requires the actual intake and per-turn prompt connections above,
tested with that launcher's frozen binding and source receipt. Target support and
transformation applicability remain the selected Compiler's decisions. Each platform
still needs its own numerical and timing evidence; a generated candidate is not an
accepted optimization.

For a fresh engineering round, use external hmz where its launcher is qualified and
retain native Lab where it is still the execution owner. Do not install this adapter
into an active or historical Run. A controlled Study's material and pass-access
treatments take precedence: this all-public-tools engineering adapter does not
implement a withheld-transformations arm.
