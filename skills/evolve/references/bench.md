# Freezing a rolling Bench comparison

`tools/evolve.py freeze-bench` supports one bounded engineering comparison: two clean
source commits, an independent Bench with `suite.json`, explicit tasks, and fresh
search with shared author controls. It generates the complete randomized allocation
before any assigned workspace exists. It creates no Run, Provider or device lease.

Prepare a draft JSON with exactly these fields:

| Field | Value |
| --- | --- |
| `hypothesis` | The maintenance mechanism and observable question. |
| `task_ids` | The full intended list of task IDs from the pinned `suite.json`. |
| `replicates` | Positive integer selected before outcomes are seen. |
| `allocation_seed` | Nonnegative integer fixing allocation order. |
| `author` | Shared fields described below. |
| `measurement_protocol` | Tracked file path relative to the Bench checkout. |
| `baseline_files` | Task ID to tracked community-baseline file path, relative to that Bench. |
| `total_author_wall_seconds` | Authorized ceiling that funds both conditions and every replicate, including confirmation. |

`author` has exactly `model`, `scaffold`, `knowledge`, `reference_access`,
`wall_time_seconds`, and `confirmation_seconds`. The first four are explicit nonempty
selections: name the model with effort, source-pinned scaffold/material selection and
reference policy. Use `none` for no extra knowledge. The default engineering allocation
is 10,800 seconds including 1,800 seconds for confirmation; a different authorized
budget must be declared before launch. Token limits are not fields in this contract.

The file references identify the Bench's code and protocol. The owning launcher must
resolve their actual pinned library/data artifacts and admit the common assay. A file
existing in Git is not proof that its baseline is executable or its timing is qualified.
Use the fixed community reference throughout; `bench-best` may be a delivery artifact
or a separately declared comparator but cannot silently replace this denominator.

```bash
python3 tools/evolve.py freeze-bench \
  --draft /research/evolution/bench-draft.json \
  --bench /checkouts/hardware-bench \
  --control /checkouts/cake-control \
  --successor /checkouts/cake-successor \
  --output /research/evolution/comparison/plan.json
```

All checkouts must be clean committed roots. The output must be outside Git checkouts.
Each comparison directory owns exactly one `plan.json`. A different comparison needs
a different directory, so two plans cannot assign the same Run workspaces.
The command freezes their commits, exact target, selected Bench tasks, shared controls,
fixed references and all condition/task/replicate workspaces. Identical retries return
the existing plan. A different plan cannot overwrite it; existing allocated workspaces
prevent a first freeze. Source identity checks here establish the version boundary,
not a new digest catalogue.

Before any actual launch, use the selected condition's own checkout to prepare its Run
through the platform's current supported entrypoint. Reconcile the prepared binding
with this frozen intent. Verify target/runtime/image/toolchain qualification, baseline
artifact identity, numerical contract, timing/confirmation protocol, author material
and fresh context. Different physical hosts require qualified comparable routes; an
alias or target string does not establish that equivalence. The two conditions differ
by system source version, which currently binds both Compiler and Executor. A narrower
Compiler-only claim needs evidence that the other changed components did not cause it.

For an external hmz launcher, use the shared
[Compiler author adapter](../../../docs/HMZ_COMPILER_TOOLS.md). Freeze its source in
the common scaffold, generate each condition's API from that condition's Compiler,
and verify that the actual author loop delivers it. Keep the platform evaluator and
budget owners. Do not retrofit active Runs or apply this all-public-tools adapter to
a Study arm that withholds transformations. Syncing the file to a platform branch
does not establish launcher adoption or device qualification.

The current native `matched_search` schema implements the four-cell E/P transfer
Study. This two-condition engineering plan is not accepted by that Study API. Do not
invent E/P flags or a cross-target transfer to make it pass. This skill coordinates
existing independent Bench execution and keeps its engineering scope explicit.

Run each frozen allocation once under the existing launch owner. Reconcile results
against the entire allocation; unsupported programs, failures and missing measurements
remain in the denominator. The same baseline artifact is remeasured with the candidate
in the qualified assay. Preserve A/A noise and paired order requirements from the
Bench protocol. Never replace a historical latency with a new reference or fill an
unsupported condition with infinite speedup.

The existing result/Finding owner records disposition and the next selected Compiler.
This plan does not create another result writer or auto-promote the successor. A flat
or negative result can close a round. Preserve the rejected source and evidence even
when the next development round continues from the prior adopted Compiler.

The format deliberately describes rolling engineering validation. It does not create
a formal heldout study, validate OS isolation, perform a fixed-program artifact assay,
or implement crash recovery. Use the owning qualified interfaces for those tasks.
No automated `advance` command is advertised by this skill.
