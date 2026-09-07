# ADR 0055 — One fixed-baseline paired assay per candidate

Status: proposed Executor successor; requires independent review and release.

A Study's old `paired_cupti` wording did not drive paired GPU execution. The
worker measured one candidate, and both receipt and Lab could accept that single
cohort result. Comparing independent authoring runs cannot establish interleaved
candidate/baseline measurement.

The successor gives `evaluation_protocol.paired_timing` the numerical assay
policy. The B300 RMSNorm, GEMM+bias and gather templates explicitly declare roles
`candidate` and `baseline`, ten alternating AB/BA pairs (five of each), 25 samples
and 42 callbacks per cohort (six estimation, eleven warmup, 25 timed), maximum CV
0.05, materiality ratio 1.05 and six pair wins. These are a new assay policy, not
inherited B200 calibration or a qualification threshold. Correct stable slower
and `close_null` candidates remain observed outcomes. The outer experimental unit
and estimand stay independent authoring runs, with three runs per arm and the same
150000-token, four-turn, three-candidate budget. Fresh confirmation is a separate
job; profiling remains a separate attribution assay.

Each candidate is measured against the same previously sealed baseline. Both
modules load in one exclusive allocation. One retained `StrictCuptiBenchmark`
loop prepares fresh poisoned outputs before every cohort, runs cold L2 without a
CUDA graph, then checks every callback's outputs and unchanged inputs. Both
participants have preflight and postflight oracle checks. Raw records bind the
actual order, positions, full sealed identities, output checks and allocation.
`derive_paired_timing` remains the only numerical derivation owner. Receipt
construction, Lab execution and Lab audit reject incomplete pairs, wrong identities,
order/count drift and single-cohort evidence under this policy.

Study templates mark runtime-owned provider, toolchain, broker and baseline
leaves with `{"binding":"campaign_lock"}`. `lab preflight --execution-bindings`
resolves a closed external locator document into a Campaign Lock. It binds the
qualified provider/anchor, current Executor, exact runtime configuration and
baseline bundle without changing the Study, its source references, model,
reasoning effort, feature policy, budgets or analysis. Qualification files,
baseline bundles, runtime configurations, Campaign Locks and new evidence remain
outside source worktrees. Existing broker request bundles are read without
rewriting their historical request or Executor identity. The baseline must match
the frozen Compiler's projected kernel and launch commitments; Python and JSON
authoring identities may differ.

The current Ralph request projects its verified immutable TASK.md and AGENTS.md
plus the dynamic StateCard into the actual provider prompt. The provider archives
the same canonical bundle that supplied the request. This preserves two-file
ownership without requiring an unavailable file-reading tool. It proves content
delivery, not that the model internally read a file. The qualifier must consume
the same projection; its immutable TASK owns both qualification turns.

The new closed provider surface explicitly disables `code_mode` and
`code_mode_only` as well as the previously disabled host and shell surfaces.
This responds to the CLI startup error but is not provider qualification. Startup
and in-turn errors remain rejected; the changed configuration requires a new
real two-turn receipt. Process transport variables such as `ALL_PROXY` are passed
through existing environment sanitization but are not captured in provider
configuration identity, so transport replay is not established by that receipt.

Historical experiments remain in pinned Git. The current Lab uses only the
Ralph TASK.md/AGENTS.md interface; no legacy authoring interface is restored.
New live native Campaigns require the explicit paired policy and current closed provider surface;
removing the policy cannot re-enable single measurement. Frozen source, Compiler
locks, Executor descriptors and prior evidence remain unchanged. This change
needs an Executor successor including the new binding and paired-validation
modules, host qualification, an actual paired B300 canary and independent review
before scientific runs. No Compiler primitive or calibration is added.
