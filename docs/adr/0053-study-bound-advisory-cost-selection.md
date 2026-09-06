# ADR 0053: Study-bound external advisory candidate order

Status: accepted software contract; model compatibility and GPU benefit require their own evidence.

The Lab's candidate-set path builds and seals every submission before spending GPU
time on a bounded search subset. Released `Compiler.rank` requires calibration
qualification; an external `EmpiricalCostModel` does not supply that qualification.
The current Flash-KMeans authoring environment also cannot consume the existing
FMA/GEMM models as though they described the same operator or runtime.

An optional Open Cake arm field declares one advisory selection policy:

```json
"candidate_selection": {"kind": "external_empirical_advisory_v1"}
```

Only the Open Cake arm of the existing Flash-KMeans/direct-CUDA assay, using
candidate-set `matched_search`, `claim_scope=artifact_optimization_only` and the closed
matched event vocabulary, admits this policy. Native Triton pairing and other Workloads
refuse policy activation because their assay context has not been established here;
their existing no-policy authoring, execution and replay paths remain supported.
Scientific comparison, system qualification, portfolio and the direct CUDA arm do not
admit this policy. A stable Study template
declares policy only. Model selection is an execution input, supplied explicitly to
`Lab.preflight(study_path, empirical_cost_model_path=...)` or the same CLI:

```sh
python -m open_cake_ir.cli --project-root /path/to/open-cake-ir lab preflight \
  /external/study.json --empirical-cost-model /external/model.json \
  --output /external/campaign.lock.json
python -m open_cake_ir.cli --project-root /path/to/open-cake-ir lab execute \
  --lock /external/campaign.lock.json --runtime-config /external/runtime.json \
  --evidence-root /external/new-evidence
python -m open_cake_ir.cli --project-root /path/to/open-cake-ir lab audit \
  --lock /external/campaign.lock.json --evidence-root /external/new-evidence
```

These commands describe the existing lifecycle; the policy does not authorize an
experiment or waive its admission gates. Preflight requires policy and model together,
reads the external model once and freezes its complete document under the resolved arm
in the CampaignLock. The retained Run authority includes this lock. Composition and
replay consume the frozen model, never a mutable external path. No Study successor per
model or Executor is necessary, and no frozen model is automatically relabeled.

The existing model context fields express one exact comparison boundary. The Lab
derives the following projection from the frozen Workload, selected case and Executor:

| Field | Required value |
| --- | --- |
| `timer` | `flashinfer.testing.utils.bench_gpu_time_with_cupti;use_cuda_graph=false` |
| `cache_protocol` | `cold_l2_cache=true` |
| `runtime.compiler_version` | Executor `host_environment.packages.triton` |
| `runtime.executor_revision` | Exact Executor canonical content identity |
| `input_scope` | Compact, key-sorted JSON string containing `case_id` and `workload_contract_sha256` |

The whole context must equal this projection, including the exact runtime key set.
The Executor identity binds the assay source, actual FlashInfer helper and admitted
host closure. Its identity is a comparison boundary, not evidence of hardware truth.
The cache field records the actual helper argument; it does not infer a cache byte
count or equivalence to another flush implementation. Missing or different facts,
unsupported runtime declarations, a different Workload/case, or different Compiler
content identity or target produce explicit abstention. Live composition still admits
the actual Executor host. Offline replay uses the frozen execution context, not the
auditor's host. The private Lab projection is the sole owner of this comparison format;
the existing assay and Executor remain the owners of timing behavior and runtime facts.

`EnvironmentResult.empirical_cost` is separate from calibrated `cost`. Assessment,
Workload/tensor admission and successful compilation precede an estimate. Only a
complete launchable set with covered comparable predictions is sorted, ascending by
`predicted_kernel_us`; equal predictions retain provider order. If any launchable
candidate is unknown, every launchable candidate retains provider order. Rejections
remain last in provider order. There is no partial known-subset sort, fallback to a
different cost model, empirical-range pruning or promotion based on predictions.
The existing semantic duplicate collapse, search budget and confirmatory acceptance
remain authoritative.

When the policy is bound, `candidate_set_filtered` adds `candidate_selection` with
the applied decision/reason; each order row adds `empirical_cost` with the existing
model/Compiler identity, target, coverage, point estimate, empirical range and refusal
reason, or null for a rejection. Arbitrary supplier context/provenance maps and raw
observations remain in the frozen model, not next-turn feedback. The common reference
producer likewise projects only the selection kind and existing model/Compiler/target
identity fields into the author-visible `run-authority.json` and Ralph `TASK.md`;
neither interface receives model curves or arbitrary supplier metadata. The complete
CampaignLock remains unchanged. The same decision and rows reach the next Turn on the same
provider thread, including unknown and unmeasured candidates. No-policy archives keep
their prior shape. Audit reconstructs predictions from retained submissions and the
frozen model, recreates the stable order from provider order, and requires the exact
retained projection. A measured material inversion can use the existing cost-model
diagnosis; provider order alone is not a model-error claim.

The delivered v51 FMA/GEMM models remain outside this path: their Compiler binding is
stale against the current release, and their templates, input contracts, Kineto
timer/cache context and runtime do not match the Flash assay. Positive
Flash fixtures establish software wiring only. This change adds no coefficients,
coverage, calibrated ranking qualification, scientific comparison, GPU savings or
performance result. Compiler code is unchanged. The Lab and CLI change requires an
independently reviewed Executor successor; old descriptors and evidence stay frozen.
