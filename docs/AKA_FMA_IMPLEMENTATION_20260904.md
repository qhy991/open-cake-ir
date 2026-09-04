# Bounded FP32 FMA implementation — 2026-09-04

## Result

The [approved FMA design](AKA_IR_OWNER_REVIEW_20260904.md) is implemented in
the isolated `codex/aka-fma-ir-successor-20260904` branch. Tested Compiler sources are
commit `1271432c22e7ba6986bfe25a0009c7058b499433`; subsequent documentation does not
change that source projection. The existing cycle derived `open-cake-ir-sm100a-v41-draft`.
This is a draft implementation, not a released Compiler or a complete AKA parent.
Log and cosine are not implemented by this change.

The implementation extends existing `elementwise` with three same-shaped FP32 register
reads and one result, a required `ptx.fma.rn.f32` contract, and no scalar/broadcast
fields. Schema and parser agree. The verifier rejects bad arity, result count, dtype,
memory space, shape, missing edges, wrong operation/instruction pairing and unsupported
targets/backends. In particular, a tanh instruction cannot be borrowed for FMA, nor FMA
for tanh. The ordinary template coverage test now checks all three operand positions.

Triton emits explicit non-FTZ/non-saturating RN-even inline PTX. Nested FMA and the
separately rounded multiply producer retain distinct dependencies. Work accounting
charges two arithmetic operations per output; existing pressure analysis already walks
all read edges, and a new test confirms that all three distinct inputs remain accounted
for. No physical-register, latency, performance or precision-promotion model was added.

## Verification

All remote checks ran as `qhy-sol` on `verda-b200x4` in a new isolated directory:
`/home/qhy-sol/aka-fma-v41-20260904-V6w5BZ/`.
The pre-existing `/home/qhy-sol/kda-runtime-venv/bin/python` provides Python 3.12.3,
Torch 2.12.1+cu130 and Triton 3.7.1; the existing system package directory supplies
jsonschema. No dependency installation, credential inspection, other-account access,
GPU allocation or experiment submission was performed. `CUDA_VISIBLE_DEVICES=""`
was set, and compilation used an explicit CUDA-100 target rather than a detected GPU.

| Check | Observed result | Boundary |
| --- | --- | --- |
| FMA focused tests | 16/16 passed, included in the 208 below | Parser/schema, adversarial assessment, lowering, work, pressure, and real GPU-free compilation |
| Draft-compatible shared regressions | 208/208 passed | Explicit IR, verifier, Triton, CuTe, analysis, work and FMA modules |
| Five released-identity observation tests | 5/5 passed on exact predecessor `ff8903d` | Separate v40 snapshot, not v41 draft coverage |
| Historical Corpus Gate | 68/68 matched, 91 bound sources | All old expected findings and lowering identities unchanged |
| Remote Gate reconstruction | Existing release tool `--prepare-gate --verify` passed | Rebuilt against exact transferred source commit |
| Ordinary FMA offline compile | 1 `fma.rn.f32`, 9,560-byte cubin | CPU compiler evidence only |
| Nested/rounded-producer offline compile | 2 `fma.rn.f32`, separate `mul.f32`, 9,824-byte cubin | CPU compiler evidence only |
| Release cycle | Exit 3, exact Gate not externally approved | Expected stop; no successor release lock |

The two compile tests inspect actual PTX, not just generated Python strings. Neither
PTX contains an FMA FTZ or saturation modifier. Their cache, PTX, cubins and logs remain
outside Git under the remote directory above:

- `draft-regressions-v2.log`: final 208-test run and both compile observations.
- `v40-replay.log`: the five separate released-identity tests.
- `triton-cache-v2/`: compiler outputs, including the two named FMA kernel PTX files.
- `fma-focused.log` and `compiler-regressions.log`: preserved initial attempts.

The first local test invocation did not load because macOS system Python lacked
jsonschema. This was not counted as a pass, and the local environment was not repaired.
The first remote shared suite had one stale binary-arity coverage assertion and five
old-lock/new-Target errors. The former was corrected by using the canonical template
path for FMA and checking three placeholders; the five historical tests were not edited.
The final draft run enumerated and excluded exactly those five, then ran them separately
on the unchanged predecessor. This is not a claim that all 213 tests pass against v41.

No GPU numerical correctness, sanitizer, throughput, timing, profiler or framework
end-to-end result exists for the new primitive. PTX/cubin existence cannot supply those
claims. The previously published 56 AKA dynamic-valid results remain unrelated old
Compiler evidence.

## Frozen history and pending Corpus admission

The current v40 lock had been used by external Lab runs but was absent from the local
witness discovery globs. Before source edits, its lock, Gate, approval and source-set
were preserved in `compiler/releases/v40/`; the normal cycle then derived v41 without
hand-choosing an id. Current `compiler/revision.lock.json` and
`compiler/release-approval.json` remain identical to the predecessor. No Executor,
calibration, old Schedule or `corpus/manifest.json` was changed.

Four new candidate Schedules are source-bound but **not yet admitted to the canonical
Corpus manifest**:

- `fma-b8-smoke.json`: positive three-input FMA;
- `fma-chain-b8-smoke.json`: positive nested FMA with a rounded product;
- `fma-b8-smoke-arity-drift.json`: negative missing operand;
- `fma-b8-smoke-instruction-drift.json`: negative wrong instruction meaning.

[The proposed four manifest entries](AKA_FMA_CORPUS_PROPOSAL_20260904.json) reuse the
existing Corpus schema. They are a review input, not a second Corpus authority. Their
boolean/finding expectations reflect the stated contract and focused tests; their
identity fields bind the candidate artifacts for a later reviewed adoption. A match
against freshly generated identities is not counted as independent correctness evidence.
The 68/68 historical Gate must not be described as a 72/72 FMA-inclusive Gate.

## Next gate

An independent reviewer must inspect the source change and four proposed entries,
including adversarial malformed public API inputs, operand ordering, producer rounding,
unsupported backends and analysis consistency. Only a separately reviewed adoption may
append those entries to `corpus/manifest.json`; do not regenerate old expectations.
Then prepare the FMA-inclusive Gate and obtain the external approval required by
[ADR 0030](adr/0030-compiler-release-approval-is-external.md). This automation did not
write that approval, release v41, merge main or start GPU work.

The existing `Review published AKA IR results` task completed its latest turn without
a returned assistant message or review artifact. It supplies **no independent concurrence**.
The implementation and test evidence above stand on their own and remain ready for an
independent review; task completion alone is not approval.
