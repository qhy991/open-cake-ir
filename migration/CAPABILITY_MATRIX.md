# Final legacy capability migration matrix

Authority: `cake-repro@2fa79092...`, tree `b02d730b...`, 146-record manifest. “Implemented” means a current owner
exists behind one public Interface; live qualification is named separately where it has passed.

| Legacy capability/evidence | Canonical new owner | Current path | Acceptance evidence | State |
| --- | --- | --- | --- | --- |
| r16 parametric Schedule and negative drift | Compiler | `Compiler.assess/lower` | compiler contract tests + Corpus cases 1–2 | implemented, verified |
| r25 full assignment Schedule | Compiler | same Interface, closed imported lowering subset | Corpus cases 3–4 | implemented, verified |
| r31 TinyGEMM2 family | Compiler + Workload/Evaluation | same Compiler Interface + `evaluate_tinygemm` | Corpus cases 5–6 + Evaluation test | implemented, verified |
| exact B200/sm_100a limits | Target inside Compiler Revision | `compiler/targets/sm_100a.json` | target resource/instruction tests | implemented, verified |
| compiler promotion | Corpus Gate + approval | persistent Gate Report → Revision lock | release verify command | implemented, verified |
| r37 tie-aware correctness | Workload + common Evaluation | `evaluate_flash_kmeans` | both-arm r37 fixture parity | implemented, replayed |
| r39 paired timing | common Evaluation | `derive_paired_timing` | 250 retained samples | implemented, replayed |
| r40 missing resumed sandbox | provider boundary | `CodexInvocationBuilder` | initial/resume equivalence test | implemented, verified |
| r41 bracketed duplicate terminal | provider boundary | `normalize_codex_turn` | single/duplicate/nonidentical tests | implemented, verified |
| r41 clean-card race | Evaluation | `evaluate_with_admission_recovery` | exact zero-work predicate contract test | implemented, contract-verified; no live recovery event |
| r42 Turn/token/checkpoint control | Lab | `Lab.execute` + `project_checkpoints` | two-Turn zero-GPU test, actual r42 index | implemented, verified without GPU |
| r42 unavailable Estimand | Analysis Plan | `Lab.audit` | actual r42 103579/329934 fixture | preserved, replayed projection |
| provider/scaffold/model treatment | Authoring Environment | Campaign Lock + `CodexRunProvider` | G7 r4 closed-feature two-Turn qualification + G8 r6 | implemented, live qualified |
| Open Cake toolchain | Authoring Environment | `TritonToolchainBuilder` | external-anchored remote CUBIN qualification + G8 r6 receipts | implemented, live qualified |
| direct CUDA toolchain | Authoring Environment | `NvccToolchainBuilder` | closed manifest/artifact contract + G8 r6 receipts | implemented, live qualified |
| r43 frozen seed and three shapes | Lab → Compiler | `KernelSeed.schedule_for` then public Compiler | deterministic three-specialist test | implemented, verified |
| r43 source-layer custody failure | Candidate artifact contract | lowering and expanded source are distinct roles | regression test | implemented, verified |
| r44 headline-only tensor gate | Evaluation/Driver | `CudaTensorContract` + `LoadedCudaCandidate` | held-out b32/persistent load tests | implemented, verified without GPU |
| r45 exact dispatcher | PortfolioArtifact + Evaluation | `ExactShapeDispatcher` | supported/unsupported route test | implemented, verified |
| r45 persistent three-module lifecycle | Evaluation | `LoadedCudaCandidate` | one-load/multi-launch/one-unload test | implemented, verified without GPU |
| r45 30 raw cohorts and route accounting | Evaluation + Evidence | `CuptiPortfolioAssay` / raw receipt / semantic replay | actual 49,728-byte r45 result replay | implemented, replayed |
| r45 bounded Claim View | Lab Analysis | `audit_portfolio` derived from raw receipt | correctness true; held-out dispatcher stability false | implemented, replayed |
| r43/r44 failure archives | historical Evidence | final bundle + manifest + legacy indexes | exact index/tree digests | preserved, not active |
| r41/r42 HMAC receipts | historical Evidence | exact bytes in bundle; key remains external | recorded as previously verified local HMAC | preserved, not re-attested |
| serving | no owner | none | remote Stage 7/8 explicitly deferred | deliberately absent |

## Canonical-path status

- New-source contract suite, Compiler release verification, legacy manifest regeneration and raw r45 semantic replay
  are local acceptance evidence.
- G7 r4 qualifies the current provider boundary; G8 r6 qualifies the two-arm B200 system path without producing a
  scientific comparison. The only remaining migration gate is the explicitly approved G9 Git/sole-owner cutover.
- No legacy `rXX` runner, versioned verifier, registry or success/failure archive path is imported as active code.
