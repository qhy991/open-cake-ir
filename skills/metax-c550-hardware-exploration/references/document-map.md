# MetaX C550 document and evidence map

Reviewed 2026-09-26. This is a source-finding aid, not a copied manual or a capability declaration. Recheck the portal and version whenever using it.

## Official MetaX documentation

| Source | Useful for | Version/access notes |
|---|---|---|
| [MetaX Developer Center: programming references](https://developer.metax-tech.com/doc?primary_category=%E7%BC%96%E7%A8%8B%E5%8F%82%E8%80%83) | MXMACA releases, EID/error codes, API and programming docs for C500/C600 | Search index showed 3.9.0 release notes (updated 2026-09-24) and EID (2026-09-22). Open/download the exact version before using details. |
| [MetaX Developer Center: AI application docs](https://developer.metax-tech.com/doc?category_ids=59) | `mcTriton`, `mcPyTorch`, `mcApex`; backend support, extensions and environment configuration | Index showed the C500/C600 `mcTriton` user guide at 3.9.0, updated 2026-09-22. Treat this as the first reference for the Triton route, then match it to the actual runtime. |
| [MetaX Developer Center: performance tools](https://developer.metax-tech.com/doc?primary_category=%E7%BC%96%E7%A8%8B%E5%8F%82%E8%80%83) | `mcProfiler`, `mcTracer`, `mxvs` | The current index lists all three at 3.9.0. `mcProfiler` describes SOL/Roofline views and Memory/Computing/Scheduling metrics; `mcTracer` captures runtime/kernel activity; `mxvs` lists memory-bandwidth, compute and stress tests. See the linked [mcProfiler manual](https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf), [mcTracer contents](https://developer.metax-tech.com/api/client/document/preview/1190/index.html) and [mxvs contents](https://developer.metax-tech.com/api/client/document/preview/996/index.html). |
| [C500/C600 programming quickstart contents](https://developer.metax-tech.com/api/client/document/preview/459/index.html) | MXMACA programming model, C++ extensions, runtime APIs, libraries and environment setup | The portal index showed quickstart 3.8.3.x, but this preview TOC reports 2.33.0.x. Verify the downloaded document/version in the authorized portal before using it; do not merge the two under one version label. |
| [C550 product page](https://www.metax-tech.com/prod.html?cid=107&id=35) | Public product-level facts | Lists 64 GB per module and 896 GB/s eight-GPU full-interconnect bandwidth. The latter is interconnect, not per-GPU DRAM bandwidth; the page does not establish a usable kernel roofline. |
| [Official profiler-field Q&A](https://developer.metax-tech.com/forum/t/guan-yu-mcprofilerjie-guo-de-yi-xie-zi-duan-de-yi-wen/674/post/3312/) | Supplemental explanations for profiler fields such as `wsm_stall` and `vls_pipeline_stall` | Treat forum replies as contextual supplemental evidence. Bind them to the profiler version and retain the original question/answer locator. The discussion notes that fuller metric documentation was still planned. |
| [FlagTree backend/version matrix](https://github.com/flagos-ai/FlagTree#multi-backend-support) | Find the MetaX backend's maintained Triton branch and guide before investigating an API mismatch | The public `main` branch listed MetaX on Triton 3.6 when reviewed on 2026-09-26. That does not replace the captured C550 3.1 distribution; compare the actual imported `triton` module and distribution in each container. |
| [Triton `range`](https://triton-lang.org/main/python-api/generated/triton.language.range.html) and [`static_range`](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html) API pages | Distinguish a loop's pipelining and partial-unroll attributes from a compile-time unroll directive | These describe current upstream APIs, not MACA acceptance or numerical behavior. The exact installed MetaX interpreter, TTIR/TTGIR and device result decide the local route. |
| [Triton `fma` API](https://triton-lang.org/main/python-api/generated/triton.language.fma.html) | Identify a fused ternary source operation distinct from separate multiply and add | The API page does not establish MACA rounding or device support. For the current C550 route, compare the emitted IR and a distinguishing exact-rational oracle such as `tools/probe_fma_fp32.py` against complete device output. |

The public index exposes titles and download entries, but full-page extraction may fail or require an authenticated portal session. Some indexed PDF material is marked “MetaX Confidential” or “沐曦专有信息”. Confirm access and redistribution terms before downloading or embedding full text. Keep source links and short, attributable notes when full-text redistribution is not authorized.

## Repository sources: examples, not hardware authority

- [MetaX-MACA/TileOPs-Metax](https://github.com/MetaX-MACA/TileOPs-Metax/blob/dev/README.md) describes manifest-driven operator signatures, workloads and roofline formulas for Agent-oriented kernel development. It is an actively developed TileLang-based operator library and identifies C500-specific implementations. Use it to find hypotheses, workload manifests and candidate patterns. Pin the reviewed commit and validate every pattern through Cake's own target and evaluator; this is not an Open-Cake Compiler capability contract.
- [MetaX-MACA/mcoplib C500 optimization guide](https://github.com/MetaX-MACA/mcoplib/blob/main/Skills/optimized-cuda-kernels/references/c500-optimization-guide.md) is a useful search-term source, but its C500/CUDA framing and internally inconsistent bandwidth figures make it unsuitable as an unreviewed source of Target facts. Do not use its A100-relative guidance or stated peaks as C550 limits without independent, version-matched verification.
- [Torch-FL MetaX low-precision matrix operations](https://github.com/flagos-ai/Torch-FL/blob/main/docs/vendors/metax/installation.md#low-precision-matrix-operations) document that project's software FP8/FP4 path: decode to BF16, then ordinary device GEMM. It is a separate implementation and numerical contract, not evidence that direct FP8 `tl.dot` lowers on C550 or that the Cake Workload oracle would accept BF16 intermediate rounding.
- Vendor samples, framework plugins and optimized kernels can show what that repository implemented for its own version and workload. Record repository, commit, license, product/runtime, workload and whether the content contains a complete implementation. Do not expose low-level implementations to a clean-start authoring arm unless the frozen reference-access assignment permits them.

## Open-Cake canonical owners

In the MetaX platform worktree, use these as current repository context and check them at the task's commit:

- `docs/metax-c550.md` — current supported path, software identities, verified capability boundary, MCPTI measurement and profiler limitations.
- `docs/metax-c550-bringup.md` — C550 host/device probes, versioned environment observations and the bring-up history.
- `compiler/targets/xcore1002.json` — canonical Target facts. Physical device ID, native code-generation family, code object and Triton compatibility API fields are distinct.
- `runtime/hosts/xcore1002.json` — captured host/runtime/tool versions; use the current campaign's captured record, not a stale example.
- `findings/` and the platform's existing result/evidence paths — measured defects and target-specific outcomes. Do not copy results into a second hand-maintained catalog.

## Minimum provenance for extracted knowledge

For each reused fact or hypothesis, retain the source title and URL, product scope (C500 family vs exact C550), SDK/driver/Triton version, document release/date, section/page, access/license status, and confidence class (`vendor_doc`, `device_probe`, `measured_receipt`, `vendor_QA`, or `sample_hypothesis`). Store a concise paraphrase and the question it can answer; do not copy full manuals or credentials.

Resolve disagreements in this order: report them; inspect the exact versioned vendor source and captured C550 host; run only the authorized read-only/device probe or qualified measurement needed to distinguish them; update the canonical Target/host or the relevant finding only through its owning workflow. A documentation statement alone never qualifies correctness, timing quality, profiler attribution or a performance ceiling.
