# B300 ranked-tile effect-to-CUDA audit

This is the NVIDIA task's review ledger for `RankedTileEffects` schema 2.
`native_cuda_ranked_tile.source_event_map` locates 27 protocol edges and all
25 Cake math operations in the emitted source. The map is an inspection aid;
the B300 device and oracle runs, not the comments, establish the tested
correctness scope.

| Effect | Owner and emitted mechanism | Boundary checked |
| --- | --- | --- |
| Payload | `dispatch_source_wave` reserves one remote slot per `(source rank, token, destination rank)` with `atom.relaxed.sys.global.add.s32`; CTA threads copy BF16 hidden, then `st.release.sys` publishes `payload_ready`. `gather_wave_tiles` uses `ld.acquire.sys` before reading that slot. Local routes read local hidden. | `model_ranked_tile_pointer_run.prepare` counts unique remote destinations. `aff2ea61` and `3c75a716` device reports observed 618/312/306/300 slots on the mixed case and 1,536/0/0/0 at the hot-8 capacity boundary. |
| Expert bin | The source CTA reserves each `(expert,row)` with a system-scope returned-old atomic, records source/token/route identity and payload slot, and publishes row/route readiness with system release. The destination gather acquires the route flag and checks expert and payload identity. | `test_ranked_tile_effects` refuses GPU-scope bin reservation. Changed-route CPU plans and B300 manifests cover every route once. |
| Logical tile | `derive_wave_order` sorts route keys for one expert; `assign_wave_tiles` emits expert IDs and padded tile keys with a checked 255-slot bound. `publish_wave_ready` releases all tile data for one source-completion or wave-end event; the worker acquires that event. | Dense 192-tile evidence checks all 20 event counts. Full tiles publish after the source chunk that filled them; publication at the exact 128th-row arrival *inside* that kernel is not claimed. |
| Stage task | `expand_tasks_for_wave` derives 24/128/32 work units from the three admitted ProgramMaps. The worker's successful GPU-scope `atomicCAS(task_head, old, old+1)` returns the old subtile and co-realizes reservation and claim. A system-scope event handoff publishes the task set. Predecessor CTA completion uses GPU-scope acquire and acq-rel increment. | `compose_model_ranked_tile_stages` binds the stage counts, six execution groups, 49,152 B declared allocation and 49,200 B emitted CTA footprint. The four-rank oracle checks every stage unit completed. Independent review should decide whether one atomic may satisfy both IR fields `reservation` and `claim`; adding a second cosmetic atomic would change queue semantics. |
| Return | `scatter_returns` derives `(source rank,item,route)` from the tile route key, writes one FP32 contribution to that deterministic source slot, then publishes `return_ready` with system release. `wait_returns` acquires every local slot before the Cake combine. | Changed-route full-chain verification checks down-tile producer rows against returned contributions bitwise, all 16,384 ready flags, and final output against an independent FP64 oracle. |
| Steal | A communication-class worker CTA uses a bounded GPU-scope CAS permit and then claims the same stage head and runs the same Cake stage body as a computation CTA. Actual stolen units and permit count are checked before ABI launch returns. | The rank-local plan `[74,1,74,1]` with budgets `[5888,0,5888,0]` observed steals `[5888,0,5888,0]`; three launches on one handle had bitwise-identical output. `c` classifies worker CTAs; dispatch itself runs in separate kernels and `c` does not reserve that many physical SMs. |

`ranked_tile_b300_host.cu` reads device names, compute capability and SM count
from the Target-bound renderer, verifies peer access and native peer atomics,
and refuses duplicate/out-of-range expert IDs before publishing. The ABI
requires an isolated process: a partial multi-rank enqueue failure poisons
the handle and `destroy` refuses to free storage beneath possibly live GPU
waiters. `ranked_tile_launch.prepare_ranked_tiles` checks tensor shape, dtype,
device placement, exact byte spans, aliasing, rank-local plans and this
process-isolation condition before binding.

The exact B300 mixed-route public Compiler→Evaluation replay at `f7cb1ac5`
passed three controls with zero FP64-oracle failures and zero repeated-output
bit mismatches. This does not establish arbitrary concurrent source arrival,
a device-fault cleanup proof, a common Workload launch manifest, or qualified
CUPTI/L2-reset latency. No automatic Compiler pass or Lab schedule rule is
promoted from it.
