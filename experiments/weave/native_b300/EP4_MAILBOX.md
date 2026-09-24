# B300 four-GPU BF16 EP mailbox development reference

`ep4_mailbox.cu` is direct CUDA/PTX for the `balanced` case of the distinct
`weave-ep4-bf16-moe-b300-v1` CPU Workload Contract. Its fixed development
geometry is four ranks, eight source tokens per rank, eight experts with two
owned by each GPU, top-2 routes, BF16 hidden width 16 and FFN width 32.

Each GPU holds one peer-accessible mailbox. Communication CTAs claim their
source-rank routes, reserve a destination mailbox slot with a system-scope PTX
atomic, write the BF16 token and route metadata, then publish a system-scope
release flag. Computation CTAs claim ready inbound slots, use their local
expert's BF16 up/gate and down weights, write one FP32 contribution into the
source GPU's mailbox, and publish another system-scope release flag. Source
CTAs wait for complete chunks, acquire both route flags and combine into BF16.
The communication CTA class may claim computation slots from the same inbound
queue before combine, within an explicit steal budget.

The host launch checks exact B300 identity, cooperative occupancy, peer access
and native peer atomics for every directed pair before it starts the four
kernels. Source and destination allocations, resets, input preparation, GPU
admission and the independent post-lease oracle belong to the standalone test
driver outside the repository. The source has no Cake IR admission and proves
no numerical correctness, progress, SM split, overlap or performance until
that driver runs. It is a hardware reference for future rank placement,
remote-transfer and system-scope effects in Cake Program lowering.

## B300-M4 development evidence

The fixed source at `8eac0fac` compiled with CUDA 13.1 for `sm_103a` using
34 registers, one barrier, 128 bytes of shared memory and no spills. Each
device run requested exactly four exclusive broker GPUs. Its independent CPU
oracle ran after broker completion and compared every BF16 output element,
checked all inputs unchanged, and checked the destination queue counts.

| Workload case and plan | Broker job | Numerical/queue result | Steal on rank 0 |
| --- | --- | --- | ---: |
| `balanced`, per-rank `c=12/36/72/120`, `K=2` | `gpuq-da2ced47bcce` | 4/4 ranks passed, maximum absolute error 0 | 0 |
| `skew_to_rank0`, rank 0 `c=120,K=2,budget=32` | `gpuq-78bb976509e0` | 4/4 numerical and queue checks passed; preregistered steal coverage failed | 0 |
| Same skew inputs, rank 0 `c=147,K=1,budget=32` | `gpuq-03ea4f2a8403` | 4/4 ranks passed, maximum absolute error 0 | 14 |

The third plan leaves one regular compute CTA on rank 0 and exercises the
shared inbound queue's steal path. The failed coverage row is retained rather
than reinterpreted as success. Broker status after each completed job no
longer listed this task's allocation. Source, CPU input/oracle, compile log,
broker receipts, device snapshots and post-release reports are outside the
checkout under `open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/`:

- `cake-weave-ep4-b300-m4-8eac0fac/`
- `cake-weave-ep4-skew-b300-m4-8eac0fac/`
- `cake-weave-ep4-skew-c147-b300-m4-8eac0fac/`

This reference sends one mailbox message per route; it does not implement the
paper's deduplication of a remote token used by multiple experts. The tested
geometry is intentionally small and no latency, NVLink bandwidth, SM overlap,
large-model or end-to-end serving claim follows. Cake still needs explicit
rank placement, remote-transfer and system-scope queue/handoff analyses before
its Compiler can generate this distributed program.
