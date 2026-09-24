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
