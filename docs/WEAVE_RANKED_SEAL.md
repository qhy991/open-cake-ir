# Exact B300 ranked-tile pre-launch seal

The distributed model EP4 Workload is frozen separately from the Compiler's
typed Program, ranked effects and combine Schedule. A candidate must bind
both to one rank-local `communication_ctas`/`chunks`/`steal_budget` plan and
one compiled CUDA host-library byte stream before any device state is made.
`RankedTileLaunchManifest` and `RankedTileCandidate` provide this NVIDIA
Evaluation boundary. The manifest retains the Workload identity, case,
exact target and Compiler commit, complete typed documents, toolchain
requirements, source map and ordered rank plans. The candidate retains
immutable manifest, emitted source and ELF bytes. At load, the library on
disk must match those bytes and the source and plan must match the current
lowering and Workload.
The loader requires the caller to assert an isolated process explicitly;
a partial multi-rank launch can leave device waiters live, so the adapter
may not silently claim process isolation on the caller's behalf.

This is deliberately a **pre-launch byte binding**, not a claim that NVCC
produced the ELF from the retained source. The CPU build command/report must
establish that provenance separately, and the compiled ABI query still
refuses an incompatible library before CUDA state creation. The module does
not yet turn a distributed launch into the repository's common
`LaunchableCandidate`, append-only Evaluation receipt or qualified timer.
It does not add a digest catalogue; the byte comparison happens once at the
candidate-to-library handoff, where a mismatch changes the next action.

At `c65531f6`, the 14 related contracts and 179-case Corpus Gate pass in a
clean detached worktree. The tests exercise manifest round-trip, wrong
Workload case, rank-divergent temporal plans, malformed source-map extent,
wrong library bytes and refusal before the device loader. B300-M4 is not
reachable from the current host, so no device library was loaded through
this new boundary. Promotion disposition: retain the task branch pending
one real four-GPU sealed replay and common Evaluation integration.
