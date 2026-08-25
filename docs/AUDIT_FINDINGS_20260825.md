# Audit findings register, 2026-08-25

This is the current-`main` disposition of the external audit originally performed at
`f503de4` and rechecked at `e351b67`. The original prose lived in a divergent old
worktree; every conclusion below was reproduced or falsified again before changing its
disposition.

## Disposition summary

| # | Finding | Current disposition |
| --- | --- | --- |
| 1 | `EpilogueFormula` was parsed but ignored | fixed in Compiler v23 |
| 2 | tutorial linked an obsolete, unparsable Schedule | fixed at the documentation authority |
| 3 | committed Evidence can fail custody checks after clone | fixed by orthogonal archive-integrity and filesystem-custody facts |
| 4 | historical ranking checker duplicates current order | refuted; historical/current split is intentional |
| 5 | residency provenance was operator-supplied | fixed; host and time derive from the run |
| 6 | Compiler release writes its own approval | fixed and independently exercised by Compiler v25 |
| 7 | revision identities cause Study successor churn | fixed by stable templates and exact CampaignLocks |
| 8 | lowering profile is both routing and workload constraint | fixed in Compiler v24 |
| 9 | emitter preconditions were late failures | fixed in Compiler v23 |
| 10 | tiled accesses used the coordinate owner's extent instead of the accessed Buffer's extent | fixed in Compiler v25; B200 successor retains two separate numerical failures |
| 11 | TinyGEMM2 regenerated a different input distribution and compared output bitwise to its own oracle | fixed by Workload v2 and Executor v29; current checked-asset launch passes |

## Re-verification and action

### 1 and 9: backend semantics now fail before lowering

The defect remained present: changing the full CuTe-DSL assignment epilogue from
`centroid_sq_minus_two_dot` to `bias_add_bf16_round` left the Schedule accepted and
lowering-eligible, while source still contained centroid arithmetic. Backend preflight
is now the single owner of emitter-only requirements. Assessment projects those
requirements as lowering-blocking Findings, and direct emitter use consumes the same
preflight. The formula drift is a reviewed negative Corpus case rather than only a unit
test. ADR 0027 records the boundary.

TinyGEMM2 already expressed and checked the four-part CTA sum. Its epilogue formula was
only indirectly protected by a whole-Schedule digest; asset preflight now explicitly
requires `bias_add_bf16_round`. The kernel remains a closed source asset and truthfully
reports `generated=false`: schedule-level reduction semantics are solved, full code
generation is not.

### 2: fix the live tutorial, retain frozen history

`examples/gpu/flash-kmeans-b32-smoke.json` still carries the removed whole-operator MMA
formula and does not parse. It is retained because frozen Executor closures refer to it.
The actual quickstart already defaults to the valid `-v2` successor; the documentation
link now points to that same authority, and a contract test parses the linked example.
Runtime example JSON files are not Schedules, so “every JSON under examples parses as a
Schedule” would be an invalid gate.

### 3: custody and archive integrity are different claims

Executor v27 now gives `EvidenceStore.audit_run` two orthogonal results. A read-only audit
can verify the authority, event chain, terminal seal and referenced object bytes under
weak clone-time modes, while `filesystem_custody_verified` separately observes the owner
and modes of those exact paths. `EvidenceStore.writer` retains strict live-custody
admission.

A byte-identical copied G8 Campaign now replays with archive integrity and semantic replay
true, filesystem custody false and system qualification false. Content tampering still
fails archive integrity. The old normalization helper was removed: changing modes just
before audit cannot prove continuous historical custody and is no longer necessary for
read-only inspection. ADR 0031 owns the boundary without adding an archive format,
signature service or second auditor.

### 4 and 5: one refuted, one fixed

The old v6 ranking checker intentionally replays the historical total order; current
ranking uses the newer preorder implementation. Editing the old checker would corrupt
negative evidence. Residency instruments now obtain UTC time and hostname from the
machine. Frozen calibration-v6 tooling is unchanged.

### 6: successor preparation no longer approves itself

`release_compiler_cycle.sh` now prepares the full failing-capable Corpus Gate but never
creates or changes `release-approval.json`. A missing, malformed or stale approval stops
the cycle with the prior lock and approval bytes intact. After a reviewer outside that
automation writes an approval bound to the exact Gate digest, rerunning the same command
consumes it through the existing release validator. An executable contract proves both
the refusal and success transitions.

The v24 approval predates this repair and records the releasing repository owner, so it
is not retroactively independent evidence. Compiler v25 closes the successor limitation:
the tmux 882 window1 reviewer independently recomputed the final Gate digest, checked all
48 source receipts and the 32-case semantic diff, refused two stale intermediate Gates,
then wrote only the approval bound to the final Gate. The release automation consumed
that approval without a second release implementation. ADR 0030 owns this boundary
without adding signatures or accounts.

### 7: stable design templates, exact execution locks

The defect was reproduced: five current zero-GPU Study fixtures differed from their
predecessors only because Compiler or Executor identity advanced. ADR 0028 now separates
stable experimental design from execution authority. A `template` Study has one legal
revision spelling, `{"binding":"current_release"}`. `Lab.preflight` resolves both
authorities exactly once; the resulting CampaignLock retains the Study digest plus exact
Compiler and Executor id, path, and digest. A `frozen` Study still requires exact
references and cannot follow current state. The two freeze tools replace both bindings
atomically before emitting a frozen successor.

The five current fixtures are now templates. Historical v37 files were not edited, and an
Executor release no longer requires five Study copies. This does not pretend that an old
unexecuted frozen Study verifies against a newer source tree: only terminal evidence earns
a copied Executor source archive.

### 8: route and Workload now have separate owners

Compiler v24 removes `metadata.profile` without a compatibility alias. Schedule syntax
has one typed `lowering = {backend, entry_point}` record. Its three backend values name
materialization mechanisms, not operators; the two generated-backend rows are independent
of the thirteen corpus program slices. Global Buffers derive the executable signature, so
the unused and already-wrong ABI labels are gone.

The Lab remains the Workload consumer: it checks the opaque Workload digest and exact
external tensor contract for the admitted case. The Compiler checks Schedule/Target
legality and backend support. As failure-capable evidence, a GEMM bias-extent variant now
lowers without a named-Workload exception, while the Lab would reject a tensor mismatch
inside a frozen Flash-KMeans Run.

TinyGEMM2 is the bounded exception, not a counterexample. Its route names one checked
source asset; the four-part CTA sum, BF16-round epilogue and whole-Schedule semantic pin
block only lowering, not IR acceptance. The reduction drift Corpus case is therefore
`accepted=true, lowering_eligible=false`. ADR 0029 records the ownership boundary.

Frozen pre-v24 observations and ranking calibrations keep their old Schedule/source
bytes. They were not re-labelled as v24 evidence. The frozen v24 diagnostic attempt and
the v25 successor now provide the historical/current split for generated lowering, while
a successor ranking calibration remains missing. The v25 generated partition remains
12/14. TinyGEMM2 separately has a passing v25 checked-asset observation; neither it nor
the historical r31 launch is relabelled as v26 evidence. The v26 Corpus Gate is 33/33 and
the zero-GPU suite passes 407 pytest items plus 284 subtests. Its new KDA weighted-combine
case has a separate preregistered B200 correctness check. The first attempt retained an
infrastructure failure before compilation because the worker could not import Torch; a
successor binds the CPU-preflighted site-packages path. These gates validate the
ownership migration and the new type relations but do not substitute for either GPU
evidence or the missing ranking successor.

### 10: an access is bounded by the Buffer it addresses

The frozen v24 plan selected the complete 14-case generated-lowering domain before GPU
work. Its one broker job retained 11 passes, one numerical GEMM failure and two crashes.
The batched Flash-KMeans result audit indexed a rank-three distance tensor as though its
output were rank one. The GEMM shape-drift oracle also assumed that bias width equalled
output width and crashed before compilation. More importantly, emitted source masked its
128-element bias with the 256-element matrix axis, allowing an out-of-bounds read.

ADR 0032 assigns one owner to the bound: `AccessMap.indices[position]` addresses the
accessed Buffer's `shape[position]`. Program axes and loops still own work decomposition,
not every Buffer reusing their coordinate. No new primitive, profile, layout abstraction
or compatibility mode was added. The observer now flattens batched tie rows with a
shape-consistency exception, and the independent GEMM oracle implements the pre-existing
masked-zero semantics for a shorter bias.

The preregistered v25 successor (`gpuq-948eb76ab045`, shared B200, no timing, zero retries)
wrote all 14 records after compile and launch. Flash-KMeans now passes and GEMM shape
drift now reaches a durable result, closing both crash paths. Twelve records pass. Both
GEMM rows remain failed under the unchanged `1e-5` maximum-absolute tolerance, with
maximum deviation `3.814697265625e-05` and 3,997/3,993 violating elements respectively.
That is retained numerical evidence, not a reason to widen the threshold or claim the
generated partition passed. The v25 TinyGEMM2 checked-asset observation passes separately
and does not change those generated-lowering dispositions. v26 leaves its source asset
and Schedule unchanged, so it neither relabels nor needlessly repeats that observation.

### 11: materialized input and parent output are separate authorities

The frozen TinyGEMM2 v1 Contract named the upstream and historical generator but reduced
its case to `mode=random_normal`. The historical generator divides input and weight by
eight before BF16 conversion; the current adapter did not. Because the same adapter also
created the FP32 oracle, its CPU test was internally consistent while exercising a
different distribution. A second dormant error treated bitwise equality to that oracle
as bitwise equality to the pinned upstream parent, even though retained r31 evidence
shows those outputs are close but not bitwise identical.

ADR 0033 leaves v1 unchanged and makes v2 the current authority. Its sole case pins five
raw receipts: input, weight, bias, independent FP32-linear/BF16 oracle and retained
upstream-parent output. One preregistered shared-B200 regeneration initially failed before
materialization because the broker worker could not traverse the temporary worktree; the
failure is retained. Its custody-only successor (`gpuq-f5c26074936e`, no kernel, no
timing, zero retries) reproduced all four generated receipts exactly. The parent receipt
remains bound to the separate r31 upstream-parent invocation.

Executor v28 restores `/8` scaling, verifies each generated receipt before candidate
launch, and implements the two declared predicates independently: output bytes must
equal the pinned parent and output values must satisfy the unchanged FP32-oracle
tolerance. An inventory plan that binds exact Executor descriptor bytes now witnesses
that revision during release, preventing another observed descriptor from being reclaimed
in place.

The first attempt to reach the current binary exposed two staged-import closure failures;
both happened before materialization and are retained. The next shared-B200 attempt
(`gpuq-e425666c5de2`, no timing, zero retries) passed all 17 byte-authority checks and all
three generated input receipts, then failed before candidate launch. The adapter had
evaluated the receipt-pinned historical CPU oracle with CUDA matmul, producing different
BF16 bytes. Executor v29 makes the existing declaration executable without adding a mode:
it moves all three operands to CPU, calls FP32 linear, rounds to BF16, and compares output
and oracle metrics on CPU. This repairs the second self-consistent oracle defect.

The preregistered successor (`gpuq-b6053f720096`, shared B200, no timing, zero retries)
then passed all 17 byte-authority checks and the four generated receipts. It launched the
pinned CUBIN exactly once with zero fallback, synchronized and unloaded the module,
matched the retained parent output bitwise, and matched the independent CPU oracle at
maximum absolute error `0.000244140625` under the unchanged `0.01/0.01` tolerance. The
raw observation is retained byte-for-byte. Neither r31 nor the failed v28 attempt was
relabelled; the current checked-asset partition now passes without authorizing a
performance or scientific claim.
