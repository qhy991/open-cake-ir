# ADR 0076: Complete Programs and independent transfer Runs

Status: accepted design; implementation and acceptance are tracked in the task branch.
Supersedes ADR 0075's ownership of composition structure and ADR 0073's requirement
that every execution begins with a matched Study. Historical runs use their pinned source.

## Contract

One path must serve ordinary kernel optimization, faithful reference reproduction, and
controlled cross-target transfer. Compiler owns executable structure and deterministic
rewrites; Evaluation owns device storage, launch, oracle and measurement. Lab owns
frozen execution permissions and budgets. Study owns treatment assignment and analysis,
never candidate syntax, backend routing or the execution state machine.

A Program declares its exact target, public tensors and ordered complete Schedules.
Its first domain is static same-device, same-stream composition: immutable public inputs,
fresh outputs, one writer per tensor and no overlapping bindings. Global state and scratch
are outside this composition domain; single-Schedule execution retains its own existing
contract. Every intermediate has a producer and consumer. The only initial view is
explicit insertion/removal of singleton axes with unchanged storage order and dtype.
This accommodates QSA's actual boundary without introducing a layout algebra.

Program syntax, use-def legality and per-stage Compiler assessment have one owner.
Device adapters must verify contiguous storage, nonaliasing and view identity before launch.
A transformation selects named stages in a complete Program, proves its own applicability,
and returns a complete successor or a localized refusal. Fusion proves that the removed
intermediate is private and has exactly the selected consumer; it preserves the public
ABI, explicit low-precision rounding and all unselected stages. Existing pass domains and
hardware evidence sets are unchanged. No transform implies device correctness or speedup.

A Run freezes Workload, source and executor, authoring environment, material and transform
permissions, common evaluation, agent configuration and budgets. It is the execution
authority that replaces the execution responsibilities of CampaignLock. Engineering runs
need no fictitious comparison or estimand. A Study freezes assignments before execution
and binds the same Runs. Treatment identity cannot affect representation or lowering.

Wall time allocates a fixed confirmation reserve inside the total Run budget. Search
admission stops at total minus reserve; the confirmation phase begins at search closure
and cannot borrow unused search time. A search Turn and an Evaluation remain atomic for
evidence collection, bounded by their existing adapter timeouts. This is a scheduling and
acceptance contract, not hardware preemption: retain actual overruns, and never classify
a token, authoring-time, search-time or confirmation-time overrun as success at budget.
All Study conditions freeze the same allocation and stopping semantics before execution.

Authors submit complete implementations or explicit transform requests. Requests refer
only to this Run's accepted implementations or authorized baselines. An action is retained
separately from the immutable resulting candidate. All candidates use the same filtering,
oracle, search accounting and fresh confirmation. Permission refusals and failed rewrites
are recorded; they are not silently converted into submitted implementations.

Knowledge consists of mechanism text, source/counterexample evidence references and optional
Compiler transform references. It contains no duplicate hardware support or measurement
registry. Frozen material selection and call grants implement the E/P factors; actual use
is recorded. Withheld material and code must be inaccessible to the author process,
not merely omitted from its prompt. Cross-run candidates and context cannot flow between
conditions. Discovery feeds successor knowledge and studies, never a running allocation.

## P1–P8 and acceptance

P1: single Schedules remain ergonomic; composition adds only genuine multi-stage structure.
P2: launches, storage and explicit rewrites remain inspectable. P3: one Program owns
composition; no parallel Lab graph. P4: bindings and dtype/shape errors fail at construction.
P5: complete use-def information supports local transformation proofs. P6: focused
before/after and refusal contracts accompany the full unchanged Corpus Gate. P7: dataflow
analysis and consumers migrate with the model. P8: same-stream launch completion and
explicit rounding define the hardware behavior; estimates never authorize acceptance.

Acceptance requires ordinary optimization without Study, reference reproduction with a
private-intermediate fusion plus public-output/multiple-consumer counterexamples, and four
same-Cake E/P conditions with enforced isolation, shared accounting and fresh confirmation.
Native environments, exact target refusals, independent replay, append-only evidence and
historical replay at original commits remain supported. The new execution entry replaces
old dispatch and parsing after validation; compatibility remains only at real external
input boundaries. Implementation completion requires independent Compiler review and
integration through main into all maintained platform branches.
