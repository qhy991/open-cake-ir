# Architecture

Six diagrams for the six things that are hard to see from the source tree. They describe
what the code does today, including where it falls short of
[`PAPER_CONTRACT.md`](PAPER_CONTRACT.md) — a diagram that flatters the implementation is
worse than none.

Vocabulary is defined in [`CONTEXT-MAP.md`](../CONTEXT-MAP.md); claim boundaries live in
[`ACCEPTANCE_GATES.md`](ACCEPTANCE_GATES.md).

---

## 1. Contexts and the dependency direction

The repository's central structural claim: the Compiler is usable on its own, and the
arrow never points back at it. Verified by a contract test that imports
`open_cake_ir.compiler` in a fresh interpreter and asserts no Lab, Evaluation or
Evidence module was loaded.

```mermaid
graph TD
    subgraph product["Open Cake Compiler — the product core"]
        C["<b>Compiler</b><br/>typed Schedule IR · exact Target<br/>contract verifier · lowering"]
    end

    subgraph app["Research Lab — a dependent application"]
        L["<b>Lab</b><br/>Study Contract · Campaign Lock<br/>Turn loop · Analysis"]
    end

    subgraph shared["Shared supporting contexts"]
        EV["<b>Evaluation</b><br/>Workload-owned oracle<br/>measurement protocol"]
        ED["<b>Evidence</b><br/>no-follow CAS · hash-chained ledger<br/>terminal seal · read-only replay"]
    end

    L -->|"freezes one immutable Revision"| C
    L --> EV
    EV --> ED
    ED -.->|"Run Audits feed the<br/>pre-declared Analysis"| L

    C -.->|"never"| L
    linkStyle 4 stroke:#c00,stroke-dasharray:4 3
```

A runtime observation may motivate a compiler change, but only an external Corpus Gate
and a human merge can create the next Revision — see diagram 5.

---

## 2. The Compiler pipeline

`assess` composes three owners; `lower` is a fourth stage that has not caught up.

```mermaid
graph LR
    S["Schedule<br/><i>JSON</i>"] --> IR

    subgraph assess["Compiler.assess"]
        direction TB
        IR["<b>ir.py</b><br/>typed Schedule<br/><i>structural admissibility</i>"]
        VER["<b>verifier.py</b><br/>target-derived hard gates"]
        TGT["<b>target.py</b><br/>exact hardware contract"]
        PROF["profile rules<br/><i>semantic digest pin</i>"]
        IR --> VER
        TGT --> VER
        IR --> PROF
    end

    VER --> A["<b>Assessment</b><br/>accepted · lowering_eligible<br/>localized Findings"]
    PROF --> A
    A --> LOW["<b>Compiler.lower</b>"]
    LOW --> ART["target source<br/><i>Triton · CuTe-DSL · CUDA</i>"]

    IR -. "structural violation<br/>becomes a Finding,<br/>not an exception" .-> A

    style LOW fill:#fff3cd,stroke:#b8860b
    style PROF fill:#fff3cd,stroke:#b8860b
```

`lower` generates the warp-specialized profile from its Schedule — warp dispatch,
mbarrier storage and participants, TMA descriptors, TMEM custody, the loop nest and every
operation body — verified on a B200 at 128/128 exact assignments. The two remaining
profiles still stamp a checked-in file, which is why the box is amber; each will follow
once its Schedule carries the commitments emission requires. `profile rules` pin a
whole-document digest for those two, and that pin disappears with the file it protects.

### Findings

Every finding names the offending path and the violated contract. The four categories
are the ones the paper's harness reports; the three severities are its three
dispositions.

```mermaid
graph TD
    F["<b>Finding</b><br/>code · path · message"] --> CAT & SEV

    subgraph CAT["category"]
        C1["schedule_semantics<br/><i>structural invariants</i>"]
        C2["hardware_conformance<br/><i>resource · instruction · architecture</i>"]
        C3["data_consistency<br/><i>dataflow · producer/consumer</i>"]
        C4["program_safety<br/><i>synchronization · ordering · hazards</i>"]
    end

    subgraph SEV["severity"]
        S1["blocking<br/><i>rejects, with a reason</i>"]
        S2["report<br/><i>a likely limit</i>"]
        S3["hint<br/><i>a commitment declined</i>"]
    end
```

---

## 3. What a Schedule commits to

The paper has the agent write down five concrete hardware commitments. Each one the
Schedule declines is a decision the backend makes instead, invisibly.

```mermaid
graph TB
    subgraph decl["Declarations"]
        R["<b>roles</b><br/>named warp groups"]
        AL["<b>allocations</b><br/>shared bytes · tensor columns"]
        B["<b>buffers</b><br/>shape · dtype · stages<br/>byte_offset · swizzle"]
        P["<b>pipelines</b><br/>stages"]
        BA["<b>barriers</b><br/>producers · consumers<br/>count · mechanism"]
        PM["<b>program_map</b> / <b>grid</b><br/><b>tile_loops</b> · <b>access_maps</b>"]
        OP["<b>operations</b><br/>kind · reads · writes<br/>waits · signals · depends_on"]
    end

    subgraph derived["Lowering derives"]
        D1["warp identity<br/>and dispatch"]
        D2["mbarrier storage,<br/>counts and phases"]
        D3["SMEM offsets,<br/>TMEM column range"]
        D4["TMA descriptors"]
        D5["loop trip counts"]
    end

    R --> D1
    BA --> D2
    P --> D2
    AL --> D3
    B --> D3
    OP --> D4
    PM --> D5
```

For the retained warp-specialized Schedule the emitter derives all thirteen module
constants its hand-written artifact hardcodes, and generates a kernel that runs correctly
from those declarations alone. See
[`tests/contracts/test_ir_sufficiency.py`](../tests/contracts/test_ir_sufficiency.py),
which compares them directly.

---

## 4. A Study Run

The Lab owns every transition. The provider and the evaluator only return observations.

```mermaid
stateDiagram-v2
    direction TB
    state "provider<br/>one candidate authored" as provider
    state "environment<br/>compile / lower / seal" as environment
    state "evaluation<br/>oracle then timing" as evaluation
    state "fault<br/>classified by live stage" as fault
    state budget <<choice>>

    [*] --> run_started
    run_started --> provider: Turn N
    provider --> environment: candidate sealed
    environment --> rejected: compile or assess fails
    environment --> evaluation: launchable artifact
    rejected --> budget: findings become feedback
    evaluation --> budget: receipt becomes feedback

    budget --> provider: budget remains
    budget --> checkpoints: budget exhausted

    provider --> fault: any exception
    environment --> fault
    evaluation --> fault
    fault --> checkpoints
    checkpoints --> endpoint
    endpoint --> sealed: terminal archive
    sealed --> [*]
```

The paper's inner loop has four stages — generate several structurally distinct
candidates, **filter** them with construction checks, verifier gates and cost-model
ranking before spending GPU time, evaluate the survivors, then route the evidence. The
prompt here says *write exactly one candidate*, so there is no candidate set and the
filter stage has nowhere to exist. That is the largest remaining divergence; diagram 6
places it.

---

## 5. Compiler Revision lifecycle

Non-obvious and load-bearing: **any edit to a Revision-bound source invalidates the
released lock.** That is deliberate, and it is why wiring new code into the Compiler is
a gated event rather than a commit.

```mermaid
graph TD
    E["edit Compiler source"] --> D["<b>revision.json</b><br/>state: draft<br/><i>no source-hash check</i>"]
    D --> G["<b>Corpus Gate</b><br/>six cases · exact finding codes<br/>exact lowering digests"]
    G -->|"any case differs"| STOP["release refused"]
    G -->|"6/6 matched"| AP["<b>release-approval.json</b><br/>binds the gate digest<br/>records who authorized it"]
    AP --> L["<b>revision.lock.json</b><br/>state: released<br/>binds every source by digest"]
    L --> AR["<b>compiler/releases/vN/</b><br/>immutable history"]

    L -.->|"a frozen Study binds<br/>one exact Revision"| SC["Study Contract"]
    E -.->|"invalidates"| L
    SC -.->|"cannot be mutated —<br/>mint a successor"| SC2["Study Contract vN+1"]

    style STOP fill:#f8d7da,stroke:#c00
    linkStyle 7 stroke:#c00,stroke-dasharray:4 3
```

`scratchpad/release_v4.sh` in the branch history drives the cycle end to end, because a
single compiler edit requires all of it again.

---

## 6. Alignment with the paper

Green is implemented and exercised; amber exists but is short of the paper; red is
absent. Sources: [`PAPER_CONTRACT.md`](PAPER_CONTRACT.md) and
[arXiv:2608.12629v1](https://arxiv.org/abs/2608.12629).

```mermaid
graph LR
    W["Workload Contract<br/>shapes · oracle · tolerances"] --> AE
    AE["Authoring Environment<br/>Cake IR arm · direct CUDA arm"] --> CH

    subgraph CH["Compiler harness"]
        direction TB
        T1["typed IR + construction checks"]
        T2["verifier hard gates"]
        T3["cost-model ranking"]
        T4["deterministic lowering"]
    end

    CH --> FIL["<b>filter</b><br/>rank and prune<br/>before GPU time"]
    FIL --> EX["compile → oracle → CUPTI"]
    EX --> EVD["retained evidence"]
    EVD -->|inner loop| AE
    EVD -->|"outer loop<br/>corpus gate + human merge"| CH

    style T1 fill:#d4edda,stroke:#28a745
    style T2 fill:#d4edda,stroke:#28a745
    style T4 fill:#fff3cd,stroke:#b8860b
    style T3 fill:#f8d7da,stroke:#c00
    style FIL fill:#f8d7da,stroke:#c00
    style W fill:#d4edda,stroke:#28a745
    style AE fill:#d4edda,stroke:#28a745
    style EX fill:#d4edda,stroke:#28a745
    style EVD fill:#d4edda,stroke:#28a745
```

| Element | State |
| --- | --- |
| Workload Contract | implemented |
| Authoring Environment, both arms | implemented |
| Typed IR and construction checks | implemented, on the product path since Revision v4 |
| Verifier hard gates, four categories | implemented, on the product path since Revision v4 |
| Compile → external oracle → GPU measurement | implemented, B200-verified |
| Retained evidence and the outer loop gate | implemented; stronger than the paper describes |
| Deterministic lowering | partial — `lower` generates the warp-specialized profile from its Schedule, verified on B200 at 128/128; the other two profiles still stamp a checked-in file |
| Cost-model ranking | absent — `calibration_coverage` is empty |
| The filter stage | absent — one candidate per Turn leaves nothing to rank |

The two red boxes are one problem. A cost model exists to order a candidate set; until a
Turn produces more than one candidate there is nothing for it to order, and the paper's
stated mechanism — *cheap analyses rank and filter candidates before they reach expensive
GPU runs* — has no place in the control flow.
