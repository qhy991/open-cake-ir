# Read results and locate failures

[中文原文](../../wiki/results.md) · [English home](../README.md)

Ask what a check examined before asking whether it passed. A green flag only covers its own boundary.

[Guide index](README.md) · [Formal scope labels](../GLOSSARY.md#evidence-strength-and-scope)

| Result | What it supports | What remains separate |
| --- | --- | --- |
| accepted=true | Modeled structural and semantic rules passed | Backend eligibility and GPU execution |
| lowering_eligible=true | This backend may produce source | Toolchain compilation and launch |
| generated=true | Source came from Schedule operations | Numerical checks and measurement |
| generated=false | An admitted fixed source asset was selected | Arbitrary source generation |
| Corpus passed=true | Positive and negative dispositions matched | GPU correctness |
| GPU correctness passed | Bound inputs, outputs, and state passed | Timing, other shapes, and model behavior |
| Measurement accepted | These samples met the protocol | Other deployments or broader conclusions |

## Assessment and command outcomes

Accepted and lowering_eligible are separate. A structurally valid plan can be unsupported by a backend. Assess exit status only reports acceptance, not permission to lower or run.

Chinese text output uses 命令未完成 for command/input failures such as a missing file, unreadable JSON, version mismatch, or existing output, with nonzero status. Findings may block acceptance, block lowering only, or provide information. Do not treat all Findings as errors or measured performance.

## Follow the location

`operations[3].parameters` is the fourth operation's parameters; `access_maps[2]` is the third mapping. Default JSON retains original English findings and full analysis.

| Code or message | Meaning and action |
| --- | --- |
| RESIDENCY_BOUND | Declared resources constrain concurrent work; distinguish actual backend allocation |
| TARGET_UNSUPPORTED | Check the exact target; do not silently substitute hardware |
| FMA arity/instruction finding | Compare operand count and numerical contract to the complete example |
| STORE_EDGE_COUNT | Check missing or extra value/destination edges |
| LOAD_ACCESS_SHAPE_MISMATCH | Align the accessed value domain and register result, without hidden replication |
| ACCESS_DIMENSION_COORDINATE_RANGE | Check the dimension actually addressed by each vector |
| TRITON_ARANGE_RANGE_UNSUPPORTED | Choose legal generated spans and masks; global size need not be a power of two |
| STATE_STORE_PROGRAM_OWNER | Program mapping does not prove ownership of state coordinates |
| STATE_STORE_PROGRAM_AXIS_COVERAGE | A required program axis is missing or duplicated |
| Executor Revision file differs | Use the exact frozen source or create a successor |
| Source/lock mismatch | Replay the complete historical Git revision instead of editing expectations |

Retain failures before a successor repair. Do not overwrite receipts or change old modes to manufacture historical custody.

## Reading speed

At matched conditions, baseline 20 ms and candidate 10 ms means 2x speed and 50% less time. State whether timing covers one kernel, an operator Program, model forward, or serving. A small component gain can produce little overall gain.

Use matched repeated samples, not a slow baseline sample against a best candidate sample. A median is the middle sorted observation; identical summary statistics do not make different protocols comparable. Uncertainty and acceptance rules belong to the predeclared Study.

Not-run means no execution. Unknown means insufficient evidence. Invalid means the evidence or protocol cannot support the intended interpretation. Correct but unstable timing retains both facts. An intact archive without verified custody stays inspectable but does not automatically support qualification. Conclusions should point to the oracle, complete arrays, fixed versions, and raw records.
