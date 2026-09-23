# Python candidate bundle optimization

Write one `candidate-set.py` file. Import the Cake frontend once, then put complete
`@cake.schedule(...)` functions in the order you want the Lab to consider them.
Each function is one candidate. When this Run grants a Compiler rewrite, place a
static `cake.transform(parent="...", transformation="...", parameters={...})`
declaration at its desired position; use the candidate ids returned in feedback.
Do not execute this file or write Schedule/Program JSON. The Lab parses its AST,
seals the raw UTF-8 bytes and derives the ordered candidate/action set.

Preserve the frozen Workload, public tensor ABI, exact target and lowering route.
Propose structurally distinct choices rather than formatting or name variants.
Static findings and cost estimates filter candidates; only the external oracle,
paired timing and profiler evidence support correctness and performance claims.
The Lab owns GPU allocation, evaluation and stopping. Do not change TASK.md or
AGENTS.md, call another compiler, or claim a serving result from this Run.
