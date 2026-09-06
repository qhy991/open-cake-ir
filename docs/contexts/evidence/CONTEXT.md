# Evidence context

Evidence preserves observed bytes and event order without redefining the Study question.
Canonical definitions are in the
[`Glossary`](../../GLOSSARY.md#evidence-and-report-terms).

## Owned terms

- [`Evidence Object`](../../GLOSSARY.md#evidence-object)
- [`Event Ledger`](../../GLOSSARY.md#event-ledger)
- [`Terminal Archive`](../../GLOSSARY.md#terminal-archive)
- [`Integrity`](../../GLOSSARY.md#integrity) and
  [`Filesystem Custody`](../../GLOSSARY.md#filesystem-custody)
- [`Protocol Adherence`](../../GLOSSARY.md#protocol-adherence)
- [`Run Audit`](../../GLOSSARY.md#run-audit)
- [`Study Report`](../../GLOSSARY.md#study-report)
- [`Claim View`](../../GLOSSARY.md#claim-view)

## Responsibilities

- Store immutable bytes without following links outside the custody boundary.
- Append ordered, typed event references under one writer.
- Produce one Terminal Archive for every terminal Run, including failures and deviations.
- Reconstruct Integrity, Filesystem Custody, Protocol Adherence, and endpoint observations
  without provider, GPU, network, or checkout mutation.
- Apply the preregistered Analysis Plan to accepted Run Audits.
- Derive disposable current views without turning them into authorities.

Evidence does not decide Workload correctness, define an Estimand after execution, or
reinterpret an old Run under a successor policy.

## Relationships

- An Event Ledger references Evidence Objects.
- Every terminal Run has one Terminal Archive.
- A Run Audit reads but never changes retained Evidence.
- A Study Report applies the frozen Study analysis to Run Audits.
- A Claim View can be deleted and regenerated from accepted Study Reports.

## Boundary examples

An archive can have complete, untampered bytes while current filesystem custody is not
verified. That archive remains inspectable but cannot support promotion or a claim-bearing
projection. A provider failure can likewise leave an intact archive while Protocol
Adherence or endpoint availability records the failure.
