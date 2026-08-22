# Evidence context

The Evidence context preserves observed bytes and event order without redefining the study question.

## Language

**Evidence Object**:
An immutable byte sequence identified by its content digest.
_Avoid_: Artifact path, output file

**Event Ledger**:
An append-only ordered record of references to Evidence Objects and observed transitions.
_Avoid_: Mutable state file, log directory

**Terminal Archive**:
The complete replayable evidence for a Run regardless of success, failure or protocol deviation.
_Avoid_: Success archive, failure archive

**Integrity**:
Whether archived bytes, references and event ordering are complete and untampered.
_Avoid_: Scientific validity

**Protocol Adherence**:
Whether observed execution followed the frozen Study Contract and Campaign admission.
_Avoid_: Integrity, candidate quality

**Run Audit**:
A pure reconstruction of Integrity, Protocol Adherence and endpoint observations for one Run.
_Avoid_: Study result

**Study Report**:
A preregistered analysis of Run Audits containing inclusion, estimate, uncertainty and availability.
_Avoid_: Estimand, run summary

**Claim View**:
A deletable projection of supported claims and explicit unknowns from accepted Study Reports.
_Avoid_: Claim owner, README status

## Relationships

- An **Event Ledger** references one or more **Evidence Objects**.
- Every Run produces one **Terminal Archive**, including a failed or invalid Run.
- A **Run Audit** never changes Evidence Objects or the Event Ledger.
- A **Study Report** applies the Study Contract's analysis plan to one or more Run Audits.
- A **Claim View** can be deleted and rebuilt without losing a fact.

## Example dialogue

> **Auditor:** “The provider failed, but all bytes and events verify. Is the archive invalid?”
>
> **Researcher:** “No. Integrity may pass while Protocol Adherence records a fault and the endpoint is missing.”

## Flagged ambiguities

- “valid” previously mixed Integrity, Protocol Adherence and study inclusion; these are now separate facts.
- “audit” previously defined the estimand after execution; the **Study Report** may estimate only a preregistered Estimand.

