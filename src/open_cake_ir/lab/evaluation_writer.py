"""One live Evaluation-to-evidence transaction, shared by search and confirmation.

This writer does not decide selection or qualification. The independent replay
readers reconstruct those decisions from the retained artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Callable, Mapping

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.store import RunLedger

from .archive import (
    _archive_evaluation_receipt,
    _archive_logical_attempt,
    _validate_receipt_authority,
    _logical_attempt_document, _evaluation_receipt_document,
)
from ._documents import _canonical_json_bytes
from .faults import RunProtocolFault
from .contracts import RunEvaluator
from .ralph import RalphController


@dataclass(frozen=True)
class EvaluationWriter:
    """Borrow validated run dependencies; own no competing lifecycle or policy."""

    evidence: EvidenceStore
    ledger: RunLedger
    evaluator: RunEvaluator
    ralph: RalphController
    case_id: str
    workload_sha256: str
    protocol_sha256: str
    evaluation_protocol: Mapping[str, object]
    execution: Mapping[str, object]
    _job_ids: set = field(default_factory=set, repr=False)

    def evaluate(self, candidate: LaunchableCandidate, *, purpose: str, turn=None, source_turn=None) -> EvaluationReceipt:
        if (turn is None) == (source_turn is None) or (purpose == 'confirmatory' and source_turn is None):
            raise ValueError('Evaluation needs one search turn or a terminal nomination source')
        origin = {'turn': turn} if turn is not None else {'source_turn': source_turn}
        self.ledger.append("evaluation_attempt_started", {
            **origin, "purpose": purpose, "candidate_sha256": candidate.candidate_sha256,
        })
        self.ralph.record_evaluation(purpose)
        attempt = self.evaluator.evaluate(candidate, case_id=self.case_id, purpose=purpose)
        receipt = attempt.final_receipt
        try:
            jobs = [item.job_id for item in attempt.attempts]
            if len(jobs) != len(set(jobs)) or self._job_ids.intersection(jobs):
                raise ValueError('Evaluation reused an earlier broker job')
            for item in attempt.attempts:
                request = json.loads(item.artifact_payloads['evaluator_request'])
                if request.get('purpose') != purpose or request.get('case_id') != self.case_id:
                    raise ValueError('Evaluation worker purpose or case differs from this invocation')
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            payloads = {'rejected_attempt_ledger':_canonical_json_bytes(_logical_attempt_document(attempt))}
            for index,item in enumerate(attempt.attempts,1):
                payloads.update({f'rejected_attempt_{index}_{role}':raw for role,raw in item.artifact_payloads.items()})
            if receipt is not None:
                payloads['rejected_evaluation_receipt'] = _canonical_json_bytes(_evaluation_receipt_document(receipt))
                payloads.update({f'rejected_receipt_{role}':raw for role,raw in receipt.artifact_payloads.items()})
            raise RunProtocolFault('harness_fault',str(error),artifact_payloads=payloads) from error
        self._job_ids.update(jobs)
        self.ledger.append(
            "evaluation_attempt_completed",
            {
                **origin,
                "purpose": purpose,
                "candidate_sha256": candidate.candidate_sha256,
                "objects": _archive_logical_attempt(self.evidence, attempt),
            },
        )
        if receipt is None:
            # The broker already said why it produced none. Repeating it here keeps the
            # fault readable without reopening the archived attempts.
            last = attempt.attempts[-1] if attempt.attempts else None
            reason = (f"admitted={last.admitted} error={last.error!r} job_id={last.job_id!r}"
                      if last is not None else "no broker attempt was made")
            raise RuntimeError(f"{purpose} Evaluation has no final receipt: {reason}")
        _validate_receipt_authority(
            receipt,
            candidate=candidate,
            workload_sha256=self.workload_sha256,
            protocol_sha256=self.protocol_sha256,
            case_id=self.case_id,
            purpose=purpose,
            evaluation_protocol=self.evaluation_protocol,
            fixed_baseline=self.execution.get("fixed_baseline", {}).get("candidate"),
        )
        references = _archive_evaluation_receipt(self.evidence, receipt)
        self.ledger.append(
            "candidate_evaluated",
            {
                **origin,
                "purpose": purpose,
                "candidate_sha256": candidate.candidate_sha256,
                "objects": references,
                **({"elapsed_wall_seconds": round(self.ralph.elapsed_wall_seconds, 6)}
                   if purpose == "confirmatory" else {}),
            },
        )
        return receipt
