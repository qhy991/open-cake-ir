"""One live Evaluation-to-evidence transaction, shared by search and confirmation.

This writer does not decide selection or qualification. The independent replay
readers reconstruct those decisions from the retained artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.store import RunLedger

from .archive import (
    _archive_evaluation_receipt,
    _archive_logical_attempt,
    _validate_receipt_authority,
)
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
                **({"elapsed_wall_seconds": self.ralph.elapsed_wall_seconds}
                   if purpose == "confirmatory" else {}),
            },
        )
        return receipt
