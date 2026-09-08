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
    clock: Callable[[], float]
    run_started_at: float | None

    def evaluate(self, candidate: LaunchableCandidate, *, purpose: str, turn: int) -> EvaluationReceipt:
        self.ralph.record_evaluation(purpose)
        attempt = self.evaluator.evaluate(candidate, case_id=self.case_id, purpose=purpose)
        receipt = attempt.final_receipt
        self.ledger.append(
            "evaluation_attempt_completed",
            {
                "turn": turn,
                "purpose": purpose,
                "candidate_sha256": candidate.candidate_sha256,
                "objects": _archive_logical_attempt(self.evidence, attempt),
            },
        )
        if receipt is None:
            raise RuntimeError(f"{purpose} Evaluation has no final receipt")
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
                "turn": turn,
                "purpose": purpose,
                "candidate_sha256": candidate.candidate_sha256,
                "objects": references,
                **({"elapsed_wall_seconds": self.clock() - self.run_started_at}
                   if purpose == "confirmatory" and self.run_started_at is not None else {}),
            },
        )
        return receipt
