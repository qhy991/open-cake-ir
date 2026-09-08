"""Typed external/protocol deviations mapped by the sole Lab terminal path."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


@dataclass(frozen=True)
class ReportedProviderUsage:
    """Complete native usage observed for one invocation, independent of acceptance."""

    event_contract: str
    thread_id: str
    provider_tokens: int

    def __post_init__(self) -> None:
        if (not isinstance(self.event_contract, str) or not self.event_contract
                or self.event_contract != self.event_contract.strip()
                or not isinstance(self.thread_id, str)
                or re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", self.thread_id) is None
                or type(self.provider_tokens) is not int or self.provider_tokens < 0):
            raise ValueError("reported provider usage differs")

    @property
    def document(self) -> Mapping[str, object]:
        return {"event_contract": self.event_contract, "thread_id": self.thread_id,
                "provider_tokens": self.provider_tokens}


class RunProtocolFault(RuntimeError):
    """A named non-treatment failure observed during one Run."""

    _ADHERENCE = {
        "provider_fault",
        "harness_fault",
        "custody_violation",
        "contamination",
        "broker_fault",
    }

    def __init__(
        self,
        protocol_adherence: str,
        message: str,
        *,
        artifact_payloads: Mapping[str, bytes] | None = None,
        reported_usage: ReportedProviderUsage | None = None,
    ) -> None:
        if protocol_adherence not in self._ADHERENCE:
            raise ValueError("Run protocol fault classification differs")
        if reported_usage is not None and not isinstance(reported_usage, ReportedProviderUsage):
            raise ValueError("Run protocol fault reported usage differs")
        super().__init__(message)
        self.protocol_adherence = protocol_adherence
        self.artifact_payloads = dict(artifact_payloads or {})
        self.reported_usage = reported_usage


class CandidateCompileRejected(ValueError):
    """Observed compiler rejection that may feed the next Turn."""

    def __init__(
        self,
        diagnostic: str,
        *,
        artifact_payloads: Mapping[str, bytes],
    ) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.artifact_payloads = dict(artifact_payloads)
