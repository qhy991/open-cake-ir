"""Typed external/protocol deviations mapped by the sole Lab terminal path."""

from __future__ import annotations

from typing import Mapping


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
    ) -> None:
        if protocol_adherence not in self._ADHERENCE:
            raise ValueError("Run protocol fault classification differs")
        super().__init__(message)
        self.protocol_adherence = protocol_adherence
        self.artifact_payloads = dict(artifact_payloads or {})


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
