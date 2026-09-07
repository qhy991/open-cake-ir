"""Collect typed findings in their stable public order."""

from __future__ import annotations

from ..diagnostics import Finding, FindingCategory, FindingSeverity


_SEVERITY_ORDER = {
    FindingSeverity.BLOCKING: 0,
    FindingSeverity.REPORT: 1,
    FindingSeverity.HINT: 2,
}


class _Collector:
    def __init__(self) -> None:
        self._findings: list[Finding] = []

    def add(
        self,
        code: str,
        path: str,
        message: str,
        category: FindingCategory,
        severity: FindingSeverity = FindingSeverity.BLOCKING,
    ) -> None:
        self._findings.append(Finding(code, path, message, category, severity))

    def result(self) -> tuple[Finding, ...]:
        # Stable order: severity, then category, code and path. Deterministic across
        # runs so a Corpus Gate can pin the exact sequence.
        return tuple(
            sorted(
                self._findings,
                key=lambda f: (
                    _SEVERITY_ORDER[f.severity],
                    f.category.value,
                    f.code,
                    f.path,
                ),
            )
        )
