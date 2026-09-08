"""Bounded author feedback from the rejected members retained by the Lab."""
from __future__ import annotations

from typing import Mapping

from .routing import route_rejection

_MAX_FINDINGS = 8
_MAX_TEXT = 512
_FINDING_FIELDS = ("code", "path", "category", "severity", "message", "blocks_acceptance", "blocks_lowering")


def _source_location(value: object) -> dict[str, int]:
    return {key: value[key] for key in ("line", "column", "end_line", "end_column")
            if type(value.get(key)) is int} if isinstance(value, Mapping) else {}


def rejected_peer_feedback(built, *, arm: str) -> list[dict[str, object]]:
    """Every rejected peer in provider order; count is bounded by the Study budget.

    Each peer exposes only bounded diagnostic fields, never artifact/source payloads.
    Original complete feedback stays in candidate_rejected evidence.
    """
    rows = []
    for submission, result in built:
        if result.disposition != "rejected":
            continue
        decision = route_rejection(result.feedback, arm=arm)
        row: dict[str, object] = {"candidate_sha256": submission.sha256,
            "routed_to": decision.destination, "routing_reason": decision.reason[:_MAX_TEXT]}
        truncated = len(decision.reason) > _MAX_TEXT
        for field in ("stage", "code", "error", "diagnostic"):
            value = result.feedback.get(field)
            if isinstance(value, str):
                row[field] = value[:_MAX_TEXT]
                truncated |= len(value) > _MAX_TEXT
        location = _source_location(result.feedback.get("source_location"))
        if location:
            row["source_location"] = location
        findings = result.feedback.get("findings", [])
        blocking = [item for item in findings if isinstance(item, Mapping)
                    and (item.get("blocks_acceptance") or item.get("blocks_lowering"))] if isinstance(findings, (list, tuple)) else []
        projected = []
        for finding in blocking[:_MAX_FINDINGS]:
            item = {}
            for field in _FINDING_FIELDS:
                value = finding.get(field)
                if isinstance(value, str):
                    item[field] = value[:_MAX_TEXT]
                    truncated |= len(value) > _MAX_TEXT
                elif type(value) is bool:
                    item[field] = value
            location = _source_location(finding.get("source_location"))
            if location:
                item["source_location"] = location
            projected.append(item)
        row["findings"] = projected
        row["omitted_findings"] = max(0, len(blocking) - _MAX_FINDINGS)
        row["text_truncated"] = truncated
        rows.append(row)
    return rows
