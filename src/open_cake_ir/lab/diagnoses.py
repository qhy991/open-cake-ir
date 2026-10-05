"""Bounded author feedback from the rejected members retained by the Lab."""
from __future__ import annotations

from typing import Mapping

from .routing import route_rejection

_MAX_FINDINGS = 8
_MAX_TEXT = 512
_FINDING_TEXT_FIELDS = ("code", "path", "category", "severity", "message")
_FINDING_BOOLEAN_FIELDS = ("blocks_acceptance", "blocks_lowering")


def _source_location(value: object) -> dict[str, int]:
    return {key: value[key] for key in ("line", "column", "end_line", "end_column")
            if type(value.get(key)) is int} if isinstance(value, Mapping) else {}


def findings_feedback(findings, *, blocking_only=False):
    """Bound declared Finding fields without leaking arbitrary source/artifact payloads."""
    eligible = [item for item in findings if isinstance(item, Mapping)
                and (not blocking_only or item.get('blocks_acceptance') or item.get('blocks_lowering'))
                ] if isinstance(findings, (list, tuple)) else []
    projected = []
    truncated = False
    for finding in eligible[:_MAX_FINDINGS]:
        item = {}
        for field in _FINDING_TEXT_FIELDS:
            value = finding.get(field)
            if isinstance(value, str):
                item[field] = value[:_MAX_TEXT]
                truncated |= len(value) > _MAX_TEXT
        for field in _FINDING_BOOLEAN_FIELDS:
            value = finding.get(field)
            if type(value) is bool:
                item[field] = value
        location = _source_location(finding.get('source_location'))
        if location:
            item['source_location'] = location
        projected.append(item)
    return {'findings': projected, 'omitted_findings': max(0, len(eligible) - _MAX_FINDINGS),
            'text_truncated': truncated}


def validate_findings_feedback(value):
    """Validate a retained bounded observation, without inventing omitted diagnostics."""
    if (not isinstance(value, Mapping) or set(value) != {'findings', 'omitted_findings', 'text_truncated'}
        or not isinstance(value['findings'], list) or len(value['findings']) > _MAX_FINDINGS
        or type(value['omitted_findings']) is not int or value['omitted_findings'] < 0
        or type(value['text_truncated']) is not bool
        or findings_feedback(value['findings'])['findings'] != value['findings']):
        raise ValueError('retained bounded Environment diagnostics differ')


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
        diagnostics = findings_feedback(result.feedback.get('findings', []), blocking_only=True)
        row.update(diagnostics)
        row['text_truncated'] |= truncated
        rows.append(row)
    return rows
