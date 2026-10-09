"""Bounded author view of this Run's observations; Evidence remains authoritative.

The live Run accumulates only projections of the facts it archives. Replay rebuilds
this view from earlier events and independently validated receipts. No author text
can change these facts, nomination, qualification, or cross-Run knowledge access.
"""
from copy import deepcopy
import json
from open_cake_ir.evaluation.core import _plain_json

from .selection import _receipt_latency_ms, _receipt_qualifies
from .diagnoses import validate_findings_feedback

_MAX_EVALUATIONS = 12
_MAX_REJECTIONS = 8
_MAX_ACTIONS = 8
_MAX_TEXT = 512
_MAX_PROFILE_BYTES = 2048
_MAX_HISTORY_BYTES = 24576


def profile_observation(attribution):
    feedback = attribution.attribution_feedback if attribution is not None else None
    if feedback is None:
        return None
    value = _plain_json(feedback)
    if len(json.dumps(value, ensure_ascii=False).encode('utf-8')) <= _MAX_PROFILE_BYTES:
        return value
    return {'kind': str(value.get('kind', 'profile'))[:128], 'summary_omitted': True,
            'details': 'retained attribution receipt at this turn and candidate'}


def diagnostic_observation(diagnostics):
    """Compact the existing bounded diagnostic projection, without new fields."""
    validate_findings_feedback(diagnostics)
    value = deepcopy(diagnostics)
    findings = value['findings']
    value['omitted_findings'] += max(0, len(findings) - 2)
    value['findings'] = findings[:2]
    return value


def evaluated_observation(turn, receipt, attribution=None, *, diagnostics=None):
    qualified = _receipt_qualifies(receipt)
    row = {'turn': turn, 'candidate_sha256': receipt.candidate_sha256,
           'event_kind': 'candidate_evaluated', 'purpose': 'search',
           'candidate_disposition': receipt.candidate_disposition,
           'measurement_quality': receipt.measurement_quality,
           'search_qualified': qualified,
           'latency_ms': _receipt_latency_ms(receipt) if qualified else None}
    timing = receipt.timing
    row['baseline_comparison'] = ({key: _plain_json(timing[key])
        for key in ('classification', 'speedup', 'statistic', 'pooled_means_ms', 'pooled_medians_ms') if key in timing}
        if qualified and timing is not None else None)
    row['profile'] = profile_observation(attribution)
    row['diagnostics'] = diagnostic_observation(diagnostics if diagnostics is not None else
        {'findings': [], 'omitted_findings': 0, 'text_truncated': False})
    return row


def rejected_observation(turn, feedback):
    row = {'turn': turn, 'event_kind': 'candidate_rejected', **deepcopy(feedback)}
    row.update(diagnostic_observation({key: row[key]
        for key in ('findings', 'omitted_findings', 'text_truncated')}))
    return row


def action_observation(turn, row):
    if row['kind'] != 'transform':
        return None
    return {'turn': turn, 'event_kind': 'author_actions_resolved', 'ordinal': row['ordinal'],
            **{key: row.get(key) for key in ('parent', 'transformation', 'candidate_sha256', 'reason')},
            'message': str(row.get('message', ''))[:_MAX_TEXT],
            'text_truncated': len(str(row.get('message', ''))) > _MAX_TEXT}


def optimization_history(evaluations, rejections, actions):
    """Summarize actual past observations, including evaluated non-winners.

    Best means lowest qualified search latency. It is not a final confirmation,
    a compiler estimate, or a lesson imported from another arm or Run.
    """
    qualified = [row for row in evaluations if row['search_qualified']]
    best = min(qualified, key=lambda row: row['latency_ms']) if qualified else None
    view = deepcopy({
        'schema_version': 1, 'scope': 'current_run_observations_only',
        'selection_authority': 'advisory search history; final confirmation remains separate',
        'best_qualified_search': best,
        'evaluations': evaluations[-_MAX_EVALUATIONS:],
        'rejections': rejections[-_MAX_REJECTIONS:],
        'transforms': actions[-_MAX_ACTIONS:],
        'total': {'evaluations': len(evaluations), 'rejections': len(rejections), 'transforms': len(actions)},
        'omitted': {'evaluations': max(0, len(evaluations)-_MAX_EVALUATIONS),
                    'rejections': max(0, len(rejections)-_MAX_REJECTIONS),
                    'transforms': max(0, len(actions)-_MAX_ACTIONS)},
    })
    # A count bound alone is insufficient for long backend diagnostic text.
    # Keep the best qualified fact; trim older detail and expose every omission.
    while len(json.dumps(view, ensure_ascii=False).encode('utf-8')) > _MAX_HISTORY_BYTES:
        sections = [name for name in ('evaluations', 'rejections', 'transforms') if view[name]]
        if not sections:
            raise ValueError('best qualified observation exceeds bounded history domain')
        name = max(sections, key=lambda key: len(json.dumps(view[key], ensure_ascii=False).encode('utf-8')))
        view[name].pop(0)
        view['omitted'][name] += 1
    return view
