"""Candidate-bound author feedback, derived from retained filter and receipt facts."""
from __future__ import annotations

from open_cake_ir.evaluation.timing import timing_statistic

from collections.abc import Mapping
from types import SimpleNamespace

from .diagnoses import rejected_peer_feedback
from .selection import _matched_search_plan, _receipt_latency_ms, _receipt_qualifies
from ._policies import _ATTRIBUTION_EVALUATION
from .generated_source import candidate_source_feedback
from open_cake_ir.evaluation.refusals import EvaluationRefusal


def baseline_comparison_feedback(specification, timing):
    """The fixed black-box opponent; never its implementation or mutable incumbent."""
    medians = timing.get(f'pooled_{timing_statistic(timing)}s_ms')
    fixed = specification.document['execution'].get('fixed_baseline')
    selection = fixed.get('selection') if isinstance(fixed, Mapping) else None
    return {
        'source': selection.get('source') if isinstance(selection, Mapping) else 'campaign_fixed_baseline',
        'baseline_latency_ms': medians.get('baseline') if isinstance(medians, Mapping) else None,
        'candidate_speedup': timing.get('speedup'),
        **({'statistic': 'mean'} if timing_statistic(timing) == 'mean' else {}),
        'measurement_quality_passed': timing.get('measurement_quality_passed'),
    }


def evaluated_feedback(receipt, attribution, diagnostics, specification):
    """Only this candidate's search result; neither qualification nor confirmation is inferred."""
    protocol = specification.document['evaluation_protocol']
    result = {
        'correctness_passed': receipt.correctness_passed,
        'candidate_disposition': receipt.candidate_disposition,
        'measurement_quality': receipt.measurement_quality,
        'search_qualified': _receipt_qualifies(receipt) and not isinstance(attribution, EvaluationRefusal),
        'search_latency_ms': _receipt_latency_ms(receipt),
        **diagnostics,
    }
    if isinstance(receipt, EvaluationRefusal):
        result['evaluation_refusal'] = receipt.document
    if isinstance(attribution, EvaluationRefusal):
        result['attribution_refusal'] = attribution.document
    if isinstance(receipt.timing, Mapping):
        result['baseline_comparison'] = baseline_comparison_feedback(specification, receipt.timing)
    if 'attribution_evaluation' in protocol:
        # A legacy post-confirmation attribution did not exist when this Turn ended.
        profile = (attribution.attribution_feedback
                   if attribution is not None and protocol['attribution_evaluation'] == _ATTRIBUTION_EVALUATION
                   else None)
        result['profile'] = dict(profile) if profile is not None else None
    return result


def derive_turn_feedback(*, turn, candidates, filter_rows, selected, receipts, attributions,
                         rejected_feedback, actions, arm, specification, selection_summary=None,
                         launchables=None, authored=None):
    """One pure projection for live execution and replay; candidate count is Run-bounded.

    Filter rows retain bounded Environment diagnostics before GPU time. Receipts own
    evaluation facts. All identities are existing action/filter/receipt references;
    Opt-in source views use the same sealed candidates; no new identity scheme or
    artifact authority enters the author handoff.
    """
    by_candidate = {row['candidate_sha256']: row for row in filter_rows}
    indices = {}
    for action in actions:
        if action['candidate_sha256'] is not None:
            indices.setdefault(action['candidate_sha256'], action['ordinal'])
    if set(candidates) != set(by_candidate) or set(candidates) != set(indices):
        raise ValueError('feedback candidates differ from resolved actions and filter')
    _, collapsed = _matched_search_plan(filter_rows,
        specification.document['evaluation_protocol'].get('searches_per_turn', 1))
    duplicates = {row['candidate_sha256']: row['same_program_as'] for row in collapsed}
    rejected = rejected_peer_feedback([
        (SimpleNamespace(sha256=candidate), SimpleNamespace(disposition='rejected', feedback=rejected_feedback[candidate]))
        for candidate in candidates if candidate in rejected_feedback], arm=arm)
    rejections = {row['candidate_sha256']: row for row in rejected}
    results = []
    source_feedback = candidate_source_feedback(candidates=candidates,
        launchables=launchables or {}, authored=authored or {}, specification=specification)
    for candidate in candidates:
        filtered = by_candidate[candidate]
        row = {'candidate_index': indices[candidate], 'candidate_sha256': candidate,
               'selected': candidate == selected}
        if candidate in receipts:
            if filtered['disposition'] != 'launchable':
                raise ValueError('rejected candidate has an evaluation receipt')
            receipt = receipts[candidate]
            attribution = attributions.get(candidate)
            if receipt.candidate_sha256 != candidate or (attribution is not None and attribution.candidate_sha256 != candidate):
                raise ValueError('feedback receipt belongs to another candidate')
            row.update(status='evaluation_refused' if isinstance(receipt, EvaluationRefusal) else 'evaluated', **evaluated_feedback(
                receipt, attribution, filtered['diagnostics'], specification))
        elif filtered['disposition'] == 'rejected':
            row.update(status='build_rejected', **rejections[candidate])
        elif candidate in duplicates:
            row.update(status='duplicate', same_program_as=duplicates[candidate])
        else:
            row.update(status='not_evaluated', reason='searches_per_turn_limit')
        if candidate in source_feedback:
            row['generated_source'] = source_feedback[candidate]
        results.append(row)

    if selected in receipts:
        selected_result = next(row for row in results if row['candidate_sha256'] == selected)
        result = {'kind': 'evaluation', **{key: value for key, value in selected_result.items()
            if key not in {'candidate_index', 'candidate_sha256', 'selected', 'status', 'generated_source'}}}
    elif selected is not None:
        # Primary guidance obeys the same bounds as the peer projection.
        result = {key: value for key, value in rejections[selected].items() if key != 'candidate_sha256'}
    else:
        result = {'stage': 'authoring'}
    result.update(source_turn=turn, selected_candidate_sha256=selected, candidate_results=results)
    if (not candidates or any(row['kind'] != 'submit' or row['action_sha256'] != row['candidate_sha256'] for row in actions)):
        result['author_actions'] = list(actions)
    if rejected:
        result['rejected_candidates'] = rejected
    if selection_summary is not None:
        result['candidate_selection'] = {**selection_summary, 'order': list(filter_rows)}
    return result
