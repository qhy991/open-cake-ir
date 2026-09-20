"""Execution controls shared by independent Runs and external Study preparation."""
from ._documents import _object, differs
from ._policies import _ATTRIBUTION_EVALUATION, _LEGACY_ATTRIBUTION_EVALUATION, untimed
from .ralph import RalphBudget


def validate_run_controls(document):
    budget = _object(document.get('budget'), 'run.budget')
    evaluation = _object(document.get('evaluation_protocol'), 'run.evaluation_protocol')
    attribution_evaluation = evaluation.get('attribution_evaluation')
    no_timed_assay = untimed(evaluation)
    from open_cake_ir.evaluation.paired import paired_protocol
    from open_cake_ir.evaluation.platforms import PLATFORMS
    paired_protocol(evaluation)
    searches_allowed = ({'correctness_only'} if no_timed_assay else
                        {f'correctness_then_paired_{row.measurement_source}' for row in PLATFORMS.values()
                         if row.measurement_source is not None})
    search = evaluation.get('search_evaluation')
    if search not in searches_allowed or evaluation.get('confirmatory_evaluation') != f'fresh_fixed_candidate_{search}':
        raise ValueError('Run evaluation requires correctness and fresh fixed-candidate confirmation')
    admitted_attribution = {
        None, _LEGACY_ATTRIBUTION_EVALUATION, _ATTRIBUTION_EVALUATION,
        *(("correctness_only",) if no_timed_assay else ()),
    }
    if attribution_evaluation not in admitted_attribution:
        raise differs(
            "Run attribution Evaluation",
            expected=sorted(admitted_attribution, key=str), observed=attribution_evaluation,
        )
    checkpoints = budget.get("checkpoints")
    limit = budget.get("limit")
    maximum_turns = budget.get("maximum_turns")
    maximum_candidates_per_turn = budget.get("maximum_candidates_per_turn", 1)
    budget_fields = {
        "unit", "limit", "checkpoints", "maximum_turns", "maximum_candidates_per_turn",
        "wall_time_seconds", "active_authoring_time_seconds", "evaluation_limits", "maximum_compilations",
    }
    if (
        set(budget) != budget_fields
        or budget.get("unit") != "provider_tokens"
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or not isinstance(checkpoints, list)
        or not checkpoints
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in checkpoints)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != limit
        or not isinstance(maximum_turns, int)
        or isinstance(maximum_turns, bool)
        or maximum_turns <= 0
        or not isinstance(maximum_candidates_per_turn, int)
        or isinstance(maximum_candidates_per_turn, bool)
        or maximum_candidates_per_turn <= 0
    ):
        raise differs(
            "Run budget grid",
            expected={"fields": sorted(budget_fields), "unit": "provider_tokens",
                      "limit": "positive int", "checkpoints": "sorted positive ints ending at limit",
                      "maximum_turns": "positive int", "maximum_candidates_per_turn": "positive int"},
            observed={key: budget.get(key) for key in sorted(set(budget) | budget_fields)},
        )
    RalphBudget.from_mapping(budget)
    run_protocol = _object(document.get("run_protocol"), "run.run_protocol")
    expected_run_protocol = {"independent_thread": True, "workspace_seed": "task_agents_only",
                             "automatic_retries": 0, "replacement_runs": 0}
    if any(run_protocol.get(key) != value for key, value in expected_run_protocol.items()):
        raise differs(
            "Run Run Protocol", expected=expected_run_protocol,
            observed={key: run_protocol.get(key) for key in expected_run_protocol},
        )
    # How many candidates a Turn search-evaluates. Checked here because a Study that
    # asks for none, or for a word, would otherwise fault partway through a run --
    # and a run that faults has already spent the GPU time this Lab exists to gate.
    searches = evaluation.get("searches_per_turn", 1)
    if not isinstance(searches, int) or isinstance(searches, bool) or searches < 1:
        raise differs("Run searches_per_turn", expected="int >= 1", observed=searches)
    if searches > maximum_candidates_per_turn:
        raise ValueError(
            "Run searches_per_turn exceeds maximum_candidates_per_turn: "
            f"{searches} > {maximum_candidates_per_turn}"
        )
    ralph_limits = _object(budget.get("evaluation_limits"), "run.budget.evaluation_limits")
    required_attribution = searches if attribution_evaluation == _ATTRIBUTION_EVALUATION else 0
    if (
        int(ralph_limits.get("search", 0)) < searches
        or int(ralph_limits.get("confirmatory", 0)) < 1
        or int(ralph_limits.get("attribution", 0)) < required_attribution
    ):
        raise ValueError(
            "Ralph budget cannot admit one complete Turn: needs search >= "
            f"{searches}, confirmatory >= 1, attribution >= {required_attribution}; "
            f"evaluation_limits {dict(ralph_limits)!r}"
        )
    # How much faster the measurement has to be before the order counts as wrong.
    # A Study that searches more than one candidate has to say, because without it
    # every inversion inside the noise would be routed to the cost model as a defect
    # -- and the loss surface is a plateau, so most inversions are inside the noise
    # (`docs/ANALYSIS_CALIBRATION.md`).
    materiality = evaluation.get("search_materiality_ratio")
    if searches > 1:
        if not isinstance(materiality, float) or not 1.0 < materiality < 100.0:
            raise ValueError(
                "a Study searching more than one candidate declares "
                f"search_materiality_ratio: expected a float in (1.0, 100.0), observed {materiality!r}"
            )
    elif materiality is not None:
        # No second candidate to compare against, so a ratio here would state a
        # threshold nothing can cross.
        raise ValueError(
            f"search_materiality_ratio without searches_per_turn above one: {materiality!r}"
        )
