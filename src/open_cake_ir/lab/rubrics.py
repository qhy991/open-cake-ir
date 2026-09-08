"""Author guidance derived only from the feedback already in a Ralph StateCard.

This projection does not judge candidates or change controller decisions. Evidence
pointers name the original feedback in the retained provider reference bundle.
"""

from __future__ import annotations

from typing import Mapping


_ABSENT = object()
_SOURCE = "/state_card/previous_feedback"


def derive_rubric(feedback: object = _ABSENT) -> dict[str, object]:
    """Explain reported evidence without fetching resources or inferring a route.

    An absent field, explicit unknown, and malformed value remain distinct. In
    particular, ``qualified`` is the search receipt's correctness disposition,
    while ``confirmed`` is the controller's separate confirmatory decision.
    """

    criteria: list[dict[str, object]] = []
    document = feedback if isinstance(feedback, Mapping) else {}
    input_state = (
        "absent" if feedback is _ABSENT else
        "unknown" if feedback is None else
        "malformed" if not isinstance(feedback, Mapping) else None
    )

    def field(name: str, allowed: tuple[str, ...]) -> str:
        if input_state is not None:
            return input_state
        value = document.get(name, _ABSENT)
        if value is _ABSENT:
            return "absent"
        if value is None or value == "unknown":
            return "unknown"
        if not isinstance(value, str):
            return "malformed"
        return value if value in allowed else "unknown"

    def add(name: str, state: str, fields: tuple[str, ...], guidance: str) -> None:
        criteria.append({
            "criterion": name,
            "state": state,
            "evidence": [f"{_SOURCE}/{key}" for key in fields if key in document],
            "guidance": guidance,
        })

    kind = field("kind", ("initial", "evaluation"))
    stage = field("stage", ("assessment", "compile", "built"))
    admission = (
        "initial" if kind == "initial" else
        "observed" if kind == "evaluation" or stage == "built" else
        "refused" if stage in ("assessment", "compile") else
        input_state or ("malformed" if "malformed" in (kind, stage) else "unknown")
    )
    add("admission", admission, ("kind", "stage", "error", "diagnostic"), {
        "initial": "Propose structurally distinct candidates within TASK.md and AGENTS.md; no candidate evidence exists yet.",
        "observed": "The feedback reports a built or evaluated artifact. Keep the same contract and inspect the separate correctness and timing evidence.",
        "refused": "Inspect the reported stage and local diagnostic before the next candidate. A pre-evaluation refusal is not a GPU correctness failure or authorization to change the Compiler.",
    }.get(admission, "Admission is not established by this feedback. Use the frozen authoring contract; do not infer a candidate defect or Compiler gap."))

    correctness = field("candidate_disposition", ("qualified", "correctness_rejected"))
    add("correctness", correctness, ("candidate_disposition",), {
        "qualified": "Search correctness is reported as passing for the evaluated contract. Keep that contract; this field alone does not establish confirmatory acceptance or framework correctness.",
        "correctness_rejected": "Search correctness is reported as rejected. Inspect the available oracle diagnostic and repair the candidate before interpreting latency as an improvement.",
    }.get(correctness, "GPU correctness is not established here. Static admission, missing evidence and unknown values do not establish an oracle result."))

    measurement = field("measurement_quality", ("stable", "unstable", "not_measured"))
    add("measurement", measurement, ("measurement_quality", "search_latency_ms"), {
        "stable": "Search timing is reported as stable. Compare only within the frozen workload and timing boundary; inspect confirmation separately.",
        "unstable": "Search timing is reported as unstable. Do not explain its apparent latency difference as an optimization benefit; inspect measurement evidence before changing a performance hypothesis.",
        "not_measured": "Search timing was not measured. There is no latency comparison to explain.",
    }.get(measurement, "Timing quality is not established. A latency value alone does not establish stable timing or a speedup."))

    confirmed = document.get("confirmed", _ABSENT)
    confirmation = input_state or (
        "confirmed" if confirmed is True else
        "unconfirmed" if confirmed is False else
        "absent" if confirmed is _ABSENT else
        "unknown" if confirmed is None or confirmed == "unknown" else "malformed"
    )
    add("confirmation", confirmation, ("confirmed", "confirmed_latency_ms"), {
        "confirmed": "The controller reports confirmatory qualification under this evaluation contract. It does not establish a causal mechanism, cross-workload generalization or serving benefit.",
        "unconfirmed": "The controller has not confirmed this candidate. This flag alone does not say whether confirmation was absent, incorrect or unstable; retain that uncertainty.",
    }.get(confirmation, "Confirmatory qualification is not established. Do not promote search timing or a missing confirmation field into acceptance."))

    profile = document.get("profile", _ABSENT)
    profile_state = input_state or (
        "absent" if profile is _ABSENT else
        "unavailable" if profile is None else
        "observed" if isinstance(profile, Mapping) and profile else
        "unknown" if isinstance(profile, Mapping) or profile == "unknown" else "malformed"
    )
    add("profile", profile_state, ("profile",),
        "Inspect the visible profile's metrics, scope, provenance and unavailable observations when forming the next hypothesis. Presence of a profile is not evidence of a bottleneck or causal mechanism."
        if profile_state == "observed" else
        "No usable profile is established by this field. Keep the mechanism unknown; do not replace missing profiler observations with static estimates or a latency number.")

    findings = document.get("findings", _ABSENT)
    findings_state = input_state or (
        "absent" if findings is _ABSENT else
        "unknown" if findings is None or findings == "unknown" else
        "malformed" if not isinstance(findings, list) or any(
            not isinstance(row, Mapping)
            or any(key in row and not isinstance(row[key], str)
                   for key in ("path", "category", "code", "message", "severity"))
            or any(key in row and not isinstance(row[key], bool)
                   for key in ("blocks_acceptance", "blocks_lowering"))
            for row in findings
        ) else
        "reported" if findings else "none"
    )
    add("findings", findings_state, ("findings",),
        "Use each finding's path and category to locate the declared contract. Respect blocks_acceptance and blocks_lowering separately: a lowering capability refusal need not mean the candidate is wrong. Reports and hints describe only their modeled domain; record gaps for independent review, without editing the frozen Compiler."
        if findings_state == "reported" else
        "No localized findings are established here. An empty or unavailable findings list does not prove GPU safety, modeled resource coverage or absence of Compiler limitations.")

    return {"schema_version": 1, "kind": "evidence_rubric_v1", "source": _SOURCE, "criteria": criteria}
