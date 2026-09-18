"""What an attribution profile taken inside evaluate says about itself, whichever platform took it.

Metal's compute-stage timestamps and the AMDGCN roctracer activity are read by their own
modules, which own the raw representation and its projection. The identity half of each
record -- whose candidate and case, which kernel, which local-broker job under which
allocation, that the instrumented launch was separate, which Evaluation policy -- was
checked twice with the same fields and only the vendor's name changed. It is checked
here once; each caller passes its own kind, its own job prefix, its own projection and
its own words, so every refusal still names the platform it belongs to.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Callable, Mapping

from open_cake_ir.serialization import canonical_json_bytes

ATTRIBUTION_EVALUATIONS = frozenset({
    "correctness_then_profile", "correctness_then_profile_each_search_survivor"})


def load_instrumented_profile(payload: bytes, *, kind: str, job_prefix: str, label: str,
                              summary: Callable[[object], Mapping], raw_name: str,
                              expected_candidate_sha256: str, expected_case_id: str,
                              expected_protocol_sha256: str | None,
                              identity: Callable[[Mapping], bool] = lambda document: True,
                              policy: Callable[[Mapping], bool] = lambda evaluation: True,
                              ) -> dict:
    """Read a retained in-evaluate profile, refusing one that is not this candidate's.

    `identity` and `policy` are the platform's own extra checks on the document and on
    its Evaluation policy; they run inside the same refusal as the shared fields so a
    reader gets one statement per class of mismatch.
    """
    document = json.loads(payload)
    if (not isinstance(document, dict) or document.get("kind") != kind
            or document.get("candidate_sha256") != expected_candidate_sha256
            or document.get("case_id") != expected_case_id
            or not isinstance(document.get("kernel_name"), str)
            or not document["kernel_name"].isidentifier()
            or not isinstance(document.get("job_id"), str)
            or re.fullmatch(rf"{job_prefix}-[0-9a-f]{{12}}", document["job_id"]) is None
            or document["job_id"] == f"{job_prefix}-000000000000"
            or document.get("allocation_mode") != "local_serialized"
            or document.get("external_gpu_activity") != "not_excluded"
            or document.get("separate_instrumented_launch") is not True
            or not identity(document)):
        raise ValueError(f"{label} attribution profile identity differs")
    evaluation = document.get("evaluation_protocol")
    if (not isinstance(evaluation, Mapping) or evaluation.get("case_id") != expected_case_id
            or evaluation.get("attribution_evaluation") not in ATTRIBUTION_EVALUATIONS
            or not policy(evaluation)):
        raise ValueError(f"{label} attribution Evaluation policy differs")
    if (expected_protocol_sha256 is not None
            and sha256(canonical_json_bytes(evaluation)).hexdigest() != expected_protocol_sha256):
        raise ValueError(f"{label} attribution Evaluation identity differs")
    if document.get("summary") != summary(document.get("raw")):
        raise ValueError(f"{label} profile summary differs from raw {raw_name}")
    return document
