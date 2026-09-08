"""Read-only compatibility boundary for the final legacy r45 result."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import json
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.tasks.flash_kmeans.portfolio import PortfolioArtifact, PortfolioCaseObservation, PortfolioEvaluationReceipt, evaluate_portfolio_observations
from open_cake_ir.evaluation.workload import WorkloadContract




def replay_legacy_r45_result(
    value: object,
    workload: WorkloadContract,
    *,
    expected_raw_fixture_sha256: str,
    expected_result_sha256: str,
    legacy_contract_sha256: str,
    broker_job_id: str,
) -> tuple[PortfolioArtifact, PortfolioEvaluationReceipt]:
    """Bind both the raw r45 fixture document and its self-hash preimage."""

    if not isinstance(value, Mapping):
        raise ValueError("legacy r45 result must be an object")
    if sha256(_canonical_json_bytes(value)).hexdigest() != expected_raw_fixture_sha256:
        raise ValueError("legacy r45 raw fixture bytes differ")
    result_preimage = dict(value)
    embedded_result_sha256 = result_preimage.pop("result_sha256", None)
    if embedded_result_sha256 != expected_result_sha256:
        raise ValueError("legacy r45 embedded result SHA256 differs")
    if sha256(_canonical_json_bytes(result_preimage)).hexdigest() != expected_result_sha256:
        raise ValueError("legacy r45 canonical result preimage differs")
    compile_results = value.get("compile")
    cases = value.get("cases")
    route = value.get("route")
    dispatcher = value.get("dispatcher")
    gpu = value.get("gpu")
    if not all(isinstance(item, Mapping) for item in (compile_results, cases, route, dispatcher, gpu)):
        raise ValueError("legacy r45 result sections differ")
    compile_results = cast(Mapping[str, Mapping[str, object]], compile_results)
    cases = cast(Mapping[str, Mapping[str, object]], cases)
    candidates: dict[str, LaunchableCandidate] = {}
    for case_id, compile_result in compile_results.items():
        artifacts = compile_result.get("artifacts")
        lowering = compile_result.get("lowering")
        launch_spec = compile_result.get("launch_spec")
        if not all(isinstance(item, Mapping) for item in (artifacts, lowering, launch_spec)):
            raise ValueError(f"legacy r45 {case_id} compile lineage differs")
        artifacts = cast(Mapping[str, Mapping[str, object]], artifacts)
        lowering = cast(Mapping[str, object], lowering)
        launch_spec = cast(Mapping[str, object], launch_spec)
        roles = {
            "lowered_source": str(lowering["source_sha256"]),
            "compiler_expanded_source": str(artifacts["source"]["sha256"]),
            "ttir": str(artifacts["ttir"]["sha256"]),
            "ttgir": str(artifacts["ttgir"]["sha256"]),
            "llir": str(artifacts["llir"]["sha256"]),
            "ptx": str(artifacts["ptx"]["sha256"]),
            "cubin": str(artifacts["cubin"]["sha256"]),
            "launch_manifest": sha256(_canonical_json_bytes(launch_spec)).hexdigest(),
        }
        candidate = LaunchableCandidate(
            candidate_sha256=str(lowering["specialist_sha256"]),
            target=str(launch_spec["target"]),
            entry_point=str(launch_spec["kernel_name"]),
            artifact_roles=roles,
            launch_spec_sha256=roles["launch_manifest"],
        )
        if compile_result.get("cubin_sha256") != roles["cubin"]:
            raise ValueError(f"legacy r45 {case_id} CUBIN identity differs")
        candidates[case_id] = candidate
    artifact = PortfolioArtifact.build(
        workload,
        "81237074bd16f926b3fadc891c9684e11b69dfb7d574277c7c4aee1227674a34",
        candidates,
    )
    gpu_admitted = cast(Mapping[str, object], cast(Mapping[str, object], gpu)["admitted"])
    observations: dict[str, PortfolioCaseObservation] = {}
    for case_id, case in cases.items():
        candidate = candidates[case_id]
        observations[case_id] = PortfolioCaseObservation(
            direct_preflight_correct=cast(Mapping[str, object], case["specialist_preflight"])[
                "passed"
            ]
            is True,
            dispatcher_preflight_correct=cast(
                Mapping[str, object], case["dispatcher_preflight"]
            )["passed"]
            is True,
            postflight_correct=cast(Mapping[str, object], case["postflight"])["passed"]
            is True,
            kernel_cohorts_ms=tuple(
                tuple(float(sample) for sample in cast(list[object], cohort["samples_ms"]))
                for cohort in cast(
                    list[Mapping[str, object]],
                    cast(Mapping[str, object], case["kernel_timing"])["cohorts"],
                )
            ),
            dispatcher_cohorts_ms=tuple(
                tuple(float(sample) for sample in cast(list[object], cohort["samples_ms"]))
                for cohort in cast(
                    list[Mapping[str, object]],
                    cast(Mapping[str, object], case["dispatcher_timing"])["cohorts"],
                )
            ),
            correctness_receipts={
                "direct_preflight": cast(
                    Mapping[str, object], case["specialist_preflight"]
                ),
                "dispatcher_preflight": cast(
                    Mapping[str, object], case["dispatcher_preflight"]
                ),
                "postflight": cast(Mapping[str, object], case["postflight"]),
            },
            candidate_record_sha256=candidate.canonical_sha256,
            cubin_sha256=candidate.artifact_roles["cubin"],
            launch_spec_sha256=candidate.launch_spec_sha256,
            module_admission={
                "candidate_record_sha256": candidate.canonical_sha256,
                "cubin_sha256": candidate.artifact_roles["cubin"],
                "launch_spec_sha256": candidate.launch_spec_sha256,
                "module_loaded": True,
                "gpu_uuid": str(gpu_admitted["device_uuid_sha256"]),
                "broker_job_id": broker_job_id,
            },
        )
    route_counts = dict(cast(Mapping[str, object], route))
    dispatcher_counters = cast(
        Mapping[str, object], cast(Mapping[str, object], dispatcher)["counters"]
    )
    route_counts.update(
        {
            "candidate_kernel_calls": 999,
            "dispatcher_kernel_calls": dispatcher_counters["kernel_calls"],
            "selections": {
                str(name).removeprefix("r43-"): count
                for name, count in cast(
                    Mapping[str, object], dispatcher_counters["selections"]
                ).items()
            },
        }
    )
    unsupported = cast(
        Mapping[str, object], cast(Mapping[str, object], dispatcher)["unsupported_probe"]
    )
    receipt = evaluate_portfolio_observations(
        artifact,
        observations,
        evaluation_protocol_sha256=legacy_contract_sha256,
        route_counts=route_counts,
        unsupported_kernel_call_delta=int(unsupported["kernel_call_delta"]),
    )
    return artifact, receipt
