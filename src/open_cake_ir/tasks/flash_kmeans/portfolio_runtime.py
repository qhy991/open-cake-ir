"""Concrete B200 assay for one frozen exact-shape portfolio."""

from __future__ import annotations
from open_cake_ir.evaluation.benchmark import CuptiBenchmark,StrictCuptiBenchmark

import json
import math
import time
import warnings
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Protocol, Sequence

from open_cake_ir.tasks.flash_kmeans.cuda import CudaTensorContract
from open_cake_ir.evaluation.cuda_driver import LoadedCudaCandidate, TensorLike
from open_cake_ir.tasks.flash_kmeans.workload import assignment_raw_sha256, classify_flash_kmeans_output, flash_kmeans_oracle, generate_flash_kmeans_case
from open_cake_ir.tasks.flash_kmeans.portfolio import ExactShapeDispatcher, PortfolioArtifact, PortfolioCaseObservation, PortfolioEvaluationReceipt, ExactShape, evaluate_portfolio_observations
from open_cake_ir.evaluation.workload import WorkloadContract


@dataclass
class _CaseState:
    case_id: str
    key: ExactShape
    tokens: object
    centroids: object
    centroid_sq: object
    output: object
    oracle: object

    @property
    def arguments(self) -> tuple[TensorLike, ...]:
        return self.tokens, self.centroids, self.centroid_sq, self.output


class _LoadedLauncher:
    def __init__(
        self,
        loaded: Mapping[str, LoadedCudaCandidate],
        by_candidate: Mapping[str, str],
        *,
        stream: object,
    ) -> None:
        self._loaded = loaded
        self._by_candidate = by_candidate
        self._stream = stream

    def launch(self, candidate, arguments) -> None:
        case_id = self._by_candidate[candidate.candidate_sha256]
        key = ExactShape(
            int(arguments[0].shape[0]),
            int(arguments[0].shape[1]),
            int(arguments[1].shape[1]),
            int(arguments[0].shape[2]),
        )
        self._loaded[case_id].launch(
            arguments,
            tensor_contract=CudaTensorContract(*key.as_tuple()),
            stream=self._stream,
        )


class CuptiPortfolioAssay:
    """Run r45's complete fixed sequence and return raw replayable observations."""

    def __init__(
        self,
        *,
        artifact: PortfolioArtifact,
        workload: WorkloadContract,
        evaluation_protocol: Mapping[str, object],
        loaded_candidates: Mapping[str, LoadedCudaCandidate],
        cupti_benchmark: CuptiBenchmark,
        synchronize: Callable[[], None],
        flush_l2: Callable[[], None],
        stream: object,
        device: str,
    ) -> None:
        if artifact.workload_sha256 != workload.canonical_sha256:
            raise ValueError("Portfolio Assay Workload differs")
        if set(loaded_candidates) != {entry.case_id for entry in artifact.entries}:
            raise ValueError("Portfolio Assay loaded-candidate set differs")
        for entry in artifact.entries:
            if loaded_candidates[entry.case_id].candidate.canonical_sha256 != entry.candidate.canonical_sha256:
                raise ValueError("Portfolio Assay loaded candidate differs from manifest")
        self.artifact = artifact
        self.protocol = json.loads(
            json.dumps(evaluation_protocol, sort_keys=True, separators=(",", ":"))
        )
        self.protocol_sha256 = sha256(
            json.dumps(self.protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._workload = workload
        self._loaded = dict(loaded_candidates)
        self._cupti = cupti_benchmark
        self._synchronize = synchronize
        self._flush_l2 = flush_l2
        self._stream = stream
        self._device = device
        self._observer: Callable[[str, Mapping[str, object]], None] | None = None

    def set_observer(
        self, observer: Callable[[str, Mapping[str, object]], None]
    ) -> None:
        self._observer = observer

    def prepare(self, _campaign_lock: object) -> PortfolioArtifact:
        return self.artifact

    def _emit(self, kind: str, payload: Mapping[str, object]) -> None:
        if self._observer is not None:
            self._observer(kind, payload)

    def _case_state(self, case_id: str) -> _CaseState:
        case = self._workload.case(case_id)
        key = ExactShape.from_case(case)
        tokens, centroids = generate_flash_kmeans_case(
            self._workload, case_id, device=self._device
        )
        torch = __import__("torch")
        centroids_fp32 = centroids.to(torch.float32)
        centroid_sq = (centroids_fp32 * centroids_fp32).sum(
            dim=-1, dtype=torch.float32
        ).contiguous()
        output = torch.empty(
            (key.batch, key.tokens), dtype=torch.int32, device=tokens.device
        )
        oracle = flash_kmeans_oracle(
            self._workload, tokens, centroids, case_id=case_id
        )
        return _CaseState(
            case_id, key, tokens, centroids, centroid_sq, output, oracle
        )

    def _correctness(self, state: _CaseState) -> dict[str, object]:
        passed, metrics = classify_flash_kmeans_output(
            self._workload,
            state.tokens,
            state.centroids,
            state.output,
            state.oracle,
            case_id=state.case_id,
        )
        output_sha256, output_size = assignment_raw_sha256(state.output)
        return {
            "passed": passed,
            "metrics": metrics,
            "output": {"sha256": output_sha256, "size_bytes": output_size},
        }

    def evaluate(self, _campaign_lock: object) -> PortfolioEvaluationReceipt:
        entries = {entry.case_id: entry for entry in self.artifact.entries}
        states = {case_id: self._case_state(case_id) for case_id in entries}
        by_candidate = {
            entry.candidate.candidate_sha256: entry.case_id
            for entry in self.artifact.entries
        }
        dispatcher = ExactShapeDispatcher(
            self.artifact,
            _LoadedLauncher(
                self._loaded,
                by_candidate,
                stream=self._stream,
            ),
        )
        direct_receipts: dict[str, Mapping[str, object]] = {}
        dispatch_receipts: dict[str, Mapping[str, object]] = {}
        post_receipts: dict[str, Mapping[str, object]] = {}
        kernel_cohorts: dict[str, tuple[tuple[float, ...], ...]] = {}
        dispatch_cohorts: dict[str, tuple[tuple[float, ...], ...]] = {}
        l2_flush_calls = 0
        try:
            for case_id, state in states.items():
                self._loaded[case_id].launch(
                    state.arguments,
                    tensor_contract=CudaTensorContract(*state.key.as_tuple()),
                    stream=self._stream,
                )
                self._synchronize()
                direct_receipts[case_id] = self._correctness(state)
                self._emit(
                    "direct_preflight",
                    {"case_id": case_id, "receipt": direct_receipts[case_id]},
                )
            if not all(item["passed"] is True for item in direct_receipts.values()):
                raise ValueError("portfolio direct preflight correctness failed")

            for case_id, state in states.items():
                dispatcher.dispatch(state.arguments)
                self._synchronize()
                dispatch_receipts[case_id] = self._correctness(state)
                self._emit(
                    "dispatcher_preflight",
                    {"case_id": case_id, "receipt": dispatch_receipts[case_id]},
                )
            if not all(item["passed"] is True for item in dispatch_receipts.values()):
                raise ValueError("portfolio dispatcher preflight correctness failed")

            torch = __import__("torch")
            unsupported = (
                torch.empty((2, 512, 128), dtype=torch.bfloat16, device=self._device),
                torch.empty((2, 1024, 128), dtype=torch.bfloat16, device=self._device),
                torch.empty((2, 1024), dtype=torch.float32, device=self._device),
                torch.empty((2, 512), dtype=torch.int32, device=self._device),
            )
            before_unsupported = dispatcher.kernel_calls
            try:
                dispatcher.dispatch(unsupported)
            except ValueError as error:
                if str(error) != "dispatcher shape is unsupported":
                    raise
            else:
                raise ValueError("portfolio unsupported shape was accepted")
            unsupported_delta = dispatcher.kernel_calls - before_unsupported
            if unsupported_delta != 0:
                raise ValueError("portfolio unsupported shape launched a kernel")
            self._emit(
                "unsupported_probe",
                {"kernel_call_delta": unsupported_delta, "rejected": True},
            )

            for case_id, state in states.items():
                cohorts: list[tuple[float, ...]] = []
                loaded = self._loaded[case_id]

                def direct_launch() -> None:
                    loaded.launch(
                        state.arguments,
                        tensor_contract=CudaTensorContract(*state.key.as_tuple()),
                        stream=self._stream,
                    )

                for _ in range(5):
                    before = loaded.launch_calls
                    samples = tuple(
                        float(value)
                        for value in self._cupti(
                            direct_launch,
                            dry_run_iters=11,
                            repeat_iters=25,
                            cold_l2_cache=True,
                            use_cuda_graph=False,
                        )
                    )
                    if len(samples) != 25 or loaded.launch_calls - before != 36:
                        raise ValueError("portfolio CUPTI cohort route differs")
                    cohorts.append(samples)
                    self._emit(
                        "kernel_cohort",
                        {
                            "case_id": case_id,
                            "cohort_index": len(cohorts) - 1,
                            "samples_ms": list(samples),
                            "route_calls": loaded.launch_calls - before,
                        },
                    )
                kernel_cohorts[case_id] = tuple(cohorts)

            for case_id, state in states.items():
                cohorts = []
                for _ in range(5):
                    before_selection = dispatcher.selections[case_id]
                    before_kernel = dispatcher.kernel_calls
                    samples: list[float] = []
                    for iteration in range(30):
                        self._flush_l2()
                        self._synchronize()
                        l2_flush_calls += 1
                        started_ns = time.perf_counter_ns() if iteration >= 5 else None
                        dispatcher.dispatch(state.arguments)
                        self._synchronize()
                        if started_ns is not None:
                            value = (time.perf_counter_ns() - started_ns) / 1e6
                            if not math.isfinite(value) or value <= 0:
                                raise ValueError("portfolio host timing sample differs")
                            samples.append(value)
                    if (
                        len(samples) != 25
                        or dispatcher.selections[case_id] - before_selection != 30
                        or dispatcher.kernel_calls - before_kernel != 30
                    ):
                        raise ValueError("portfolio dispatcher cohort route differs")
                    cohorts.append(tuple(samples))
                    self._emit(
                        "dispatcher_cohort",
                        {
                            "case_id": case_id,
                            "cohort_index": len(cohorts) - 1,
                            "samples_ms": samples,
                            "selection_calls": 30,
                            "kernel_calls": 30,
                        },
                    )
                dispatch_cohorts[case_id] = tuple(cohorts)

            for case_id, state in states.items():
                dispatcher.dispatch(state.arguments)
                self._synchronize()
                post_receipts[case_id] = self._correctness(state)
                self._emit(
                    "postflight",
                    {"case_id": case_id, "receipt": post_receipts[case_id]},
                )
            if not all(item["passed"] is True for item in post_receipts.values()):
                raise ValueError("portfolio postflight correctness failed")

            candidate_kernel_calls = sum(item.launch_calls for item in self._loaded.values())
            route_counts: dict[str, object] = {
                "automatic_retries": 0,
                "compiler_invocations": len(entries),
                "module_loads": len(entries),
                "module_unloads": len(entries),
                "direct_preflight_calls": len(entries),
                "dispatcher_preflight_calls": len(entries),
                "cupti_candidate_calls": 540,
                "host_dispatch_calls": 450,
                "l2_flush_calls": l2_flush_calls,
                "dispatcher_postflight_calls": len(entries),
                "unsupported_probes": 1,
                "candidate_kernel_calls": candidate_kernel_calls,
                "dispatcher_kernel_calls": dispatcher.kernel_calls,
                "fallback_calls": dispatcher.fallback_calls,
                "selections": dict(dispatcher.selections),
            }
            observations = {
                case_id: PortfolioCaseObservation(
                    direct_preflight_correct=bool(direct_receipts[case_id]["passed"]),
                    dispatcher_preflight_correct=bool(
                        dispatch_receipts[case_id]["passed"]
                    ),
                    postflight_correct=bool(post_receipts[case_id]["passed"]),
                    kernel_cohorts_ms=kernel_cohorts[case_id],
                    dispatcher_cohorts_ms=dispatch_cohorts[case_id],
                    correctness_receipts={
                        "direct_preflight": direct_receipts[case_id],
                        "dispatcher_preflight": dispatch_receipts[case_id],
                        "postflight": post_receipts[case_id],
                    },
                    candidate_record_sha256=entries[
                        case_id
                    ].candidate.canonical_sha256,
                    cubin_sha256=entries[case_id].candidate.artifact_roles["cubin"],
                    launch_spec_sha256=entries[case_id].candidate.launch_spec_sha256,
                    module_admission={
                        "candidate_record_sha256": entries[
                            case_id
                        ].candidate.canonical_sha256,
                        "cubin_sha256": entries[case_id].candidate.artifact_roles[
                            "cubin"
                        ],
                        "launch_spec_sha256": entries[
                            case_id
                        ].candidate.launch_spec_sha256,
                        "module_loaded": True,
                        "gpu_uuid": self._loaded[case_id].admission.gpu_uuid,
                        "broker_job_id": self._loaded[
                            case_id
                        ].admission.broker_job_id,
                    },
                )
                for case_id in entries
            }
            return evaluate_portfolio_observations(
                self.artifact,
                observations,
                evaluation_protocol_sha256=self.protocol_sha256,
                route_counts=route_counts,
                unsupported_kernel_call_delta=unsupported_delta,
            )
        finally:
            teardown_errors = []
            for loaded in self._loaded.values():
                if not loaded.closed:
                    try:
                        loaded.close(synchronize=self._synchronize)
                    except BaseException as error:
                        teardown_errors.append(error)
            if teardown_errors:
                raise RuntimeError(
                    f"portfolio teardown failed for {len(teardown_errors)} module(s)"
                )
