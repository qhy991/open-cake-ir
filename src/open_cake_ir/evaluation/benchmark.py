"""Shared CUPTI timing policy, independent of task semantics."""
from __future__ import annotations
from typing import Callable,Protocol,Sequence
import warnings

class CuptiBenchmark(Protocol):
    """The retained FlashInfer CUPTI timing boundary."""

    def __call__(
        self,
        function: Callable[[], None],
        *,
        dry_run_iters: int,
        repeat_iters: int,
        cold_l2_cache: bool,
        use_cuda_graph: bool,
    ) -> Sequence[float]: ...


class StrictCuptiBenchmark:
    """Forbid CUDA-event/graph fallback around the retained CUPTI helper."""

    def __init__(self, helper_module: object) -> None:
        required = (
            "bench_gpu_time_with_cupti",
            "bench_gpu_time_with_cuda_event",
            "bench_gpu_time_with_cudagraph",
        )
        if any(not callable(getattr(helper_module, name, None)) for name in required):
            raise ValueError("CUPTI helper surface differs")
        self._helper = helper_module

    def __call__(
        self,
        function: Callable[[], None],
        *,
        dry_run_iters: int,
        repeat_iters: int,
        cold_l2_cache: bool,
        use_cuda_graph: bool,
    ) -> Sequence[float]:
        original_event = self._helper.bench_gpu_time_with_cuda_event
        original_graph = self._helper.bench_gpu_time_with_cudagraph

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("CUPTI timing fallback is forbidden")

        self._helper.bench_gpu_time_with_cuda_event = forbidden
        self._helper.bench_gpu_time_with_cudagraph = forbidden
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                values = self._helper.bench_gpu_time_with_cupti(
                    function,
                    dry_run_iters=dry_run_iters,
                    repeat_iters=repeat_iters,
                    cold_l2_cache=cold_l2_cache,
                    use_cuda_graph=use_cuda_graph,
                )
            if captured:
                raise RuntimeError("CUPTI helper emitted a warning")
            return values
        finally:
            self._helper.bench_gpu_time_with_cuda_event = original_event
            self._helper.bench_gpu_time_with_cudagraph = original_graph
