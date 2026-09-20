"""MCPTI single-dispatch timing and separately collected C550 attribution.

The measured interval is Kernel8.end - Kernel8.start, excluding host launch and the
explicit preceding reset. Each cold sample follows a streamed FP32 fill of four times
the Target's L2 size, on the same stream. This is a declared reset protocol, not an
assertion that all caches are physically empty. Every device operation is accounted
for; a second kernel or an unobserved reset invalidates the entire cohort.
"""
from __future__ import annotations

import math
from typing import Mapping

from .metax_activity import activity_collector

TIMER = "mcpti_concurrent_kernel_start_end_ns"
RESET = "fp32_fill_ones_4x_declared_l2_same_stream_before_each_sample"
# MACA 3.5.3 mcpti_runtime_cbid.h. Only the ordinary Torch launch and the sealed
# module launch are qualified. Batch/cooperative/graph launches cannot masquerade
# as one dispatch even if their activity record happens to carry the expected name.
_LAUNCH_CBIDS = frozenset({56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 333, 394})
_SINGLE_LAUNCH_CBIDS = frozenset({56, 60})


def kernel_records(activity: Mapping) -> list[dict]:
    """Validate raw capture coverage before deriving any duration or attribution."""
    if (not isinstance(activity, Mapping) or activity.get("source") != "mcpti_activity"
            or activity.get("api_version") != 18 or activity.get("dropped_records") != 0
            or type(activity.get("dropped_records")) is not int
            or type(activity.get("pending_buffers")) is not int or activity["pending_buffers"] != 0
            or not isinstance(activity.get("records"), (list, tuple))):
        raise ValueError("MACA activity source, ABI or dropped-record coverage differs")
    kernels, correlations, launches = [], set(), {}
    for item in activity["records"]:
        if not isinstance(item, Mapping) or type(item.get("kind")) is not int:
            raise ValueError("MACA activity record differs")
        if item["kind"] in (4, 5):
            if (item["kind"] != 5
                    or any(type(item.get(key)) is not int or item[key] <= 0
                           for key in ("correlation", "cbid", "start_ns", "end_ns"))
                    or item["end_ns"] < item["start_ns"]
                    or type(item.get("return_value")) is not int or item["return_value"] != 0):
                raise ValueError("MACA activity contains a failed, unmodeled or unidentified API call")
            if item["cbid"] in _LAUNCH_CBIDS:
                if (item["cbid"] not in _SINGLE_LAUNCH_CBIDS or item["correlation"] in launches):
                    raise ValueError("MACA launch API is not a unique qualified single dispatch")
                launches[item["correlation"]] = item
            continue
        if item["kind"] != 10:
            raise ValueError("MACA cohort contains a copy, memset or unmodeled device operation")
        if (not isinstance(item.get("name"), str) or not item["name"]
                or any(type(item.get(k)) is not int or item[k] < 0 for k in (
                    "device", "context", "stream", "start_ns", "end_ns", "correlation",
                    "registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes", "local_bytes_per_thread"))
                or item["start_ns"] <= 0 or item["end_ns"] <= item["start_ns"]
                or item["correlation"] <= 0 or item["correlation"] in correlations
                or item["device"] != 0
                or any(not isinstance(item.get(axis), (list, tuple)) or len(item[axis]) != 3
                       or any(type(n) is not int or n <= 0 for n in item[axis]) for axis in ("grid", "block"))):
            raise ValueError("MACA kernel timestamps, launch identity or resources differ")
        correlations.add(item["correlation"])
        kernels.append(dict(item))
    if correlations != set(launches):
        raise ValueError("MACA launch APIs and device dispatches do not correspond one to one")
    if any(launches[item["correlation"]]["start_ns"] > item["end_ns"] for item in kernels):
        raise ValueError("MACA device activity precedes its launch API")
    return sorted(kernels, key=lambda item: (item["start_ns"], item["correlation"]))


def _identity(record):
    return (record["name"], tuple(record["grid"]), tuple(record["block"]),
            record["device"], record["context"], record["stream"])


def dispatch_samples(activity: Mapping, *, kernel_name: str, grid, block,
                     repeats: int, reset_record: Mapping | None) -> list[float]:
    """Attribute complete dispatches, not a substring match or an event-count residual."""
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("MACA sample count must be positive")
    records = kernel_records(activity)
    stride = 2 if reset_record is not None else 1
    if len(records) != repeats * stride:
        raise ValueError("MACA observed dispatch count differs from reset/candidate sequence")
    samples, previous = [], None
    stream = None
    for index in range(repeats):
        reset = records[stride * index] if reset_record is not None else None
        candidate = records[stride * index + stride - 1]
        if (candidate["name"] != kernel_name or tuple(candidate["grid"]) != tuple(grid)
                or tuple(candidate["block"]) != tuple(block)):
            raise ValueError("MACA timed kernel differs from the sealed name/grid/block")
        current_stream = (candidate["device"], candidate["context"], candidate["stream"])
        if stream is not None and current_stream != stream:
            raise ValueError("MACA sample changed context or stream")
        stream = current_stream
        if reset is not None and (
                _identity(reset) != _identity(reset_record) or reset["name"] == kernel_name
                or (reset["device"], reset["context"], reset["stream"]) != current_stream
                or reset["end_ns"] > candidate["start_ns"]):
            raise ValueError("MACA cache reset did not precede the matching sample on its stream")
        first = reset if reset is not None else candidate
        if previous is not None and previous > first["start_ns"]:
            raise ValueError("MACA serialized samples overlap")
        previous = candidate["end_ns"]
        duration = (candidate["end_ns"] - candidate["start_ns"]) / 1e6
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("MACA sample duration is invalid")
        samples.append(duration)
    return samples


class McptiDispatchBenchmark:
    def __init__(self, manifest, *, activity_library: str, l2_cache_bytes: int):
        if type(l2_cache_bytes) is not int or l2_cache_bytes <= 0 or l2_cache_bytes % 4:
            raise ValueError("MACA timing requires a declared positive FP32-aligned L2 capacity")
        self.manifest = manifest
        self.l2_cache_bytes = l2_cache_bytes
        self._collector = activity_collector(activity_library)
        self._reset = None
        self._reset_record = None
        self.last_activity = None
        self.non_target_dispatches = None
        self.resolution_us = None

    def _collect(self, function):
        import torch
        torch.cuda.synchronize()
        self._collector.begin()
        try:
            function()
            torch.cuda.synchronize()
        except BaseException as primary:
            try:
                torch.cuda.synchronize()
            finally:
                try:
                    self._collector.finish()
                except BaseException as teardown:
                    from .loaders import LifecycleError
                    raise LifecycleError(primary, teardown) from primary
            raise
        return self._collector.finish()

    def _prepare_reset(self):
        import torch
        if self._reset is None:
            observed = torch.cuda.get_device_properties(0).L2_cache_size
            if observed != self.l2_cache_bytes:
                raise ValueError("MACA runtime L2 capacity differs from the Target")
            # Bytes = four times the declared L2, elements = bytes / sizeof(FP32).
            self._reset = torch.empty(self.l2_cache_bytes, dtype=torch.float32, device="cuda:0")
            self._reset.fill_(1.0)
            activity = self._collect(lambda: self._reset.fill_(1.0))
            records = kernel_records(activity)
            if len(records) != 1 or records[0]["name"] == self.manifest.kernel_name:
                raise ValueError("MACA reset is not one independently identified device fill")
            self._reset_record = records[0]

    def __call__(self, function, *, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph):
        import torch
        if use_cuda_graph:
            raise ValueError("MACA dispatch timing does not measure graph replay")
        if type(cold_l2_cache) is not bool or any(type(v) is not int or v <= 0 for v in (dry_run_iters, repeat_iters)):
            raise ValueError("MACA timing iteration or reset contract differs")
        if cold_l2_cache:
            self._prepare_reset()
        for _ in range(dry_run_iters):
            function()
        torch.cuda.synchronize()
        def cohort():
            for _ in range(repeat_iters):
                if cold_l2_cache:
                    self._reset.fill_(1.0)
                function()
        activity = self._collect(cohort)
        reset = self._reset_record if cold_l2_cache else None
        samples = dispatch_samples(activity, kernel_name=self.manifest.kernel_name,
            grid=self.manifest.grid, block=self.manifest.block, repeats=repeat_iters, reset_record=reset)
        self.last_activity = {"timer": TIMER, "cache_policy": RESET if cold_l2_cache else "none",
            "l2_cache_bytes": self.l2_cache_bytes, "reset_bytes": 4 * self.l2_cache_bytes if cold_l2_cache else 0,
            "reset_record": reset, "activity": activity}
        self.non_target_dispatches = 0  # Proven above; any extra device activity is refused.
        unique = sorted(set(int(round(sample * 1e6)) for sample in samples))
        self.resolution_us = min((b - a for a, b in zip(unique, unique[1:])), default=0) / 1000 or None
        return samples
