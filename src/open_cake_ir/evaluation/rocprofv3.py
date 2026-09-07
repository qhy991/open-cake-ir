"""Fail-closed projection of exact rocprofv3 kernel-trace CSV evidence."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast


_REQUIRED_COLUMNS = {
    "Kind",
    "Agent_Id",
    "Dispatch_Id",
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
}
_RESOURCE_COLUMNS = (
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
)
_STATS_COLUMNS = {
    "Name",
    "Calls",
    "TotalDurationNs",
    "AverageNs",
    "Percentage",
    "MinNs",
    "MaxNs",
    "StdDev",
}


@dataclass(frozen=True)
class Rocprofv3KernelTraceExpectation:
    """Expected target dispatches for one isolated profiler arm."""

    kernel_name: str
    dispatch_count: int
    workgroup_size: tuple[int, int, int]
    grid_size: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_name, str) or not self.kernel_name.strip():
            raise ValueError("rocprofv3 kernel name must be non-empty")
        if type(self.dispatch_count) is not int or self.dispatch_count <= 0:
            raise ValueError("rocprofv3 dispatch count must be positive")
        for name, value in (
            ("workgroup_size", self.workgroup_size),
            ("grid_size", self.grid_size),
        ):
            if (
                not isinstance(value, tuple) or len(value) != 3
                or any(type(item) is not int or item <= 0 for item in value)
            ):
                raise ValueError(f"rocprofv3 {name} must contain three positive integers")


def find_kernel_trace_csv(output_root: Path) -> Path:
    """Locate the one raw kernel trace without depending on rocprofv3's PID prefix."""

    return _find_one(output_root, "*_kernel_trace.csv", "kernel-trace CSV")


def _find_one(output_root: Path, pattern: str, context: str) -> Path:
    if output_root.is_symlink():
        raise ValueError("rocprofv3 output root custody differs")
    root = output_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("rocprofv3 output root custody differs")
    matches = tuple(
        path
        for path in sorted(root.rglob(pattern))
    )
    if any(path.is_symlink() or not path.is_file() for path in matches):
        raise ValueError("rocprofv3 output file custody differs")
    if len(matches) != 1:
        raise ValueError(f"rocprofv3 must produce exactly one {context}")
    return matches[0]


def find_kernel_stats_csv(output_root: Path) -> Path:
    """Locate the one raw kernel stats CSV for cross-output call validation."""

    return _find_one(output_root, "*_kernel_stats.csv", "kernel-stats CSV")


def find_results_json(output_root: Path) -> Path:
    """Locate the one raw rocprofv3 JSON result for an independent dispatch join."""

    return _find_one(output_root, "*_results.json", "results JSON")


def _integer(row: dict[str, str], column: str, *, positive: bool = False) -> int:
    raw = row.get(column)
    try:
        value = int(raw) if raw is not None else -1
    except ValueError as error:
        raise ValueError(f"rocprofv3 column {column!r} must be an integer") from error
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"rocprofv3 column {column!r} must be {qualifier}")
    return value


def parse_kernel_trace_csv(
    payload: bytes,
    expectation: Rocprofv3KernelTraceExpectation,
) -> dict[str, object]:
    """Validate dispatch identity/count/geometry and project no timing decision."""

    if not isinstance(payload, bytes):
        raise ValueError("rocprofv3 CSV payload must contain bytes")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("rocprofv3 kernel trace is not UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(source, newline=""), strict=True)
    fields = reader.fieldnames
    if (
        fields is None
        or len(fields) != len(set(fields))
        or not _REQUIRED_COLUMNS.issubset(fields)
    ):
        raise ValueError("rocprofv3 kernel-trace columns differ")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError("rocprofv3 CSV syntax differs") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("rocprofv3 CSV row fields differ")

    dispatch_ids: list[int] = []
    agents: set[str] = set()
    resources: set[tuple[int, ...]] = set()
    expected_workgroup = expectation.workgroup_size
    expected_grid = expectation.grid_size
    for row in rows:
        if row.get("Kind") != "KERNEL_DISPATCH":
            raise ValueError("rocprofv3 trace contains a non-kernel-dispatch row")
        start = _integer(row, "Start_Timestamp")
        end = _integer(row, "End_Timestamp")
        if end < start:
            raise ValueError("rocprofv3 dispatch timestamps differ")
        if row.get("Kernel_Name") != expectation.kernel_name:
            continue
        agent = row.get("Agent_Id")
        if not agent:
            raise ValueError("rocprofv3 trace Agent_Id differs")
        agents.add(agent)
        dispatch_ids.append(_integer(row, "Dispatch_Id"))
        workgroup = tuple(
            _integer(row, f"Workgroup_Size_{axis}", positive=True)
            for axis in "XYZ"
        )
        grid = tuple(
            _integer(row, f"Grid_Size_{axis}", positive=True) for axis in "XYZ"
        )
        if workgroup != expected_workgroup or grid != expected_grid:
            raise ValueError("rocprofv3 dispatch geometry differs")
        resources.add(tuple(_integer(row, name) for name in _RESOURCE_COLUMNS))
    if len(dispatch_ids) != expectation.dispatch_count:
        raise ValueError("rocprofv3 target dispatch count differs")
    if len(set(dispatch_ids)) != len(dispatch_ids):
        raise ValueError("rocprofv3 dispatch identifiers are not unique")
    if len(agents) != 1 or len(resources) != 1:
        raise ValueError("rocprofv3 target dispatch metadata is not invariant")

    resource_values = next(iter(resources))
    return {
        "schema_version": 1,
        "kind": "rocprofv3_kernel_trace_projection_v1",
        "kernel_name": expectation.kernel_name,
        "dispatch_count": len(dispatch_ids),
        "raw_kernel_dispatch_count": len(rows),
        "non_target_dispatch_count": len(rows) - len(dispatch_ids),
        "agent_id": next(iter(agents)),
        "launch": {
            "workgroup_size": list(expected_workgroup),
            "grid_size": list(expected_grid),
        },
        "resources": {
            "lds_allocation_block_bytes": resource_values[0],
            "lds_allocation_granularity_bytes": 512,
            "scratch_size": resource_values[1],
            "vgpr_count": resource_values[2],
            "accum_vgpr_count": resource_values[3],
            "sgpr_count": resource_values[4],
            "occupancy_derived": False,
        },
        "timestamps_validated": True,
        "timestamps_projected": False,
        "duration_used_for_timing_or_promotion": False,
    }


def parse_kernel_stats_csv(
    payload: bytes,
    *,
    kernel_name: str,
    expected_calls: int,
) -> dict[str, object]:
    """Validate stats call count while leaving all profiler durations in raw evidence."""

    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError("rocprofv3 kernel name must be non-empty")
    if type(expected_calls) is not int or expected_calls <= 0:
        raise ValueError("rocprofv3 expected calls must be positive integer")
    if not isinstance(payload, bytes):
        raise ValueError("rocprofv3 CSV payload must contain bytes")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("rocprofv3 kernel stats is not UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(source, newline=""), strict=True)
    fields = reader.fieldnames
    if fields is None or len(fields) != len(set(fields)) or set(fields) != _STATS_COLUMNS:
        raise ValueError("rocprofv3 kernel-stats columns differ")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError("rocprofv3 CSV syntax differs") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("rocprofv3 CSV row fields differ")
    matches = [row for row in rows if row.get("Name") == kernel_name]
    if len(matches) != 1:
        raise ValueError("rocprofv3 kernel stats target row count differs")
    row = matches[0]
    calls = _integer(row, "Calls", positive=True)
    if calls != expected_calls:
        raise ValueError("rocprofv3 kernel stats Calls differs")
    for column in _STATS_COLUMNS - {"Name", "Calls"}:
        try:
            value = float(cast(str, row.get(column)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"rocprofv3 stats {column!r} must be numeric") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"rocprofv3 stats {column!r} must be finite and non-negative")
    return {
        "schema_version": 1,
        "kind": "rocprofv3_kernel_stats_projection_v1",
        "kernel_name": kernel_name,
        "calls": calls,
        "raw_kernel_count": len(rows),
        "duration_columns_validated": True,
        "duration_values_projected": False,
        "duration_used_for_timing_or_promotion": False,
    }


def _strict_json(payload: bytes) -> Mapping[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"rocprofv3 JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid(value: str) -> object:
        raise ValueError(f"rocprofv3 JSON contains non-finite number {value}")

    if not isinstance(payload, bytes):
        raise ValueError("rocprofv3 JSON payload must contain bytes")
    try:
        document = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("rocprofv3 result is not valid UTF-8 JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("rocprofv3 result must contain an object")
    return cast(Mapping[str, object], document)


def _json_integer(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"rocprofv3 JSON {context} must be {qualifier} integer")
    return value


def _xyz(value: object, context: str) -> tuple[int, int, int]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y", "z"}:
        raise ValueError(f"rocprofv3 JSON {context} differs")
    return tuple(
        _json_integer(value[axis], f"{context}.{axis}", positive=True)
        for axis in "xyz"
    )


def parse_results_json(
    payload: bytes,
    expectation: Rocprofv3KernelTraceExpectation,
) -> dict[str, object]:
    """Join JSON dispatches to formatted symbols and validate the target independently."""

    document = _strict_json(payload)
    tools = document.get("rocprofiler-sdk-tool")
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], Mapping):
        raise ValueError("rocprofv3 JSON process set differs")
    tool = cast(Mapping[str, object], tools[0])
    symbols_value = tool.get("kernel_symbols")
    buffers = tool.get("buffer_records")
    if not isinstance(symbols_value, list) or not isinstance(buffers, Mapping):
        raise ValueError("rocprofv3 JSON kernel records differ")
    dispatches = buffers.get("kernel_dispatch")
    if not isinstance(dispatches, list):
        raise ValueError("rocprofv3 JSON kernel dispatches differ")
    symbols: dict[int, Mapping[str, object]] = {}
    for value in symbols_value:
        if not isinstance(value, Mapping):
            raise ValueError("rocprofv3 JSON kernel symbol differs")
        kernel_id = _json_integer(value.get("kernel_id"), "kernel_id")
        if kernel_id == 0:
            # rocprofv3 sizes this table by max kernel id; index zero and holes are
            # zero-id placeholders, not dispatchable symbols.
            continue
        if kernel_id in symbols:
            raise ValueError("rocprofv3 JSON kernel_id is duplicated")
        symbols[kernel_id] = cast(Mapping[str, object], value)

    target_dispatch_ids: list[int] = []
    agents: set[int] = set()
    resources: set[tuple[int, int, int, int, int]] = set()
    symbol_resources: set[tuple[int, int, int, int, int]] = set()
    for value in dispatches:
        if not isinstance(value, Mapping):
            raise ValueError("rocprofv3 JSON dispatch record differs")
        info = value.get("dispatch_info")
        if not isinstance(info, Mapping):
            raise ValueError("rocprofv3 JSON dispatch_info differs")
        kernel_id = _json_integer(info.get("kernel_id"), "dispatch kernel_id", positive=True)
        symbol = symbols.get(kernel_id)
        if symbol is None:
            raise ValueError("rocprofv3 JSON dispatch has no kernel symbol")
        name = symbol.get("formatted_kernel_name")
        if not isinstance(name, str) or not name:
            raise ValueError("rocprofv3 JSON formatted kernel name differs")
        start = _json_integer(value.get("start_timestamp"), "start_timestamp")
        end = _json_integer(value.get("end_timestamp"), "end_timestamp")
        if end < start:
            raise ValueError("rocprofv3 JSON dispatch timestamps differ")
        if name != expectation.kernel_name:
            continue
        if (
            _xyz(info.get("workgroup_size"), "workgroup_size")
            != expectation.workgroup_size
            or _xyz(info.get("grid_size"), "grid_size") != expectation.grid_size
        ):
            raise ValueError("rocprofv3 JSON dispatch geometry differs")
        target_dispatch_ids.append(
            _json_integer(info.get("dispatch_id"), "dispatch_id")
        )
        agent = info.get("agent_id")
        if not isinstance(agent, Mapping) or set(agent) != {"handle"}:
            raise ValueError("rocprofv3 JSON agent_id differs")
        agents.add(_json_integer(agent.get("handle"), "agent_id.handle", positive=True))
        group = _json_integer(info.get("group_segment_size"), "group_segment_size")
        private = _json_integer(
            info.get("private_segment_size"), "private_segment_size"
        )
        resources.add(
            (
                group,
                private,
                _json_integer(symbol.get("arch_vgpr_count"), "arch_vgpr_count"),
                _json_integer(symbol.get("accum_vgpr_count"), "accum_vgpr_count"),
                _json_integer(symbol.get("sgpr_count"), "sgpr_count"),
            )
        )
        symbol_resources.add(
            (
                _json_integer(symbol.get("group_segment_size"), "symbol group_segment_size"),
                _json_integer(
                    symbol.get("private_segment_size"), "symbol private_segment_size"
                ),
                _json_integer(symbol.get("arch_vgpr_count"), "symbol arch_vgpr_count"),
                _json_integer(
                    symbol.get("accum_vgpr_count"), "symbol accum_vgpr_count"
                ),
                _json_integer(symbol.get("sgpr_count"), "symbol sgpr_count"),
            )
        )
    if len(target_dispatch_ids) != expectation.dispatch_count:
        raise ValueError("rocprofv3 JSON target dispatch count differs")
    if len(set(target_dispatch_ids)) != len(target_dispatch_ids):
        raise ValueError("rocprofv3 JSON target dispatch ids are duplicated")
    if len(agents) != 1 or len(resources) != 1 or resources != symbol_resources:
        raise ValueError("rocprofv3 JSON target dispatch metadata differs")
    resource = next(iter(resources))
    return {
        "schema_version": 1,
        "kind": "rocprofv3_json_dispatch_projection_v1",
        "kernel_name": expectation.kernel_name,
        "dispatch_count": len(target_dispatch_ids),
        "raw_kernel_dispatch_count": len(dispatches),
        "launch": {
            "workgroup_size": list(expectation.workgroup_size),
            "grid_size": list(expectation.grid_size),
        },
        "resources": {
            "group_segment_size": resource[0],
            "private_segment_size": resource[1],
            "vgpr_count": resource[2],
            "accum_vgpr_count": resource[3],
            "sgpr_count": resource[4],
            "occupancy_derived": False,
        },
        "timestamps_validated": True,
        "timestamps_projected": False,
        "duration_used_for_timing_or_promotion": False,
    }


def validate_cross_output_agreement(
    trace: Mapping[str, object],
    stats: Mapping[str, object],
    result_json: Mapping[str, object],
) -> None:
    """Require three independent rocprofv3 output surfaces to agree on the target."""

    if any(not isinstance(value, Mapping) for value in (trace, stats, result_json)):
        raise ValueError("rocprofv3 projections must be objects")
    _json_integer(stats.get("calls"), "stats.calls", positive=True)
    for label, projection in (("trace", trace), ("result", result_json)):
        launch = projection.get("launch")
        if not isinstance(launch, Mapping) or any(
            not isinstance(launch.get(name), list)
            for name in ("workgroup_size", "grid_size")
        ):
            raise ValueError(f"rocprofv3 {label} launch projection differs")
        # Projected geometry retains the raw dispatch expectation's domain.
        Rocprofv3KernelTraceExpectation(
            kernel_name=projection.get("kernel_name"),
            dispatch_count=projection.get("dispatch_count"),
            workgroup_size=tuple(launch["workgroup_size"]),
            grid_size=tuple(launch["grid_size"]),
        )
    trace_resources = trace.get("resources")
    json_resources = result_json.get("resources")
    if not isinstance(trace_resources, Mapping) or not isinstance(
        json_resources, Mapping
    ):
        raise ValueError("rocprofv3 resource projections differ")
    group_segment = _json_integer(
        json_resources.get("group_segment_size"), "result.group_segment_size"
    )
    lds_allocation = _json_integer(
        trace_resources.get("lds_allocation_block_bytes"), "trace.lds_allocation_block_bytes"
    )
    granularity = _json_integer(
        trace_resources.get("lds_allocation_granularity_bytes"),
        "trace.lds_allocation_granularity_bytes", positive=True,
    )
    resource_pairs = (
        ("scratch_size", "private_segment_size"),
        ("vgpr_count", "vgpr_count"),
        ("accum_vgpr_count", "accum_vgpr_count"),
        ("sgpr_count", "sgpr_count"),
    )
    for trace_name, json_name in resource_pairs:
        _json_integer(trace_resources.get(trace_name), f"trace.{trace_name}")
        _json_integer(json_resources.get(json_name), f"result.{json_name}")
    expected_lds_allocation = (
        (group_segment + granularity - 1) // granularity
    ) * granularity
    shared_resources_agree = (
        lds_allocation == expected_lds_allocation
        and all(
            trace_resources[trace_name] == json_resources[json_name]
            for trace_name, json_name in resource_pairs
        )
        and trace_resources.get("occupancy_derived") is False
        and json_resources.get("occupancy_derived") is False
    )
    if (
        trace.get("kernel_name") != stats.get("kernel_name")
        or trace.get("kernel_name") != result_json.get("kernel_name")
        or trace.get("dispatch_count") != stats.get("calls")
        or trace.get("dispatch_count") != result_json.get("dispatch_count")
        or trace.get("launch") != result_json.get("launch")
        or not shared_resources_agree
    ):
        raise ValueError("rocprofv3 CSV, stats and JSON projections disagree")


__all__ = [
    "Rocprofv3KernelTraceExpectation",
    "find_kernel_stats_csv",
    "find_kernel_trace_csv",
    "find_results_json",
    "parse_kernel_stats_csv",
    "parse_kernel_trace_csv",
    "parse_results_json",
    "validate_cross_output_agreement",
]
