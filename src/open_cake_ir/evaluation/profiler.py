"""Typed, replayable Nsight Compute attribution evidence.

The raw CSV emitted by NCU is the observation.  ``metrics`` and ``summary`` are
projections kept beside it for bounded agent feedback; loading the artifact reparses the
raw bytes and refuses either projection when it differs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, cast

_DIGEST = re.compile(r"^[0-9a-f]{64}$")

NCU_ATTRIBUTION_METRICS = (
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_warps",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
)

_OCCUPANCY_LIMITS = {
    "launch__occupancy_limit_registers": "registers",
    "launch__occupancy_limit_shared_mem": "shared_memory",
    "launch__occupancy_limit_blocks": "blocks",
    "launch__occupancy_limit_warps": "warps",
}

_SIGNALS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "active_warps_pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (
        "long_scoreboard_stall_pct"
    ),
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "barrier_stall_pct",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _metric_rows(raw_stdout: str, kernel_name: str) -> dict[str, dict[str, object]]:
    lines = raw_stdout.splitlines()
    header_index: int | None = None
    required_columns = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    for index, line in enumerate(lines):
        try:
            columns = next(csv.reader([line]))
        except csv.Error:
            continue
        if required_columns <= set(columns):
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU attribution CSV header is missing")

    observed: dict[str, dict[str, object]] = {}
    try:
        rows = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        for row in rows:
            metric = row.get("Metric Name")
            if metric not in NCU_ATTRIBUTION_METRICS:
                continue
            if metric in observed or row.get("Kernel Name") != kernel_name:
                raise ValueError("NCU attribution metric identity differs")
            unit = row.get("Metric Unit")
            raw_value = row.get("Metric Value")
            if not isinstance(unit, str) or not unit or not isinstance(raw_value, str):
                raise ValueError("NCU attribution metric unit or value is missing")
            value = float(raw_value.replace(",", ""))
            if not math.isfinite(value) or value < 0:
                raise ValueError("NCU attribution metric value differs")
            observed[metric] = {"unit": unit, "value": value}
    except (csv.Error, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("NCU attribution"):
            raise
        raise ValueError("NCU attribution CSV differs") from error
    if set(observed) != set(NCU_ATTRIBUTION_METRICS):
        missing = sorted(set(NCU_ATTRIBUTION_METRICS) - set(observed))
        raise ValueError(f"NCU attribution metrics are incomplete: {', '.join(missing)}")
    return observed


def _summary(metrics: Mapping[str, object]) -> dict[str, object]:
    values = {
        name: float(_object(metrics[name], f"profile.metrics.{name}")["value"])
        for name in NCU_ATTRIBUTION_METRICS
    }
    limits = {
        resource: values[metric]
        for metric, resource in _OCCUPANCY_LIMITS.items()
    }
    resident = min(limits.values())
    return {
        "occupancy": {
            "binding_resources": sorted(
                resource for resource, value in limits.items() if value == resident
            ),
            "registers_per_thread": values["launch__registers_per_thread"],
            "resident_ctas_per_sm": resident,
        },
        "signals": {
            signal: values[metric] for metric, signal in _SIGNALS.items()
        },
    }


def build_ncu_attribution_profile(
    *,
    candidate_sha256: str,
    case_id: str,
    kernel_name: str,
    ncu_version: str,
    ncu_executable_sha256: str,
    stdout: bytes,
    stderr: bytes,
) -> bytes:
    """Build the sole canonical profile artifact from one raw NCU invocation."""

    try:
        raw_stdout = stdout.decode("utf-8")
        raw_stderr = stderr.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("NCU attribution output is not UTF-8") from error
    metrics = _metric_rows(raw_stdout, kernel_name)
    document = {
        "schema_version": 1,
        "kind": "ncu_kernel_attribution",
        "candidate_sha256": candidate_sha256,
        "case_id": case_id,
        "kernel_name": kernel_name,
        "tool": {
            "version": ncu_version,
            "executable_sha256": ncu_executable_sha256,
        },
        "raw": {"stdout": raw_stdout, "stderr": raw_stderr},
        "metrics": metrics,
        "summary": _summary(metrics),
    }
    payload = _canonical_json_bytes(document)
    # The loader is the evidence boundary. Building through it prevents the worker and
    # replay path from accepting different spellings of the same profile.
    load_ncu_attribution_profile(
        payload,
        expected_candidate_sha256=candidate_sha256,
        expected_case_id=case_id,
    )
    return payload


def load_ncu_attribution_profile(
    payload: bytes,
    *,
    expected_candidate_sha256: str,
    expected_case_id: str,
) -> Mapping[str, object]:
    """Reparse raw NCU output and validate every retained projection."""

    try:
        document = _object(json.loads(payload), "profile")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("NCU attribution profile is not JSON") from error
    if payload != _canonical_json_bytes(document):
        raise ValueError("NCU attribution profile is not canonical JSON")
    if set(document) != {
        "schema_version",
        "kind",
        "candidate_sha256",
        "case_id",
        "kernel_name",
        "tool",
        "raw",
        "metrics",
        "summary",
    } or document.get("schema_version") != 1 or document.get("kind") != (
        "ncu_kernel_attribution"
    ):
        raise ValueError("NCU attribution profile fields differ")
    kernel_name = document.get("kernel_name")
    if (
        document.get("candidate_sha256") != expected_candidate_sha256
        or _DIGEST.fullmatch(expected_candidate_sha256) is None
        or document.get("case_id") != expected_case_id
        or not isinstance(kernel_name, str)
        or not kernel_name
    ):
        raise ValueError("NCU attribution profile authority differs")
    tool = _object(document.get("tool"), "profile.tool")
    raw = _object(document.get("raw"), "profile.raw")
    metrics = _object(document.get("metrics"), "profile.metrics")
    summary = _object(document.get("summary"), "profile.summary")
    if (
        set(tool) != {"version", "executable_sha256"}
        or not isinstance(tool.get("version"), str)
        or not tool["version"]
        or not isinstance(tool.get("executable_sha256"), str)
        or _DIGEST.fullmatch(cast(str, tool["executable_sha256"])) is None
        or set(raw) != {"stdout", "stderr"}
        or not isinstance(raw.get("stdout"), str)
        or not isinstance(raw.get("stderr"), str)
        or set(metrics) != set(NCU_ATTRIBUTION_METRICS)
    ):
        raise ValueError("NCU attribution profile tool, raw output or metrics differ")
    for name, value in metrics.items():
        row = _object(value, f"profile.metrics.{name}")
        if (
            set(row) != {"unit", "value"}
            or not isinstance(row.get("unit"), str)
            or not row["unit"]
            or not isinstance(row.get("value"), float)
            or not math.isfinite(cast(float, row["value"]))
            or cast(float, row["value"]) < 0
        ):
            raise ValueError("NCU attribution metric projection differs")
    reparsed = _metric_rows(cast(str, raw["stdout"]), kernel_name)
    derived_summary = _summary(reparsed)
    if _canonical_json_bytes(metrics) != _canonical_json_bytes(
        reparsed
    ) or _canonical_json_bytes(summary) != _canonical_json_bytes(derived_summary):
        raise ValueError("NCU attribution projection differs from raw CSV")
    detached = json.loads(_canonical_json_bytes(document))
    return MappingProxyType(cast(dict[str, object], detached))


def ncu_attribution_feedback(profile: Mapping[str, object]) -> Mapping[str, object]:
    """Project only bounded actionable facts into the next authoring Turn."""

    summary = _object(profile.get("summary"), "profile.summary")
    return MappingProxyType(
        {
            "kind": "ncu_kernel_attribution",
            "kernel_name": profile["kernel_name"],
            "occupancy": dict(_object(summary["occupancy"], "profile.occupancy")),
            "signals": dict(_object(summary["signals"], "profile.signals")),
        }
    )
