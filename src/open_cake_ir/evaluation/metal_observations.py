"""Pure validation of observed Metal timestamps; no runtime or task dependency."""
from __future__ import annotations

from hashlib import sha256
import json
import math
import re
from typing import Mapping
from .artifacts import METAL_TARGETS

METAL_TIMER = "MTLCommandBuffer.GPUStartTime/GPUEndTime"
METAL_CACHE = "warm_no_explicit_flush"
METAL_PROFILE_KIND = "metal_compute_stage_timestamps_v1"


def command_buffer_ms(observation: Mapping) -> float:
    if (not isinstance(observation, Mapping) or observation.get("completed") is not True
            or type(observation.get("launch_index")) is not int or observation["launch_index"] < 0):
        raise ValueError("Metal command completion or launch index differs")
    start, end = observation.get("gpu_start_seconds"), observation.get("gpu_end_seconds")
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or not 0 < start < end):
        raise ValueError("Metal command-buffer timestamps must be finite, positive and ordered")
    return (end - start) * 1000.0


def validate_host(host: object) -> dict:
    fields = {"device_name", "device_registry_id", "operating_system", "target"}
    if (not isinstance(host, Mapping) or set(host) != fields
            or any(not isinstance(value, str) or not value for value in host.values())
            or host["target"] not in METAL_TARGETS
            or not host["device_registry_id"].isdigit() or int(host["device_registry_id"]) <= 0):
        raise ValueError("Metal observation host identity differs")
    return dict(host)


def validate_launch_sequence(commands: list[Mapping]) -> None:
    ordered = sorted(commands, key=lambda row: row['launch_index'])
    if [row['launch_index'] for row in ordered] != list(range(len(ordered))):
        raise ValueError('Metal observations duplicate or omit physical launches')
    previous_end = 0.0
    for row in ordered:
        command_buffer_ms(row)
        if row['gpu_start_seconds'] < previous_end:
            raise ValueError('Metal command timestamps contradict the serialized launch order')
        previous_end = row['gpu_end_seconds']


def validate_command_samples(record: Mapping, *, route_calls: int, sample_count: int) -> None:
    commands = record.get("command_buffers")
    if not isinstance(commands, list) or len(commands) != route_calls:
        raise ValueError("Metal raw command-buffer coverage differs")
    observed = []
    seen = set()
    for index, command in enumerate(commands):
        value = command_buffer_ms(command)
        if command["launch_index"] in seen:
            raise ValueError("Metal command-buffer launch identity is duplicated")
        seen.add(command["launch_index"])
        timed = index >= route_calls - sample_count
        if command.get("timed") is not timed:
            raise ValueError("Metal warmup/sample boundary differs")
        if timed:
            observed.append(value)
    samples = record.get("samples_ms")
    if (not isinstance(samples, list) or len(samples) != len(observed)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
                   for value, expected in zip(samples, observed))):
        raise ValueError("Metal samples differ from raw command-buffer timestamps")


def metal_profile_summary(raw: Mapping) -> dict:
    if (not isinstance(raw, Mapping) or raw.get("counter_set") != "Timestamp"
            or raw.get("counter") != "GPUTimestamp" or raw.get("sampling_boundary") != "compute_stage"
            or raw.get("units") != "raw_device_timestamp_units"
            or raw.get("sample_indices") != [0, 1] or raw.get("resolved_bytes") != 16):
        raise ValueError("Metal profile counter representation differs")
    ticks = raw.get("timestamps")
    if (not isinstance(ticks, list) or len(ticks) != 2
            or any(type(value) is not int for value in ticks) or not 0 < ticks[0] < ticks[1] < 2**64 - 1):
        raise ValueError("Metal profile timestamp samples differ")
    command_buffer_ms(raw.get("command_buffer"))
    if raw["command_buffer"].get("timed") is not False:
        raise ValueError("Metal attribution timestamps cannot be timing samples")
    return {"coverage": "compute_stage_timestamps", "timestamp_delta": ticks[1] - ticks[0],
            "units": "raw_device_timestamp_units",
            "not_collected": ["occupancy", "bandwidth", "instruction_counters"],
            "timing_use": "attribution_only"}


def load_metal_profile(payload: bytes, *, expected_candidate_sha256: str, expected_case_id: str, expected_protocol_sha256: str | None = None) -> dict:
    document = json.loads(payload)
    if (not isinstance(document, dict) or document.get("kind") != METAL_PROFILE_KIND
            or document.get("candidate_sha256") != expected_candidate_sha256
            or document.get("case_id") != expected_case_id
            or not isinstance(document.get("kernel_name"), str) or not document["kernel_name"].isidentifier()
            or not isinstance(document.get("job_id"), str) or re.fullmatch(r"metal-[0-9a-f]{12}", document["job_id"]) is None
            or document["job_id"] == "metal-000000000000"
            or document.get("allocation_mode") != "local_serialized" or document.get("external_gpu_activity") != "not_excluded"
            or document.get("separate_instrumented_launch") is not True
            or document.get("archive_miss_policy") != "failOnBinaryArchiveMiss"):
        raise ValueError("Metal attribution profile identity differs")
    from .paired import paired_protocol, validation_case_ids, PAIRED_METAL_KIND
    evaluation = document.get("evaluation_protocol")
    if (paired_protocol(evaluation) is None or evaluation["paired_timing"]["kind"] != PAIRED_METAL_KIND
            or evaluation.get("case_id") != expected_case_id
            or evaluation.get("attribution_evaluation") not in {"correctness_then_profile", "correctness_then_profile_each_search_survivor"}):
        raise ValueError("Metal attribution Evaluation policy differs")
    validation_case_ids(evaluation)
    if expected_protocol_sha256 is not None and sha256(json.dumps(evaluation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != expected_protocol_sha256:
        raise ValueError("Metal attribution Evaluation identity differs")
    validate_host(document.get("host"))
    if document.get("summary") != metal_profile_summary(document.get("raw")):
        raise ValueError("Metal profile summary differs from raw timestamp samples")
    return document
