"""The AMDGCN attribution profile: what roctracer saw for one instrumented dispatch.

Nsight Compute is CUDA's profiler and this platform does not have it, so for a long time
`hsaco` declared no attribution at all and an authoring Turn on a DCU got a latency and
nothing to act on. That refusal was right about Nsight and wrong as a permanent answer:
the device does have an in-process activity API, the same roctracer that
`HipDispatchBenchmark` already reads, and this module says exactly what it can and cannot
see.

**What it collects.** One separate instrumented launch, outside the timed cohorts, and the
per-dispatch device time roctracer attributes to the candidate's own kernel by name --
plus a count of any dispatch it did not name, so a candidate that quietly did part of its
work in a second kernel is visible rather than invisible.

**What it does not collect, and says so.** Occupancy, achieved bandwidth and instruction
counters. Nsight reports those for a cubin and this does not, so `not_collected` names
them in the record itself. An omitted field reads as "does not constrain" and means "was
not looked at", and a reader will assume the generous one; the Metal profile states its
own gap the same way and for the same reason.

**What the number is not.** The instrumented dispatch is attribution, never a latency. It
runs once, without the device-state reset the paired assay applies before every timed
sample, so it is not comparable to a cohort median and `timing_use` says so in the record.
"""

from __future__ import annotations

from typing import Mapping

from open_cake_ir.compiler.target import CodeObject

from .attribution import load_instrumented_profile
from .platforms import PLATFORMS

HIP_PROFILE_KIND = "hip_dispatch_activity_v1"

# Nsight's three, named rather than omitted. The projection republishes this list to the
# authoring Turn so the absence is part of the feedback and not an inference from silence.
NOT_COLLECTED = ("occupancy", "bandwidth", "instruction_counters")


def collect_hip_dispatch_activity(launch, kernel_name: str) -> dict:
    """Run one launch under roctracer and return what it attributed, unsummarised.

    `launch` is the callable that issues exactly one dispatch, the same one the cohorts
    drive, so the profiled launch is the candidate's own and not a re-creation of it.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    if not isinstance(kernel_name, str) or not kernel_name:
        raise ValueError("a dispatch profile attributes one named kernel")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as session:
        launch()
        torch.cuda.synchronize()
    device_events = [event for event in session.events()
                     if getattr(event, "device_time", 0.0)]
    named = [event for event in device_events
             if kernel_name in getattr(event, "name", "")]
    if len(named) != 1:
        raise ValueError(
            f"the profiler attributed {len(named)} dispatches of {kernel_name!r} to one "
            "instrumented launch")
    return {
        "source": "roctracer_device_activity",
        "instrumented_dispatches": 1,
        "device_time_us": float(getattr(named[0], "device_time", 0.0)),
        "non_target_dispatches": len(device_events) - len(named),
        "not_collected": list(NOT_COLLECTED),
    }


def hip_profile_summary(raw: Mapping) -> dict:
    """Project the raw activity, refusing a representation this did not produce."""

    if (not isinstance(raw, Mapping) or raw.get("source") != "roctracer_device_activity"
            or raw.get("instrumented_dispatches") != 1
            or list(raw.get("not_collected") or ()) != list(NOT_COLLECTED)):
        raise ValueError("HIP profile activity representation differs")
    device_time = raw.get("device_time_us")
    if type(device_time) is not float or not 0.0 < device_time < 1e9:
        raise ValueError("HIP profile device time differs")
    non_target = raw.get("non_target_dispatches")
    if type(non_target) is not int or non_target < 0:
        raise ValueError("HIP profile non-target dispatch count differs")
    return {
        "coverage": "per_dispatch_device_time",
        "device_time_us": device_time,
        "non_target_dispatches": non_target,
        "not_collected": list(NOT_COLLECTED),
        # The same sentence the Metal profile carries, for the same reason: one
        # uninstrumented-state dispatch is evidence about where time went, never a latency.
        "timing_use": "attribution_only",
    }


def load_hip_profile(payload: bytes, *, expected_candidate_sha256: str, expected_case_id: str,
                     expected_protocol_sha256: str | None = None) -> dict:
    """Read a retained HIP profile, checking it is this candidate's and this case's."""

    return load_instrumented_profile(
        payload, kind=HIP_PROFILE_KIND, job_prefix=PLATFORMS[CodeObject.HSACO].local_job_prefix,
        label="HIP", summary=hip_profile_summary, raw_name="activity",
        expected_candidate_sha256=expected_candidate_sha256, expected_case_id=expected_case_id,
        expected_protocol_sha256=expected_protocol_sha256)


def hip_attribution_feedback(profile: Mapping[str, object]) -> Mapping[str, object]:
    """Project only bounded actionable facts into the next authoring Turn."""

    summary = profile.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("HIP profile summary differs")
    return {
        "kind": "hip_dispatch_attribution",
        "kernel_name": profile["kernel_name"],
        **{key: summary[key] for key in (
            "coverage", "device_time_us", "non_target_dispatches", "not_collected",
            "timing_use")},
    }
