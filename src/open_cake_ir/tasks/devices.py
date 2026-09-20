"""Portable device registry: one backend names one Target, one route and its contracts.

A task's mathematics is device-independent, so a Workload should not be an Apple artifact
that a CUDA run has to re-derive. What actually differs between an Apple and an NVIDIA
run of the same Workload is small and enumerable, and this module is where it is written
down instead of being spread through each family's authoring code:

* the Compiler Target and the one device it admits;
* the lowering route the Schedule declares (`metal` or `triton`);
* how a run reaches that device -- `gpu_run` for the cluster allocator that hands out one
  exclusive GPU, `local_broker` for a single machine that serializes its own. This is not
  the route: a DCU lowers through Triton like a B200 and is reached like an Apple device,
  and inferring one axis from the other refused every DCU launch with "Triton execution
  requires the existing gpu-run allocator";
* the measurement source a timed assay is allowed to name. `metal` and `cupti` are
  the two that exist; `None` means this backend has no named timing source yet, and a
  Study for it is refused rather than declared against whichever source the code reaches
  by falling through. A DCU has no CUPTI, and a study that said `paired_cupti` on one
  would have labelled its evidence with a profiler that is not installed;
* instruction contracts come from each Target document and the typed IR registry;
  this task registry does not keep a second copy of their admission;
* whether the route can tile a row whose width is not a power of two. The Metal emitter
  stripes any width over 32 lanes; Triton's `tl.arange` requires a positive power-of-two
  span, so a Workload frozen at an odd width has no Triton Schedule to compare against
  and this registry refuses to create one rather than emitting a Schedule that cannot
  lower.

This is the one registry. `open_cake_ir.tasks.apple` is a read-only Metal-only view of
it for code that is genuinely Apple-specific; it holds no rows of its own, because two
tables for one fact is how five task families ended up admitting every backend while two
admitted only Metal.
"""
from __future__ import annotations

import math
from pathlib import Path

# The largest `tl.arange` span the Triton backend admits, from its own refusal text.
TRITON_MAXIMUM_TILE = 1 << 20
# `snapshotPayloadLimit` in evaluation/metal/observer.swift. The native observer charges a
# participant's whole tensor ABI for every launch it holds in one snapshot cohort, so the
# cohort's pending payload is that ABI times the cohort's route calls.
SNAPSHOT_PAYLOAD_LIMIT = 64 * 1024 * 1024

BACKENDS = {
    "metal-m1-pro": {"target": "apple_gpu_family7", "device_name": "Apple M1 Pro",
                     "provenance_token": "M1_Pro", "route": "metal",
                     "allocation": "local_broker",
                     "timing_source": "metal",
                    "power_of_two_width": False},
    "metal-m2": {"target": "apple_gpu_family8", "device_name": "Apple M2",
                 "provenance_token": "M2", "route": "metal",
                 "allocation": "local_broker",
                 "timing_source": "metal",
                    "power_of_two_width": False},
    "metal-m4": {"target": "apple_gpu_family9", "device_name": "Apple M4",
                 "provenance_token": "M4", "route": "metal",
                 "allocation": "local_broker",
                 "timing_source": "metal",
                    "power_of_two_width": False},
    "triton-b200": {"target": "sm_100a", "device_name": "NVIDIA B200",
                    "provenance_token": "B200", "route": "triton",
                    "allocation": "gpu_run",
                    "timing_source": "cupti",
                    "power_of_two_width": True},
    "triton-b300": {"target": "sm_103a", "device_name": "NVIDIA B300",
                    "provenance_token": "B300", "route": "triton",
                    "allocation": "gpu_run",
                    "timing_source": "cupti",
                    "power_of_two_width": True},
    # Hygon DCU. Instruction contracts are read from its Target document.
    "triton-dcu": {"target": "gfx938", "device_name": "BW1101",
                   "provenance_token": "BW1101", "route": "triton",
                   "allocation": "local_broker",
                   "timing_source": "hip_dispatch",
                   "power_of_two_width": True},
    # Strix Halo, an RDNA3.5 iGPU on ROCm 7.2.1. It reaches its device the way the DCU
    # does -- one visible device on one machine, serialized by the local broker -- and
    # lowers through Triton like a B200, which is why those two axes are separate rows.
    # `timing_source` was None until something measured on this device. `HipDispatchBenchmark`
    # was then run here against the gfx1151-rmsnorm-b8-smoke kernel, ROCm 7.2.1, torch
    # 2.9.1: 25 dispatches with the device reset before each, median 31.858us, min 31.217,
    # max 36.226, CV 0.038, 23 distinct values out of 25 -- a resolved cohort, not a
    # quantum. Without the reset the same cohort reads CV 0.52, which is why the assay
    # takes one. Its misattribution guard was checked too: a cohort it cannot attribute to
    # the named kernel is refused, not averaged.
    #
    # Two device-side sources disagree on this kernel by 31% -- `rocprofv3 --kernel-trace`
    # read 24.224us against this assay's 31.858us, both with the device reset. Retained,
    # with what it does and does not bound, in F-2026-09-17-002; a ranking under one source
    # is unaffected, an absolute latency is not supported, and the two are not comparable.
    "triton-gfx1151": {"target": "gfx1151", "device_name": "AMD Radeon Graphics",
                       "provenance_token": "gfx1151", "route": "triton",
                       "allocation": "local_broker",
                       "timing_source": "hip_dispatch",
                       "power_of_two_width": True},
    "triton-metax": {"target": "xcore1002", "device_name": "MetaX C550",
                     "provenance_token": "C550", "route": "triton",
                     "allocation": "local_broker",
                     "timing_source": "mcpti_dispatch",
                     "power_of_two_width": True},
}


def timing_source(backend: str) -> str | None:
    """Name the measurement source a timed assay may declare for this backend.

    `None` is a real answer and the caller must handle it: the backend exists, lowers and
    builds, and nothing has yet measured a latency on it under a named timer. Returning a
    default here would put a profiler's name on evidence that profiler never produced.
    """
    return BACKENDS[backend]["timing_source"]


def backend_for_target(target: object) -> str | None:
    """Name the single backend that admits one Compiler target, or None."""
    return next((name for name, device in BACKENDS.items() if device["target"] == target), None)


def device_name(target: object) -> str:
    """Return the exact device a task admits for one bound Target."""
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError("Workload target has no admitted device")
    return BACKENDS[backend]["device_name"]


def admit_width(backend: str, columns: int) -> None:
    """Refuse a frozen row width the backend's route provably cannot tile."""
    device = BACKENDS[backend]
    if not device["power_of_two_width"]:
        return
    if columns > TRITON_MAXIMUM_TILE or columns & (columns - 1):
        raise ValueError(
            f"{backend} tiles a row with tl.arange, which requires a positive power-of-two "
            f"span no larger than {TRITON_MAXIMUM_TILE}; {columns} is not one")


ALLOCATIONS = frozenset({"gpu_run", "local_broker"})


def allocation(backend: str) -> str:
    """How a run on this backend reaches its device."""
    value = BACKENDS[backend]["allocation"]
    if value not in ALLOCATIONS:
        raise ValueError(f"{backend} declares an unknown allocation {value!r}")
    return value


# The broker job mode each allocation admits a run under: the cluster allocator issues an
# exclusive lease, the local broker serializes one machine's device. Stated as a table so
# the Study check and the worker read the same fact, and a third allocation is a row here
# rather than an `else` in either.
_JOB_MODES = {"gpu_run": "exclusive", "local_broker": "local_serialized"}


def allocation_mode(target: object) -> str:
    """The job mode a run on this target's backend is admitted under, as the row declares."""
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError(
            f"target {target!r} is admitted by no registered backend; how it reaches its "
            "device cannot be read from its owner")
    return _JOB_MODES[allocation(backend)]


def admit_dtype(backend: str, dtype: str) -> None:
    """Refuse a Workload dtype the backend's lowering route cannot name.

    Read from the Compiler's own backend vocabulary rather than restated here, so a task
    is available on exactly the devices whose route can express its ABI. A hard-coded
    list of backend names says the same thing until a sixth device exists, and then says
    something false.
    """
    from open_cake_ir.compiler.backends import BACKENDS as _ROUTES
    from open_cake_ir.compiler.ir import DType, LoweringBackend

    route = LoweringBackend(BACKENDS[backend]["route"])
    if DType(dtype) not in _ROUTES[route].module.SUPPORTED_DTYPES:
        raise ValueError(
            f"{backend} lowers through {route.value}, which cannot name dtype {dtype!r}")


def admit_operations(backend: str, kinds: tuple[str, ...]) -> None:
    """Refuse a backend whose Target does not admit every operation kind a task needs.

    A task knows its own body; what a Target admits is a hardware commitment recorded in
    the Target document. Asking it is how a task family stays available on exactly the
    devices that can express it, instead of naming the devices that could when it was
    written.
    """
    from open_cake_ir.compiler.target import Target

    root = Path(__file__).resolve().parents[3]
    target_id = BACKENDS[backend]["target"]
    admitted = {kind.value for kind in
                Target.load(root / "compiler" / "targets" / f"{target_id}.json").operation_kinds}
    missing = [kind for kind in kinds if kind not in admitted]
    if missing:
        raise ValueError(
            f"{backend} targets {target_id}, which does not admit operation "
            f"{'kinds' if len(missing) > 1 else 'kind'} {', '.join(repr(k) for k in missing)}")


def tanh_contract(backend: str) -> str:
    """Name the device's admitted tanh instruction contract, or refuse the task.

    Each Target names this function in its own vocabulary and no Target admits another's
    spelling, which is the point of naming a contract at all. A device whose Target
    declares none is refused here by name: the alternative is emitting some other
    Target's spelling and finding out at lowering, or worse, lowering against numerics
    nobody measured on this hardware.
    """
    from open_cake_ir.compiler.ir.instruction_contracts import contract
    from open_cake_ir.compiler.ir.vocabulary import DType, ElementwiseOp
    from open_cake_ir.compiler.target import Target

    root = Path(__file__).resolve().parents[3]
    target_id = BACKENDS[backend]["target"]
    target = Target.load(root / "compiler" / "targets" / f"{target_id}.json")
    names = []
    for name in sorted(target.instruction_contracts):
        record = contract(name)
        if (record is not None and record.elementwise_op is ElementwiseOp.TANH
                and record.elementwise_dtype is DType.FP32):
            names.append(name)
    if not names:
        raise ValueError(
            f"{backend} has no admitted tanh instruction contract, so no task in this "
            "family that needs one has a Schedule on it; admitting one is a Target "
            "change with its own hardware evidence")
    if len(names) != 1:
        raise ValueError(f"{backend} admits multiple FP32 tanh contracts {names!r}; "
                         "the task needs an explicit instruction choice")
    return names[0]


def admit_cohort_payload(workload, case_id: str, route_calls_per_cohort: int) -> None:
    """Refuse a shape whose snapshot cohort cannot fit the observer's payload bound.

    The bound is enforced inside the pinned native observer, which refuses the cohort
    before a single dispatch. A task that exceeds it therefore fails deterministically at
    its first evaluation, after the campaign has already spent provider tokens authoring
    candidates for it -- which is what F-2026-09-10-002 recorded for every multi-buffer
    optimizer task at the launcher's default shape. Checking the same arithmetic here
    turns that into a refusal at launch, naming the shape that would fit.

    This does not change the bound. Charging a cohort for what correctness actually needs
    is the real fix and it lives in the observer, whose bytes are pinned by the Executor
    host environment; until that happens this keeps campaigns from paying to discover it.
    """
    if type(route_calls_per_cohort) is not int or route_calls_per_cohort <= 0:
        raise ValueError("cohort route-call count must be a positive integer")
    tensors = workload.tensor_abi(case_id)
    elements = sum(math.prod(argument.shape) for argument in tensors)
    per_launch = elements * 4
    pending = per_launch * route_calls_per_cohort
    if pending > SNAPSHOT_PAYLOAD_LIMIT:
        admissible = SNAPSHOT_PAYLOAD_LIMIT // (len(tensors) * 4 * route_calls_per_cohort)
        raise ValueError(
            f"snapshot cohort would hold {pending / 2**20:.1f} MiB against the observer's "
            f"{SNAPSHOT_PAYLOAD_LIMIT // 2**20} MiB bound: this Workload declares "
            f"{len(tensors)} FP32 tensors of {elements // len(tensors)} elements and the "
            f"protocol holds {route_calls_per_cohort} route calls per cohort. At this ABI "
            f"the largest admissible tensor is {admissible} elements")
