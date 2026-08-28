#!/usr/bin/env python3
"""Measure the peak rates a Target may declare, so no utilisation quotes an unsourced one.

`Target.peak` is what turns a measured time into an MFU or a BWU. Every other fact on a
Target was read from the device or cited to a document, and a peak copied off a
specification slide would be the one number in the Compiler that nobody could reproduce --
in a repository whose entire argument is that its claims can be repeated.

So this measures them, and it is explicit about what it measured:

* **Bandwidth** is a triad over arrays far larger than L2: two reads and one write per
  element, timed with CUDA events. That is achieved bandwidth, not the memory system's
  theoretical product of clock and bus width, and it is the more useful ceiling of the
  two -- no kernel reaches the theoretical one, so a BWU against it understates every
  kernel by the same unmeasured factor.

* **Arithmetic** is a large square matmul in the contract's operand dtype, through the
  vendor library rather than through a generated kernel. cuBLAS is the strongest
  contraction available on this device, so its rate is an upper bound on what a Schedule
  issuing that contract can reach, and a utilisation against it therefore stays at or
  below one for the right reason. Measuring it with our own emitter would make the peak a
  function of the thing being evaluated.

Both are recorded as `source: microbenchmark`, which is a claim about the best rate this
repository has observed rather than about the architecture. The distinction is in
`PeakSource`, and it decides what exceeding one means: against a microbenchmark it means
a better kernel exists, and against a specification it means something is wrong.

A contract with no probe here is left out of the block rather than given a neighbouring
dtype's rate. `tcgen05` and the block-scaled FP8 contraction have no vendor-library
equivalent that is unambiguously the same instruction, so a Schedule issuing one gets no
arithmetic utilisation until someone measures it. That is the honest state and it is
visible in the emitted block.

Run under the broker in exclusive mode: this is a timing measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.target import Peak, Target, TargetParseError  # noqa: E402
from open_cake_ir.evaluation.timing import summarize_cohort  # noqa: E402

# The Study protocol's measurement-quality rule, reused rather than restated: a cohort
# whose coefficient of variation exceeds this is not a measurement anything may be
# divided by, and a peak taken from an unstable cohort would be quoted forever.
MAXIMUM_CV = 0.05
SAMPLES = 25
WARMUP = 5

# Which vendor-library operation is admitted as an upper bound for which declared
# contract. Absence is deliberate: a contract without a row here yields no rate.
BF16_DOT = "triton.dot.bf16_fp32"
FP32_DOT = "triton.dot.fp32_ieee"


def derive_rate(quantity: float, samples_ms: list[float]) -> tuple[float, dict]:
    """Rate implied by one cohort, refusing a cohort too unstable to quote.

    The median is the declared rate rather than the fastest sample. A minimum is the
    luckiest launch of the cohort and moves every time the instrument runs; a median
    under a coefficient-of-variation gate is reproducible, which is the property a
    number written into a content-bound Target has to have. The raw samples are retained
    either way, so a reader who wants a different convention can derive it.
    """

    summary = summarize_cohort(samples_ms)
    if summary["cv"] > MAXIMUM_CV:
        raise SystemExit(
            f"cohort coefficient of variation {summary['cv']:.4f} exceeds {MAXIMUM_CV}"
        )
    return quantity / (float(summary["median_ms"]) / 1000.0), summary


def build_peak_block(
    *,
    observed_at: str,
    bandwidth_bytes_per_second: float | None,
    arithmetic_flops_per_second: Mapping[str, float],
) -> dict:
    """Assemble a `peak` block in the exact shape `Target.from_dict` admits."""

    block: dict[str, object] = {}
    if bandwidth_bytes_per_second is not None:
        block["memory_bandwidth"] = {
            "bytes_per_second": bandwidth_bytes_per_second,
            "source": "microbenchmark",
            "observed_at": observed_at,
        }
    if arithmetic_flops_per_second:
        block["arithmetic"] = {
            contract: {
                "flops_per_second": value,
                "source": "microbenchmark",
                "observed_at": observed_at,
            }
            for contract, value in sorted(arithmetic_flops_per_second.items())
        }
    return block


def _cohort(callable_under_test, torch) -> list[float]:
    """One warmed cohort of event-timed samples, in milliseconds."""

    for _ in range(WARMUP):
        callable_under_test()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_under_test()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="compiler/targets/sm_100a.json")
    parser.add_argument("--out")
    parser.add_argument("--observed-at")
    parser.add_argument(
        "--stream-elements",
        type=int,
        default=1 << 27,
        help="fp32 elements per triad array; the default moves 1.5GiB per sample",
    )
    parser.add_argument("--matmul-size", type=int, default=8192)
    arguments = parser.parse_args()

    kernelinfra_result = os.environ.get("KERNELINFRA_RESULT")
    if kernelinfra_result:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if (
            os.environ.get("KERNELINFRA_STAGE_KIND") != "profile"
            or not os.environ.get("KERNELINFRA_RUN_ID")
            or not visible
            or "," in visible
            or not os.environ.get("KERNELINFRA_STAGE_DIR")
        ):
            raise SystemExit("broker-visible target peak environment differs")
        out = Path(os.environ["KERNELINFRA_STAGE_DIR"]).resolve(strict=True) / "target-peak.json"
    elif arguments.out:
        out = Path(arguments.out)
    else:
        parser.error("--out is required outside a GPU Infra stage")
    if out.exists():
        raise SystemExit(f"{out} already exists; a new measurement gets a new file")

    target = Target.load(ROOT / arguments.target)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; a peak is measured, not assumed")
    device_name = torch.cuda.get_device_name(0)
    if device_name not in target.device_names:
        # A rate measured on another device pasted into this Target would be an invented
        # number with a real measurement's provenance, which is worse than no rate.
        raise SystemExit(
            f"device {device_name!r} is not one this Target declares: "
            + ", ".join(target.device_names)
        )

    observed_at = arguments.observed_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    measurements: dict[str, object] = {}

    elements = arguments.stream_elements
    a = torch.empty(elements, dtype=torch.float32, device="cuda").uniform_(-1, 1)
    b = torch.empty(elements, dtype=torch.float32, device="cuda").uniform_(-1, 1)
    c = torch.empty_like(a)
    moved = 3 * elements * 4  # two reads and one write of fp32
    samples = _cohort(lambda: torch.add(a, b, out=c), torch)
    bandwidth, bandwidth_summary = derive_rate(float(moved), samples)
    measurements["memory_bandwidth"] = {
        "probe": "fp32 triad c = a + b",
        "elements": elements,
        "bytes_per_sample": moved,
        "samples_ms": samples,
        "summary": bandwidth_summary,
        "bytes_per_second": bandwidth,
    }
    del a, b, c
    torch.cuda.empty_cache()

    arithmetic: dict[str, float] = {}
    if BF16_DOT in target.instruction_contracts:
        size = arguments.matmul_size
        left = torch.empty((size, size), dtype=torch.bfloat16, device="cuda").normal_()
        right = torch.empty((size, size), dtype=torch.bfloat16, device="cuda").normal_()
        flops = 2.0 * size * size * size
        samples = _cohort(lambda: torch.matmul(left, right), torch)
        rate, summary = derive_rate(flops, samples)
        arithmetic[BF16_DOT] = rate
        measurements[BF16_DOT] = {
            "probe": f"cuBLAS bf16 matmul, {size}x{size}x{size}",
            "flops_per_sample": flops,
            "samples_ms": samples,
            "summary": summary,
            "flops_per_second": rate,
        }
        del left, right
        torch.cuda.empty_cache()

    if FP32_DOT in target.instruction_contracts:
        size = arguments.matmul_size
        left = torch.empty((size, size), dtype=torch.float32, device="cuda").normal_()
        right = torch.empty((size, size), dtype=torch.float32, device="cuda").normal_()
        flops = 2.0 * size * size * size
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            samples = _cohort(lambda: torch.matmul(left, right), torch)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        rate, summary = derive_rate(flops, samples)
        arithmetic[FP32_DOT] = rate
        measurements[FP32_DOT] = {
            "probe": f"cuBLAS IEEE fp32 matmul, {size}x{size}x{size}, TF32 disabled",
            "flops_per_sample": flops,
            "samples_ms": samples,
            "summary": summary,
            "flops_per_second": rate,
        }
        del left, right
        torch.cuda.empty_cache()

    unmeasured = sorted(
        contract
        for contract in target.instruction_contracts
        if contract not in arithmetic
    )
    block = build_peak_block(
        observed_at=observed_at,
        bandwidth_bytes_per_second=bandwidth,
        arithmetic_flops_per_second=arithmetic,
    )
    # Built through the parser, so a block this instrument emits is a block the Target
    # admits. Discovering otherwise while editing a released Target is the failure this
    # prevents.
    try:
        Peak.from_dict(block, target.instruction_contracts, "target.peak")
    except TargetParseError as error:
        raise SystemExit(f"the emitted peak block is not admissible: {error}") from error

    record = {
        "schema_version": 1,
        "observed_at": observed_at,
        "purpose": "target_peak_rates",
        "host": socket.gethostname(),
        "device": device_name,
        "target": {"target_id": target.target_id, "path": arguments.target},
        "protocol": {
            "samples_per_cohort": SAMPLES,
            "warmup": WARMUP,
            "maximum_cv": MAXIMUM_CV,
            "declared_rate": "cohort median",
        },
        "measurements": measurements,
        "unmeasured_contracts": unmeasured,
        "peak": block,
    }
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"peak": block, "unmeasured_contracts": unmeasured}, indent=1))
    print(f"wrote {out}")
    print(
        "\nThis block is not installed. Adding it to the Target changes the Target's "
        "bytes,\nwhich the Compiler Revision is content bound to -- it belongs in a "
        "release, not an edit."
    )
    if kernelinfra_result:
        result_path = Path(kernelinfra_result).absolute()
        if result_path.exists() or result_path.is_symlink():
            raise SystemExit("refusing to overwrite GPU Infra stage result")
        result_path.write_text(
            json.dumps(
                {
                    "schema": "kernelinfra.stage-result.v1",
                    "status": "passed",
                    "validity": "valid",
                    "summary": "B200 target peak cohorts passed the stability gate",
                    "workloads": [],
                    "artifacts": {"target_peak": "target-peak.json"},
                    "metrics": {
                        "peak": block,
                        "unmeasured_contracts": unmeasured,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
