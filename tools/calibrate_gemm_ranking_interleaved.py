#!/usr/bin/env python3
"""Measure or replay the drift-controlled GEMM 3-to-2 ranking calibration.

The frozen plan owns the domain, protocol, source closure and decision. ``measure`` builds
and correctness-checks the complete domain before timing, then visits every eligible
candidate once per round in cyclic rotations. ``check`` is GPU-free and never writes or
regenerates a measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import socket
import statistics
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_cases import ORACLES, build_inputs  # noqa: E402
from open_cake_ir.compiler.analysis import residency_upper_bound  # noqa: E402
from open_cake_ir.compiler.core import Compiler  # noqa: E402
from open_cake_ir.compiler.ir import Schedule  # noqa: E402
from open_cake_ir.compiler.ranking import (  # noqa: E402
    Cost,
    cost as ranking_cost,
    rank_for_cut,
)
from open_cake_ir.compiler.target import Target  # noqa: E402


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _resolve_owned(relative: str) -> Path:
    path = (ROOT / relative).resolve(strict=True)
    if ROOT not in path.parents or path.is_symlink():
        raise ValueError(f"repository custody differs: {relative}")
    return path


def _load_plan(plan_path: Path) -> dict[str, object]:
    plan = _read_json(plan_path)
    if plan.get("schema_version") != 1 or plan.get("state") != "frozen":
        raise ValueError("calibration plan is not frozen schema v1")
    sources = plan.get("authority", {}).get("evaluation_sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("calibration source closure is empty")
    for source in sources:
        path = _resolve_owned(str(source["path"]))
        if sha256(path.read_bytes()).hexdigest() != source["raw_sha256"]:
            raise ValueError(f"evaluation source differs: {source['path']}")
    return plan


def _gemm_variant(
    base: dict[str, object], size: int, tile: int, registers: int, warps: int
) -> dict[str, object]:
    document = json.loads(json.dumps(base))
    shapes = {buffer["name"]: buffer for buffer in document["buffers"]}
    k = shapes["a_tile"]["shape"][1]
    shapes["a"]["shape"][0] = size
    shapes["c"]["shape"][0] = size
    shapes["a_tile"]["shape"] = [tile, k]
    shapes["b_tile"]["shape"] = [tile, k]
    for name in ("acc", "c_tile"):
        shapes[name]["shape"] = [tile, tile]
    shapes["bias_tile"]["shape"] = [tile]
    for axis in document["program_map"]["axes"]:
        axis["tile"] = tile
    for operation in document["operations"]:
        if operation["kind"] == "mma":
            operation["parameters"]["tile_shape"] = [tile, tile, k]
    document["roles"][0]["warps"] = list(range(warps))
    document["residency"] = {
        "ctas_per_multiprocessor": 1,
        "registers_per_thread": registers,
    }
    document["schedule_id"] = f"gemm-t{tile}-r{registers}-w{warps}"
    return document


def _expected_ids(domain: dict[str, object]) -> list[str]:
    return [
        f"gemm-t{tile}-r{registers}-w{warps}"
        for tile, registers, warps in itertools.product(
            domain["tiles"], domain["registers_per_thread"], domain["warps"]
        )
    ]


def _revision_document(authority: dict[str, object]) -> dict[str, object] | None:
    revision_id = str(authority["revision_id"])
    candidates = [_resolve_owned(str(authority["path"]))]
    match = re.fullmatch(r"open-cake-ir-(?:sm100a-)?(v[1-9]\d*)", revision_id)
    if match is not None:
        archive = ROOT / "compiler" / "releases" / match.group(1) / "revision.lock.json"
        if archive.is_file():
            candidates.append(archive.resolve(strict=True))
    for path in candidates:
        document = _read_json(path)
        if (
            document.get("revision_id") == revision_id
            and sha256(_canonical_bytes(document)).hexdigest()
            == authority["canonical_sha256"]
        ):
            return document
    return None


def _measure(plan_path: Path, observed_at: str, output: Path) -> int:
    if output.exists() or output.is_symlink():
        raise ValueError(f"{output} already exists; a measurement gets a new file")
    if not output.parent.resolve(strict=True).is_dir():
        raise ValueError("measurement output parent differs")
    plan = _load_plan(plan_path)
    authority = plan["authority"]
    domain = plan["evaluation_domain"]
    protocol = plan["protocol"]
    schedule_authority = authority["schedule"]
    revision_authority = authority["compiler_revision"]
    if _revision_document(revision_authority) is None:
        raise ValueError("calibration Compiler Revision differs")
    if domain["extent"].get("symbol") != "M":
        raise ValueError("this instrument requires a GEMM M domain")

    import torch

    schedule_path = _resolve_owned(str(schedule_authority["path"]))
    schedule_bytes = schedule_path.read_bytes()
    if sha256(schedule_bytes).hexdigest() != schedule_authority["raw_sha256"]:
        raise ValueError("calibration Schedule differs")
    base = json.loads(schedule_bytes)
    if base.get("metadata", {}).get("profile") != "gemm_bias_b1_smoke":
        raise ValueError("this instrument requires the GEMM profile")
    compiler = Compiler.load(ROOT, str(revision_authority["path"]))
    target = Target.from_dict(
        _read_json(_resolve_owned(f"compiler/targets/{base['target']}.json"))
    )
    oracle = ORACLES["gemm_bias_b1_smoke"]
    workspace = Path(tempfile.mkdtemp(prefix="cake-ranking-interleaved-"))
    flush = torch.empty(
        int(protocol["l2_flush_bytes_per_sample"]), dtype=torch.uint8, device="cuda"
    )

    rows: list[dict[str, object]] = []
    launches: list[object] = []
    excluded: list[dict[str, object]] = []
    inputs = None
    expected = None
    revision = None
    extent = int(domain["extent"]["value"])
    for tile, registers, warps in itertools.product(
        domain["tiles"], domain["registers_per_thread"], domain["warps"]
    ):
        document = _gemm_variant(base, extent, tile, registers, warps)
        assessment = compiler.assess(document)
        identity = {
            "schedule_id": document["schedule_id"],
            "tile": tile,
            "registers_per_thread": registers,
            "warps": warps,
        }
        if not assessment.lowering_eligible:
            excluded.append(
                {
                    **identity,
                    "disposition": "refused_by_compiler",
                    "finding_codes": [item.code for item in assessment.findings],
                }
            )
            continue
        revision = {
            "revision_id": assessment.compiler_revision_id,
            "revision_sha256": assessment.compiler_revision_sha256,
        }
        typed = Schedule.from_dict(json.loads(assessment.schedule_bytes))
        bound = residency_upper_bound(typed, target)
        hypothesis = ranking_cost(typed, target)
        if bound is None or bound.binding is None:
            raise RuntimeError("eligible candidate has no residency analysis")
        lowering = compiler.lower(assessment)
        module_path = workspace / f"{document['schedule_id'].replace('-', '_')}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        specification = importlib.util.spec_from_file_location(module_path.stem, module_path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        launch = getattr(module, lowering.entry_point)
        if inputs is None:
            torch.manual_seed(int(protocol["input_seed"]))
            inputs = build_inputs(document, torch)
            expected, _ = oracle(inputs, torch)
        observed = launch(*inputs)
        torch.cuda.synchronize()
        deviation = (observed - expected).abs().max().item()
        if deviation > float(protocol["correctness_tolerance"]):
            excluded.append(
                {**identity, "disposition": "incorrect", "max_deviation": deviation}
            )
            continue
        grid_x, grid_y, grid_z = assessment.analysis["grid"]
        rows.append(
            {
                **identity,
                "ctas": grid_x * grid_y * grid_z,
                "ctas_per_multiprocessor_upper_bound": bound.ctas_per_multiprocessor,
                "binding_resource": bound.binding.resource,
                "multiprocessor_count": target.occupancy.multiprocessor_count,
                "ranked_by_hypothesis": hypothesis is not None,
                "max_deviation": deviation,
                "timing_samples_ms": [],
            }
        )
        launches.append(launch)

    if revision is None or inputs is None:
        raise RuntimeError("the domain produced no lowering-eligible candidate")
    expected_ids = _expected_ids(domain)
    if (
        len(expected_ids) != domain["candidate_count"]
        or len(rows) + len(excluded) != len(expected_ids)
    ):
        raise RuntimeError("calibration domain coverage differs")

    def rotation(round_index: int) -> list[int]:
        start = round_index % len(rows)
        return [*range(start, len(rows)), *range(0, start)]

    for round_index in range(int(protocol["warmup_rounds"])):
        for index in rotation(round_index):
            launches[index](*inputs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for round_index in range(int(protocol["timing_rounds"])):
        for index in rotation(round_index):
            flush.zero_()
            torch.cuda.synchronize()
            start_event.record()
            launches[index](*inputs)
            end_event.record()
            end_event.synchronize()
            rows[index]["timing_samples_ms"].append(start_event.elapsed_time(end_event))
    for row in rows:
        row["median_ms"] = statistics.median(row["timing_samples_ms"])

    device = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": 4,
        "observed_at": observed_at,
        "purpose": "candidate_ranking_calibration",
        "host": socket.gethostname(),
        "evaluation_sources": authority["evaluation_sources"],
        "schedule": {
            "path": schedule_authority["path"],
            "schedule_id": base["schedule_id"],
            "profile": "gemm_bias_b1_smoke",
            "raw_sha256": sha256(schedule_bytes).hexdigest(),
        },
        "target": base["target"],
        "compiler_revision": revision,
        "device": {
            "name": device.name,
            "multiprocessor_count": device.multi_processor_count,
        },
        "evaluation_domain": domain,
        "protocol": protocol,
        "excluded": excluded,
        "rows": rows,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {output}: {len(rows)} measured, {len(excluded)} excluded")
    return 0


def _read_record(relative: str) -> tuple[dict[str, object], str]:
    path = _resolve_owned(relative)
    payload = path.read_bytes()
    return json.loads(payload), sha256(payload).hexdigest()


def _check(plan_path: Path) -> dict[str, object]:
    plan = _load_plan(plan_path)
    authority = plan["authority"]
    domain = plan["evaluation_domain"]
    protocol = plan["protocol"]
    execution = plan["execution"]
    decision = plan["decision"]
    if _revision_document(authority["compiler_revision"]) is None:
        raise ValueError("calibration Compiler Revision differs")
    schedule = authority["schedule"]
    if sha256(_resolve_owned(schedule["path"]).read_bytes()).hexdigest() != schedule["raw_sha256"]:
        raise ValueError("calibration Schedule differs")
    expected_ids = _expected_ids(domain)
    if len(expected_ids) != domain["candidate_count"] or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("declared candidate domain differs")

    repetitions = []
    observed_at: set[str] = set()
    sample_payloads: set[str] = set()
    expected_sources = {
        (item["role"], item["path"], item["raw_sha256"])
        for item in authority["evaluation_sources"]
    }
    for relative in execution["measurement_records"]:
        record, raw_digest = _read_record(relative)
        observed_sources = {
            (item["role"], item["path"], item["raw_sha256"])
            for item in record.get("evaluation_sources", [])
        }
        if (
            record.get("schema_version") != 4
            or record.get("purpose") != "candidate_ranking_calibration"
            or observed_sources != expected_sources
            or record.get("schedule")
            != {
                "path": schedule["path"],
                "schedule_id": schedule["schedule_id"],
                "profile": schedule["profile"],
                "raw_sha256": schedule["raw_sha256"],
            }
            or record.get("target") != authority["target"]["id"]
            or record.get("device")
            != {
                "name": authority["target"]["device_name"],
                "multiprocessor_count": authority["target"]["multiprocessor_count"],
            }
            or record.get("evaluation_domain") != domain
            or record.get("protocol") != protocol
            or record.get("compiler_revision")
            != {
                "revision_id": authority["compiler_revision"]["revision_id"],
                "revision_sha256": authority["compiler_revision"]["canonical_sha256"],
            }
        ):
            raise ValueError(f"measurement authority or protocol differs: {relative}")
        rows = record.get("rows", [])
        excluded = record.get("excluded", [])
        members = rows + excluded
        if (
            len(members) != len(expected_ids)
            or {item.get("schedule_id") for item in members} != set(expected_ids)
            or len({item.get("schedule_id") for item in members}) != len(members)
        ):
            raise ValueError(f"measurement domain coverage differs: {relative}")
        if any(item.get("disposition") != "refused_by_compiler" for item in excluded):
            raise ValueError(f"incorrect or unknown exclusion: {relative}")
        if any(not row.get("ranked_by_hypothesis") for row in rows):
            raise ValueError(f"eligible row is outside the ranking domain: {relative}")
        for row in rows:
            samples = row.get("timing_samples_ms", [])
            if (
                len(samples) != protocol["timing_rounds"]
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in samples)
                or not math.isclose(
                    statistics.median(samples), row["median_ms"], rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ValueError(f"timing samples differ: {relative}")

        decisive = 0
        abstained = 0
        worst_ratio = 0.0
        worst_set: list[str] = []
        retained: list[str] = []
        for candidate_set in itertools.combinations(rows, int(decision["candidate_set_size"])):
            costs = [
                Cost(
                    schedule_id=str(row["schedule_id"]),
                    ctas=int(row["ctas"]),
                    ctas_per_multiprocessor=int(row["ctas_per_multiprocessor_upper_bound"]),
                    binding_resource=str(row["binding_resource"]),
                    device_fill=int(row["ctas"])
                    / (
                        int(row["ctas_per_multiprocessor_upper_bound"])
                        * int(row["multiprocessor_count"])
                    ),
                )
                for row in candidate_set
            ]
            ordered_costs = rank_for_cut(costs, int(decision["survivor_count"]))
            if ordered_costs is None:
                abstained += 1
                continue
            decisive += 1
            row_by_id = {str(row["schedule_id"]): row for row in candidate_set}
            survivor = min(
                float(row_by_id[item.schedule_id]["median_ms"])
                for item in ordered_costs[: int(decision["survivor_count"])]
            )
            best = min(float(row["median_ms"]) for row in candidate_set)
            ratio = survivor / best
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_set = [str(row["schedule_id"]) for row in candidate_set]
                retained = [
                    item.schedule_id
                    for item in ordered_costs[: int(decision["survivor_count"])]
                ]
        if decisive == 0:
            raise ValueError(f"calibration has no decisive subset: {relative}")
        sample_digest = sha256(
            _canonical_bytes([row["timing_samples_ms"] for row in rows])
        ).hexdigest()
        if record["observed_at"] in observed_at or sample_digest in sample_payloads:
            raise ValueError("measurement repetitions are not distinct")
        observed_at.add(record["observed_at"])
        sample_payloads.add(sample_digest)
        repetitions.append(
            {
                "path": relative,
                "raw_sha256": raw_digest,
                "observed_at": record["observed_at"],
                "measured_candidate_count": len(rows),
                "compiler_refused_count": len(excluded),
                "decisive_candidate_set_count": decisive,
                "abstained_candidate_set_count": abstained,
                "maximum_decisive_survivor_regret_ratio": worst_ratio,
                "worst_candidate_set": worst_set,
                "retained_survivors": retained,
                "passed": worst_ratio
                <= float(decision["maximum_survivor_regret_ratio"]),
            }
        )
    if len(repetitions) != execution["independent_repetitions"]:
        raise ValueError("measurement repetition count differs")
    return {
        "schema_version": 1,
        "calibration_id": plan["calibration_id"],
        "plan": {
            "path": plan_path.resolve(strict=True).relative_to(ROOT).as_posix(),
            "raw_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        },
        "repetitions": repetitions,
        "decision": {
            "maximum_survivor_regret_ratio": decision["maximum_survivor_regret_ratio"],
            "all_repetitions_passed": all(item["passed"] for item in repetitions),
            "promotion_scope": decision["promotion_scope"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--plan", required=True)
    measure.add_argument("--observed-at", required=True)
    measure.add_argument("--out", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--plan", required=True)
    check.add_argument("--out")
    arguments = parser.parse_args()
    plan_path = _resolve_owned(arguments.plan)
    try:
        if arguments.operation == "measure":
            return _measure(plan_path, arguments.observed_at, Path(arguments.out).resolve())
        result = _check(plan_path)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if arguments.out:
            output = Path(arguments.out).resolve()
            if output.exists() or output.is_symlink():
                raise ValueError(f"{output} already exists")
            output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0 if result["decision"]["all_repetitions_passed"] else 1
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
