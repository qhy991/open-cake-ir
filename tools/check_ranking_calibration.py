#!/usr/bin/env python3
"""Replay one frozen finite-domain ranking calibration and emit its decision."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _ranking_key(row: dict[str, object]) -> tuple[float, int, str]:
    resident = int(row["ctas_per_multiprocessor_upper_bound"])
    capacity = resident * int(row["multiprocessor_count"])
    return (
        -int(row["ctas"]) / capacity,
        -resident,
        str(row["schedule_id"]),
    )


def _expected_ids(domain: dict[str, object]) -> list[str]:
    return [
        f"gemm-t{tile}-r{registers}-w{warps}"
        for tile, registers, warps in itertools.product(
            domain["tiles"], domain["registers_per_thread"], domain["warps"]
        )
    ]


def _read_record(root: Path, relative: str) -> tuple[dict[str, object], str]:
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or path.is_symlink():
        raise ValueError(f"measurement custody differs: {relative}")
    payload = path.read_bytes()
    return json.loads(payload), sha256(payload).hexdigest()


def evaluate(root: Path, plan_path: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("state") != "frozen":
        raise ValueError("calibration plan is not frozen schema v1")

    authority = plan["authority"]
    domain = plan["evaluation_domain"]
    protocol = plan["protocol"]
    decision = plan["decision"]
    execution = plan["execution"]
    set_size = int(decision["candidate_set_size"])
    survivor_count = int(decision["survivor_count"])
    if not 0 < survivor_count < set_size:
        raise ValueError("ranking cut must retain a non-empty proper subset")
    expected_ids = _expected_ids(domain)
    if len(expected_ids) != domain["candidate_count"] or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise ValueError("declared candidate domain differs")

    source_rows = authority["evaluation_sources"]
    for source in source_rows:
        if sha256((root / source["path"]).read_bytes()).hexdigest() != source["raw_sha256"]:
            raise ValueError(f"evaluation source differs: {source['path']}")
    schedule_authority = authority["schedule"]
    if (
        sha256((root / schedule_authority["path"]).read_bytes()).hexdigest()
        != schedule_authority["raw_sha256"]
    ):
        raise ValueError("calibration Schedule differs")
    revision_authority = authority["compiler_revision"]
    revision_id = revision_authority["revision_id"]
    revision_paths = [root / revision_authority["path"]]
    match = re.fullmatch(r"open-cake-ir-sm100a-(v\d+)", revision_id)
    if match is not None:
        # Early frozen plans named the then-current lock path. Once that exact release
        # advances into immutable history, id and digest remain the authorities and the
        # lifecycle's canonical archive is the bounded compatibility location.
        revision_paths.append(
            root / "compiler/releases" / match.group(1) / "revision.lock.json"
        )
    revision = next(
        (
            candidate
            for path in revision_paths
            if path.is_file()
            for candidate in [json.loads(path.read_text(encoding="utf-8"))]
            if candidate.get("revision_id") == revision_id
            and sha256(_canonical_bytes(candidate)).hexdigest()
            == revision_authority["canonical_sha256"]
        ),
        None,
    )
    if revision is None:
        raise ValueError("calibration Compiler Revision differs")

    repetitions = []
    observed_at: set[str] = set()
    sample_payloads: set[str] = set()
    expected_sources = {
        (item["role"], item["path"], item["raw_sha256"]) for item in source_rows
    }
    for relative in execution["measurement_records"]:
        record, raw_sha256 = _read_record(root, relative)
        schedule = record.get("schedule", {})
        target = authority["target"]
        if record.get("schema_version") != 3 or record.get("purpose") != "candidate_ranking_calibration":
            raise ValueError(f"measurement contract differs: {relative}")
        if record.get("compiler_revision") != {
            "revision_id": revision_authority["revision_id"],
            "revision_sha256": revision_authority["canonical_sha256"],
        }:
            raise ValueError(f"measurement Compiler differs: {relative}")
        if (
            schedule.get("path") != schedule_authority["path"]
            or schedule.get("profile") != schedule_authority["profile"]
            or schedule.get("raw_sha256") != schedule_authority["raw_sha256"]
            or record.get("target") != target["id"]
            or record.get("device")
            != {
                "name": target["device_name"],
                "multiprocessor_count": target["multiprocessor_count"],
            }
            or record.get("evaluation_domain") != domain
            or record.get("protocol") != protocol
        ):
            raise ValueError(f"measurement authority or domain differs: {relative}")
        observed_sources = {
            (item["role"], item["path"], item["raw_sha256"])
            for item in record.get("evaluation_sources", [])
        }
        if observed_sources != expected_sources:
            raise ValueError(f"measurement source closure differs: {relative}")

        rows = record.get("rows", [])
        excluded = record.get("excluded", [])
        members = rows + excluded
        if (
            len(members) != domain["candidate_count"]
            or {item.get("schedule_id") for item in members} != set(expected_ids)
            or len({item.get("schedule_id") for item in members}) != len(members)
            or any(
                item.get("schedule_id")
                != f"gemm-t{item.get('tile')}-r{item.get('registers_per_thread')}-w{item.get('warps')}"
                for item in members
            )
        ):
            raise ValueError(f"measurement domain coverage differs: {relative}")
        if any(item.get("disposition") != "refused_by_compiler" for item in excluded):
            raise ValueError(f"incorrect or unknown exclusion: {relative}")
        if len(rows) < set_size:
            raise ValueError(f"no pruning decision is covered: {relative}")
        if any(not row.get("ranked_by_hypothesis") for row in rows):
            raise ValueError(f"eligible row is outside the ranking domain: {relative}")
        for row in rows:
            samples = row.get("timing_samples_ms", [])
            if len(samples) != protocol["timing_samples_per_candidate"] or not math.isclose(
                statistics.median(samples),
                row["median_ms"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"timing summary differs: {relative}")

        worst_ratio = 0.0
        worst_set: list[str] = []
        survivor_set: list[str] = []
        combination_count = 0
        for candidate_set in itertools.combinations(rows, set_size):
            ordered = sorted(candidate_set, key=_ranking_key)
            best = min(float(row["median_ms"]) for row in candidate_set)
            survivor = min(
                float(row["median_ms"]) for row in ordered[:survivor_count]
            )
            ratio = survivor / best
            combination_count += 1
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_set = [str(row["schedule_id"]) for row in candidate_set]
                survivor_set = [
                    str(row["schedule_id"]) for row in ordered[:survivor_count]
                ]

        passed = worst_ratio <= float(decision["maximum_survivor_regret_ratio"])
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
                "raw_sha256": raw_sha256,
                "observed_at": record["observed_at"],
                "measured_candidate_count": len(rows),
                "compiler_refused_count": len(excluded),
                "candidate_set_count": combination_count,
                "maximum_survivor_regret_ratio": worst_ratio,
                "worst_candidate_set": worst_set,
                "retained_survivors": survivor_set,
                "passed": passed,
            }
        )

    if len(repetitions) != execution["independent_repetitions"]:
        raise ValueError("measurement repetition count differs")
    return {
        "schema_version": 1,
        "calibration_id": plan["calibration_id"],
        "plan": {
            "path": plan_path.resolve(strict=True).relative_to(root).as_posix(),
            "raw_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        },
        "repetitions": repetitions,
        "decision": {
            "maximum_survivor_regret_ratio": decision[
                "maximum_survivor_regret_ratio"
            ],
            "all_repetitions_passed": all(item["passed"] for item in repetitions),
            "promotion_scope": decision["promotion_scope"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default="contracts/calibrations/gemm-b200-ranking-m512-v6.json",
    )
    parser.add_argument("--out")
    arguments = parser.parse_args()
    plan_path = (ROOT / arguments.plan).resolve(strict=True)
    result = evaluate(ROOT, plan_path)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.out:
        output = (ROOT / arguments.out).resolve()
        if output.exists() or output.is_symlink():
            raise SystemExit(f"{output} already exists")
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["decision"]["all_repetitions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
