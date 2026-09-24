#!/usr/bin/env python3
"""Offline admission probes for Weave-style spatial, temporal and steal schedules.

Run from a clean, pinned checkout. This generates source but does not compile a
GPU binary, allocate a device, or measure latency.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler


def schedule(name: str) -> dict:
    return json.loads((ROOT / "corpus" / "schedules" / name).read_text())


def check(compiler: Compiler, label: str, document: dict, *, eligible: bool,
          codes: tuple[str, ...] = (), source_fragments: tuple[str, ...] = ()) -> dict:
    assessment = compiler.assess(document)
    found = sorted({finding.code for finding in assessment.findings})
    if assessment.lowering_eligible is not eligible or any(code not in found for code in codes):
        raise AssertionError(f"{label}: eligibility={assessment.lowering_eligible}, findings={found}")
    result: dict = {"eligible": assessment.lowering_eligible, "findings": found}
    if eligible:
        source = compiler.lower(assessment).source
        missing = [fragment for fragment in source_fragments if fragment not in source]
        if missing:
            raise AssertionError(f"{label}: generated source lacks {missing}")
        result["source_fragments"] = list(source_fragments)
        constants = dict(re.findall(r"^\s+(TOTAL_TILES|NUM_CTAS)=(\d+),$", source, re.MULTILINE))
        if constants:
            result["persistent_constants"] = {key: int(value) for key, value in constants.items()}
    return result


def claimed_work() -> dict:
    """Make the returned atomic index drive a visible load and output store."""
    document = schedule("atomic-reservation-b8-smoke.json")
    document["program_map"]["persistent"] = True
    document["residency"]["ctas_per_multiprocessor"] = 1
    # More logical work tiles than resident CTAs, including the all-to-one-expert
    # case where one counter must cover every returned old value.
    for buffer in document["buffers"]:
        if buffer["name"] in {"expert_ids", "positions"}:
            buffer["shape"][0] = 256
    document["buffers"].insert(1, {
        "name": "work_tiles", "space": "global", "dtype": "int32",
        "shape": [4, 2048], "mode": "input",
    })
    document["buffers"].append({
        "name": "claimed_tiles", "space": "register", "dtype": "int32",
        "shape": [8], "mode": "scratch",
    })
    document["operations"].insert(2, {
        "id": "load_claimed", "kind": "load", "role": "compute",
        "reads": ["work_tiles", "expert_id_tile", "position_tile"],
        "writes": ["claimed_tiles"], "depends_on": ["reserve_positions"],
        "parameters": {"movement": "global", "reuse": "streamed"},
    })
    store = next(op for op in document["operations"] if op["id"] == "store_positions")
    store["reads"] = ["claimed_tiles"]
    store["depends_on"] = ["load_claimed"]
    document["access_maps"].append({
        "operation": "load_claimed", "buffer": "work_tiles",
        "indices": [
            {"source": "buffer", "name": "expert_id_tile"},
            {"source": "buffer", "name": "position_tile"},
        ],
        "boundary": "mask_tiled_axes",
    })
    return document


def main() -> int:
    compiler = Compiler.load(ROOT)
    if compiler.commit is None:
        raise RuntimeError("probe requires a clean checkout at an exact commit")
    result: dict = {"source_commit": compiler.commit, "device_test": "not_run"}

    persistent = schedule("rmsnorm-b128-persistent.json")
    result["static_persistent"] = check(
        compiler, "static_persistent", persistent, eligible=True,
        source_fragments=("for _work in tl.range(tl.program_id(0), TOTAL_TILES, NUM_CTAS):",),
    )
    constants = result["static_persistent"]["persistent_constants"]
    if constants["TOTAL_TILES"] <= constants["NUM_CTAS"]:
        raise AssertionError("positive persistent control did not traverse more tiles than CTAs")

    split = deepcopy(persistent)
    split["roles"] = [
        {"name": "compute", "execution_groups": [0, 1]},
        {"name": "comm", "execution_groups": [2, 3]},
    ]
    result["two_cta_roles"] = check(
        compiler, "two_cta_roles", split, eligible=False, codes=("TRITON_ROLE_COUNT",),
    )
    cross_role = deepcopy(split)
    next(op for op in cross_role["operations"] if op["id"] == "load_gamma")["role"] = "comm"
    result["cross_role_dependency"] = check(
        compiler, "cross_role_dependency", cross_role, eligible=False,
        codes=("OP_CROSS_ROLE_RACE",),
    )
    with_barrier = deepcopy(cross_role)
    with_barrier["barriers"] = [{
        "name": "ready", "count": 1, "producers": ["comm"],
        "consumers": ["compute"], "mechanism": "barrier.sync",
    }]
    next(op for op in with_barrier["operations"] if op["id"] == "load_gamma")["signals"] = ["ready"]
    next(op for op in with_barrier["operations"] if op["id"] == "weight")["waits"] = ["ready"]
    result["cross_role_barrier"] = check(
        compiler, "cross_role_barrier", with_barrier, eligible=False,
        codes=("TRITON_BARRIER_UNSUPPORTED", "TRITON_ROLE_COUNT"),
    )

    loop = schedule("top-k-streaming-merge2-exact-half-k256-b8-smoke.json")
    result["static_chunk_loop"] = check(compiler, "static_chunk_loop", loop, eligible=True)
    route_stop = deepcopy(loop)
    route_stop["tile_loops"][0]["stop"]["program"] = "routing_count"
    result["routing_driven_chunk_loop"] = check(
        compiler, "routing_driven_chunk_loop", route_stop, eligible=False,
        codes=("LOOP_STOP_PROGRAM_UNKNOWN",),
    )

    work = claimed_work()
    result["persistent_atomic_claim"] = check(
        compiler, "persistent_atomic_claim", work, eligible=True,
        source_fragments=("tl.atomic_add(", "claimed_tiles = tl.load(",
                          "for _work in tl.range(tl.program_id(0), TOTAL_TILES, NUM_CTAS):"),
    )
    constants = result["persistent_atomic_claim"]["persistent_constants"]
    if constants["TOTAL_TILES"] <= constants["NUM_CTAS"]:
        raise AssertionError("claimed-work control did not actually reuse resident CTAs")
    steal = deepcopy(work)
    steal["roles"] = split["roles"]
    for op in steal["operations"]:
        if op["id"] in {"load_expert_ids", "reserve_positions"}:
            op["role"] = "comm"
    result["comm_to_compute_handoff"] = check(
        compiler, "comm_to_compute_handoff", steal, eligible=False,
        codes=("OP_CROSS_ROLE_RACE",),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
