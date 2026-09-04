"""Recompute every collected output independently of the judge's recorded verdict."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from oracle import ELEMENTS, FAMILIES, MODES, SHAPE, equivalent, expected_bits, inputs


def verify(root: Path) -> dict:
    paths = sorted(root.rglob("*.complete-output.json"))
    counts = Counter()
    checked = 0
    for path in paths:
        row = json.loads(path.read_text())
        mode, family = row["mode"], row["family"]
        assert mode in MODES and family in FAMILIES and row["shape"] == list(SHAPE), path
        before = list(inputs(family))
        assert row["input_bits_before"] == before == row["input_bits_after"], path
        expected = [expected_bits(mode, *triple) for triple in zip(*before)]
        actual = row["actual_bits"]
        assert len(actual) == ELEMENTS and all(type(x) is int and 0 <= x < 2**32 for x in actual), path
        assert all(equivalent(e, a) for e, a in zip(expected, actual)), path
        assert row["expected_bits"] == expected and row["abi_valid"] is True, path
        assert row["case_id"] == f"{mode}-{family}", path
        counts[row["case_id"]] += 1
        checked += len(actual)
    # Exactly one complete-output set from each of the three mandatory stages.
    assert counts == Counter({f"{m}-{f}": 3 for m in MODES for f in FAMILIES}), counts
    return {"status": "passed", "artifacts_checked": len(paths),
            "output_elements_recomputed": checked, "case_counts": dict(counts),
            "gpu_rerun": False, "performance_measured": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact_root), indent=2, sort_keys=True))
