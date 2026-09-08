#!/usr/bin/env python3
"""Read retained diagnosis counts from sealed Evidence roots; never reroute history."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.routing import DESTINATIONS
from open_cake_ir.serialization import canonical_json_bytes


def summarize(roots) -> dict[str, object]:
    """Archive integrity is checked; counts are archived decisions, not new findings.

    Semantic replay remains owned by each Campaign's pinned Executor. Custody is
    reported separately and is never repaired or required for historical counting.
    """
    groups = {}
    runs = []
    seen = {}
    for root in sorted({Path(value).resolve(strict=True) for value in roots}):
        evidence = EvidenceStore.open(root)
        for run_path in sorted((root / "runs").iterdir()):
            if not run_path.is_dir() or run_path.is_symlink():
                raise ValueError("diagnosis Run directory differs")
            audit = evidence.audit_run(run_path.name)
            if not audit.archive_integrity:
                raise ValueError(f"Run archive integrity failed: {root} / {run_path.name}")
            identity = (audit.authority_sha256, audit.run_id)
            if identity in seen:
                if seen[identity] != audit.terminal_seal_sha256:
                    raise ValueError("conflicting sealed histories for the same Campaign Run")
                continue  # The same sealed Run copied or named twice is one observation.
            seen[identity] = audit.terminal_seal_sha256
            authority = json.loads((run_path / "authority.json").read_text())["authority"]
            execution = authority.get("execution", {})
            resolved = authority.get("resolved_inputs", {})
            if not isinstance(execution, dict) or not isinstance(resolved, dict):
                raise ValueError("diagnosis summary Campaign authority differs")
            provenance = {"executor_revision": execution.get("executor_revision"),
                          "evidence_policy": resolved.get("evidence_policy")}
            if not isinstance(provenance["executor_revision"], dict) or not isinstance(provenance["evidence_policy"], dict):
                raise ValueError("diagnosis summary requires retained Executor and evidence policy")
            key = canonical_json_bytes(provenance)
            group = groups.setdefault(key, {**provenance, "counts": Counter(), "run_count": 0})
            group["run_count"] += 1
            counts = Counter()
            for event in evidence.replay_events(audit.run_id):
                if event["kind"] not in {"candidate_rejected", "diagnosis_routed"}:
                    continue
                payload = event["payload"]
                destination = payload.get("routed_to")
                if destination not in DESTINATIONS:
                    raise ValueError("retained diagnosis destination differs")
                # Rejections and set-level collapse/ranking diagnoses are distinct
                # occurrences. Preserve their kind rather than inventing a combined total.
                counts[f"{event['kind']}:{destination}"] += 1
            group["counts"].update(counts)
            runs.append({"root": str(root), "run_id": audit.run_id,
                         "campaign_id": authority.get("campaign_id"),
                         "archive_integrity": True,
                         "filesystem_custody_verified": audit.filesystem_custody_verified,
                         "counts": dict(sorted(counts.items()))})
    return {"schema_version": 1,
            "domain": "archive-integrity-checked retained diagnosis counts; no semantic reclassification or promotion",
            "semantic_replay": "use each Campaign's pinned Executor audit",
            "groups": [{**groups[key], "counts": dict(sorted(groups[key]["counts"].items()))}
                       for key in sorted(groups)], "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_roots", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.evidence_roots)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"diagnosis summary refused: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
