#!/usr/bin/env python3
"""Read retained TaskLab reports for a documentation snapshot; never audit or promote.

Run on the host that owns --runs. Stdout is a public, allowlisted projection; raw
reports, prompts, runtime configuration and artifact bytes are not copied. Redirect
it to a new external file before reviewing it for publication in the results gallery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_workspace(workspace: Path) -> dict:
    report = json.loads((workspace / "report.json").read_text())
    if (workspace/'run.json').exists():
        if (workspace/'campaign-lock.json').exists():
            raise ValueError('workspace has two execution authorities')
        lock = json.loads((workspace/'run.json').read_text())
        audit,replay = report.get('audit'),report.get('replay')
        if not isinstance(audit,dict) or not isinstance(replay,dict):
            raise ValueError('Run report audit or replay fields differ')
        audits = [audit]
        verified = (report.get('run_id')==lock['run_id']==audit.get('run_id')==replay.get('run_id')
                    and audit.get('archive_integrity') is True and audit.get('filesystem_custody_verified') is True
                    and replay.get('refusals')==[])
        performance = report.get('performance',{})
    else:
        lock = json.loads((workspace / "campaign-lock.json").read_text())
        audits = report.get('run_audits',[])
        verified = (all(report.get(key) is True for key in (
            'archive_integrity_passed','filesystem_custody_verified','semantic_replay_passed'))
            and report.get('campaign_complete') is True)
        performance = report.get("descriptive", {}).get("performance", {})
    workload = lock["workload"]["workload_id"]
    rows = performance.get("rows", [])
    retained = []
    for audit in audits:
        endpoint = audit.get("endpoint") or {}
        best = endpoint.get("best_candidate_sha256")
        candidate = next((r for r in rows if r.get("role") == "candidate"
                          and r.get("run_id") == audit["run_id"]
                          and r.get("candidate_id") == best
                          and r.get("latency_ms") == endpoint.get("best_confirmed_latency_ms")), None)
        baseline = next((r for r in rows if candidate is not None
                         and r.get("role") == "baseline"
                         and r.get("run_id") == candidate.get("run_id")
                         and r.get("confirmation_event") == candidate.get("confirmation_event")), None)
        qualified = (
            verified
            and audit.get("protocol_adherence") == "adhered"
            and audit.get("endpoint_observation") == "qualified"
            and candidate is not None and baseline is not None
        )
        retained.append({
            "run": audit["run_id"],
            "status": "reported_qualified" if qualified else "not_qualified",
            "protocol_adherence": audit.get("protocol_adherence"),
            "endpoint_observation": audit.get("endpoint_observation"),
            "terminal_reason": endpoint.get("terminal_reason"),
            "candidate_id": best,
            "candidate_ms": candidate["latency_ms"] if qualified else None,
            "baseline_ms": baseline["latency_ms"] if qualified else None,
            "paired_speedup": candidate["speedup"] if qualified else None,
            "confirmation_event": candidate["confirmation_event"] if qualified else None,
            "confirmations": [{
                "turn": r["turn"], "event": r["confirmation_event"],
                "candidate_ms": r["latency_ms"], "paired_speedup": r["speedup"],
            } for r in sorted(rows, key=lambda r: r.get("confirmation_event", 0))
                if qualified and r.get("role") == "candidate" and r.get("run_id") == audit["run_id"]],
        })
    revision = lock.get("compiler_revision", {})
    return {
        "workspace": str(workspace), "workload": workload,
        "target": lock["execution"]["target"],
        "compiler_revision": revision.get("revision_id", revision.get("compiler_revision_id", revision.get("path"))),
        "baseline_selection": lock["execution"]["fixed_baseline"].get("selection", {}).get("source", "unreported"),
        "case": performance.get("case_id", lock.get("evaluation_protocol", {}).get("case_id")),
        "claim_scope": report.get("claim_scope"),
        "runs": retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    rows, unavailable = [], []
    for path in sorted(args.runs.glob("*/report.json")):
        try:
            rows.append(read_workspace(path.parent))
        except (OSError, ValueError, KeyError, TypeError) as error:
            unavailable.append({"workspace": str(path.parent), "reason": str(error)})
    print(json.dumps({
        "kind": "retained_report_projection", "runs_root": str(args.runs),
        "scope": "Read-only projection of retained audit reports; no fresh semantic replay, registry promotion or global-best selection.",
        "selection": "Every workspace with report.json; retain each run's recorded endpoint and its own paired confirmation baseline.",
        "workspaces": rows, "unavailable": unavailable,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
