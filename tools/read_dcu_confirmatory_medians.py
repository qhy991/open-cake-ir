#!/usr/bin/env python3
"""Read the DCU confirmatory medians out of retained campaign evidence.

`findings/data/2026-09-18-dcu-confirmatory-medians.json` is the table F-2026-09-18-002,
F-2026-09-18-003 and F-2026-09-18-005 derive their counts from. Until this existed the table had no instrument:
it was produced by a script that lived in a scratch directory on the host, so the numbers
could be read but not reproduced, and the file's own `collected` field named a source that
did not exist in the repository. That is the failure mode `tools/observe_lowered_kernel.py`
was written to close for a different record, and the same argument applies here -- evidence
that can only be renewed by editing it is not evidence.

It walks one runs directory of campaign workspaces, and for each task takes the FIRST run
that both adhered and reached a qualified endpoint, by workspace timestamp. Later qualified
runs of the same task are reported separately under `excluded_reruns` rather than dropped,
because the choice is not neutral and a reader has to be able to see what it discarded.

Nothing here times, launches or evaluates anything. It only reads sealed events and the
receipts they point at, so it can run anywhere the evidence is readable -- in practice
inside the DTK image, because the workspaces are written by root in that container.

    python3 tools/read_dcu_confirmatory_medians.py --runs /runs --out medians.json

`--out` writes only the fields this tool derives. The committed table also carries
hand-kept ones it cannot produce -- what, collected, row_selection, receipt_paths and
the transcribed footprint_sweep -- so merge into the committed file rather than
replacing it; the table's `generated_fields` names exactly the half this tool owns.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from math import gcd
from collections import defaultdict

WORKSPACE = re.compile(r"^(?P<task>.+)-(?P<day>\d{8})-(?P<time>\d{6})$")
#: The identity a campaign ran under, as its own evidence spells it. Both spellings occur:
#: the reference object carries `open-cake-ir@<sha>`, and some records carry the bare field.
REVISION = (re.compile(r"open-cake-ir@([0-9a-f]{8,40})"),
            re.compile(r'"compiler_revision_id"\s*:\s*"([^"]+)"'))


def _events(root):
    return sorted(glob.glob(os.path.join(root, "runs", "*", "events", "*.json")))


def _read(path):
    try:
        with open(path, errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def read_revision(workspace):
    """The Compiler revision one campaign ran at, receipt or no receipt.

    A run that qualified nothing still ran at a commit, and records about turn budget
    cite exactly those runs, so this must not be gated on a confirmatory receipt.
    """
    root = os.path.join(workspace, "campaign-evidence")
    if not os.path.isdir(root):
        return None
    revisions = set()
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        blob = _read(path)
        for pattern in REVISION:
            for hit in pattern.finditer(blob):
                revisions.add(hit.group(1).split("@")[-1][:8])
    if not revisions:
        return None
    if len(revisions) != 1:
        raise SystemExit(f"{os.path.basename(workspace)}: evidence names "
                         f"{len(revisions)} Compiler revisions {sorted(revisions)}; "
                         "a run has exactly one")
    return "open-cake-ir@" + revisions.pop()


def read_workspace(workspace):
    """Return this run's endpoint, confirmatory medians and Compiler revision, or None."""
    root = os.path.join(workspace, "campaign-evidence")
    if not os.path.isdir(root):
        return None
    name = os.path.basename(workspace)
    match = WORKSPACE.match(name)
    if match is None:
        return None
    found = {"task": match.group("task"), "workspace": name,
             "started": match.group("day") + "-" + match.group("time"),
             "adherence": None, "observation": None, "receipt": None,
             "candidate": None, "baseline": None, "classification": None}
    # The Compiler identity is carried by the objects a run sealed, not by its events, so
    # the revision scan covers the whole evidence root while the endpoint comes from the
    # event stream. Reading only the events finds no revision at all.
    revisions = set()
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        blob = _read(path)
        for pattern in REVISION:
            for hit in pattern.finditer(blob):
                revisions.add(hit.group(1).split("@")[-1][:8])
    for path in _events(root):
        blob = _read(path)
        if not blob:
            continue
        try:
            event = json.loads(blob)
        except ValueError:
            continue
        payload = event.get("payload", {})
        if event.get("kind") == "run_terminal":
            found["adherence"] = payload.get("protocol_adherence") or found["adherence"]
            observed = payload.get("endpoint_observation")
            if observed and observed != "missing":
                found["observation"] = observed
        for obj in payload.get("objects", []):
            if obj.get("role") != "evaluation_receipt":
                continue
            receipt = _read(os.path.join(root, obj["relative_path"]))
            try:
                record = json.loads(receipt)
            except ValueError:
                continue
            if record.get("purpose") != "confirmatory" or not record.get("timing"):
                continue
            medians = (record["timing"].get("pooled_medians_ms") or {})
            if "candidate" in medians and "baseline" in medians:
                # Six decimal places is what the assay reports; the extra binary digits a
                # float carries here are not measurement and must not reach the ledger.
                found["receipt"] = obj["relative_path"]
                found["candidate"] = round(medians["candidate"], 6)
                found["baseline"] = round(medians["baseline"], 6)
                found["classification"] = (record["timing"].get("classification")
                                           or record.get("classification"))
    if found["receipt"] is None:
        return None
    if len(revisions) != 1:
        raise SystemExit(f"{name}: evidence names {len(revisions)} Compiler revisions "
                         f"{sorted(revisions)}; a run has exactly one")
    found["compiler_revision"] = "open-cake-ir@" + revisions.pop()
    return found


def collect(runs_directory):
    """Group every qualified run by task, and record what every campaign ran at.

    The second half matters because records cite runs this table has no row for.
    F-2026-09-18-003 is about turn budget, so it cites each task's runs at two, four and
    six turns including the ones that qualified nothing -- and its Compiler stamp has to
    be supported by those runs, not by the qualified subset. Reading the revision for
    every workspace costs nothing extra here and leaves nothing for prose to assert.
    """
    by_task = defaultdict(list)
    revisions = {}
    for workspace in sorted(glob.glob(os.path.join(runs_directory, "*"))):
        if not os.path.isdir(workspace):
            continue
        revision = read_revision(workspace)
        if revision is not None:
            revisions[os.path.basename(workspace)] = revision
        found = read_workspace(workspace)
        if found is None:
            continue
        if found["adherence"] == "adhered" and found["observation"] == "qualified":
            by_task[found["task"]].append(found)
    for runs in by_task.values():
        runs.sort(key=lambda run: run["started"])
    return by_task, revisions


def table(by_task, revisions, *, floor_ms, materiality_ratio):
    def row(run):
        fields = ["candidate", "baseline", "classification", "workspace", "receipt",
                  "compiler_revision"]
        return {name: run[name] for name in fields}

    kept = {task: row(runs[0]) for task, runs in sorted(by_task.items())}
    dropped = {task: [row(run) for run in runs[1:]]
               for task, runs in sorted(by_task.items()) if len(runs) > 1}
    qualified_runs = sum(len(runs) for runs in by_task.values())
    # The grid every reported median lies on. hip_benchmark reports resolution_us per
    # cohort, from whatever steps that cohort happened to show; across a whole collection
    # the greatest common divisor of the pairwise differences is unambiguous, and it is
    # what decides whether a shape can be measured at all. F-2026-09-18-002 argued from
    # bytes for three versions without it.
    nanoseconds = sorted({round(row[key] * 1e6) for row in kept.values()
                          for key in ("candidate", "baseline")})
    quantum = 0
    for first in nanoseconds:
        for second in nanoseconds:
            quantum = gcd(quantum, abs(first - second))
    document = {
        "schema_version": 1,
        "finding": "F-2026-09-18-002",
        "target": "gfx938",
        "generated_by": "tools/read_dcu_confirmatory_medians.py",
        # Exactly the fields this tool owns. The rest of the file is prose written by the
        # author, and saying which is which is the point: an earlier version of the file
        # claimed the tool "regenerates every field below" while it emitted ten keys of
        # fifteen, so running it with --out would have deleted five.
        "generated_fields": None,
        "qualified_runs": qualified_runs,
        "qualified_tasks": len(kept),
        "floor_ms": floor_ms,
        "timer_quantum_ns": quantum or None,
        "materiality_ratio": materiality_ratio,
        "sweep_days": sorted({name.rsplit("-", 2)[1] for name in revisions}),
        "medians_ms": kept,
        "excluded_reruns": dropped,
        "campaign_revisions": dict(sorted(revisions.items())),
    }
    # Derived from the document rather than listed beside it, so the two cannot disagree.
    document["generated_fields"] = sorted(document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True,
                        help="directory of campaign workspaces, one per run")
    parser.add_argument("--out", help="write here instead of stdout")
    parser.add_argument("--floor-ms", type=float, default=0.005439,
                        help="the dispatch floor F-2026-09-18-002 measured on this target")
    parser.add_argument("--materiality-ratio", type=float, default=1.05,
                        help="the Study's declared materiality ratio")
    arguments = parser.parse_args(argv)
    by_task, revisions = collect(arguments.runs)
    document = table(by_task, revisions, floor_ms=arguments.floor_ms,
                     materiality_ratio=arguments.materiality_ratio)
    rendered = json.dumps(document, indent=1) + "\n"
    if arguments.out:
        with open(arguments.out, "w") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
