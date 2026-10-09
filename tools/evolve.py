#!/usr/bin/env python3
"""Read external development evidence for the evolve skill. Never dispatch work.

The supported allocation is the retained BW1100 static two-host handoff. This is a
format adapter, not a native Lab Run audit or an independent Bench evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile


ALLOCATION_SCHEMA = "bw1100.round5.static-handoff.v1"
REVIEW_KIND = "external_development_review"
IDENTITY_FIELDS = ("task", "compiler", "gateway", "image", "case_ids", "model",
                   "reference_access", "rows", "columns", "depth", "wall_time_seconds",
                   "search_seconds", "confirmation_seconds")
STATES = {"confirmed", "failed", "incomplete", "invalid", "release_pending", "unobserved"}


def read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def allocation_rows(path):
    document = read_object(path)
    if document.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError("unsupported allocation format; use its owning reader")
    rows = document.get("assignments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("allocation has no assigned tasks")
    tasks, locations = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("allocation member is not an object")
        task, host, root = row.get("task"), row.get("host"), row.get("root")
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task):
            raise ValueError("invalid assigned task name")
        if not isinstance(host, str) or host not in document.get("hosts", {}):
            raise ValueError("assigned host is not declared")
        if not isinstance(root, str) or not Path(root).is_absolute() or ".." in Path(root).parts:
            raise ValueError("assigned root must be an absolute path without traversal")
        if task in tasks or (host, root) in locations:
            raise ValueError("duplicate task or run location in allocation")
        tasks.add(task)
        locations.add((host, root))
        if row.get("compiler") != document.get("source", {}).get("compiler"):
            raise ValueError("assignment differs from the allocation Compiler")
        cases = row.get("case_ids")
        if (not isinstance(cases, list) or not cases or
                not all(isinstance(case, str) and case for case in cases) or len(set(cases)) != len(cases)):
            raise ValueError("assigned cases must be explicit and unique")
        if any(key not in row for key in IDENTITY_FIELDS if key != "depth"):
            raise ValueError("assignment is missing a development binding field")
        # The handoff retains the source scaffold's gateway in each assignment.
        # prepare_destination.py binds the destination host's qualified gateway.
        host_config = document["hosts"][host]
        if host_config.get("role") not in {"source", "destination"} or not isinstance(host_config.get("gateway"), str):
            raise ValueError("host gateway binding is missing")
        if host_config["role"] == "source" and row["gateway"] != host_config["gateway"]:
            raise ValueError("source gateway differs from the retained assignment")
        if host_config["role"] == "destination" and row.get("intaken") is not False:
            raise ValueError("destination handoff cannot replace an already intaken Run")
    rows = [{**row, "source_gateway": row["gateway"],
             "gateway": document["hosts"][row["host"]]["gateway"]} for row in rows]
    return document, rows


def _identity(row):
    return {key: row.get(key) for key in ("host", "root", "source_gateway", *IDENTITY_FIELDS)}


def _summary(rows):
    counts = Counter(row["state"] for row in rows)
    return {"assigned": len(rows), "states": dict(sorted(counts.items())),
            "ready_for_maintenance_review": all(row["state"] in {"confirmed", "failed"} for row in rows),
            "performance_claim": False}


def _review_task(assignment, *, mirror=None):
    original = Path(assignment["root"])
    root = Path(mirror) / original.relative_to("/") if mirror is not None else original
    issues = []

    def locator(relative):
        return f"{assignment['host']}:{original / relative}"

    def read(relative, *, required=True):
        path = root / relative
        try:
            return read_object(path)
        except FileNotFoundError:
            if required:
                issues.append(f"missing: {relative}")
        except (OSError, ValueError) as error:
            issues.append(f"unreadable {relative}: {type(error).__name__}")
        return None

    binding = read("campaign/development-binding.json")
    done = read("campaign/DONE.json", required=False)
    endpoint = read("campaign/ENDPOINT.json", required=done is not None)
    if binding is not None:
        for key in IDENTITY_FIELDS:
            if binding.get(key) != assignment.get(key):
                issues.append(f"binding differs from allocation: {key}")
        if binding.get("root") != str(original):
            issues.append("binding differs from allocated root")
    if endpoint is not None:
        for key in ("task", "compiler", "gateway", "wall_time_seconds"):
            if endpoint.get(key) != assignment.get(key):
                issues.append(f"endpoint differs from allocation: {key}")

    candidates = []
    for directory in sorted((root / "campaign/evaluations").glob("*")):
        if not directory.is_dir():
            continue
        relative = str(directory.relative_to(root))
        outcome = read(relative + "/outcome.json", required=False)
        assessment = read(relative + "/assessment.json", required=False)
        candidates.append({"id": directory.name,
            "reported_status": outcome.get("status") if outcome else "missing_outcome",
            "phase": outcome.get("phase") if outcome else None,
            "outcome": locator(relative + "/outcome.json") if outcome is not None else None,
            "assessment": locator(relative + "/assessment.json") if assessment is not None else None,
            "candidate": locator(relative + "/candidate.py"),
            "emitted_source": locator(relative + "/source.py"),
            "blocking_codes": [item.get("code") for item in (assessment or {}).get("findings", [])
                               if isinstance(item, dict) and (item.get("blocks_acceptance") or item.get("blocks_lowering"))]})

    best = (endpoint or {}).get("best")
    confirmation = (endpoint or {}).get("confirmation")
    for label, record, phase in (("best", best, "search"), ("confirmation", confirmation, "confirmation")):
        if record is None:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9_-]{0,63}", record["id"]):
            issues.append(f"invalid {label} identity")
            continue
        outcome = read(f"campaign/evaluations/{record['id']}/outcome.json")
        if outcome != record:
            issues.append(f"{label} differs from retained evaluation outcome")
        if record.get("compiler") != assignment["compiler"] or record.get("phase") != phase:
            issues.append(f"{label} Compiler or phase differs")
        if record.get("status") == "accepted":
            checks = record.get("checks")
            if (not isinstance(checks, list) or any(not isinstance(check, dict) for check in checks)
                    or [check.get("case_id") for check in checks] != assignment["case_ids"]
                    or not all(check.get("passed") is True for check in checks)):
                issues.append(f"{label} accepted without all assigned cases")

    terminal = done is not None and done.get("terminal") is True
    if done is not None and not terminal:
        issues.append("DONE does not establish a terminal result")
    if done is not None and done.get("status") not in {"completed", "failed", "no_qualified_result"}:
        issues.append("unknown DONE status")
    confirmed = (isinstance(best, dict) and best.get("status") == "accepted"
                 and isinstance(confirmation, dict) and confirmation.get("status") == "accepted")
    if terminal and done.get("status") == "completed" and (not confirmed or (endpoint or {}).get("error")):
        issues.append("completed record lacks a consistent final confirmation")
    released = bool(terminal and done.get("device_release_verified") is True
                    and (endpoint or {}).get("device_release_verified") is True)
    state = ("invalid" if issues else "incomplete" if not terminal else
             "release_pending" if not released else "confirmed" if done["status"] == "completed" else "failed")
    return {**_identity(assignment), "state": state, "issues": issues,
            "reported_done_status": done.get("status") if done else None,
            "release_verified_by_owner": released,
            "nominee": best.get("id") if isinstance(best, dict) else None,
            "baseline": {"kind": "canonical_development_starter",
                         "source": locator("campaign/candidates/starter.py"),
                         "workload": locator("campaign/original-workload.json")},
            "evidence": {"binding": locator("campaign/development-binding.json"),
                         "endpoint": locator("campaign/ENDPOINT.json"),
                         "done": locator("campaign/DONE.json"),
                         "journal": locator("campaign/JOURNAL.md"),
                         "profiles": locator(".local/profile"),
                         "release_receipts": locator(".local")},
            "candidates": candidates,
            "search_close_rationale": ((endpoint or {}).get("search_close") or {}).get("rationale"),
            "error": (endpoint or {}).get("error")}


def inspect_development(allocation, *, host, mirror=None):
    document, assignments = allocation_rows(allocation)
    if host not in document["hosts"]:
        raise ValueError("requested host is not in the allocation")
    rows = []
    for row in assignments:
        if row["host"] != host:
            continue
        try:
            rows.append(_review_task(row, mirror=mirror))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            rows.append({**_identity(row), "state": "invalid",
                         "issues": [f"malformed task evidence: {type(error).__name__}: {error}"]})
    return {"schema_version": 1, "kind": REVIEW_KIND, "host": host,
            "allocation": str(Path(allocation).resolve()),
            "scope": "read-only projection of external development records; no native Run audit, device replay or performance claim",
            "assigned_total": len(assignments), "summary": _summary(rows), "rows": rows}


def merge_development(allocation, reports):
    _, assignments = allocation_rows(allocation)
    expected = {row["task"]: row for row in assignments}
    observed = {}
    for path in reports:
        report = read_object(path)
        if report.get("schema_version") != 1 or report.get("kind") != REVIEW_KIND:
            raise ValueError("not an external development review")
        for row in report.get("rows", []):
            task = row.get("task")
            if task not in expected or _identity(row) != _identity(expected[task]):
                raise ValueError("review row differs from the assigned task and controls")
            if task in observed:
                raise ValueError("duplicate task observation; do not count a mirror twice")
            if report.get("host") != row.get("host"):
                raise ValueError("review row belongs to another host")
            if row.get("state") not in STATES:
                raise ValueError("unknown observation state")
            observed[task] = row
    rows = [observed.get(row["task"], {**_identity(row), "state": "unobserved",
                                    "issues": ["no host observation supplied"]}) for row in assignments]
    return {"schema_version": 1, "kind": REVIEW_KIND, "host": None,
            "allocation": str(Path(allocation).resolve()),
            "scope": "joined host snapshots; not a new live read or semantic audit; diagnostic samples are not Bench speedups",
            "sources": [str(Path(path).resolve()) for path in reports],
            "summary": _summary(rows), "rows": rows}


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True,
                                   stderr=subprocess.PIPE, timeout=30).strip()


def _checkout(path):
    root = Path(path).resolve(strict=True)
    if _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise ValueError("pass the checkout root")
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"freeze requires a clean committed checkout: {root}")
    return {"root": str(root), "commit": _git(root, "rev-parse", "HEAD")}


def _tracked_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("a Bench reference must be a repository-relative file")
    _git(root, "cat-file", "-e", f"HEAD:{relative}")
    path = Path(root).resolve(strict=True) / relative
    if path != path.resolve() or not path.is_file():
        raise ValueError(f"Bench reference is not a file: {relative}")


def freeze_bench(draft_path, *, bench, control, successor, output):
    """Freeze one rolling fresh-search cohort; this is not a Study or launch API.

    The frozen document is the intent owner. Run creation must bind its controls
    through the existing platform launcher, which still owns admission and execution.
    """
    draft = read_object(draft_path)
    fields = {"hypothesis", "task_ids", "replicates", "allocation_seed", "author",
              "measurement_protocol", "baseline_files", "total_author_wall_seconds"}
    if set(draft) != fields:
        raise ValueError("Bench draft fields differ; see the evolve Bench reference")
    if not isinstance(draft["hypothesis"], str) or not draft["hypothesis"].strip():
        raise ValueError("name the hypothesis before freezing")
    bindings = {"bench": _checkout(bench), "control": _checkout(control), "successor": _checkout(successor)}
    for compiler_root in (control, successor):
        _tracked_file(compiler_root, "compiler/revision.json")
    if bindings["control"]["commit"] == bindings["successor"]["commit"]:
        raise ValueError("no source change: record No promotion or use a separately declared material comparison")
    _tracked_file(bench, "suite.json")
    suite = read_object(Path(bench) / "suite.json")
    available = {task["id"] for task in suite["tasks"]}
    tasks = draft["task_ids"]
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) for task in tasks)
            or len(set(tasks)) != len(tasks) or not set(tasks) <= available):
        raise ValueError("Bench tasks must be an explicit unique subset of the pinned suite")
    author = draft["author"]
    if not isinstance(author, dict) or set(author) != {
            "model", "scaffold", "knowledge", "reference_access", "wall_time_seconds", "confirmation_seconds"}:
        raise ValueError("author controls differ; tokens are accounting only")
    for key in ("model", "scaffold", "knowledge", "reference_access"):
        if not isinstance(author[key], str) or not author[key].strip():
            raise ValueError(f"author {key} requires an explicit shared selection or none")
    wall, confirmation = author["wall_time_seconds"], author["confirmation_seconds"]
    if type(wall) is not int or type(confirmation) is not int or not 0 < confirmation < wall:
        raise ValueError("wall budget must include a positive confirmation reserve")
    replicates, seed = draft["replicates"], draft["allocation_seed"]
    if type(replicates) is not int or replicates < 1 or type(seed) is not int or seed < 0:
        raise ValueError("replicates and allocation seed must be predeclared integers")
    ceiling = draft["total_author_wall_seconds"]
    if type(ceiling) is not int or ceiling < len(tasks) * 2 * replicates * wall:
        raise ValueError("total author time cannot fund all preassigned runs including confirmation")
    baselines = draft["baseline_files"]
    if not isinstance(baselines, dict) or set(baselines) != set(tasks):
        raise ValueError("every task requires its fixed community baseline; no incumbent denominator")
    for relative in [draft["measurement_protocol"], *baselines.values()]:
        _tracked_file(bench, relative)
    output = Path(output).absolute()
    if output.name != "plan.json":
        raise ValueError("each cohort directory owns one plan.json; use a new directory for a new segment")
    if output != output.resolve() or any((parent / ".git").exists() for parent in (output.parent, *output.parents)):
        raise ValueError("freeze output must be canonical and outside Git checkouts")
    rng = random.Random(seed)
    allocations = []
    for index, task in enumerate(tasks, 1):
        for replicate in range(1, replicates + 1):
            conditions = ["control", "successor"]
            rng.shuffle(conditions)
            for condition in conditions:
                allocations.append({"task": task, "condition": condition, "replicate": replicate,
                                    "workspace": str(output.parent / "runs" / f"t{index:03d}-r{replicate}-{condition}")})
    frozen = {"schema_version": 1, "kind": "rolling_bench_fresh_search", **draft,
              "target": suite["target_arch"], "bindings": bindings, "allocations": allocations,
              "claim_scope": "engineering system-version comparison; no pure Compiler attribution or heldout claim",
              "stopping_rule": "all preassigned runs classified; no replacements or budget reset",
              "baseline_measurement": "remeasure the fixed community baseline in each common assay",
              "execution_owner": "existing platform launcher and GPU admission; this file grants no execution authority"}
    data = (json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("conflicting frozen cohort; create a new comparison segment")
        return frozen
    if any(Path(row["workspace"]).exists() for row in allocations):
        raise ValueError("freeze precedes every assigned workspace; do not collect runs after seeing their results")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".evolve-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != data:
                raise ValueError("another coordinator froze different inputs") from None
    finally:
        temporary.unlink()
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-development", help="read the local host's assigned external task roots")
    inspect.add_argument("--allocation", required=True, type=Path)
    inspect.add_argument("--host", required=True)
    inspect.add_argument("--mirror", type=Path, help="read a local snapshot under MIRROR/<original absolute path>")
    merge = commands.add_parser("merge-development", help="join host snapshots against the original assignment denominator")
    merge.add_argument("--allocation", required=True, type=Path)
    merge.add_argument("reports", nargs="+", type=Path)
    freeze = commands.add_parser("freeze-bench", help="freeze a rolling engineering fresh-search cohort; does not launch")
    freeze.add_argument("--draft", required=True, type=Path)
    freeze.add_argument("--bench", required=True, type=Path)
    freeze.add_argument("--control", required=True, type=Path)
    freeze.add_argument("--successor", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-development":
            result = inspect_development(args.allocation, host=args.host, mirror=args.mirror)
        elif args.command == "merge-development":
            result = merge_development(args.allocation, args.reports)
        else:
            result = freeze_bench(args.draft, bench=args.bench, control=args.control,
                                  successor=args.successor, output=args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"evolve inspection refused: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
