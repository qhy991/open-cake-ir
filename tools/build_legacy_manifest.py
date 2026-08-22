#!/usr/bin/env python3
"""Build or verify the bounded legacy source manifest from a pinned Git checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git(checkout: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _safe_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source-set path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe source-set path: {value!r}")
    return value


def build_manifest(source_set_path: Path, checkout: Path) -> list[dict[str, Any]]:
    source_set = _load_object(source_set_path)
    if source_set.get("schema_version") != 2:
        raise ValueError("source-set schema_version must be 2")
    source_set_id = source_set.get("source_set_id")
    repository = source_set.get("repository")
    entries = source_set.get("entries")
    exclusions = source_set.get("exclusions")
    durability = source_set.get("durability")
    if (
        not isinstance(source_set_id, str)
        or not isinstance(repository, Mapping)
        or not isinstance(entries, list)
        or not isinstance(exclusions, list)
        or not isinstance(durability, Mapping)
    ):
        raise ValueError("source-set root fields differ")
    revision = repository.get("revision")
    expected_tree = repository.get("tree")
    if not isinstance(revision, str) or not isinstance(expected_tree, str):
        raise ValueError("repository revision/tree must be strings")
    observed_revision = _git(checkout, "rev-parse", f"{revision}^{{commit}}")
    observed_tree = _git(checkout, "show", "-s", "--format=%T", revision)
    if observed_revision != revision or observed_tree != expected_tree:
        raise ValueError("legacy repository revision or tree differs")
    if set(durability) != {"bundle_path", "sha256", "size_bytes", "complete_history"}:
        raise ValueError("source-set durability fields differ")
    bundle_relative = _safe_path(durability.get("bundle_path"))
    project_root = source_set_path.resolve(strict=True).parents[1]
    bundle_path = (project_root / bundle_relative).resolve(strict=True)
    if project_root not in bundle_path.parents or bundle_path.is_symlink() or not bundle_path.is_file():
        raise ValueError("legacy Git bundle custody differs")
    bundle_payload = bundle_path.read_bytes()
    if (
        durability.get("sha256") != sha256(bundle_payload).hexdigest()
        or durability.get("size_bytes") != len(bundle_payload)
        or durability.get("complete_history") is not True
    ):
        raise ValueError("legacy Git bundle bytes differ")
    with tempfile.TemporaryDirectory(prefix="open-cake-bundle-verify-") as directory:
        verify_repository = Path(directory) / "verify.git"
        subprocess.run(
            ["git", "init", "--bare", "-q", str(verify_repository)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(verify_repository), "bundle", "verify", str(bundle_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

    records: list[dict[str, Any]] = [
        {
            "kind": "source_set",
            "repository": repository.get("name"),
            "revision": revision,
            "schema_version": 2,
            "source_set_id": source_set_id,
            "tree": expected_tree,
            "exclusions": exclusions,
            "origin_main": repository.get("origin_main"),
            "ahead_of_origin": repository.get("ahead_of_origin"),
            "initial_observed_origin_main": repository.get(
                "initial_observed_origin_main"
            ),
            "initial_observed_ahead_of_origin": repository.get(
                "initial_observed_ahead_of_origin"
            ),
            "durability": dict(durability),
        }
    ]
    observed_paths: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("source-set entry must be an object")
        kind = raw_entry.get("kind")
        path = _safe_path(raw_entry.get("path"))
        role = raw_entry.get("role")
        disposition = raw_entry.get("disposition")
        if path in observed_paths:
            raise ValueError(f"duplicate source-set path: {path}")
        observed_paths.add(path)
        if kind not in {"file", "tree"} or not isinstance(role, str) or not isinstance(disposition, str):
            raise ValueError(f"invalid source-set entry: {path}")
        object_id = _git(checkout, "rev-parse", f"{revision}:{path}")
        object_kind = _git(checkout, "cat-file", "-t", object_id)
        record: dict[str, Any] = {
            "disposition": disposition,
            "git_object": object_id,
            "kind": kind,
            "path": path,
            "revision": revision,
            "role": role,
            "schema_version": 2,
            "source_set_id": source_set_id,
        }
        if kind == "file":
            if object_kind != "blob":
                raise ValueError(f"{path} is not a Git blob")
            payload = _git(checkout, "show", f"{revision}:{path}", binary=True)
            assert isinstance(payload, bytes)
            record["raw_sha256"] = sha256(payload).hexdigest()
            record["size_bytes"] = len(payload)
            if path.endswith(".json"):
                parsed = json.loads(payload)
                record["canonical_json_sha256"] = sha256(
                    _canonical_json_bytes(parsed)
                ).hexdigest()
        else:
            if object_kind != "tree":
                raise ValueError(f"{path} is not a Git tree")
            names = _git(checkout, "ls-tree", "-r", "--name-only", f"{revision}:{path}")
            assert isinstance(names, str)
            record["file_count"] = len(names.splitlines()) if names else 0
        records.append(record)
    return records


def _manifest_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-set", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    records = build_manifest(args.source_set, args.checkout.resolve(strict=True))
    expected = _manifest_bytes(records)
    if args.verify:
        if args.output.read_bytes() != expected:
            raise SystemExit("legacy manifest differs from pinned source set")
        return 0
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite legacy manifest")
    args.output.write_bytes(expected)
    args.output.chmod(0o644)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
