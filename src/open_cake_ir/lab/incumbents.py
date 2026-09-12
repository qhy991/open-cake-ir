"""Append-only per-task incumbents derived from audited Campaign evidence.

The registry is an EvidenceStore: promotions are sealed Runs, artifact bytes live in its
CAS, and the current incumbent is a replay projection.  There is deliberately no mutable
``current.json`` and no second performance ledger.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, cast

from open_cake_ir.evaluation.paired import (
    candidate_from_identity,
    paired_protocol,
)
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence.custody import external_path
from open_cake_ir.serialization import canonical_json_bytes

from .contracts import CampaignLock, CampaignRef, StudyReport
from .executor import ExecutorRevision


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN = re.compile(r"^incumbent-([0-9a-f]{64})-([0-9]{12})$")
_PROMOTION_KIND = "task_incumbent_promoted_v1"
_KEY_KIND = "task_incumbent_key_v1"


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class TaskIncumbentKey:
    """One exact comparison cell whose winner may advance independently."""

    workload_id: str
    workload_sha256: str
    case_id: str
    target: str
    backend: str
    evaluation_protocol_sha256: str

    def __post_init__(self) -> None:
        for name in ("workload_id", "case_id", "target", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"incumbent key {name} differs")
        _digest(self.workload_sha256, "incumbent key workload")
        _digest(self.evaluation_protocol_sha256, "incumbent key protocol")

    @classmethod
    def from_values(
        cls,
        *,
        workload_id: str,
        workload_sha256: str,
        case_id: str,
        target: str,
        backend: str,
        evaluation_protocol: Mapping[str, object],
    ) -> "TaskIncumbentKey":
        return cls(
            workload_id=workload_id,
            workload_sha256=workload_sha256,
            case_id=case_id,
            target=target,
            backend=backend,
            evaluation_protocol_sha256=sha256(
                canonical_json_bytes(evaluation_protocol)
            ).hexdigest(),
        )

    @classmethod
    def from_campaign(cls, lock: CampaignLock) -> "TaskIncumbentKey":
        workload = _object(lock.document["workload"], "campaign workload")
        execution = _object(lock.document["execution"], "campaign execution")
        evaluation = _object(
            lock.document["evaluation_protocol"], "campaign evaluation protocol"
        )
        resolved = _object(lock.document["resolved_inputs"], "campaign resolved inputs")
        arms = _object(resolved["arm_environments"], "campaign arm environments")
        open_cake = _object(arms["open_cake"], "open_cake environment")
        route = _object(open_cake["lowering_route"], "open_cake lowering route")
        return cls.from_values(
            workload_id=str(workload["workload_id"]),
            workload_sha256=_digest(
                workload["canonical_sha256"], "campaign workload"
            ),
            case_id=str(evaluation["case_id"]),
            target=str(execution["target"]),
            backend=str(route["backend"]),
            evaluation_protocol=evaluation,
        )

    @classmethod
    def from_dict(cls, value: object) -> "TaskIncumbentKey":
        document = _object(value, "task incumbent key")
        if set(document) != {
            "schema_version",
            "kind",
            "workload_id",
            "workload_sha256",
            "case_id",
            "target",
            "backend",
            "evaluation_protocol_sha256",
        } or document.get("schema_version") != 1 or document.get("kind") != _KEY_KIND:
            raise ValueError("task incumbent key fields differ")
        return cls(
            workload_id=str(document["workload_id"]),
            workload_sha256=str(document["workload_sha256"]),
            case_id=str(document["case_id"]),
            target=str(document["target"]),
            backend=str(document["backend"]),
            evaluation_protocol_sha256=str(
                document["evaluation_protocol_sha256"]
            ),
        )

    @property
    def canonical_sha256(self) -> str:
        return sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": _KEY_KIND,
            "workload_id": self.workload_id,
            "workload_sha256": self.workload_sha256,
            "case_id": self.case_id,
            "target": self.target,
            "backend": self.backend,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
        }


def validate_baseline_selection(value: object) -> dict[str, object]:
    """Validate the Campaign-bound explanation of one fixed baseline choice."""

    selection = _object(value, "fixed baseline selection")
    if set(selection) != {
        "schema_version",
        "policy",
        "source",
        "incumbent_key",
        "promotion_run_id",
        "registry_root",
    } or selection.get("schema_version") != 1:
        raise ValueError("fixed baseline selection fields differ")
    policy = selection.get("policy")
    source = selection.get("source")
    key = selection.get("incumbent_key")
    promotion = selection.get("promotion_run_id")
    registry_root = selection.get("registry_root")
    if policy == "starter_reference":
        valid = (
            source == "starter_reference"
            and key is None
            and promotion is None
            and registry_root is None
        )
    elif policy == "explicit_fixed_bundle":
        valid = (
            source == "explicit"
            and key is None
            and promotion is None
            and registry_root is None
        )
    elif policy == "exact_incumbent_or_reference":
        incumbent_key = TaskIncumbentKey.from_dict(key)
        if not isinstance(registry_root, str):
            raise ValueError("fixed baseline incumbent registry path differs")
        external_path(registry_root)
        promotion_match = (
            _RUN.fullmatch(promotion) if isinstance(promotion, str) else None
        )
        valid = (
            source == "starter_reference" and promotion is None
        ) or (
            source == "task_incumbent"
            and promotion_match is not None
            and promotion_match.group(1) == incumbent_key.canonical_sha256
        )
    else:
        valid = False
    if not valid:
        raise ValueError("fixed baseline selection policy differs")
    return dict(selection)


@contextmanager
def _writer_lock(root: Path) -> Iterator[None]:
    """Serialize current-read plus append without changing the custody-bound root."""

    lock_path = root.with_name(root.name + ".writer.lock")
    if lock_path.is_symlink():
        raise ValueError("incumbent writer lock must not be a symlink")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("incumbent writer lock custody differs")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class TaskIncumbentRegistry:
    """Replay and materialize exact-task incumbents from one custody-bearing store."""

    def __init__(self, store: EvidenceStore):
        self.store = store
        self.root = store.root

    @classmethod
    def open(cls, root: str | Path) -> "TaskIncumbentRegistry":
        return cls(EvidenceStore.open(external_path(root)))

    @classmethod
    def open_if_exists(
        cls, root: str | Path
    ) -> "TaskIncumbentRegistry | None":
        path = external_path(root)
        return cls(EvidenceStore.open(path)) if path.exists() else None

    @staticmethod
    def key_for_launch(
        *,
        workload_id: str,
        workload_sha256: str,
        case_id: str,
        target: str,
        backend: str,
        evaluation_protocol: Mapping[str, object],
    ) -> TaskIncumbentKey:
        return TaskIncumbentKey.from_values(
            workload_id=workload_id,
            workload_sha256=workload_sha256,
            case_id=case_id,
            target=target,
            backend=backend,
            evaluation_protocol=evaluation_protocol,
        )

    def _promotion(self, run_id: str, key: TaskIncumbentKey) -> dict[str, object]:
        audit = self.store.audit_run(run_id)
        if (
            not audit.archive_integrity
            or not audit.filesystem_custody_verified
            or audit.protocol_adherence != "adhered"
            or audit.endpoint_observation != "observed"
            or audit.authority_sha256 != key.canonical_sha256
        ):
            raise ValueError(f"incumbent promotion {run_id!r} is not custody-audited")
        events = self.store.replay_events(run_id)
        if len(events) != 2 or events[0].get("kind") != _PROMOTION_KIND:
            raise ValueError("incumbent promotion event sequence differs")
        payload = _object(events[0].get("payload"), "incumbent promotion")
        expected = {
            "generation",
            "candidate",
            "predecessor",
            "comparison",
            "source",
            "objects",
        }
        if set(payload) != expected:
            raise ValueError("incumbent promotion fields differ")
        candidate = _object(payload["candidate"], "incumbent candidate")
        comparison = _object(payload["comparison"], "incumbent comparison")
        source = _object(payload["source"], "incumbent source")
        if set(comparison) != {
            "baseline_candidate_sha256",
            "classification",
            "speedup",
            "candidate_median_ms",
            "baseline_median_ms",
            "measurement_quality_passed",
            "materiality_ratio",
        } or set(source) != {
            "campaign_lock_path",
            "campaign_lock_sha256",
            "evidence_root",
            "run_id",
            "launchable_event_sequence",
            "confirmation_event_sequence",
            "evaluation_receipt_sha256",
        }:
            raise ValueError("incumbent comparison or source fields differ")
        if (
            comparison.get("classification") != "first_arm_faster"
            or comparison.get("measurement_quality_passed") is not True
            or not isinstance(comparison.get("speedup"), (int, float))
            or isinstance(comparison.get("speedup"), bool)
            or not math.isfinite(float(comparison["speedup"]))
            or not isinstance(comparison.get("materiality_ratio"), (int, float))
            or isinstance(comparison.get("materiality_ratio"), bool)
            or not math.isfinite(float(comparison["materiality_ratio"]))
            or float(comparison["materiality_ratio"]) <= 1
            or float(comparison["speedup"])
            < float(comparison["materiality_ratio"])
        ):
            raise ValueError("incumbent promotion is not a material confirmed win")
        references = payload["objects"]
        if not isinstance(references, list):
            raise ValueError("incumbent artifact references differ")
        by_role: dict[str, Mapping[str, object]] = {}
        for value in references:
            reference = _object(value, "incumbent artifact reference")
            role = reference.get("role")
            if not isinstance(role, str) or role in by_role:
                raise ValueError("incumbent artifact roles differ")
            by_role[role] = reference
        artifact_roles = _object(candidate.get("artifact_roles"), "candidate roles")
        if set(by_role) != {*artifact_roles, "confirmation_receipt"}:
            raise ValueError("incumbent artifact role coverage differs")
        payloads = {
            role: self.store.read_object(by_role[role]) for role in artifact_roles
        }
        if any(by_role[role].get("sha256") != digest for role, digest in artifact_roles.items()):
            raise ValueError("incumbent artifact identity differs")
        candidate_from_identity(candidate, payloads)
        receipt_payload = self.store.read_object(by_role["confirmation_receipt"])
        receipt = _object(json.loads(receipt_payload), "incumbent confirmation receipt")
        receipt_timing = _object(
            receipt.get("timing"), "incumbent confirmation timing"
        )
        receipt_medians = _object(
            receipt_timing.get("pooled_medians_ms"),
            "incumbent confirmation medians",
        )
        if (
            sha256(receipt_payload).hexdigest()
            != source.get("evaluation_receipt_sha256")
            or receipt.get("candidate_sha256") != candidate.get("candidate_sha256")
            or receipt.get("purpose") != "confirmatory"
            or receipt.get("correctness_passed") is not True
            or receipt.get("kernel_calls") != 1
            or receipt.get("fallback_calls") != 0
            or receipt_timing.get("classification")
            != comparison.get("classification")
            or receipt_timing.get("measurement_quality_passed") is not True
            or receipt_timing.get("speedup") != comparison.get("speedup")
            or receipt_medians.get("candidate")
            != comparison.get("candidate_median_ms")
            or receipt_medians.get("baseline")
            != comparison.get("baseline_median_ms")
        ):
            raise ValueError("incumbent confirmation receipt differs")
        return {
            "run_id": run_id,
            "terminal_seal_sha256": audit.terminal_seal_sha256,
            **dict(payload),
        }

    def _all(self) -> dict[str, list[dict[str, object]]]:
        groups: dict[str, list[dict[str, object]]] = {}
        names = sorted(path.name for path in (self.root / "runs").iterdir())
        for name in names:
            match = _RUN.fullmatch(name)
            if match is None:
                raise ValueError(f"incumbent registry Run {name!r} differs")
            key_digest, generation_text = match.groups()
            authority = json.loads(
                (self.root / "runs" / name / "authority.json").read_text()
            ).get("authority")
            key = TaskIncumbentKey.from_dict(authority)
            if key.canonical_sha256 != key_digest:
                raise ValueError("incumbent Run and key identities differ")
            promotion = self._promotion(name, key)
            if promotion.get("generation") != int(generation_text):
                raise ValueError("incumbent promotion generation differs")
            groups.setdefault(key_digest, []).append(promotion)
        return groups

    def current(self, key: TaskIncumbentKey) -> Mapping[str, object] | None:
        records = self._all().get(key.canonical_sha256, [])
        previous = None
        for generation, record in enumerate(records):
            if record["generation"] != generation or record["predecessor"] != previous:
                raise ValueError("incumbent promotion chain differs")
            comparison = _object(record["comparison"], "incumbent comparison")
            if (previous is not None
                    and comparison["baseline_candidate_sha256"]
                    != previous["candidate_sha256"]):
                raise ValueError("incumbent comparison baseline chain differs")
            candidate = _object(record["candidate"], "incumbent candidate")
            previous = {
                "run_id": record["run_id"],
                "terminal_seal_sha256": record["terminal_seal_sha256"],
                "candidate_sha256": candidate["candidate_sha256"],
            }
        return MappingProxyType(records[-1]) if records else None

    def materialize(
        self, key: TaskIncumbentKey, destination: str | Path
    ) -> tuple[Path, Mapping[str, object]] | None:
        record = self.current(key)
        if record is None:
            return None
        path = Path(destination)
        if path.exists() or path.is_symlink():
            raise ValueError("incumbent materialization destination must be new")
        path.mkdir(mode=0o750)
        references = cast(list[Mapping[str, object]], record["objects"])
        by_role = {cast(str, item["role"]): item for item in references}
        candidate = _object(record["candidate"], "incumbent candidate")
        artifact_roles = _object(candidate["artifact_roles"], "candidate roles")
        artifact_paths = {}
        for role in sorted(artifact_roles):
            filename = role + ".bin"
            output = path / filename
            with output.open("xb") as stream:
                stream.write(self.store.read_object(by_role[role]))
            output.chmod(0o440)
            artifact_paths[role] = filename
        bundle = path / "candidate.json"
        with bundle.open("xb") as stream:
            stream.write(
                canonical_json_bytes(
                    {"candidate": dict(candidate), "artifact_paths": artifact_paths}
                )
            )
        bundle.chmod(0o440)
        return bundle, record


def _bound_audit_report(
    project: Path, lock: CampaignLock, lock_path: Path, evidence_path: Path
) -> Mapping[str, object]:
    """Audit through the frozen Campaign's own Executor source, including history."""

    execution = _object(lock.document["execution"], "campaign execution")
    reference = _object(
        execution["executor_revision"], "campaign Executor Revision"
    )
    executor = ExecutorRevision.load_reference(
        project, reference, "campaign Executor Revision"
    )
    python = executor.document["host_environment"]["python"]["invocation_path"]
    bootstrap = project / "src/open_cake_ir/evaluation/source_bootstrap.py"
    command = [
        str(python),
        "-I",
        str(bootstrap),
        "open_cake_ir.cli",
        "--project-root",
        str(project),
        "lab",
        "audit",
        "--lock",
        str(lock_path),
        "--evidence-root",
        str(evidence_path),
    ]
    completed = subprocess.run(
        command,
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "campaign-bound semantic audit failed: "
            + (completed.stderr.strip().splitlines() or ["no diagnostic"])[-1]
        )
    report = _object(json.loads(completed.stdout), "campaign-bound audit report")
    if (
        report.get("study_id") != lock.study_id
        or report.get("claim_scope") != lock.claim_scope
    ):
        raise ValueError("campaign-bound audit report identity differs")
    return report


def promote_task_incumbent(
    *,
    project_root: str | Path,
    registry_root: str | Path,
    campaign_lock_path: str | Path,
    evidence_root: str | Path,
    lab=None,
    run_id: str | None = None,
) -> Mapping[str, object]:
    """Promote one material confirmed winner and seal its complete artifact bundle."""

    project = Path(project_root).resolve(strict=True)
    lock_path = Path(campaign_lock_path).resolve(strict=True)
    evidence_path = Path(evidence_root).resolve(strict=True)
    lock = CampaignLock.load(lock_path)
    if lock.claim_scope != "artifact_optimization_only":
        raise ValueError("task incumbent promotion requires artifact_optimization_only")
    if lab is None:
        report = _bound_audit_report(project, lock, lock_path, evidence_path)
        descriptive = _object(report.get("descriptive"), "campaign audit descriptive")
    else:
        campaign: CampaignRef = lab.reference_campaign(lock, evidence_path)
        live_report: StudyReport = lab.audit(campaign)
        descriptive = _object(
            live_report.descriptive, "campaign audit descriptive"
        )
    promoted = _object(
        descriptive.get("promoted_artifacts"), "promoted artifacts"
    )
    semantic_replay = _object(
        descriptive.get("semantic_replay_by_run"),
        "promotion semantic replay",
    )
    eligible = [name for name, value in promoted.items() if value is not None]
    selected_run = run_id or (eligible[0] if len(eligible) == 1 else None)
    if selected_run is None or selected_run not in eligible:
        raise ValueError("task incumbent promotion requires exactly one selected Run")
    if semantic_replay.get(selected_run) is not True:
        raise ValueError("task incumbent promotion requires selected-Run semantic replay")
    selected = _object(promoted[selected_run], "promoted artifact")
    candidate_sha = _digest(selected.get("candidate_sha256"), "promoted candidate")
    source_store = EvidenceStore.open(evidence_path)
    source_audit = source_store.audit_run(selected_run)
    if (
        not source_audit.archive_integrity
        or not source_audit.filesystem_custody_verified
        or source_audit.protocol_adherence != "adhered"
        or source_audit.authority_sha256 != lock.canonical_sha256
    ):
        raise ValueError("promoted source Run is not custody-audited")
    events = source_store.replay_events(selected_run)
    launchable = [
        event
        for event in events
        if event.get("kind") == "launchable_candidate_sealed"
        and _object(event.get("payload"), "launchable event").get(
            "candidate_sha256"
        )
        == candidate_sha
    ]
    confirmations = [
        event
        for event in events
        if event.get("kind") == "candidate_evaluated"
        and _object(event.get("payload"), "candidate event").get(
            "candidate_sha256"
        )
        == candidate_sha
        and _object(event.get("payload"), "candidate event").get("purpose")
        == "confirmatory"
    ]
    if len(launchable) != 1 or len(confirmations) != 1:
        raise ValueError("promoted candidate artifact or confirmation coverage differs")
    launch_payload = _object(launchable[0]["payload"], "launchable payload")
    confirm_payload = _object(confirmations[0]["payload"], "confirmation payload")
    source_references = launch_payload.get("objects")
    confirmation_references = confirm_payload.get("objects")
    if not isinstance(source_references, list) or not isinstance(
        confirmation_references, list
    ):
        raise ValueError("promoted candidate object references differ")
    receipt_refs = [
        _object(value, "confirmation receipt reference")
        for value in confirmation_references
        if isinstance(value, Mapping) and value.get("role") == "evaluation_receipt"
    ]
    if len(receipt_refs) != 1:
        raise ValueError("promoted candidate confirmation receipt differs")
    receipt_payload = source_store.read_object(receipt_refs[0])
    receipt = _object(json.loads(receipt_payload), "confirmation receipt")
    timing = _object(receipt.get("timing"), "confirmation timing")
    receipt_sha256 = sha256(receipt_payload).hexdigest()
    protocol = paired_protocol(
        _object(lock.document["evaluation_protocol"], "evaluation protocol")
    )
    speedup = timing.get("speedup")
    medians = _object(timing.get("pooled_medians_ms"), "confirmation medians")
    if (
        protocol is None
        or timing.get("classification") != "first_arm_faster"
        or timing.get("measurement_quality_passed") is not True
        or not isinstance(speedup, (int, float))
        or isinstance(speedup, bool)
        or not math.isfinite(float(speedup))
        or float(speedup) < protocol.materiality_ratio
        or receipt.get("correctness_passed") is not True
        or receipt.get("kernel_calls") != 1
        or receipt.get("fallback_calls") != 0
        or selected.get("evaluation_receipt_sha256") != receipt_sha256
        or selected.get("turn") != confirm_payload.get("turn")
        or selected.get("confirmed_latency_ms") != medians.get("candidate")
    ):
        raise ValueError("promoted candidate is not a material confirmed win")
    candidate_refs = [_object(value, "candidate artifact reference") for value in source_references]
    artifact_payloads = {
        cast(str, reference["role"]): source_store.read_object(reference)
        for reference in candidate_refs
    }
    candidate_identity = {
        "candidate_sha256": candidate_sha,
        "target": json.loads(artifact_payloads["launch_manifest"])["target"],
        "entry_point": json.loads(artifact_payloads["launch_manifest"])["kernel_name"],
        "artifact_roles": {
            role: sha256(payload).hexdigest()
            for role, payload in sorted(artifact_payloads.items())
        },
        "launch_spec_sha256": sha256(
            artifact_payloads["launch_manifest"]
        ).hexdigest(),
        "candidate_record_sha256": launch_payload["candidate_record_sha256"],
    }
    candidate_from_identity(candidate_identity, artifact_payloads)
    key = TaskIncumbentKey.from_campaign(lock)
    root = external_path(Path(registry_root).absolute())
    if project == root or project in root.parents or root in project.parents:
        raise ValueError("task incumbent registry must remain outside the source checkout")
    with _writer_lock(root):
        store = EvidenceStore.writer(root) if root.exists() else EvidenceStore.create(root)
        registry = TaskIncumbentRegistry(store)
        previous = registry.current(key)
        fixed = _object(lock.document["execution"], "campaign execution")
        baseline = _object(
            _object(fixed["fixed_baseline"], "fixed baseline")["candidate"],
            "fixed baseline candidate",
        )
        if previous is not None:
            incumbent_candidate = _object(previous["candidate"], "current incumbent")
            if baseline != incumbent_candidate:
                raise ValueError(
                    "campaign fixed baseline is not the current task incumbent"
                )
            predecessor = {
                "run_id": previous["run_id"],
                "terminal_seal_sha256": previous["terminal_seal_sha256"],
                "candidate_sha256": incumbent_candidate["candidate_sha256"],
            }
            generation = int(previous["generation"]) + 1
        else:
            predecessor = None
            generation = 0
        if baseline.get("candidate_sha256") == candidate_sha:
            raise ValueError("promoted candidate does not advance the incumbent")
        registry_references = []
        for reference in candidate_refs:
            role = cast(str, reference["role"])
            stored = store.put(
                artifact_payloads[role], media_type=cast(str, reference["media_type"])
            )
            registry_references.append(stored.reference(role))
        receipt_object = store.put(receipt_payload, media_type="application/json")
        registry_references.append(receipt_object.reference("confirmation_receipt"))
        run_name = f"incumbent-{key.canonical_sha256}-{generation:012d}"
        ledger = store.start_run(
            run_name,
            authority_sha256=key.canonical_sha256,
            authority=key.as_dict(),
        )
        comparison = {
            "baseline_candidate_sha256": baseline["candidate_sha256"],
            "classification": timing["classification"],
            "speedup": timing["speedup"],
            "candidate_median_ms": medians["candidate"],
            "baseline_median_ms": medians["baseline"],
            "measurement_quality_passed": True,
            "materiality_ratio": protocol.materiality_ratio,
        }
        source = {
            "campaign_lock_path": str(lock_path),
            "campaign_lock_sha256": lock.canonical_sha256,
            "evidence_root": str(evidence_path),
            "run_id": selected_run,
            "launchable_event_sequence": launchable[0]["sequence"],
            "confirmation_event_sequence": confirmations[0]["sequence"],
            "evaluation_receipt_sha256": receipt_sha256,
        }
        ledger.append(
            _PROMOTION_KIND,
            {
                "generation": generation,
                "candidate": candidate_identity,
                "predecessor": predecessor,
                "comparison": comparison,
                "source": source,
                "objects": registry_references,
            },
        )
        ledger.seal(
            protocol_adherence="adhered",
            endpoint_observation="observed",
            endpoint={
                "candidate_sha256": candidate_sha,
                "generation": generation,
            },
        )
        current = TaskIncumbentRegistry.open(root).current(key)
        if current is None:
            raise ValueError("incumbent promotion did not publish a current record")
        return current
