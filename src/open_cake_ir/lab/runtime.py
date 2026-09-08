"""Concrete composition primitives for canonical Lab execution."""

from __future__ import annotations

import grp, json, os, pwd, re, stat, tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol

from open_cake_ir.evaluation import (
    BrokerAttempt,
    EvaluationReceipt,
    LaunchableCandidate,
    LogicalEvaluationAttempt,
    evaluate_with_admission_recovery,
)
from open_cake_ir.evaluation.paired import (
    candidate_identity,
    paired_protocol,
    validate_pair_candidates,
    validate_receipt_policy,
    validate_paired_broker,
)

from .executor import ExecutorRevision
from .faults import RunProtocolFault
from .process import (
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)
from .runtime_config import (
    _canonical_json_bytes,
    _runtime_command,
    _runtime_positive_int,
    _runtime_string,
    load_runtime_config,
    broker_execution_sha256,
)


_BROKER_JOB_OBSERVATION = re.compile(
    rb"(?m)^\[gpu-run\] accepted job (gpuq-[0-9a-f]{12})\b"
)

_WORKER_JOB_PLACEHOLDER = "gpuq-000000000000"

class BrokerSubmitter(Protocol):
    """Submit one immutable candidate to the pinned evaluator/broker route."""

    def submit(
        self,
        candidate: LaunchableCandidate,
        *,
        case_id: str,
        purpose: str,
        attempt: int,
    ) -> BrokerAttempt:
        """Return the full job/receipt observation for exactly one broker job."""

class BoundedBrokerEvaluator:
    """Concrete RunEvaluator enforcing the sole admitted infrastructure resubmission."""

    def __init__(
        self,
        protocol: Mapping[str, object],
        submitter: BrokerSubmitter,
    ) -> None:
        self.protocol = json.loads(_canonical_json_bytes(protocol))
        self.protocol_sha256 = sha256(_canonical_json_bytes(protocol)).hexdigest()
        self._submitter = submitter

    def evaluate(
        self,
        candidate: LaunchableCandidate,
        *,
        case_id: str,
        purpose: str,
    ) -> LogicalEvaluationAttempt:
        """Retain every job while permitting only the exact zero-work race once."""

        return evaluate_with_admission_recovery(
            candidate,
            lambda attempt: self._submitter.submit(
                candidate,
                case_id=case_id,
                purpose=purpose,
                attempt=attempt,
            ),
        )

class CommandBrokerSubmitter:
    """Invoke one closed evaluator/broker command and retain all returned raw bytes."""

    _REMOVED_ENVIRONMENT = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        workload_path: str | Path,
        workload_sha256: str,
        protocol_sha256: str,
        cwd: str | Path,
        executor: ExecutorRevision,
        service_user: str,
        service_group: str,
        timeout_seconds: int = 1800,
        evaluation_protocol: Mapping[str, object] | None = None,
        baseline: LaunchableCandidate | None = None,
        workload_loader: Callable | None = None,
    ) -> None:
        self._command = _runtime_command(command)
        _runtime_positive_int(timeout_seconds, "runtime_config.broker.timeout_seconds")
        _runtime_string(service_user, "runtime_config.broker.service_user")
        _runtime_string(service_group, "runtime_config.broker.service_group")
        self._workload_path = Path(workload_path).resolve(strict=True)
        self._workload_sha256 = workload_sha256
        self._protocol_sha256 = protocol_sha256
        self._cwd = Path(cwd).resolve(strict=True)
        self._executor = executor
        self._service_uid = pwd.getpwnam(service_user).pw_uid
        self._service_gid = grp.getgrnam(service_group).gr_gid
        self._timeout = timeout_seconds
        self._protocol = json.loads(_canonical_json_bytes(evaluation_protocol)) if evaluation_protocol is not None else None
        self._baseline = baseline
        self._load_workload = workload_loader
        if baseline is not None and not callable(workload_loader):
            raise ValueError('paired broker requires the task Workload loader')
        if self._protocol is not None:
            if sha256(_canonical_json_bytes(self._protocol)).hexdigest() != protocol_sha256:
                raise ValueError('broker evaluation protocol differs')
            if (paired_protocol(self._protocol) is not None) != (baseline is not None):
                raise ValueError('broker paired policy and fixed baseline differ')
        elif baseline is not None:
            raise ValueError('broker baseline requires an explicit evaluation protocol')

    def _read_output_artifact(self, root: Path, value: object, role: str) -> bytes:
        if not isinstance(value, str) or not value:
            raise ValueError(f"evaluator artifact {role!r} path differs")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or "\\" in value:
            raise ValueError(f"evaluator artifact {role!r} path is unsafe")
        unresolved = root / value
        if unresolved.is_symlink():
            raise ValueError(f"evaluator artifact {role!r} custody differs")
        path = unresolved.resolve(strict=True)
        if root not in path.parents:
            raise ValueError(f"evaluator artifact {role!r} escapes output root")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid not in {os.geteuid(), self._service_uid}
                or metadata.st_gid != self._service_gid
                or metadata.st_size <= 0
                or metadata.st_size > 256 * 1024 * 1024
            ):
                raise ValueError(f"evaluator artifact {role!r} metadata differs")
            payload = os.read(descriptor, metadata.st_size)
            if len(payload) != metadata.st_size or os.read(descriptor, 1):
                raise ValueError(f"evaluator artifact {role!r} changed during read")
            return payload
        finally:
            os.close(descriptor)

    def submit(
        self,
        candidate: LaunchableCandidate,
        *,
        case_id: str,
        purpose: str,
        attempt: int,
    ) -> BrokerAttempt:
        """Run one job; BoundedBrokerEvaluator decides whether another is admitted."""

        if (
            attempt not in {1, 2}
            or purpose not in {"search", "confirmatory", "attribution"}
            or not candidate.artifact_payloads
        ):
            raise ValueError("broker attempt number or Candidate custody differs")
        if self._baseline is not None:
            workload = self._load_workload(self._workload_path)
            if workload.canonical_sha256 != self._workload_sha256:
                raise ValueError('broker Workload authority differs')
            validate_pair_candidates(candidate, self._baseline, workload, case_id)
        with tempfile.TemporaryDirectory(prefix="open-cake-evaluation-") as directory:
            root = Path(directory).resolve()
            os.chown(root, -1, self._service_gid)
            root.chmod(0o2770)
            artifact_paths: dict[str, str] = {}
            for role, payload in candidate.artifact_payloads.items():
                path = root / f"candidate-{role}"
                path.write_bytes(payload)
                os.chown(path, -1, self._service_gid)
                path.chmod(0o640)
                artifact_paths[role] = path.name
            evaluator_arguments = {
                "candidate_sha256": candidate.candidate_sha256,
                "candidate_record_sha256": candidate.canonical_sha256,
                "target": candidate.target,
                "entry_point": candidate.entry_point,
                "artifact_roles": dict(candidate.artifact_roles),
                "artifact_paths": artifact_paths,
                "launch_spec_sha256": candidate.launch_spec_sha256,
                "workload_path": str(self._workload_path),
                "workload_sha256": self._workload_sha256,
                "evaluation_protocol_sha256": self._protocol_sha256,
                "case_id": case_id,
                "purpose": purpose,
                "attempt": attempt,
                "executor_revision": dict(self._executor.reference),
            }
            if self._protocol is not None:
                evaluator_arguments['evaluation_protocol'] = self._protocol
            if self._baseline is not None:
                baseline_paths = {}
                for role, payload in self._baseline.artifact_payloads.items():
                    path = root / f'baseline-{role}'
                    path.write_bytes(payload)
                    os.chown(path, -1, self._service_gid)
                    path.chmod(0o640)
                    baseline_paths[role] = path.name
                evaluator_arguments['baseline'] = {
                    **candidate_identity(self._baseline), 'artifact_paths': baseline_paths,
                }
            request_path = root / "request.json"
            result_path = root / "result.json"
            request_path.write_bytes(_canonical_json_bytes(evaluator_arguments))
            os.chown(request_path, -1, self._service_gid)
            request_path.chmod(0o640)
            command = [
                *self._command,
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ]
            environment = sanitized_environment(tuple(self._REMOVED_ENVIRONMENT))
            try:
                completed = run_supervised(
                    command,
                    cwd=self._cwd,
                    environment=environment,
                    timeout_seconds=self._timeout,
                )
            except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
                raise RunProtocolFault(
                    "broker_fault",
                    str(error),
                    artifact_payloads={
                        "broker_stdout": error.stdout,
                        "broker_stderr": error.stderr,
                    },
                ) from error
            if not result_path.is_file() or result_path.is_symlink():
                raise RunProtocolFault(
                    "broker_fault",
                    f"evaluator command exited {completed.returncode} without a result",
                    artifact_payloads={
                        "broker_stdout": completed.stdout,
                        "broker_stderr": completed.stderr,
                    },
                )
            metadata = result_path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid not in {os.geteuid(), self._service_uid}
                or metadata.st_gid != self._service_gid
                or metadata.st_size <= 0
                or metadata.st_size > 16 * 1024 * 1024
            ):
                raise ValueError("evaluator result metadata differs")
            worker_result_bytes = result_path.read_bytes()
            if len(worker_result_bytes) != metadata.st_size:
                raise ValueError("evaluator result changed during read")
            if completed.returncode != 0:
                raise RunProtocolFault(
                    "broker_fault",
                    f"evaluator/broker command exited {completed.returncode}",
                    artifact_payloads={
                        "broker_stdout": completed.stdout,
                        "broker_stderr": completed.stderr,
                        "broker_result": worker_result_bytes,
                    },
                )
            worker_result = json.loads(worker_result_bytes)
            result = worker_result
            if not isinstance(result, Mapping) or set(result) != {
                "schema_version",
                "job_id",
                "mode",
                "admitted",
                "error",
                "failure_class",
                "counters",
                "receipt",
            } or result.get("schema_version") != 1:
                raise ValueError("evaluator command result fields differ")
            job_observations = _BROKER_JOB_OBSERVATION.findall(completed.stderr)
            if len(job_observations) != 1:
                raise ValueError("broker job observation coverage differs")
            observed_job_id = job_observations[0].decode("ascii")
            if result.get("job_id") not in {observed_job_id, _WORKER_JOB_PLACEHOLDER}:
                raise ValueError("worker and broker job identities differ")
            result = dict(result)
            result["job_id"] = observed_job_id
            result_bytes = _canonical_json_bytes(result)
            counters = result["counters"]
            if not isinstance(counters, Mapping) or set(counters) != {
                "compiler_invocations",
                "module_loads",
                "preflight_calls",
                "kernel_calls",
                "timing_samples",
                "fallback_calls",
            }:
                raise ValueError("evaluator command counters differ")
            receipt: EvaluationReceipt | None = None
            receipt_value = result["receipt"]
            if receipt_value is not None:
                if not isinstance(receipt_value, Mapping) or set(receipt_value) != {
                    "correctness_passed",
                    "correctness",
                    "kernel_calls",
                    "fallback_calls",
                    "timing",
                    "artifacts",
                }:
                    raise ValueError("evaluator receipt fields differ")
                artifact_refs = receipt_value["artifacts"]
                expected_artifacts = (
                    {"correctness_output", "launch_receipt", "profile"}
                    if purpose == "attribution"
                    else {"correctness_output", "launch_receipt", "timing_samples"}
                )
                if (
                    not isinstance(artifact_refs, Mapping)
                    or set(artifact_refs) != expected_artifacts
                ):
                    raise ValueError("evaluator receipt artifact roles differ")
                payloads = {
                    str(role): self._read_output_artifact(root, path, str(role))
                    for role, path in artifact_refs.items()
                }
                correctness = receipt_value["correctness"]
                timing = receipt_value["timing"]
                if not isinstance(correctness, Mapping) or (
                    timing is not None and not isinstance(timing, Mapping)
                ):
                    raise ValueError("evaluator receipt observation fields differ")
                receipt = EvaluationReceipt(
                    candidate_sha256=candidate.candidate_sha256,
                    workload_sha256=self._workload_sha256,
                    evaluation_protocol_sha256=self._protocol_sha256,
                    purpose=purpose,
                    case_id=case_id,
                    correctness_passed=receipt_value["correctness_passed"] is True,
                    correctness=dict(correctness),
                    kernel_calls=int(receipt_value["kernel_calls"]),
                    fallback_calls=int(receipt_value["fallback_calls"]),
                    launch_receipt_sha256=sha256(payloads["launch_receipt"]).hexdigest(),
                    timing=dict(timing) if timing is not None else None,
                    artifact_payloads=payloads,
                )
                if self._protocol is not None:
                    validate_receipt_policy(receipt, self._protocol,
                        candidate_identity(self._baseline) if self._baseline is not None else None,
                        candidate)
                validate_paired_broker(receipt, observed_job_id, counters)

            evaluator_authority = dict(evaluator_arguments)
            evaluator_authority.pop("attempt")
            evaluator_arguments_sha256 = sha256(
                _canonical_json_bytes(evaluator_authority)
            ).hexdigest()
            return BrokerAttempt(
                job_id=str(result["job_id"]),
                mode=str(result["mode"]),
                candidate_sha256=candidate.candidate_sha256,
                manifest_sha256=candidate.launch_spec_sha256,
                policy_sha256=self._protocol_sha256,
                evaluator_arguments_sha256=evaluator_arguments_sha256,
                admitted=result["admitted"] is True,
                error=str(result["error"]) if result["error"] is not None else None,
                compiler_invocations=int(counters["compiler_invocations"]),
                module_loads=int(counters["module_loads"]),
                preflight_calls=int(counters["preflight_calls"]),
                kernel_calls=int(counters["kernel_calls"]),
                timing_samples=int(counters["timing_samples"]),
                fallback_calls=int(counters["fallback_calls"]),
                receipt=receipt,
                artifact_payloads={
                    "broker_record": result_bytes,
                    "evaluator_result": worker_result_bytes,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
