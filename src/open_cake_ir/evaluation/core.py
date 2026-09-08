"""Common Evaluation from sealed launchable artifacts to append-only receipts."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from .artifacts import allowed_artifact_roles, executable_role

from .profiler import load_ncu_attribution_profile, ncu_attribution_feedback
from .workload import WorkloadContract

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: object) -> object:
    """Detach supported JSON values from caller-owned mutable containers."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical record object keys must be strings")
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("canonical record contains a non-JSON value")


def _freeze_json(value: object) -> object:
    """Recursively freeze a detached JSON value without changing its content."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class LaunchableCandidate:
    """Sealed arm output at the seam before common correctness/timing."""

    candidate_sha256: str
    target: str
    entry_point: str
    artifact_roles: Mapping[str, str]
    launch_spec_sha256: str
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        executable = executable_role(self.target)
        allowed_roles = allowed_artifact_roles(self.target)
        if (
            _DIGEST.fullmatch(self.candidate_sha256) is None
            or _DIGEST.fullmatch(self.launch_spec_sha256) is None
            or not self.entry_point
            or not self.artifact_roles
            or executable not in self.artifact_roles
            or any(
                role not in allowed_roles or _DIGEST.fullmatch(digest) is None
                for role, digest in self.artifact_roles.items()
            )
        ):
            raise ValueError("LaunchableCandidate identity or artifacts differ")
        if self.artifact_payloads and (
            set(self.artifact_payloads) != set(self.artifact_roles)
            or any(
                not isinstance(payload, bytes)
                or not payload
                or sha256(payload).hexdigest() != self.artifact_roles[role]
                for role, payload in self.artifact_payloads.items()
            )
        ):
            raise ValueError("LaunchableCandidate artifact bytes differ from their seals")
        if (
            "lowered_source" in self.artifact_roles
            and "compiler_expanded_source" in self.artifact_roles
            and self.artifact_roles["lowered_source"]
            == self.artifact_roles["compiler_expanded_source"]
        ):
            raise ValueError("lowering input and compiler-expanded source are distinct artifacts")
        object.__setattr__(
            self,
            "artifact_roles",
            MappingProxyType(dict(self.artifact_roles)),
        )
        object.__setattr__(
            self,
            "artifact_payloads",
            MappingProxyType(dict(self.artifact_payloads)),
        )

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "candidate_sha256": self.candidate_sha256,
                    "target": self.target,
                    "entry_point": self.entry_point,
                    "artifact_roles": dict(sorted(self.artifact_roles.items())),
                    "launch_spec_sha256": self.launch_spec_sha256,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class EvaluationProtocol:
    """Frozen assay purpose and Workload case."""

    protocol_id: str
    purpose: str
    workload_sha256: str
    case_id: str
    timing: str

    def __post_init__(self) -> None:
        if (
            not self.protocol_id
            or self.purpose not in {"search", "confirmatory", "attribution"}
            or _DIGEST.fullmatch(self.workload_sha256) is None
            or not self.case_id
            or self.timing not in {"none", "paired_cupti", "paired_metal"}
            # A profiler serialises kernels and inflates every span it observes, so an
            # attribution assay cannot also be a timing source. Making that structural
            # rather than a note means a profiled run has no latency to be mistaken for
            # a measurement.
            or (self.purpose == "attribution" and self.timing != "none")
        ):
            raise ValueError("EvaluationProtocol differs")

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "protocol_id": self.protocol_id,
                    "purpose": self.purpose,
                    "workload_sha256": self.workload_sha256,
                    "case_id": self.case_id,
                    "timing": self.timing,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class LaunchObservation:
    """Observed common launch result; arm build details remain behind the seam."""

    output: object
    kernel_calls: int
    fallback_calls: int
    launch_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kernel_calls != 1
            or self.fallback_calls != 0
            or _DIGEST.fullmatch(self.launch_receipt_sha256) is None
        ):
            raise ValueError("LaunchObservation violates common route policy")


@dataclass(frozen=True)
class EvaluationReceipt:
    """Append-only common Evaluation observation."""

    candidate_sha256: str
    workload_sha256: str
    evaluation_protocol_sha256: str
    purpose: str
    case_id: str
    correctness_passed: bool
    correctness: Mapping[str, object]
    kernel_calls: int
    fallback_calls: int
    launch_receipt_sha256: str
    timing: Mapping[str, object] | None
    artifact_payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            _DIGEST.fullmatch(self.candidate_sha256) is None
            or _DIGEST.fullmatch(self.workload_sha256) is None
            or _DIGEST.fullmatch(self.evaluation_protocol_sha256) is None
            or _DIGEST.fullmatch(self.launch_receipt_sha256) is None
            or self.purpose not in {"search", "confirmatory", "attribution"}
            or not self.case_id
            or self.kernel_calls != 1
            or self.fallback_calls != 0
            or (self.purpose == "attribution" and self.timing is not None)
            # The profile describes this launch, not the earlier confirmatory one. An
            # incorrect observed launch therefore cannot be attribution evidence even
            # when the same sealed Candidate happened to pass before profiling.
            or (self.purpose == "attribution" and not self.correctness_passed)
        ):
            raise ValueError("EvaluationReceipt identity or route differs")
        if self.artifact_payloads:
            expected_roles = (
                {"correctness_output", "launch_receipt", "profile"}
                if self.purpose == "attribution"
                else {"correctness_output", "launch_receipt", "timing_samples"}
            )
            if (
                set(self.artifact_payloads) != expected_roles
                or any(not isinstance(payload, bytes) or not payload for payload in self.artifact_payloads.values())
                or sha256(self.artifact_payloads["launch_receipt"]).hexdigest()
                != self.launch_receipt_sha256
            ):
                raise ValueError("EvaluationReceipt artifact custody differs")
            try:
                correctness_raw = json.loads(self.artifact_payloads["correctness_output"])
                timing_raw = (
                    None
                    if self.purpose == "attribution"
                    else json.loads(self.artifact_payloads["timing_samples"])
                )
                launch_raw = json.loads(self.artifact_payloads["launch_receipt"])
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("EvaluationReceipt raw artifacts are not JSON") from error
            if self.purpose == "attribution":
                profile_document = json.loads(self.artifact_payloads["profile"])
                if profile_document.get("kind") == "metal_compute_stage_timestamps_v1":
                    from .metal_observations import load_metal_profile
                    profile = load_metal_profile(self.artifact_payloads["profile"],
                        expected_candidate_sha256=self.candidate_sha256, expected_case_id=self.case_id,
                        expected_protocol_sha256=self.evaluation_protocol_sha256)
                    from .paired import validate_metal_correctness_checks, validation_case_ids
                    from .metal_observations import validate_launch_sequence
                    validate_metal_correctness_checks(correctness_raw, (*validation_case_ids(profile["evaluation_protocol"]), self.case_id))
                    validate_launch_sequence([row["command_buffer"] for row in correctness_raw["launches"]])
                    if (correctness_raw["launches"][-1]["command_buffer"] != profile["raw"]["command_buffer"]
                            or launch_raw.get("allocation_mode") != "local_serialized" or launch_raw.get("external_gpu_activity") != "not_excluded"
                            or launch_raw.get("host") != profile["host"] or launch_raw.get("job_id") != profile["job_id"]
                            or launch_raw.get("instrumented_command") != profile["raw"]["command_buffer"]):
                        raise ValueError("Metal attribution launch differs from instrumented profile")
                else:
                    load_ncu_attribution_profile(
                        self.artifact_payloads["profile"],
                        expected_candidate_sha256=self.candidate_sha256,
                        expected_case_id=self.case_id,
                    )
            if not isinstance(correctness_raw, Mapping) or not isinstance(
                launch_raw, Mapping
            ):
                raise ValueError("EvaluationReceipt raw correctness or launch receipt differs")
            raw_metrics = correctness_raw.get("metrics", correctness_raw)
            if not isinstance(raw_metrics, Mapping) or dict(raw_metrics) != dict(
                self.correctness
            ):
                raise ValueError("EvaluationReceipt correctness summary differs from raw output")
            derived_correctness = correctness_raw.get("passed")
            if derived_correctness is None and {
                "tie_aware_distance_match",
                "max_chosen_distance_excess",
            } <= set(raw_metrics):
                derived_correctness = bool(raw_metrics["tie_aware_distance_match"]) and (
                    raw_metrics["max_chosen_distance_excess"] == 0.0
                )
            if derived_correctness is not None and (
                derived_correctness is True
            ) != self.correctness_passed:
                raise ValueError("EvaluationReceipt correctness disposition differs from raw output")
            if launch_raw.get("candidate_sha256") not in {
                None,
                self.candidate_sha256,
            }:
                raise ValueError("EvaluationReceipt launch candidate differs")
            paired_raw = isinstance(timing_raw, Mapping) and timing_raw.get('kind') in {'fixed_baseline_paired_cupti_v1', 'fixed_baseline_paired_metal_v1'}
            if paired_raw:
                from .paired import validate_paired_receipt
                validate_paired_receipt(self, timing_raw, correctness_raw, launch_raw)
            if self.timing is not None and not paired_raw:
                cohorts_value = (
                    timing_raw.get("cohorts_ms")
                    if isinstance(timing_raw, Mapping)
                    else [timing_raw]
                )
                if not isinstance(cohorts_value, list) or not cohorts_value:
                    raise ValueError("EvaluationReceipt timing samples differ")
                cohorts: list[list[float]] = []
                for cohort in cohorts_value:
                    if not isinstance(cohort, list) or not cohort:
                        raise ValueError("EvaluationReceipt timing cohort differs")
                    values = [float(value) for value in cohort]
                    if any(not math.isfinite(value) or value <= 0 for value in values):
                        raise ValueError("EvaluationReceipt timing sample differs")
                    cohorts.append(values)
                pooled = statistics.median(value for cohort in cohorts for value in cohort)
                stable = all(
                    (
                        math.sqrt(
                            sum((value - statistics.fmean(cohort)) ** 2 for value in cohort)
                            / len(cohort)
                        )
                        / statistics.fmean(cohort)
                    )
                    <= 0.05
                    for cohort in cohorts
                )
                if (
                    not math.isclose(
                        float(self.timing.get("pooled_median_ms")),
                        pooled,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or (self.timing.get("measurement_quality_passed") is True)
                    != stable
                ):
                    raise ValueError("EvaluationReceipt timing summary differs from raw samples")
        detached_correctness = json.loads(
            _canonical_json_bytes(_plain_json(self.correctness))
        )
        object.__setattr__(
            self,
            "correctness",
            cast(Mapping[str, object], _freeze_json(detached_correctness)),
        )
        if self.timing is not None:
            detached_timing = json.loads(
                _canonical_json_bytes(_plain_json(self.timing))
            )
            object.__setattr__(
                self,
                "timing",
                cast(Mapping[str, object], _freeze_json(detached_timing)),
            )
        object.__setattr__(
            self,
            "artifact_payloads",
            MappingProxyType(dict(self.artifact_payloads)),
        )

    @property
    def candidate_disposition(self) -> str:
        return "qualified" if self.correctness_passed else "correctness_rejected"

    @property
    def measurement_quality(self) -> str:
        if self.timing is None:
            return "not_measured"
        return "stable" if self.timing.get("measurement_quality_passed") is True else "unstable"

    @property
    def attribution_feedback(self) -> Mapping[str, object] | None:
        """Return the raw-checked profiler projection for the next authoring Turn."""

        if self.purpose != "attribution" or not self.artifact_payloads:
            return None
        if json.loads(self.artifact_payloads["profile"]).get("kind") == "metal_compute_stage_timestamps_v1":
            from .metal_observations import load_metal_profile
            profile = load_metal_profile(self.artifact_payloads["profile"],
                expected_candidate_sha256=self.candidate_sha256, expected_case_id=self.case_id,
                expected_protocol_sha256=self.evaluation_protocol_sha256)
            return {"kind": profile["kind"], **profile["summary"]}
        profile = load_ncu_attribution_profile(
            self.artifact_payloads["profile"],
            expected_candidate_sha256=self.candidate_sha256,
            expected_case_id=self.case_id,
        )
        return ncu_attribution_feedback(profile)

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "candidate_sha256": self.candidate_sha256,
                    "workload_sha256": self.workload_sha256,
                    "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
                    "purpose": self.purpose,
                    "case_id": self.case_id,
                    "correctness_passed": self.correctness_passed,
                    "correctness": _plain_json(self.correctness),
                    "kernel_calls": self.kernel_calls,
                    "fallback_calls": self.fallback_calls,
                    "launch_receipt_sha256": self.launch_receipt_sha256,
                    "timing": _plain_json(self.timing) if self.timing is not None else None,
                    "artifact_payload_sha256": {
                        role: sha256(payload).hexdigest()
                        for role, payload in sorted(self.artifact_payloads.items())
                    },
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class TensorLaunchManifest:
    """Explicit Workload tensor ABI for the existing sealed CUBIN launch boundary."""

    workload_sha256: str
    case_id: str
    tensor_abi: tuple[tuple[str, tuple[int, ...], str, str], ...]
    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    hidden_null_pointer_parameters: int

    @classmethod
    def from_dict(cls, document: object) -> 'TensorLaunchManifest':
        from .cuda_manifest import CudaKernelSpec
        if not isinstance(document, Mapping) or set(document) != {
            'schema_version', 'abi', 'workload_sha256', 'case_id', 'tensor_abi',
            'target', 'kernel_name', 'grid', 'block', 'dynamic_shared_memory_bytes',
            'hidden_null_pointer_parameters',
        } or document.get('schema_version') != 1 or document.get('abi') != 'workload_tensors_v1':
            raise ValueError('Workload tensor launch manifest fields differ')
        launch = CudaKernelSpec.from_dict({
            key: value for key, value in document.items()
            if key not in {'schema_version', 'abi', 'workload_sha256', 'case_id', 'tensor_abi'}
        })
        rows = document['tensor_abi']
        if (not isinstance(document['workload_sha256'], str)
            or _DIGEST.fullmatch(document['workload_sha256']) is None
            or not isinstance(document['case_id'], str) or not document['case_id']
            or not isinstance(rows, list) or not rows):
            raise ValueError('Workload tensor launch identity differs')
        abi = []
        for row in rows:
            if (not isinstance(row, Mapping) or set(row) != {'name', 'shape', 'dtype', 'mode'}
                or not isinstance(row['name'], str) or not row['name'].isidentifier()
                or not isinstance(row['dtype'], str) or row['dtype'] not in {'fp32', 'bf16', 'fp16', 'int32'}
                or not isinstance(row['mode'], str) or row['mode'] not in {'input', 'output'}
                or not isinstance(row['shape'], list) or not row['shape']
                or any(type(v) is not int or v <= 0 for v in row['shape'])):
                raise ValueError('Workload tensor launch ABI differs')
            abi.append((row['name'], tuple(row['shape']), row['dtype'], row['mode']))
        modes = [row[3] for row in abi]
        if len({r[0] for r in abi}) != len(abi) or 'input' not in modes or 'output' not in modes or modes != sorted(modes):
            raise ValueError('Workload tensor launch ABI order differs')
        return cls(document['workload_sha256'], document['case_id'], tuple(abi),
                   launch.target, launch.kernel_name, launch.grid, launch.block,
                   launch.dynamic_shared_memory_bytes, launch.hidden_null_pointer_parameters)

    @classmethod
    def for_workload(cls, workload: WorkloadContract, case_id: str, **launch: object) -> 'TensorLaunchManifest':
        from dataclasses import asdict
        manifest = cls.from_dict({'schema_version': 1, 'abi': 'workload_tensors_v1',
            'workload_sha256': workload.canonical_sha256, 'case_id': case_id,
            'tensor_abi': [{**asdict(t), 'shape': list(t.shape)} for t in workload.tensor_abi(case_id)],
            **launch})
        manifest.check_workload(workload, case_id)
        return manifest

    def check_workload(self, workload: WorkloadContract, case_id: str) -> None:
        expected = tuple((t.name, t.shape, t.dtype, t.mode) for t in workload.tensor_abi(case_id))
        if (self.workload_sha256 != workload.canonical_sha256 or self.case_id != case_id
            or self.tensor_abi != expected
            or self.target != workload.document['semantics'].get('target')):
            raise ValueError('sealed launch ABI differs from the selected Workload')

    @property
    def block_threads(self) -> int:
        return math.prod(self.block)

    @property
    def tensors(self) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        dtypes = {'fp32': 'torch.float32', 'bf16': 'torch.bfloat16', 'fp16': 'torch.float16', 'int32': 'torch.int32'}
        return tuple((name, shape, dtypes[dtype]) for name, shape, dtype, _ in self.tensor_abi)

    def as_dict(self) -> dict[str, object]:
        return {'schema_version': 1, 'abi': 'workload_tensors_v1',
            'workload_sha256': self.workload_sha256, 'case_id': self.case_id,
            'tensor_abi': [dict(name=n, shape=list(s), dtype=d, mode=m) for n, s, d, m in self.tensor_abi],
            'target': self.target, 'kernel_name': self.kernel_name, 'grid': list(self.grid),
            'block': list(self.block), 'dynamic_shared_memory_bytes': self.dynamic_shared_memory_bytes,
            'hidden_null_pointer_parameters': self.hidden_null_pointer_parameters}

    @property
    def canonical_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


def compare_tile_outputs(workload, before, expected, observed, after):
    """One comparison owner for fresh and already-recorded tensor launches."""
    import struct
    validation = workload.document['validation']
    mismatch = 0
    maximum_error = 0.0
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        mismatch += 1
    else:
        for name, values in expected.items():
            actual = observed[name]
            if not isinstance(actual, (list, tuple)) or len(actual) != len(values):
                mismatch += 1
                continue
            for value, reference in zip(actual, values, strict=True):
                if (not isinstance(value, (float, int)) or isinstance(value, bool)
                    or not -3.4028234663852886e38 <= value <= 3.4028234663852886e38
                    or not math.isfinite(value)):
                    mismatch += 1
                    continue
                error = abs(value - reference)
                maximum_error = max(maximum_error, error)
                if validation['comparison'] == 'bitwise_bf16':
                    mismatch += (abs(value) > 3.4028234663852886e38 or struct.pack('>f', value) != struct.pack('>f', reference))
                else:
                    mismatch += error > validation['atol'] + validation['rtol'] * abs(reference)
    unchanged = after == before and all(
        struct.pack('>d', float(a)) == struct.pack('>d', float(b))
        for name in before for a, b in zip(after[name], before[name], strict=True)
    )
    metrics = {'output_mismatches': mismatch, 'max_abs_error': maximum_error, 'inputs_unchanged': unchanged}
    return mismatch == 0 and unchanged, metrics


class LoadedTorchTensorCandidate:
    """One Workload-shaped argument set and admitted CUBIN for preflight/timing/postflight."""

    def __init__(self, candidate, manifest, inputs, admission):
        from .cuda_driver import LoadedCudaCandidate
        import torch
        self.candidate = candidate
        self.manifest = manifest
        self.admission = admission
        self.inputs = {name: list(values) for name, values in inputs.items()}
        dtypes = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'int32': torch.int32}
        self.arguments = [
            torch.tensor(inputs[name], dtype=dtypes[dtype], device='cuda:0').reshape(shape)
            if mode == 'input' else torch.full(shape,
                float('nan') if dtype != 'int32' else -(2**31),
                dtype=dtypes[dtype], device='cuda:0')
            for name, shape, dtype, mode in manifest.tensor_abi
        ]
        self.loaded = LoadedCudaCandidate.load(candidate, candidate.artifact_payloads['cubin'], manifest, admission)

    def fresh_argument_sets(self, count):
        """Prepare non-reusable outputs and finish their initialization outside timing."""
        import torch
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('fresh tensor argument count differs')
        sets = [[argument if mode == 'input' else torch.full_like(argument,
                    float('nan') if dtype != 'int32' else -(2**31))
                 for (_, _, dtype, mode), argument in zip(self.manifest.tensor_abi, self.arguments, strict=True)]
                for _ in range(count)]
        torch.cuda.synchronize()
        return sets

    def launch(self, arguments=None):
        import torch
        self.loaded.launch(self.arguments if arguments is None else arguments, tensor_contract=self.manifest,
                           stream=torch.cuda.current_stream().cuda_stream)

    def snapshot(self, arguments=None):
        arguments = self.arguments if arguments is None else arguments
        observed = {name: value.cpu().reshape(-1).tolist() for (name, _, _, mode), value
                    in zip(self.manifest.tensor_abi, arguments, strict=True) if mode == 'output'}
        after = {name: value.cpu().reshape(-1).tolist() for (name, _, _, mode), value
                 in zip(self.manifest.tensor_abi, arguments, strict=True) if mode == 'input'}
        return observed, after

    def launch_tensors(self, candidate, manifest, inputs):
        from dataclasses import asdict
        if (candidate.canonical_sha256 != self.candidate.canonical_sha256
            or manifest.canonical_sha256 != self.manifest.canonical_sha256
            or inputs != self.inputs):
            raise ValueError('loaded tensor assay input or candidate differs')
        for (_, _, dtype, mode), argument in zip(manifest.tensor_abi, self.arguments, strict=True):
            if mode == 'output':
                argument.fill_(float('nan') if dtype != 'int32' else -(2**31))
        before = self.loaded.launch_calls
        self.launch()
        observed, after = self.snapshot()
        return observed, after, {'candidate_sha256': candidate.candidate_sha256,
            'kernel_calls': self.loaded.launch_calls - before, 'fallback_calls': 0,
            'manifest_sha256': manifest.canonical_sha256, 'device_admission': asdict(self.admission),
            'module_unloaded': self.loaded.closed, 'resources': self.loaded.resources}

    def close(self):
        import torch
        self.loaded.close(synchronize=torch.cuda.synchronize)


class TorchTensorLauncher:
    """Single-launch convenience over the shared loaded-tensor lifecycle."""

    def __init__(self, admission):
        self.admission = admission

    def launch_tensors(self, candidate, manifest, inputs):
        loaded = LoadedTorchTensorCandidate(candidate, manifest, inputs, self.admission)
        try:
            observed, after, receipt = loaded.launch_tensors(candidate, manifest, inputs)
        finally:
            loaded.close()
        receipt['module_unloaded'] = loaded.loaded.closed
        return observed, after, receipt
