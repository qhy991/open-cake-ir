"""Common Evaluation from sealed launchable artifacts to append-only receipts."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler.target import CodeObject

from .artifacts import allowed_artifact_roles, executable_role
from .launch_manifest import WorkloadTensorManifest, tensor_abi_rows
from .platforms import PLATFORMS, platform_for
from .profiler import load_ncu_attribution_profile, ncu_attribution_feedback
from .workload import WorkloadContract

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# The timing an assay may declare: none, or the paired assay of a platform that names a
# timer. Read off the rows so a platform whose timer exists can declare it -- the HIP
# row's `paired_hip` had no spelling here while the assay itself was already measured.
_TIMINGS = frozenset({"none"}) | frozenset(
    row.protocol_timing for row in PLATFORMS.values() if row.protocol_timing is not None)


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
        if self.is_program:
            from .program import PROGRAM_ROLES
            allowed_roles = PROGRAM_ROLES
            executable = 'program_bundle'
        else:
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
        # Historical identity-only and opaque legacy records remain valid. A
        # composed manifest, once bytes are present, must be a complete bundle at
        # construction as well as at paired evaluation and archive replay.
        document = None
        if 'launch_manifest' in self.artifact_payloads:
            try:
                document = json.loads(self.artifact_payloads['launch_manifest'])
            except (ValueError, UnicodeError):
                pass
        if self.is_program and self.artifact_payloads:
            from .program import program_components
            program_components(self)
        bundled = 'kernel_bundle' in self.artifact_payloads
        declares_variant = isinstance(document, Mapping) and document.get('aligned_variant') is not None
        if bundled or declares_variant:
            if not bundled or not isinstance(document, Mapping):
                raise ValueError('aligned candidate requires its complete sealed kernel bundle')
            from .kernel_bundle import alignment_component
            alignment_component(self, TensorLaunchManifest.from_dict(document))

    @property
    def is_program(self):
        return 'program_bundle' in self.artifact_roles

    @property
    def kernels_per_call(self):
        if self.is_program:
            from .program import ProgramLaunchManifest
            return ProgramLaunchManifest.from_dict(json.loads(self.artifact_payloads['launch_manifest'])).kernels_per_call
        return 1

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
            or self.timing not in _TIMINGS
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
            or type(self.kernel_calls) is not int or self.kernel_calls <= 0
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
                elif profile_document.get("kind") == "hip_dispatch_activity_v1":
                    from .hip_observations import load_hip_profile
                    profile = load_hip_profile(self.artifact_payloads["profile"],
                        expected_candidate_sha256=self.candidate_sha256,
                        expected_case_id=self.case_id,
                        expected_protocol_sha256=self.evaluation_protocol_sha256)
                    if (launch_raw.get("job_id") != profile["job_id"]
                            or launch_raw.get("gpu_uuid") != profile["gpu_uuid"]):
                        raise ValueError(
                            "HIP attribution launch differs from instrumented profile")
                elif profile_document.get('kind') == 'ncu_program_attribution':
                    from .profiler import load_ncu_program_profile
                    profile = load_ncu_program_profile(self.artifact_payloads['profile'],
                        expected_candidate_sha256=self.candidate_sha256, expected_case_id=self.case_id)
                    if len(profile['stages']) != self.kernel_calls:
                        raise ValueError('Program profile kernel coverage differs from its receipt')
                elif profile_document.get("kind") != "ncu_kernel_attribution":
                    raise ValueError(
                        "no attribution source declares profile kind "
                        f"{profile_document.get('kind')!r}")
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
            # The assay owns its kind vocabulary; naming the versions again here is how
            # a successor silently falls through to the unpaired branch.
            from .paired import PAIRED_KINDS
            paired_raw = isinstance(timing_raw, Mapping) and timing_raw.get('kind') in PAIRED_KINDS
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
        # Each source is decided by the kind it declares. Nsight used to be the
        # fall-through, which read as "whatever is not Metal is CUDA" -- the shape that
        # made a third source arrive as an unexplained NCU parse failure rather than as a
        # refusal naming it. A kind nobody declares is now named in the refusal.
        kind = json.loads(self.artifact_payloads["profile"]).get("kind")
        if kind == "metal_compute_stage_timestamps_v1":
            from .metal_observations import load_metal_profile
            profile = load_metal_profile(self.artifact_payloads["profile"],
                expected_candidate_sha256=self.candidate_sha256, expected_case_id=self.case_id,
                expected_protocol_sha256=self.evaluation_protocol_sha256)
            return {"kind": profile["kind"], **profile["summary"]}
        if kind == "hip_dispatch_activity_v1":
            from .hip_observations import load_hip_profile, hip_attribution_feedback
            return hip_attribution_feedback(load_hip_profile(
                self.artifact_payloads["profile"],
                expected_candidate_sha256=self.candidate_sha256,
                expected_case_id=self.case_id,
                expected_protocol_sha256=self.evaluation_protocol_sha256))
        if kind == 'ncu_program_attribution':
            from .profiler import load_ncu_program_profile
            profile = load_ncu_program_profile(self.artifact_payloads['profile'],
                expected_candidate_sha256=self.candidate_sha256, expected_case_id=self.case_id)
            return {'kind': kind, 'stages': [{key: stage[key] for key in ('stage', 'kernel_name', 'summary')}
                                             for stage in profile['stages']]}
        if kind != "ncu_kernel_attribution":
            raise ValueError(
                f"no attribution source declares profile kind {kind!r}; this reader knows "
                "Nsight Compute, Metal compute-stage timestamps and roctracer activity")
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
class TensorLaunchManifest(WorkloadTensorManifest):
    """Explicit Workload tensor ABI for a module the host launches itself (a cubin or an hsaco)."""

    dynamic_shared_memory_bytes: int
    hidden_null_pointer_parameters: int
    pointer_alignments: Mapping[str, int] = field(default_factory=dict)
    aligned_variant: str | None = None

    abi = 'workload_tensors_v1'
    workload_mismatch = 'sealed launch ABI differs from the selected Workload'

    @classmethod
    def from_dict(cls, document: object) -> 'TensorLaunchManifest':
        from .cuda_manifest import CudaKernelSpec
        fields = {
            'schema_version', 'abi', 'workload_sha256', 'case_id', 'tensor_abi',
            'target', 'kernel_name', 'grid', 'block', 'dynamic_shared_memory_bytes',
            'hidden_null_pointer_parameters',
        }
        if not isinstance(document, Mapping) or document.get('abi') != cls.abi:
            raise ValueError('Workload tensor launch manifest fields differ')
        extra = set(document) - fields
        if (not fields <= set(document) or extra not in (set(), {'pointer_alignments'}, {'aligned_variant'})
                or document['schema_version'] != (2 if extra else 1)):
            raise ValueError('Workload tensor launch manifest fields differ')
        launch = CudaKernelSpec.from_dict({
            key: value for key, value in document.items()
            if key not in {'schema_version', 'abi', 'workload_sha256', 'case_id', 'tensor_abi',
                           'pointer_alignments', 'aligned_variant'}
        })
        rows = document['tensor_abi']
        if (not isinstance(document['workload_sha256'], str)
            or _DIGEST.fullmatch(document['workload_sha256']) is None
            or not isinstance(document['case_id'], str) or not document['case_id']
            or not isinstance(rows, list) or not rows):
            raise ValueError('Workload tensor launch identity differs')
        abi = tensor_abi_rows(rows, dtypes=frozenset({'fp32', 'bf16', 'fp16', 'fp8_e4m3', 'int32'}),
                              ascii_names=False,
                              row_error='Workload tensor launch ABI differs',
                              order_error='Workload tensor launch ABI order differs')
        alignments = document.get('pointer_alignments', {})
        if (not isinstance(alignments, Mapping) or ('pointer_alignments' in extra and not alignments)
                or any(name not in {r[0] for r in abi} or type(value) is not int
                       or value <= 0 or value & (value - 1) for name, value in alignments.items())):
            raise ValueError('sealed pointer alignment contract differs')
        variant = document.get('aligned_variant')
        if 'aligned_variant' in extra and variant != 'aligned':
            raise ValueError('sealed aligned variant must name the sole aligned kernel')
        return cls(document['workload_sha256'], document['case_id'], abi,
                   launch.target, launch.kernel_name, launch.grid, launch.block,
                   launch.dynamic_shared_memory_bytes, launch.hidden_null_pointer_parameters,
                   MappingProxyType(dict(alignments)), variant)

    @classmethod
    def for_workload(cls, workload: WorkloadContract, case_id: str, **launch: object) -> 'TensorLaunchManifest':
        from dataclasses import asdict
        version = 2 if 'pointer_alignments' in launch or 'aligned_variant' in launch else 1
        manifest = cls.from_dict({'schema_version': version, 'abi': cls.abi,
            'workload_sha256': workload.canonical_sha256, 'case_id': case_id,
            'tensor_abi': [{**asdict(t), 'shape': list(t.shape)} for t in workload.tensor_abi(case_id)],
            **launch})
        manifest.check_workload(workload, case_id)
        return manifest

    def check_validation_case(self, workload: WorkloadContract, case_id: str) -> None:
        """Admit another required input distribution without changing the primary seal."""
        self.check_workload(workload, self.case_id)
        validation = workload.document['validation']
        if (validation.get('all_cases_required') is not True
                or self.case_id != validation.get('primary_case') or case_id not in workload.case_ids):
            raise ValueError('validation case must belong to the primary Workload validation contract')
        expected = tuple((t.name, t.shape, t.dtype, t.mode) for t in workload.tensor_abi(case_id))
        if self.tensor_abi != expected:
            raise ValueError('validation case must preserve the sealed primary Workload tensor ABI')

    @property
    def tensors(self) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        dtypes = {'fp32': 'torch.float32', 'bf16': 'torch.bfloat16', 'fp16': 'torch.float16', 'fp8_e4m3': 'torch.float8_e4m3fn', 'int32': 'torch.int32'}
        return tuple((name, shape, dtypes[dtype]) for name, shape, dtype, _ in self.tensor_abi)

    def as_dict(self) -> dict[str, object]:
        specialization = ({'pointer_alignments': dict(self.pointer_alignments)} if self.pointer_alignments
                          else {'aligned_variant': self.aligned_variant} if self.aligned_variant else {})
        return {'schema_version': 2 if specialization else 1, 'abi': self.abi,
            'workload_sha256': self.workload_sha256, 'case_id': self.case_id,
            'tensor_abi': [dict(name=n, shape=list(s), dtype=d, mode=m) for n, s, d, m in self.tensor_abi],
            'target': self.target, 'kernel_name': self.kernel_name, 'grid': list(self.grid),
            'block': list(self.block), 'dynamic_shared_memory_bytes': self.dynamic_shared_memory_bytes,
            'hidden_null_pointer_parameters': self.hidden_null_pointer_parameters, **specialization}

    def check_complete_domain(self):
        if self.pointer_alignments:
            raise ValueError('alignment-restricted leaf cannot replace the complete Workload candidate')

    @property
    def module_count(self):
        return 2 if self.aligned_variant else 1

    @property
    def kernels_per_call(self):
        return 1



def _same_tensor_inputs(before, after):
    """Compare admitted tensor values, not the host sequence container spelling.

    Workloads may materialize array.array while driver snapshots return lists.
    Preserve the existing signed-zero/bit-pattern check and reject any changed
    field, length or value before accepting an unchanged-input observation.
    """
    from array import array
    import struct
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or set(before) != set(after):
        return False
    for name, values in before.items():
        actual = after[name]
        if len(actual) != len(values):
            return False
        if (isinstance(values, array) and isinstance(actual, array)
                and values.typecode == actual.typecode == 'd'):
            # Native double-array comparison keeps NaN != NaN; byte comparison
            # additionally preserves signed zero. Neither makes Python objects or
            # calls struct.pack once per matrix element.
            if values != actual or values.tobytes() != actual.tobytes():
                return False
        elif not all(a == b and struct.pack('>d', float(a)) == struct.pack('>d', float(b))
                     for a, b in zip(values, actual, strict=True)):
            return False
    return True

def compare_tile_outputs(workload, before, expected, observed, after):
    """One comparison owner for fresh and already-recorded tensor launches."""
    import struct
    validation = workload.document['validation']
    comparisons = None
    if validation['comparison'] == 'per_output':
        comparisons = validation.get('outputs')
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(expected):
            raise ValueError('per-output comparison must cover every output exactly')
        for rule in comparisons.values():
            if (not isinstance(rule, Mapping) or set(rule) != {'comparison', 'atol', 'rtol'}
                or rule['comparison'] not in {'bitwise_bf16', 'elementwise_atol_rtol', 'elementwise_atol_rtol_ieee'}
                or any(type(rule[k]) not in {int, float} or not math.isfinite(rule[k]) or rule[k] < 0
                       for k in ('atol', 'rtol'))
                or (rule['comparison'] == 'bitwise_bf16' and (rule['atol'] != 0 or rule['rtol'] != 0))):
                raise ValueError('per-output comparison rule differs')
    mismatch = 0
    maximum_error = 0.0
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        mismatch += 1
    else:
        for name, values in expected.items():
            rule = comparisons[name] if comparisons is not None else validation
            actual = observed[name]
            if not isinstance(actual, (list, tuple)) or len(actual) != len(values):
                mismatch += 1
                continue
            for value, reference in zip(actual, values, strict=True):
                if any(not isinstance(v, (float,int)) or isinstance(v,bool) for v in (value,reference)):
                    mismatch += 1
                    continue
                if not math.isfinite(value) or not math.isfinite(reference):
                    matches = (rule['comparison'] == 'elementwise_atol_rtol_ieee' and
                               ((math.isnan(value) and math.isnan(reference)) or
                                (math.isinf(value) and value == reference)))
                    mismatch += not matches
                    continue
                if not -3.4028234663852886e38 <= value <= 3.4028234663852886e38:
                    mismatch += 1
                    continue
                error = abs(value - reference)
                maximum_error = max(maximum_error, error)
                if rule['comparison'] == 'bitwise_bf16':
                    mismatch += (abs(value) > 3.4028234663852886e38 or struct.pack('>f', value) != struct.pack('>f', reference))
                else:
                    mismatch += error > rule['atol'] + rule['rtol'] * abs(reference)
    unchanged = _same_tensor_inputs(before, after)
    metrics = {'output_mismatches': mismatch, 'max_abs_error': maximum_error, 'inputs_unchanged': unchanged}
    return mismatch == 0 and unchanged, metrics


def _load_cubin(candidate, manifest, admission):
    from .cuda_driver import LoadedCudaCandidate
    return LoadedCudaCandidate.load(
        candidate, candidate.artifact_payloads['cubin'], manifest, admission)


def _load_hsaco(candidate, manifest, admission):
    from .hip_driver import LoadedHipModuleCandidate
    return LoadedHipModuleCandidate.load(
        candidate, candidate.artifact_payloads['hsaco'], manifest, admission.device_arch)


def _load_mcfatbin(candidate, manifest, admission):
    from .metax_driver import LoadedMetaxCandidate
    return LoadedMetaxCandidate.load(candidate, manifest, admission)


# The driver that retains a module for the tensor-tile path, per declared object. A row
# with no loader is refused by the object's name: a Metal binary archive is observed by
# its native observer rather than launched here.
_MODULE_LOADERS = {
    CodeObject.CUBIN: _load_cubin,
    CodeObject.HSACO: _load_hsaco,
    CodeObject.MCFATBIN: _load_mcfatbin,
}


_TORCH_DTYPE_NAMES = {'fp32':'float32','bf16':'bfloat16','fp16':'float16',
                      'fp8_e4m3':'float8_e4m3fn','int32':'int32'}


def load_torch_program(candidate,manifest,arguments,admission,loader):
    """Prepare already materialized public tensors through common Program ownership."""
    import torch
    from .program import LoadedProgram
    def allocate(spec):
        return torch.full(spec.shape,float('nan') if spec.dtype.value!='int32' else -(2**31),
            dtype=getattr(torch,_TORCH_DTYPE_NAMES[spec.dtype.value]),device=arguments[0].device)
    def span(tensor):
        if not tensor.is_contiguous():
            raise ValueError('Program tensors must have contiguous storage')
        return (str(tensor.device),tensor.data_ptr(),tensor.data_ptr()+tensor.numel()*tensor.element_size())
    loaded = LoadedProgram(candidate,manifest,admission,loader,allocate=allocate,
        view=lambda tensor,shape:tensor.view(shape),storage_span=span,
        stream=torch.cuda.current_stream().cuda_stream)
    try:
        return loaded,loaded.prepare_arguments(arguments)
    except BaseException as primary:
        try:
            loaded.close(synchronize=torch.cuda.synchronize)
        except BaseException as teardown:
            from .loaders import LifecycleError
            raise LifecycleError(primary,teardown) from primary
        raise


class LoadedTorchTensorCandidate:
    """One Workload-shaped argument set and admitted module for preflight/timing/postflight."""

    def __init__(self, candidate, manifest, inputs, admission):
        from array import array
        import torch
        self.candidate = candidate
        self.manifest = manifest
        self.admission = admission
        self.inputs = {name: array('d', values) for name, values in inputs.items()}
        dtype_names = _TORCH_DTYPE_NAMES
        dtypes = {}
        for _, _, dtype, _ in manifest.tensor_abi:
            if not hasattr(torch, dtype_names[dtype]):
                raise ValueError(f'framework cannot name required tensor dtype {dtype!r}')
            dtypes[dtype] = getattr(torch, dtype_names[dtype])
        # `cuda:0` is torch's device string for both runtimes: a ROCm build keeps the
        # `torch.cuda` namespace and maps it onto HIP. It reads like a vendor leak and is
        # not one, so it is left alone rather than aliased into a second spelling.
        self.arguments = [
            torch.tensor(inputs[name], dtype=dtypes[dtype], device='cuda:0').reshape(shape)
            if mode == 'input' else torch.full(shape,
                float('nan') if dtype != 'int32' else -(2**31),
                dtype=dtypes[dtype], device='cuda:0')
            for name, shape, dtype, mode in manifest.tensor_abi
        ]
        # The executable the target builds decides which driver retains the module. This
        # named the CUBIN one directly, so the whole tensor-tile evaluation path -- the
        # oracle, the cohorts, the receipts, none of which is CUDA's -- could only ever
        # run a CUDA candidate.
        executable = platform_for(candidate.target).code_object
        loader = _MODULE_LOADERS.get(executable)
        if loader is None:
            raise ValueError(
                f'{candidate.target!r} builds a {executable.value!r}, which this '
                'tensor-tile path has no driver for')
        if candidate.is_program:
            self.loaded,_ = load_torch_program(candidate,manifest,self.arguments,admission,loader)
        elif manifest.aligned_variant:
            from .kernel_bundle import LoadedAlignmentCandidate
            self.loaded = LoadedAlignmentCandidate(candidate, manifest, admission, loader, torch.cuda.synchronize)
        else:
            self.loaded = loader(candidate, manifest, admission)

    @property
    def module_count(self):
        return self.manifest.module_count

    def fresh_argument_sets(self, count):
        """Prepare non-reusable outputs and finish their initialization outside timing."""
        import torch
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('fresh tensor argument count differs')
        sets = [[argument if mode == 'input' else torch.full_like(argument,
                    float('nan') if dtype != 'int32' else -(2**31))
                 for (_, _, dtype, mode), argument in zip(self.manifest.tensor_abi, self.arguments, strict=True)]
                for _ in range(count)]
        if self.candidate.is_program:
            for arguments in sets:
                self.loaded.prepare_arguments(arguments)
        torch.cuda.synchronize()
        return sets

    def release_argument_sets(self, sets):
        if self.candidate.is_program:
            import torch
            torch.cuda.synchronize()
            for arguments in sets:
                self.loaded.release_arguments(arguments)

    def launch(self, arguments=None):
        import torch
        self.loaded.launch(self.arguments if arguments is None else arguments, tensor_contract=self.manifest,
                           stream=torch.cuda.current_stream().cuda_stream)

    def snapshot(self, arguments=None):
        from array import array
        import ctypes
        import torch
        arguments = self.arguments if arguments is None else arguments
        observed = {name: value.cpu().reshape(-1).tolist() for (name, _, _, mode), value
                    in zip(self.manifest.tensor_abi, arguments, strict=True) if mode == 'output'}
        after = {}
        for (name, _, _, mode), value in zip(self.manifest.tensor_abi, arguments, strict=True):
            if mode != 'input':
                continue
            host = value.detach().to(device='cpu', dtype=torch.float64).contiguous()
            values = array('d')
            values.frombytes(ctypes.string_at(host.data_ptr(), host.numel() * host.element_size()))
            after[name] = values
        return observed, after

    @property
    def validation_inputs(self):
        """Private pre-launch CPU copy, checked against each materialized case at preflight."""
        return self.inputs

    def launch_tensors(self, candidate, manifest, inputs):
        from dataclasses import asdict
        if (candidate.canonical_sha256 != self.candidate.canonical_sha256
            or manifest.canonical_sha256 != self.manifest.canonical_sha256
            or not _same_tensor_inputs(self.inputs, inputs)):
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
