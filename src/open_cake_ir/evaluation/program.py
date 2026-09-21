"""Sealed same-stream Programs at the common Evaluation boundary.

The Compiler Program is the only graph. Its stages reference independently sealed
single-kernel artifacts; this module binds those artifacts and owns their lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from threading import Lock
from collections.abc import Mapping

from open_cake_ir.compiler.ir import Program, BufferMode, MemorySpace
from open_cake_ir.serialization import canonical_json_bytes
from .launch_manifest import WorkloadTensorManifest

PROGRAM_ROLES = frozenset({'launch_manifest', 'program_bundle'})


def program_tensor_abi(program):
    return tuple((name, program.tensors[name].shape, program.tensors[name].dtype.value, mode)
                 for mode, names in (('input', program.inputs), ('output', program.outputs))
                 for name in names)


def single_kernel_lowering(lowered):
    """Use the existing kernel ABI when composition requires no runtime mapping.

    No stage, entry point, binding or tensor is renamed. The authored Program remains
    the candidate identity; this only selects its executable handoff representation.
    Views and nonidentity bindings need the ordered Program adapter instead.
    """
    program = lowered.program
    if len(program.stages) != 1:
        return None
    stage = program.stages[0]
    if (set(program.tensors) != set(program.inputs) | set(program.outputs)
        or any(name != binding.tensor or binding.singleton_view for name,binding in stage.bindings.items())
        or stage_abi(stage) != program_tensor_abi(program)):
        return None
    return lowered.lowerings[0]


def admit_program_execution(target, *, timing=False, attribution=False):
    """Admit execution separately from a complete Program's measurement coverage.

    HIP and MACA module drivers share ordered execution, but the optimization Run
    instruments still describe one dispatch. Standalone MACA Program attribution
    has its own source; it does not grant admission to the Run's measurement loop.
    """
    from open_cake_ir.compiler.target import CodeObject
    from .platforms import platform_for
    code_object = platform_for(target).code_object
    if code_object not in {CodeObject.CUBIN, CodeObject.HSACO, CodeObject.MCFATBIN}:
        raise ValueError(f'ordered Program execution is not implemented for {code_object.value}')
    if code_object is not CodeObject.CUBIN and (timing or attribution):
        purpose = 'attribution' if attribution else 'timing'
        raise ValueError(f'ordered Program {purpose} is not implemented for {code_object.value}; '
                         'the existing instrument covers one dispatch')


@dataclass(frozen=True)
class ProgramLaunchManifest:
    workload_sha256: str
    case_id: str
    program: Program
    lowered_sources: Mapping[str, str]

    abi = 'ordered_program_v1'
    workload_mismatch = 'sealed Program public ABI differs from the selected Workload'

    @classmethod
    def from_dict(cls, document):
        if (not isinstance(document, Mapping)
            or set(document) != {'schema_version', 'abi', 'workload_sha256', 'case_id', 'program', 'lowered_sources'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['abi'] != cls.abi):
            raise ValueError('Program launch manifest fields differ')
        digest = document['workload_sha256']
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or not isinstance(document['case_id'], str) or not document['case_id']):
            raise ValueError('Program Workload binding differs')
        program = Program.from_dict(document['program'])
        sources = document['lowered_sources']
        if (not isinstance(sources, Mapping) or set(sources) != {stage.name for stage in program.stages}
            or any(not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                   for value in sources.values())):
            raise ValueError('Program lowering references must bind every stage')
        return cls(digest, document['case_id'], program, MappingProxyType(dict(sources)))

    def as_dict(self):
        return {'schema_version': 1, 'abi': self.abi, 'workload_sha256': self.workload_sha256,
                'case_id': self.case_id, 'program': self.program.document, 'lowered_sources': dict(self.lowered_sources)}

    @property
    def canonical_sha256(self):
        return sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    @property
    def target(self):
        return self.program.target

    @property
    def kernel_name(self):
        # A logical candidate identity, never presented as a GPU function name.
        return self.program.program_id

    @property
    def tensor_abi(self):
        return program_tensor_abi(self.program)

    def check_workload(self, workload, case_id):
        WorkloadTensorManifest.check_workload(self, workload, case_id)

    def check_validation_case(self, workload, case_id):
        from .core import TensorLaunchManifest
        TensorLaunchManifest.check_validation_case(self, workload, case_id)

    def check_complete_domain(self):
        # Every component is checked by program_components. Program inputs do not
        # declare an address-alignment or dispatcher restriction.
        pass

    @property
    def module_count(self):
        return len(self.program.stages)

    @property
    def kernels_per_call(self):
        return len(self.program.stages)


def stage_abi(stage):
    """Compiled argument order follows the complete Schedule's global declarations."""
    return tuple((b.name, b.shape, b.dtype.value, b.mode.value)
                 for b in stage.schedule.buffers if b.space is MemorySpace.GLOBAL)


def check_triton_launch_record(candidate,manifest,source_sha256):
    """Bind the physical launch to the Triton builder's existing compile record."""
    report = json.loads(candidate.artifact_payloads.get('stage_compilation',b'null'))
    expected = {'schema_version':1,'kind':'triton_stage_compilation',
        'source_sha256':source_sha256,'target':candidate.target,
        'kernel_name':candidate.entry_point,'threads_per_cta':manifest.block[0],
        'dynamic_shared_memory_bytes':manifest.dynamic_shared_memory_bytes,
        'hidden_null_pointer_parameters':manifest.hidden_null_pointer_parameters,'grid':list(manifest.grid)}
    if report != expected or manifest.block[1:] != (1,1):
        raise ValueError('Program kernel launch differs from compiler metadata')


def program_components(candidate):
    """Validate the complete executable handoff without loading a module."""
    from .core import TensorLaunchManifest
    from .artifacts import required_build_roles
    from .kernel_bundle import unpack_candidate_bundle
    if set(candidate.artifact_roles) != PROGRAM_ROLES or set(candidate.artifact_payloads) != PROGRAM_ROLES:
        raise ValueError('Program candidate requires its complete sealed bundle')
    manifest = ProgramLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
    if (candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name
        or candidate.launch_spec_sha256 != manifest.canonical_sha256):
        raise ValueError('Program candidate and launch manifest differ')
    admit_program_execution(candidate.target)
    children = unpack_candidate_bundle(candidate.artifact_payloads['program_bundle'], target=candidate.target)
    if set(children) != {stage.name for stage in manifest.program.stages}:
        raise ValueError('Program bundle must bind every stage exactly once')
    manifests = {}
    for stage in manifest.program.stages:
        child = children[stage.name]
        if not (required_build_roles(child.target) | {'lowered_source', 'stage_compilation'}) <= set(child.artifact_roles):
            raise ValueError(f'Program stage {stage.name!r} build evidence is incomplete')
        if child.is_program or 'kernel_bundle' in child.artifact_roles:
            raise ValueError('Program stages require single-kernel artifacts without dispatch variants')
        child_manifest = TensorLaunchManifest.from_dict(json.loads(child.artifact_payloads['launch_manifest']))
        if (child.candidate_sha256 != candidate.candidate_sha256
            or child_manifest.workload_sha256 != manifest.workload_sha256
            or child_manifest.case_id != manifest.case_id
            or child_manifest.tensor_abi != stage_abi(stage)
            or child_manifest.target != stage.schedule.target
            or child_manifest.pointer_alignments or child_manifest.aligned_variant
            or child.entry_point != child_manifest.kernel_name
            or child.launch_spec_sha256 != child_manifest.canonical_sha256
            or child.artifact_roles.get('lowered_source') != manifest.lowered_sources[stage.name]):
            raise ValueError(f'Program stage {stage.name!r} artifact or ABI binding differs')
        check_triton_launch_record(child,child_manifest,manifest.lowered_sources[stage.name])
        manifests[stage.name] = child_manifest
    return manifest, children, manifests


def seal_program_candidate(lowered, children, *, candidate_sha256, workload, case_id):
    from .core import LaunchableCandidate
    from .kernel_bundle import pack_candidates
    lowered.validate_binding()
    manifest = ProgramLaunchManifest(workload.canonical_sha256, case_id, lowered.program,
        MappingProxyType({stage.name: lowering.source_sha256 for stage, lowering in zip(lowered.program.stages, lowered.lowerings, strict=True)}))
    manifest.check_workload(workload, case_id)
    if set(children) != {stage.name for stage in lowered.program.stages}:
        raise ValueError('Program build did not produce every stage')
    for stage, lowering in zip(lowered.program.stages, lowered.lowerings, strict=True):
        child = children[stage.name]
        child_manifest = json.loads(child.artifact_payloads['launch_manifest'])
        requirements = lowering.toolchain_requirements
        if (child.artifact_roles.get('lowered_source') != lowering.source_sha256
            or child.target != lowering.target
            or child.entry_point != requirements['kernel_entry_point']
            or child_manifest.get('grid') != list(requirements['grid'])):
            raise ValueError(f'Program build replaced stage {stage.name!r} lowering')
    payloads = {'launch_manifest': canonical_json_bytes(manifest.as_dict()),
                'program_bundle': pack_candidates(children)}
    return LaunchableCandidate(candidate_sha256, manifest.target, manifest.kernel_name,
        {role: sha256(payload).hexdigest() for role, payload in payloads.items()}, manifest.canonical_sha256, payloads)


class LoadedProgram:
    """Retain all modules and prepared storage; one call launches the ordered stages."""

    def __init__(self, candidate, manifest, admission, loader, *, allocate, view, check_tensor, storage_span, stream):
        checked, children, manifests = program_components(candidate)
        if checked.as_dict() != manifest.as_dict():
            raise ValueError('loaded Program manifest differs')
        self.manifest = checked
        self._allocate, self._view, self._span = allocate, view, storage_span
        self._check_tensor = check_tensor
        self._stream = stream
        self._prepared = {}
        self._lock = Lock()
        self._children = {}
        self._manifests = manifests
        self._stage_buffers = {stage.name: {b.name: b for b in stage.schedule.buffers}
                               for stage in checked.program.stages}
        try:
            for stage in checked.program.stages:
                self._children[stage.name] = loader(children[stage.name], manifests[stage.name], admission)
        except BaseException as primary:
            cleanup = None
            for child in reversed(tuple(self._children.values())):
                try:
                    child.close(synchronize=lambda: None)
                except BaseException as error:
                    from .loaders import LifecycleError
                    cleanup = error if cleanup is None else LifecycleError(cleanup, error)
            if cleanup is not None:
                raise primary from cleanup
            raise

    def prepare_arguments(self, arguments):
        """Allocate and bind each argument set before entering any timed interval."""
        program = self.manifest.program
        public = program.inputs + program.outputs
        if len(arguments) != len(public):
            raise ValueError('Program argument count differs')
        tensors = dict(zip(public, arguments, strict=True))
        for name, spec in program.tensors.items():
            if name not in tensors:
                tensors[name] = self._allocate(spec)
        spans = []
        for name, tensor in tensors.items():
            self._check_tensor(tensor, program.tensors[name])
            device, start, end = self._span(tensor)
            if (type(start) is not int or type(end) is not int or start <= 0
                or end - start != program.tensors[name].nbytes
                or spans and device != spans[0][0]
                or any(device == d and start < e and s < end for d, s, e in spans)):
                raise ValueError(f'Program tensor {name!r} storage differs or overlaps')
            spans.append((device, start, end))
        stage_arguments = {}
        for stage in program.stages:
            args = []
            for local, shape, dtype, mode in self._manifests[stage.name].tensor_abi:
                binding = stage.bindings[local]
                value = tensors[binding.tensor]
                if binding.singleton_view:
                    value = self._view(value, shape)
                    if self._span(value) != self._span(tensors[binding.tensor]):
                        raise ValueError('Program singleton view copied or changed storage')
                self._check_tensor(value, self._stage_buffers[stage.name][local])
                args.append(value)
            stage_arguments[stage.name] = args
        self._prepared[id(arguments)] = (arguments, tuple(id(value) for value in arguments), tensors,
                                        stage_arguments, tuple(spans))
        return MappingProxyType(tensors)

    @property
    def launch_calls(self):
        return sum(child.launch_calls for child in self._children.values())

    @property
    def closed(self):
        return all(child.closed for child in self._children.values())

    @property
    def resources(self):
        return {'kind': 'ordered_program', 'stages': {name: child.resources for name, child in self._children.items()}}

    def release_arguments(self, arguments):
        """Drop one fully checked cohort's output/intermediate ownership."""
        with self._lock:
            record = self._prepared.get(id(arguments))
            if record is None or record[0] is not arguments:
                raise ValueError('Program argument set is not retained')
            del self._prepared[id(arguments)]

    def launch(self, arguments, *, tensor_contract, stream, boundary=None):
        with self._lock:
            self._launch(arguments, tensor_contract=tensor_contract, stream=stream,boundary=boundary)

    def _launch(self, arguments, *, tensor_contract, stream,boundary=None):
        if tensor_contract.canonical_sha256 != self.manifest.canonical_sha256 or stream != self._stream:
            raise ValueError('Program launch contract or ordered stream differs')
        prepared = self._prepared.get(id(arguments))
        if prepared is None or prepared[0] is not arguments or prepared[1] != tuple(id(value) for value in arguments):
            raise ValueError('Program arguments must be prepared outside the timed interval')
        # A tensor object can be rebound in place after preparation. Recheck the
        # entire argument graph before the first dispatch, including retained views.
        for (name, tensor), span in zip(prepared[2].items(), prepared[4], strict=True):
            self._check_tensor(tensor, self.manifest.program.tensors[name])
            if self._span(tensor) != span:
                raise ValueError('Program tensor storage changed after preparation')
        for stage in self.manifest.program.stages:
            for (local, _, _, _), tensor in zip(self._manifests[stage.name].tensor_abi,
                                               prepared[3][stage.name], strict=True):
                self._check_tensor(tensor, self._stage_buffers[stage.name][local])
                if self._span(tensor) != self._span(prepared[2][stage.bindings[local].tensor]):
                    raise ValueError('Program stage view storage changed after preparation')
        for stage in self.manifest.program.stages:
            if boundary is not None:
                boundary(stage.name,'before')
            self._children[stage.name].launch(prepared[3][stage.name],
                tensor_contract=self._manifests[stage.name], stream=stream)
            if boundary is not None:
                boundary(stage.name,'after')

    def close(self, *, synchronize):
        with self._lock:
            self._close(synchronize=synchronize)

    def _close(self, *, synchronize):
        from .loaders import LifecycleError
        failure = None
        try:
            synchronize()
        except BaseException as error:
            failure = error
        for child in reversed(tuple(self._children.values())):
            try:
                child.close(synchronize=lambda: None)
            except BaseException as error:
                failure = error if failure is None else LifecycleError(failure, error)
        self._prepared.clear()
        if failure is not None:
            raise failure
