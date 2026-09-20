"""Sealed alignment variants; child requirements remain enforced after extraction."""
from __future__ import annotations

import base64
import json
from collections.abc import Mapping

from open_cake_ir.serialization import canonical_json_bytes
from .loaders import LifecycleError


def pack_candidates(candidates):
    from .paired import candidate_identity
    return canonical_json_bytes({'schema_version': 1, 'kernels': {
        name: {'identity': candidate_identity(candidate),
               'payloads': {role: base64.b64encode(payload).decode('ascii')
                            for role, payload in candidate.artifact_payloads.items()}}
        for name, candidate in candidates.items()}})


def unpack_candidates(parent):
    from .paired import candidate_from_identity
    return unpack_candidate_bundle(parent.artifact_payloads['kernel_bundle'], target=parent.target)


def unpack_candidate_bundle(payload, *, target):
    from .paired import candidate_from_identity
    raw = json.loads(payload)
    if (not isinstance(raw, Mapping) or set(raw) != {'schema_version', 'kernels'}
            or raw['schema_version'] != 1 or not isinstance(raw['kernels'], Mapping)
            or not raw['kernels']):
        raise ValueError('sealed kernel bundle fields differ')
    result = {}
    for name, record in raw['kernels'].items():
        if (not isinstance(name, str) or not name
                or not isinstance(record, Mapping) or set(record) != {'identity', 'payloads'}
                or not isinstance(record['payloads'], Mapping)):
            raise ValueError('sealed kernel bundle entry differs')
        payloads = {role: base64.b64decode(value, validate=True) for role, value in record['payloads'].items()}
        child = candidate_from_identity(record['identity'], payloads)
        if child.target != target:
            raise ValueError('sealed kernel bundle changes the exact target')
        result[name] = child
    return result


def alignment_component(parent, manifest):
    from .core import TensorLaunchManifest
    children = unpack_candidates(parent)
    if set(children) != {'aligned'} or manifest.aligned_variant != 'aligned':
        raise ValueError('alignment bundle must contain exactly the declared aligned kernel')
    child = children['aligned']
    other = TensorLaunchManifest.from_dict(json.loads(child.artifact_payloads['launch_manifest']))
    if (other.aligned_variant or not other.pointer_alignments
            or other.tensor_abi != manifest.tensor_abi or other.target != manifest.target
            or other.workload_sha256 != manifest.workload_sha256 or other.case_id != manifest.case_id
            or other.kernel_name != manifest.kernel_name or child.entry_point != parent.entry_point
            or other.grid != manifest.grid
            or other.hidden_null_pointer_parameters != manifest.hidden_null_pointer_parameters
            or child.candidate_sha256 != parent.candidate_sha256
            or child.launch_spec_sha256 != other.canonical_sha256
            or 'kernel_bundle' in child.artifact_roles):
        raise ValueError('aligned kernel must preserve source, target and Workload ABI')
    # Triton's asm['source'] is already compiler IR: pointer attributes and
    # temporary source locations legitimately differ between compilations.
    # Bind the authoring input, while sealing each compiler output separately.
    source_roles = ('lowered_source', 'authored_source')
    if not any(role in parent.artifact_roles for role in source_roles):
        raise ValueError('alignment bundle requires its declared source program')
    for role in source_roles:
        if child.artifact_roles.get(role) != parent.artifact_roles.get(role):
            raise ValueError('aligned kernel changes its source program')
    return child, other


class LoadedAlignmentCandidate:
    """Two complete implementations, selected from actual addresses on every call."""
    def __init__(self, parent, manifest, admission, loader, synchronize):
        child, child_manifest = alignment_component(parent, manifest)
        self.manifest = manifest
        self.aligned_manifest = child_manifest
        self.generic = loader(parent, manifest, admission)
        try:
            self.aligned = loader(child, child_manifest, admission)
        except BaseException as primary:
            try:
                self.generic.close(synchronize=synchronize)
            except BaseException as teardown:
                raise LifecycleError(primary, teardown) from primary
            raise
        self.dispatch_counts = {'generic': 0, 'aligned': 0}
        self.last_variant = None
        self.module_count = 2

    @property
    def closed(self):
        return self.generic.closed and self.aligned.closed

    @property
    def launch_calls(self):
        return self.generic.launch_calls + self.aligned.launch_calls

    @property
    def resources(self):
        return {'implementations': {'generic': self.generic.resources, 'aligned': self.aligned.resources},
                'dispatch_counts': dict(self.dispatch_counts), 'last_variant': self.last_variant}

    def launch(self, arguments, *, tensor_contract, stream):
        if tensor_contract.canonical_sha256 != self.manifest.canonical_sha256:
            raise ValueError('aligned program tensor contract differs from the sealed parent')
        pointers = {name: value.data_ptr() for (name, _, _, _), value
                    in zip(self.manifest.tensor_abi, arguments, strict=True)}
        aligns = self.aligned_manifest.pointer_alignments
        selected = ('aligned' if all(type(pointers.get(name)) is int and pointers[name] > 0
                    and pointers[name] % value == 0 for name, value in aligns.items()) else 'generic')
        # Both drivers still enforce shape/dtype/device/alias checks; the aligned
        # leaf additionally enforces its own requirements on the addresses it uses.
        contract = self.aligned_manifest if selected == 'aligned' else self.manifest
        getattr(self, selected).launch(arguments, tensor_contract=contract, stream=stream)
        self.last_variant = selected
        self.dispatch_counts[selected] += 1

    def close(self, *, synchronize):
        if self.closed:
            raise ValueError('alignment candidate modules are already closed')
        errors = []
        for loaded in (self.aligned, self.generic):
            if loaded.closed:
                continue
            try:
                loaded.close(synchronize=synchronize)
            except BaseException as error:
                errors.append(error)
        if len(errors) > 1:
            raise LifecycleError(errors[0], *errors[1:]) from errors[0]
        if errors:
            raise errors[0]
