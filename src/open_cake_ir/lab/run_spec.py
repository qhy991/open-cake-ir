"""One frozen execution authority, independent of Study design and allocation.

A Study can bind an assignment before execution. Engineering Runs omit it. Neither
condition names nor run ids select an authoring representation or backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Mapping

from ._documents import _name, _digest, _object, _canonical_json_bytes
from ._policies import _matched_evidence_policy_version
from .endpoints import endpoint_policy
from .run_controls import validate_run_controls
from .reference_access import validate_declarations


@dataclass(frozen=True)
class RunSpecification:
    _bytes: bytes

    @classmethod
    def from_dict(cls, value):
        document = _object(value, 'run')
        fields = {'schema_version', 'run_id', 'sequence', 'assignment', 'workload',
                  'compiler_revision', 'authoring', 'budget', 'run_protocol',
                  'agent_interface', 'evidence_policy', 'evaluation_protocol',
                  'execution', 'endpoint_policy', 'reference_inputs'}
        if set(document) != fields or type(document['schema_version']) is not int or document['schema_version'] != 1:
            raise ValueError('Run specification fields or schema_version differ')
        run_id = _name(document['run_id'], 'run.run_id')
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
            raise ValueError('Run id must be a safe artifact name')
        if type(document['sequence']) is not int or document['sequence'] < 1:
            raise ValueError('Run sequence must be a positive integer')
        assignment = document['assignment']
        if assignment is not None:
            assignment = _object(assignment, 'run.assignment')
            if set(assignment) != {'study_id', 'study_sha256', 'condition_id'}:
                raise ValueError('Run assignment fields differ')
            for field in ('study_id', 'condition_id'):
                _name(assignment[field], f'run.assignment.{field}')
            _digest(assignment['study_sha256'], 'run.assignment.study_sha256')
        workload = _object(document['workload'], 'run.workload')
        if set(workload) != {'workload_id', 'path', 'canonical_sha256'}:
            raise ValueError('Run Workload reference differs')
        _name(workload['workload_id'], 'run.workload.workload_id')
        _name(workload['path'], 'run.workload.path')
        _digest(workload['canonical_sha256'], 'run.workload.canonical_sha256')
        compiler = _object(document['compiler_revision'], 'run.compiler_revision')
        if set(compiler) != {'revision_id', 'path'}:
            raise ValueError('Run Compiler reference differs')
        for field in compiler:
            _name(compiler[field], f'run.compiler_revision.{field}')
        authoring = _object(document['authoring'], 'run.authoring')
        if authoring.get('environment_kind') not in {'open_cake', 'direct_cuda', 'native_triton', 'native_cute_dsl'}:
            raise ValueError('Run Authoring Environment kind differs')
        if 'compiler_revision' in authoring and authoring['compiler_revision'] != compiler:
            raise ValueError('authoring Compiler reference differs from the Run Compiler')
        validate_declarations({'author': authoring})
        references = _object(document['reference_inputs'], 'run.reference_inputs')
        expected_references = {'baseline_schedule'} if authoring['environment_kind'] in {'native_triton', 'native_cute_dsl'} else set()
        if set(references) != expected_references:
            raise ValueError('Run reference inputs differ from its authoring environment')
        for reference in references.values():
            if not isinstance(reference, Mapping) or set(reference) != {'path', 'canonical_sha256'}:
                raise ValueError('Run baseline Schedule reference differs')
            _name(reference['path'], 'run.reference.path')
            _digest(reference['canonical_sha256'], 'run.reference.canonical_sha256')
        _object(authoring.get('provider'), 'run.authoring.provider')
        validate_run_controls(document)
        interface = _object(document['agent_interface'], 'run.agent_interface')
        if interface != {'schema_version': 1, 'kind': 'task_agents_ralph_v1'}:
            raise ValueError('Run author interface differs')
        _matched_evidence_policy_version(_object(document['evidence_policy'], 'run.evidence_policy'), 'run.evidence_policy')
        evaluation = _object(document['evaluation_protocol'], 'run.evaluation_protocol')
        _name(evaluation.get('case_id'), 'run.evaluation_protocol.case_id')
        execution = _object(document['execution'], 'run.execution')
        _name(execution.get('target'), 'run.execution.target')
        executor = _object(execution.get('executor_revision'), 'run.execution.executor_revision')
        if set(executor) != {'executor_id', 'path'}:
            raise ValueError('Run Executor reference must be resolved')
        for field in executor:
            _name(executor[field], f'run.execution.executor_revision.{field}')
        policy = document['endpoint_policy']
        endpoint_policy({} if policy is None else {'endpoint_policy': policy})
        return cls(_canonical_json_bytes(document))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_bytes()))

    @property
    def document(self):
        return json.loads(self._bytes)

    @property
    def canonical_sha256(self):
        return sha256(self._bytes).hexdigest()

    @property
    def run_id(self):
        return self.document['run_id']

    @property
    def condition_id(self):
        assignment = self.document['assignment']
        return assignment['condition_id'] if assignment is not None else self.run_id

    @property
    def environment_kind(self):
        return self.document['authoring']['environment_kind']

    @property
    def terminal_policy(self):
        value = self.document['endpoint_policy']
        return {} if value is None else {'endpoint_policy': value}


@dataclass(frozen=True)
class RunRef:
    specification: RunSpecification
    evidence_root: Path
