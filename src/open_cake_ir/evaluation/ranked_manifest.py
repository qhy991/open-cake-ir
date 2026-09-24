"""Pre-seal launch binding for a distributed ranked-mailbox candidate.

The Workload owns tensor semantics and placement; the Compiler owns complete
math, effects and generated source. This manifest checks their ABI and one
case's c/K/steal plan before any build or GPU run. It is not itself a
LaunchableCandidate or an acceptance decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping

from open_cake_ir.compiler import (LoweredRankedMailbox, Program,
                                   RankedMailboxEffects, Schedule)
from open_cake_ir.compiler.ir import DType
from open_cake_ir.serialization import canonical_json_bytes


_FIELDS = {'schema_version', 'abi', 'workload_sha256', 'case_id', 'target',
           'compiler_revision_id', 'compiler_commit', 'entry_point',
           'effects', 'local_program', 'combine', 'rank_inputs',
           'tensor_bindings', 'output_binding', 'plans', 'grid_per_rank',
           'block', 'cooperative_grid', 'state_reset',
           'peer_pair_runtime_check', 'input_domain_runtime_check'}


@dataclass(frozen=True)
class RankedMailboxLaunchManifest:
    """A detached canonical document; no executable bytes are claimed."""

    _bytes: bytes
    abi = 'ranked_mailbox_v1'

    @classmethod
    def from_dict(cls, document):
        if (not isinstance(document, Mapping) or set(document) != _FIELDS
                or document.get('schema_version') != 1
                or document.get('abi') != cls.abi):
            raise ValueError('ranked launch manifest fields or ABI differ')
        digest = document['workload_sha256']
        commit = document['compiler_commit']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)
                or not isinstance(commit, str) or len(commit) != 40
                or any(char not in '0123456789abcdef' for char in commit)
                or not isinstance(document['case_id'], str)
                or not document['case_id']
                or not isinstance(document['compiler_revision_id'], str)
                or not document['compiler_revision_id']
                or not isinstance(document['target'], str) or not document['target']):
            raise ValueError('ranked launch Workload or Compiler identity differs')
        effects = RankedMailboxEffects.from_dict(document['effects'])
        local = Program.from_dict(document['local_program'])
        combine = Schedule.from_dict(document['combine'])
        analysis = effects.analyze(local, combine)
        if (document['target'] != local.target
                or document['entry_point'] != effects.lowering.entry_point):
            raise ValueError('ranked launch target or entry differs from complete math')
        rows = document['rank_inputs']
        if not isinstance(rows, list) or not rows:
            raise ValueError('ranked launch per-rank input ABI differs')
        specs = {}
        for row in rows:
            if (not isinstance(row, Mapping) or set(row) != {'name', 'shape', 'dtype'}
                    or not isinstance(row['name'], str) or not row['name']
                    or row['name'] in specs
                    or not isinstance(row['shape'], list) or not row['shape']
                    or any(type(extent) is not int or extent <= 0
                           for extent in row['shape'])):
                raise ValueError('ranked launch input row differs')
            try:
                dtype = DType(row['dtype'])
            except (ValueError, TypeError):
                raise ValueError('ranked launch input dtype differs') from None
            specs[row['name']] = (tuple(row['shape']), dtype.value)
        bindings = document['tensor_bindings']
        if (not isinstance(bindings, Mapping) or not bindings
                or any(not isinstance(name, str) or not name
                       or not isinstance(local_name, str) or not local_name
                       for name, local_name in bindings.items())
                or len(set(bindings.values())) != len(bindings)
                or set(bindings.values()) != set(specs)):
            raise ValueError('ranked launch Workload inputs need a bijective rank ABI binding')
        if document['output_binding'] != 'mailbox_output':
            raise ValueError('ranked launch output needs the declared mailbox view')
        grids = document['grid_per_rank']
        if (not isinstance(grids, list) or len(grids) != analysis.world_size
                or any(type(grid) is not list or len(grid) != 3
                       or any(type(extent) is not int or extent <= 0
                              for extent in grid)
                       for grid in grids)
                or type(grids[0][0]) is not int or grids[0][0] < 2
                or any(grid != [grids[0][0], 1, 1] for grid in grids)
                or type(document['block']) is not list
                or any(type(extent) is not int for extent in document['block'])
                or document['block'] != [32, 1, 1]
                or document['cooperative_grid'] is not True
                or document['state_reset'] != 'zero_all_rank_mailboxes_before_launch'
                or document['peer_pair_runtime_check'] is not True
                or document['input_domain_runtime_check'] is not True):
            raise ValueError('ranked launch target, grid or runtime gate differs')
        plans = document['plans']
        if (not isinstance(plans, list) or len(plans) != analysis.world_size
                or [row.get('rank') if isinstance(row, Mapping) else None
                    for row in plans] != list(range(analysis.world_size))):
            raise ValueError('ranked launch needs one ordered plan per rank')
        for row in plans:
            if (not isinstance(row, Mapping) or set(row) != {
                    'rank', 'communication_ctas', 'chunks', 'steal_budget'}
                    or type(row['rank']) is not int):
                raise ValueError('ranked launch c/K/steal fields differ')
            c, k, budget = (row[name] for name in
                            ('communication_ctas', 'chunks', 'steal_budget'))
            if (any(type(value) is not int for value in (c, k, budget))
                    or not 0 < c < grids[row['rank']][0]
                    or not 1 <= k <= analysis.items_per_rank
                    or not 0 <= budget <= analysis.compute_task_capacity_per_rank):
                raise ValueError(f'rank {row["rank"]} c/K/steal plan is out of bounds')
        return cls(canonical_json_bytes(document))

    @classmethod
    def from_lowered(cls, lowered: LoweredRankedMailbox, *, combine_document,
                     workload, case_id: str, tensor_bindings: Mapping[str, str],
                     plans):
        lowered.validate_binding()
        if Schedule.from_dict(combine_document) != lowered.combine_schedule:
            raise ValueError('ranked launch combine document differs from lowering')
        req = lowered.toolchain_requirements
        document = {
            'schema_version': 1, 'abi': cls.abi,
            'workload_sha256': workload.canonical_sha256,
            'case_id': case_id, 'target': lowered.local_program.target,
            'compiler_revision_id': lowered.compiler_revision_id,
            'compiler_commit': req.get('compiler_commit'),
            'entry_point': lowered.effects.lowering.entry_point,
            'effects': lowered.effects.document,
            'local_program': lowered.local_program.document,
            'combine': dict(combine_document),
            'rank_inputs': req.get('rank_inputs'),
            'tensor_bindings': dict(tensor_bindings),
            'output_binding': 'mailbox_output',
            'plans': list(plans),
            'grid_per_rank': req.get('grid_per_rank'),
            'block': req.get('block'),
            'cooperative_grid': req.get('cooperative_grid'),
            'state_reset': req.get('state_reset'),
            'peer_pair_runtime_check': req.get('peer_pair_runtime_check'),
            'input_domain_runtime_check': req.get('input_domain_runtime_check'),
        }
        manifest = cls.from_dict(document)
        manifest.check_lowered(lowered)
        manifest.check_workload(workload, case_id)
        return manifest

    def as_dict(self):
        return json.loads(self._bytes)

    def check_lowered(self, lowered: LoweredRankedMailbox):
        lowered.validate_binding()
        document = self.as_dict()
        req = lowered.toolchain_requirements
        if (document['target'] != lowered.local_program.target
                or document['compiler_revision_id'] != lowered.compiler_revision_id
                or document['compiler_commit'] != req.get('compiler_commit')
                or document['entry_point'] != lowered.effects.lowering.entry_point
                or document['effects'] != lowered.effects.document
                or Program.from_dict(document['local_program']) != lowered.local_program
                or Schedule.from_dict(document['combine']) != lowered.combine_schedule
                or document['rank_inputs'] != req.get('rank_inputs')
                or document['grid_per_rank'] != req.get('grid_per_rank')
                or document['block'] != req.get('block')
                or document['cooperative_grid'] != req.get('cooperative_grid')
                or document['state_reset'] != req.get('state_reset')
                or document['peer_pair_runtime_check']
                != req.get('peer_pair_runtime_check')
                or document['input_domain_runtime_check']
                != req.get('input_domain_runtime_check')):
            raise ValueError('ranked launch manifest differs from exact Compiler lowering')
        return None

    def check_workload(self, workload, case_id: str):
        document = self.as_dict()
        effects = RankedMailboxEffects.from_dict(document['effects'])
        local = Program.from_dict(document['local_program'])
        combine = Schedule.from_dict(document['combine'])
        analysis = effects.analyze(local, combine)
        if (not workload.requires_distributed_execution
                or document['workload_sha256'] != workload.canonical_sha256
                or document['case_id'] != case_id
                or document['target'] != workload.target):
            raise ValueError('ranked launch differs from the frozen distributed Workload')
        semantics = workload.document['semantics']
        topology = semantics['execution_topology']
        placement = semantics['tensor_placement']
        if (topology != {'kind': 'expert_parallel',
                         'world_size': analysis.world_size}
                or semantics['expert_placement'] != 'contiguous_equal_ranges_by_rank'
                or semantics['top_k'] != analysis.routes_per_item):
            raise ValueError('ranked launch topology or route placement differs')
        abi = workload.tensor_abi(case_id)
        inputs = {row.name: row for row in abi if row.mode == 'input'}
        outputs = [row for row in abi if row.mode == 'output']
        bindings = document['tensor_bindings']
        specs = {row['name']: row for row in document['rank_inputs']}
        if (set(inputs) != set(bindings) or len(outputs) != 1
                or placement.get(outputs[0].name) != 'rank_sharded_axis_0'
                or outputs[0].dtype != 'bf16'
                or outputs[0].shape != (analysis.world_size,
                                        analysis.items_per_rank,
                                        analysis.feature_width)):
            raise ValueError('ranked launch public output or input ABI differs')
        for public, row in inputs.items():
            spec = specs[bindings[public]]
            local_shape = tuple(spec['shape'])
            kind = placement.get(public)
            if kind == 'rank_sharded_axis_0':
                expected = (analysis.world_size,) + local_shape
            elif kind == 'expert_sharded_axis_0':
                expected = (analysis.world_size * local_shape[0],) + local_shape[1:]
            else:
                raise ValueError(f'ranked launch input {public!r} placement differs')
            if row.shape != expected or row.dtype != spec['dtype']:
                raise ValueError(f'ranked launch input {public!r} shard ABI differs')
        return None
