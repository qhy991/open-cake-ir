"""Byte-bearing seal for one exact B300 distributed ranked-tile launch.

The Workload owns public tensor semantics; the Compiler owns complete math,
effects and CUDA source. This boundary binds those two authorities to one
rank-local c/K/steal plan and the compiled host library before device loading.
It does not assert that NVCC produced the library or qualify device timing.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from open_cake_ir.compiler import Program, RankedTileEffects, Schedule
from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    NativeRankedTileLowering,
)
from open_cake_ir.serialization import canonical_json_bytes
from .ranked_tile_launch import (
    _controls, prepare_ranked_tile_case, validate_ranked_tile_case,
)
from .ranked_tile_ctypes import load_ranked_tile_ctypes
from .workload import WorkloadContract


_FIELDS = frozenset({
    'schema_version', 'abi', 'workload_sha256', 'case_id', 'target',
    'compiler_revision_id', 'compiler_commit', 'entry_point',
    'effects', 'local_program', 'combine', 'toolchain_requirements',
    'source_map', 'plans',
})


@dataclass(frozen=True)
class RankedTileLaunchManifest:
    _bytes: bytes
    abi = 'ranked_tile_b300_pointer_v3'

    @classmethod
    def from_dict(cls, document: object) -> 'RankedTileLaunchManifest':
        if (not isinstance(document, Mapping) or set(document) != _FIELDS
                or type(document.get('schema_version')) is not int
                or document['schema_version'] != 1
                or document.get('abi') != cls.abi):
            raise ValueError('ranked tile launch manifest fields or ABI differ')
        effects = RankedTileEffects.from_dict(document['effects'])
        local = Program.from_dict(document['local_program'])
        combine = Schedule.from_dict(document['combine'])
        analysis = effects.analyze(local, combine)
        req = document['toolchain_requirements']
        if (not isinstance(req, Mapping)
                or document['target'] != local.target
                or req.get('target') != local.target
                or document['entry_point'] != effects.lowering.entry_point
                or req.get('entry_point') != document['entry_point']
                or req.get('compiler_commit') != document['compiler_commit']
                or not isinstance(document['compiler_commit'], str)
                or len(document['compiler_commit']) != 40
                or any(c not in '0123456789abcdef'
                       for c in document['compiler_commit'])
                or not isinstance(document['compiler_revision_id'], str)
                or not document['compiler_revision_id']
                or not isinstance(document['workload_sha256'], str)
                or len(document['workload_sha256']) != 64
                or any(c not in '0123456789abcdef'
                       for c in document['workload_sha256'])
                or not isinstance(document['case_id'], str)
                or not document['case_id']):
            raise ValueError('ranked tile Workload or Compiler identity differs')
        rows = document['plans']
        if (not isinstance(rows, list) or len(rows) != analysis.world_size
                or [row.get('rank') if isinstance(row, Mapping) else None
                    for row in rows] != list(range(analysis.world_size))):
            raise ValueError('ranked tile manifest needs ordered rank plans')
        plans = {}
        for rank, row in enumerate(rows):
            if (not isinstance(row, Mapping) or set(row) != {
                    'rank', 'communication_ctas', 'chunks', 'steal_budget'}):
                raise ValueError('ranked tile manifest plan fields differ')
            plans[rank] = {name: row[name] for name in
                           ('communication_ctas', 'chunks', 'steal_budget')}
        _controls(plans, analysis.world_size,
                  analysis.stage_task_slots_per_rank)
        source_map = document['source_map']
        if (not isinstance(source_map, Mapping) or not source_map
                or any(not isinstance(name, str) or not name
                       or not isinstance(span, list) or len(span) != 2
                       or any(type(line) is not int or line < 1
                              for line in span)
                       for name, span in source_map.items())):
            raise ValueError('ranked tile manifest source map differs')
        return cls(canonical_json_bytes(document))

    @classmethod
    def from_lowered(cls, lowered: NativeRankedTileLowering, *,
                     combine_document: Mapping,
                     workload: WorkloadContract, case_id: str,
                     plans: Mapping[int, Mapping[str, int]]) -> 'RankedTileLaunchManifest':
        if Schedule.from_dict(combine_document) != lowered.combine_schedule:
            raise ValueError('ranked tile combine document differs from lowering')
        validate_ranked_tile_case(lowered, workload, case_id)
        checked = _controls(plans, lowered.analysis.world_size,
                            lowered.analysis.stage_task_slots_per_rank)
        req = lowered.toolchain_requirements
        document = {
            'schema_version': 1, 'abi': cls.abi,
            'workload_sha256': workload.canonical_sha256,
            'case_id': case_id, 'target': lowered.local_program.target,
            'compiler_revision_id': lowered.compiler_revision_id,
            'compiler_commit': req['compiler_commit'],
            'entry_point': lowered.effects.lowering.entry_point,
            'effects': lowered.effects.document,
            'local_program': lowered.local_program.document,
            'combine': dict(combine_document),
            'toolchain_requirements': dict(req),
            'source_map': {name: list(span) for name, span in
                           lowered.source_map.items()},
            'plans': [{'rank': rank, **dict(checked[rank])}
                      for rank in range(lowered.analysis.world_size)],
        }
        manifest = cls.from_dict(document)
        manifest.check_lowered(lowered)
        manifest.check_workload(workload, case_id, lowered)
        return manifest

    def as_dict(self) -> dict:
        return json.loads(self._bytes)

    def plans(self) -> dict[int, dict[str, int]]:
        return {row['rank']: {name: row[name] for name in
                              ('communication_ctas','chunks','steal_budget')}
                for row in self.as_dict()['plans']}

    def check_lowered(self, lowered: NativeRankedTileLowering) -> None:
        lowered.validate_binding()
        doc = self.as_dict()
        if (doc['target'] != lowered.local_program.target
                or doc['compiler_revision_id'] != lowered.compiler_revision_id
                or doc['compiler_commit']
                   != lowered.toolchain_requirements.get('compiler_commit')
                or doc['entry_point'] != lowered.effects.lowering.entry_point
                or doc['effects'] != lowered.effects.document
                or doc['local_program'] != lowered.local_program.document
                or Schedule.from_dict(doc['combine'])
                   != lowered.combine_schedule
                or doc['toolchain_requirements']
                   != dict(lowered.toolchain_requirements)
                or doc['source_map'] != {name: list(span) for name, span
                                         in lowered.source_map.items()}):
            raise ValueError('ranked tile manifest differs from exact Compiler lowering')

    def check_workload(self, workload: WorkloadContract,
                       case_id: str, lowered: NativeRankedTileLowering) -> None:
        doc = self.as_dict()
        if (doc['workload_sha256'] != workload.canonical_sha256
                or doc['case_id'] != case_id
                or doc['target'] != workload.target):
            raise ValueError('ranked tile manifest differs from frozen Workload')
        validate_ranked_tile_case(lowered, workload, case_id)


@dataclass(frozen=True)
class RankedTileCandidate:
    """The exact source, manifest and ELF bytes presented at one load boundary."""

    manifest: RankedTileLaunchManifest
    source: bytes
    library: bytes

    def __post_init__(self) -> None:
        if (not isinstance(self.manifest, RankedTileLaunchManifest)
                or not isinstance(self.source, bytes) or not self.source
                or not isinstance(self.library, bytes)
                or not self.library.startswith(b'\x7fELF')):
            raise ValueError('ranked tile candidate needs source, manifest and ELF bytes')

    @classmethod
    def seal(cls, lowered: NativeRankedTileLowering,
             manifest: RankedTileLaunchManifest, *,
             workload: WorkloadContract, case_id: str,
             library: bytes) -> 'RankedTileCandidate':
        manifest.check_lowered(lowered)
        manifest.check_workload(workload, case_id, lowered)
        if not isinstance(library, bytes) or not library.startswith(b'\x7fELF'):
            raise ValueError('ranked tile compiled host library is not ELF')
        return cls(manifest, lowered.source.encode('utf-8'), library)

    def check(self, lowered: NativeRankedTileLowering,
              workload: WorkloadContract, case_id: str,
              plans: Mapping[int, Mapping[str, int]],
              library_path: Path) -> None:
        self.manifest.check_lowered(lowered)
        self.manifest.check_workload(workload, case_id, lowered)
        checked = _controls(plans, lowered.analysis.world_size,
                            lowered.analysis.stage_task_slots_per_rank)
        if (self.manifest.plans() != {rank: dict(checked[rank])
                                     for rank in range(lowered.analysis.world_size)}
                or self.source != lowered.source.encode('utf-8')
                or self.library != Path(library_path).resolve(strict=True).read_bytes()):
            raise ValueError('ranked tile sealed source, plan or library bytes differ')


def prepare_sealed_ranked_tile_case(candidate: RankedTileCandidate,
        lowered: NativeRankedTileLowering, workload: WorkloadContract,
        case_id: str, inputs, outputs, plans, *, library_path: Path,
        pointer_of: Callable, isolated_process: bool,
        check_tensor: Callable,
        storage_span: Callable, execution_context: Callable):
    """Check the source/library/Workload seal before any device state is created."""
    if isolated_process is not True:
        raise ValueError('sealed ranked tile requires an isolated process')
    candidate.check(lowered, workload, case_id, plans, library_path)
    return prepare_ranked_tile_case(
        lowered, workload, case_id, inputs, outputs, plans,
        load_source=lambda source: load_ranked_tile_ctypes(
            source, library_path, pointer_of=pointer_of,
            isolated_process=isolated_process),
        check_tensor=check_tensor, storage_span=storage_span,
        execution_context=execution_context)
