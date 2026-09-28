"""Evaluation binding for one exact native CUDA ranked-tile lowering.

The caller owns device tensor allocation, an isolated broker-held process,
compilation/loading and the external oracle. This adapter checks the typed
Compiler source, rank-local tensor placement, controls and state lifecycle.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType

from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    NativeRankedTileLowering,
)
from open_cake_ir.compiler.ir import DType, ProgramTensor
from .workload import WorkloadContract


_PUBLIC_INPUTS = {
    'hidden_states': 'hidden',
    'expert_ids': 'expert_ids',
    'route_weights': 'route_weights',
    'w_up_gate': 'w_up_gate',
    'w_down': 'w_down',
}


def validate_ranked_tile_case(lowered: NativeRankedTileLowering,
                              workload: WorkloadContract, case_id: str) -> None:
    """Match a frozen distributed Workload to every local pointer ABI row."""
    lowered.validate_binding()
    analysis = lowered.analysis
    req = lowered.toolchain_requirements
    if (not isinstance(workload, WorkloadContract)
            or not workload.requires_distributed_execution
            or workload.target != lowered.local_program.target
            or workload.document['state'] != 'frozen'):
        raise ValueError('ranked tile needs a frozen exact-target distributed Workload')
    semantics = workload.document['semantics']
    if (semantics.get('execution_topology') != {
                'kind': 'expert_parallel', 'world_size': analysis.world_size}
            or semantics.get('expert_placement')
               != 'contiguous_equal_ranges_by_rank'
            or semantics.get('top_k') != analysis.routes_per_item
            or semantics.get('route_ids') != 'distinct_in_range_per_token'
            or semantics.get('route_weights')
               != 'finite_nonnegative_sum_one_per_token'):
        raise ValueError('ranked tile Workload topology or route domain differs')
    public = {row.name: row for row in workload.tensor_abi(case_id)}
    if set(public) != set(_PUBLIC_INPUTS) | {'output'}:
        raise ValueError('ranked tile Workload public tensor set differs')
    rank_inputs = _specs(req.get('rank_inputs'))
    if set(rank_inputs) != set(_PUBLIC_INPUTS.values()):
        raise ValueError('ranked tile Compiler input set differs from Workload binding')
    placement = semantics.get('tensor_placement')
    if not isinstance(placement, Mapping):
        raise ValueError('ranked tile Workload tensor placement differs')
    for name, local_name in _PUBLIC_INPUTS.items():
        row = public[name]
        spec = rank_inputs[local_name]
        owner = ('expert_sharded_axis_0' if name.startswith('w_')
                 else 'rank_sharded_axis_0')
        first = (analysis.world_size * spec.shape[0]
                 if owner == 'expert_sharded_axis_0'
                 else analysis.world_size)
        expected = (first,) + (spec.shape[1:] if owner == 'expert_sharded_axis_0'
                               else spec.shape)
        if (row.mode != 'input' or row.shape != expected
                or row.dtype != spec.dtype.value or placement.get(name) != owner):
            raise ValueError(f'ranked tile Workload {name!r} shard ABI differs')
    output = public['output']
    out = req.get('rank_output')
    if (not isinstance(out, Mapping)
            or output.mode != 'output'
            or output.shape != (analysis.world_size, *out.get('shape', ()))
            or output.dtype != out.get('dtype')
            or placement.get('output') != 'rank_sharded_axis_0'):
        raise ValueError('ranked tile Workload output shard ABI differs')


@dataclass(frozen=True)
class RankedTileBound:
    """Compiled ABI callbacks with one created CUDA state handle."""

    launch: Callable[[tuple[int, ...], tuple[int, ...], tuple[int, ...]], int]
    stolen: Callable[[int], int]
    payloads: Callable[[int], int]
    tile_counts: Callable[[int], tuple[int, ...]]
    destroy: Callable[[], int]


@dataclass(frozen=True)
class RankedTileExecutable:
    """Loaded source and ABI facts, before device input binding."""

    ranks: int
    source_events: int
    output_bytes: int
    isolated_process: bool
    bind: Callable


def _specs(rows) -> dict[str, ProgramTensor]:
    if not isinstance(rows, list) or not rows:
        raise ValueError('ranked tile needs declared per-rank inputs')
    specs = {}
    for row in rows:
        if (not isinstance(row, Mapping)
                or set(row) != {'name', 'shape', 'dtype'}
                or not isinstance(row['name'], str) or not row['name']
                or row['name'] in specs
                or not isinstance(row['shape'], list)
                or not row['shape']
                or any(type(n) is not int or n <= 0 for n in row['shape'])):
            raise ValueError('ranked tile input declaration differs')
        try:
            dtype = DType(row['dtype'])
        except (TypeError, ValueError):
            raise ValueError('ranked tile input dtype differs') from None
        specs[row['name']] = ProgramTensor(tuple(row['shape']), dtype)
    return specs


def _span(tensor, expected: ProgramTensor, rank: int, storage_span,
          occupied: list[tuple[int, int, int]], label: str):
    device, start, end = storage_span(tensor)
    if (type(device) is not int or device != rank
            or type(start) is not int or type(end) is not int
            or not 0 < start < end or end - start != expected.nbytes
            or any(device == other_device and start < other_end
                   and other_start < end
                   for other_device, other_start, other_end in occupied)):
        raise ValueError(f'rank {rank} {label} owner, extent or alias differs')
    occupied.append((device, start, end))


def _controls(plans, world: int, task_capacity: int):
    ranks=set(range(world))
    if not isinstance(plans, Mapping) or set(plans)!=ranks:
        raise ValueError('ranked tile needs one rank-local control plan per rank')
    controls={}
    for rank in range(world):
        row=plans[rank]
        if (not isinstance(row, Mapping)
                or set(row)!={'communication_ctas','chunks','steal_budget'}):
            raise ValueError(f'rank {rank} spatial, temporal or steal plan differs')
        c,chunks,budget=(row[name] for name in
                         ('communication_ctas','chunks','steal_budget'))
        if (any(type(value) is not int for value in (c,chunks,budget))
                or not 1<=c<96 or chunks not in (1,2,4)
                or not 0<=budget<=task_capacity):
            raise ValueError(f'rank {rank} controls exceed the B300 lowering domain')
        controls[rank]=MappingProxyType(dict(row))
    if len({controls[rank]['chunks'] for rank in range(world)})!=1:
        raise ValueError('ranked tile temporal chunks must agree across ranks')
    return MappingProxyType(controls)


def prepare_ranked_tiles(lowered: NativeRankedTileLowering,
        inputs: Mapping[int, Mapping[str, object]],
        outputs: Mapping[int, object],
        plans: Mapping[int, Mapping[str, int]], *,
        load_source: Callable, check_tensor: Callable,
        storage_span: Callable, execution_context: Callable):
    """Bind four exact-device tensors and rank-local controls before launch."""
    lowered.validate_binding()
    req = lowered.toolchain_requirements
    world = lowered.analysis.world_size
    ranks = set(range(world))
    abi = req.get('host_abi')
    if (req.get('target') != lowered.local_program.target
            or req.get('source_events') != 20
            or req.get('supported_chunks') != [1,2,4]
            or req.get('source_chunk_tokens_by_chunks')
               != {'1':512,'2':256,'4':128}
            or req.get('logical_tile_capacity') != 255
            or req.get('stage_task_capacity')
               != lowered.analysis.stage_task_slots_per_rank
            or req.get('rank_local_controls') is not True
            or req.get('state_reset') != 'zero_all_rank_mailboxes_before_launch'
            or req.get('peer_pair_runtime_check') is not True
            or req.get('input_domain_runtime_check') is not True
            or req.get('synchronous_launch') is not True
            or req.get('failed_partial_launch_requires_process_exit') is not True
            or not isinstance(abi, Mapping)
            or set(abi) != {'abi_version','ranks','source_events','bin_bytes',
                            'output_bytes','create','launch','destroy',
                            'stolen','payloads','tile_counts'}
            or any(not isinstance(name, str) or not name
                   for name in abi.values())):
        raise ValueError('ranked tile source, state or host ABI differs')
    specs = _specs(req.get('rank_inputs'))
    output_row = req.get('rank_output')
    if (not isinstance(output_row, Mapping)
            or set(output_row) != {'shape', 'dtype'}
            or output_row != {'shape':[512,2048], 'dtype':'bf16'}):
        raise ValueError('ranked tile output declaration differs')
    output_spec = ProgramTensor((512,2048), DType.BF16)
    if (not isinstance(inputs, Mapping) or set(inputs) != ranks
            or not isinstance(outputs, Mapping) or set(outputs) != ranks
            or not isinstance(plans, Mapping) or set(plans) != ranks):
        raise ValueError('ranked tile needs every rank input, output and plan')
    controls=_controls(plans,world,lowered.analysis.stage_task_slots_per_rank)
    contexts = tuple(execution_context(rank) for rank in range(world))
    spans = []
    frozen_inputs = {}
    frozen_outputs = {}
    for rank in range(world):
        row = inputs[rank]
        if not isinstance(row, Mapping) or set(row) != set(specs):
            raise ValueError(f'rank {rank} public input set differs')
        for name,tensor in row.items():
            check_tensor(tensor,specs[name])
            _span(tensor,specs[name],rank,storage_span,spans,name)
        output = outputs[rank]
        check_tensor(output,output_spec)
        _span(output,output_spec,rank,storage_span,spans,'output')
        frozen_inputs[rank] = MappingProxyType(dict(row))
        frozen_outputs[rank] = output
    executable = load_source(lowered)
    if (not isinstance(executable, RankedTileExecutable)
            or executable.ranks != world
            or executable.source_events != 20
            or executable.output_bytes != output_spec.nbytes
            or executable.isolated_process is not True
            or not callable(executable.bind)):
        raise ValueError('ranked tile compiled ABI or process isolation differs')
    bound = executable.bind(MappingProxyType(frozen_inputs),
                            MappingProxyType(frozen_outputs), contexts)
    if (not isinstance(bound, RankedTileBound)
            or any(not callable(getattr(bound, name)) for name in
                   ('launch','stolen','payloads','tile_counts','destroy'))):
        raise ValueError('ranked tile loaded ABI binder differs')
    if tuple(execution_context(rank) for rank in range(world)) != contexts:
        raise ValueError('ranked tile execution contexts changed during binding')
    return PreparedRankedTiles(
        lowered, bound, MappingProxyType(frozen_inputs),
        MappingProxyType(frozen_outputs), controls,
        execution_context, contexts)


def prepare_ranked_tile_case(lowered: NativeRankedTileLowering,
        workload: WorkloadContract, case_id: str,
        inputs: Mapping[int, Mapping[str, object]],
        outputs: Mapping[int, object],
        plans: Mapping[int, Mapping[str, int]], *,
        load_source: Callable, check_tensor: Callable,
        storage_span: Callable, execution_context: Callable):
    """Bind a checked Workload case before loading any device executable."""
    validate_ranked_tile_case(lowered, workload, case_id)
    return prepare_ranked_tiles(
        lowered, inputs, outputs, plans, load_source=load_source,
        check_tensor=check_tensor, storage_span=storage_span,
        execution_context=execution_context)


@dataclass
class PreparedRankedTiles:
    lowering: NativeRankedTileLowering
    bound: RankedTileBound
    inputs: Mapping[int, Mapping[str, object]]
    outputs: Mapping[int, object]
    controls: Mapping[int, Mapping[str, int]]
    execution_context: Callable
    contexts: tuple
    launch_calls: int = 0
    poisoned: bool = False
    closed: bool = False
    _lock: object = field(default_factory=Lock, repr=False)

    def run(self, plans: Mapping[int, Mapping[str, int]] | None = None):
        """One synchronous four-rank launch and source-owned status audit."""
        with self._lock:
            if self.closed or self.poisoned:
                raise ValueError('ranked tile state is closed or poisoned')
            if tuple(self.execution_context(rank) for rank in range(4)) != self.contexts:
                raise ValueError('ranked tile execution contexts changed')
            controls=(self.controls if plans is None else _controls(
                plans,self.lowering.analysis.world_size,
                self.lowering.analysis.stage_task_slots_per_rank))
            communication = tuple(controls[rank]['communication_ctas']
                                  for rank in range(4))
            budgets = tuple(controls[rank]['steal_budget']
                            for rank in range(4))
            chunks = tuple(controls[rank]['chunks'] for rank in range(4))
            try:
                status = self.bound.launch(communication,budgets,chunks)
                if type(status) is not int or status != 0:
                    raise ValueError(f'ranked tile launch status {status!r}')
                stolen = tuple(self.bound.stolen(rank) for rank in range(4))
                payloads = tuple(self.bound.payloads(rank) for rank in range(4))
                tile_counts = tuple(tuple(self.bound.tile_counts(rank))
                                    for rank in range(4))
            except Exception:
                self.poisoned = True
                raise
            if (any(type(value) is not int or not 0 <= value <= budgets[rank]
                    for rank,value in enumerate(stolen))
                    or any(type(value) is not int or not 0 <= value <= 1536
                           for value in payloads)
                    or any(len(row)!=20
                           or any(type(value) is not int or value<0
                                  for value in row)
                           or sum(row)>self.lowering.analysis.logical_tile_slots_per_rank
                           or any(row[5*chunks[rank]:])
                           for rank,row in enumerate(tile_counts))):
                self.poisoned = True
                raise ValueError('ranked tile stolen, payload or tile status differs')
            if tuple(self.execution_context(rank) for rank in range(4)) != self.contexts:
                self.poisoned = True
                raise ValueError('ranked tile execution contexts changed after launch')
            self.launch_calls += 1
            return self.outputs, MappingProxyType({
                'stolen_by_rank':stolen,
                'remote_payloads_by_owner':payloads,
                'chunks_by_rank':chunks,
                'tile_counts_by_rank':tile_counts,
            })

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if self.poisoned:
                raise ValueError('poisoned ranked tile requires isolated process exit')
            status = self.bound.destroy()
            if type(status) is not int or status != 0:
                self.poisoned = True
                raise ValueError(f'ranked tile destroy status {status!r}')
            self.closed = True
