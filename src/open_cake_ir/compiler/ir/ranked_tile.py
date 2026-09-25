"""Tile-keyed ranked effects and structural capacity/resource analysis.

This contract owns task/return identity and upper bounds. It authorizes no
device emission: the backend must still prove the exact Target, publication,
runtime route domain, CTA residency and progress before a ranked launch.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil

from .program import Program
from .schedule import LoweringRoute, Schedule
from .vocabulary import BufferMode, DType, MemorySpace


_WORKERS = ['communication', 'computation']
_CONTROLS = {
    'communication_ctas': 'rank_local_int32',
    'chunks': 'rank_local_int32',
    'steal_budget': 'rank_local_int32',
}
_PAYLOAD = {
    'owner': 'destination_rank',
    'key': ['source_rank', 'item', 'destination_rank'],
    'reservation': 'returned_old_atomic_system',
    'ready': 'release_acquire_system',
}
_BIN = {
    'owner': 'destination_rank',
    'key': ['destination_rank', 'expert', 'row'],
    'reservation': 'returned_old_atomic_gpu',
    'ready': 'release_acquire_system',
    'row_identity': ['source_rank', 'item', 'route'],
}
_TASK = {
    'owner': 'destination_rank',
    'key': ['destination_rank', 'expert', 'tile_index'],
    'reservation': 'returned_old_atomic_gpu',
    'ready': 'release_acquire_system',
    'claim': 'atomic_gpu',
    'payload': 'expert_bin_rows_with_valid_count',
}
_RETURN = {
    'owner': 'source_rank',
    'key': ['source_rank', 'item', 'route'],
    'reservation': 'deterministic_index',
    'ready': 'release_acquire_system',
}
_STEAL = {
    'borrower': 'communication',
    'queue': 'tile_task',
    'after': 'dispatch_chunk',
    'before': 'combine',
    'resource_transition': 'complete_tile_worker',
}
_PUBLICATION = 'full_on_capacity_partial_at_wave_threshold_and_terminal'
_INPUT_DOMAIN = 'distinct_expert_ids_in_range_per_item'
_RESET = 'zero_all_rank_mailboxes_before_launch'


@dataclass(frozen=True)
class RankedTileAnalysis:
    world_size: int
    items_per_rank: int
    routes_per_item: int
    experts: int
    experts_per_rank: int
    tile_rows: int
    feature_width: int
    maximum_chunks_per_rank: int
    partial_threshold_rows: int
    remote_payload_slots_per_rank: int
    packed_route_rows_per_rank: int
    rows_per_expert: int
    tile_task_slots_per_rank: int
    return_slots_per_rank: int
    required_execution_groups: int
    maximum_shared_bytes: int
    maximum_tensor_bytes: int


@dataclass(frozen=True)
class RankedTileEffects:
    world_size: int
    experts: int
    tile_rows: int
    maximum_chunks_per_rank: int
    partial_threshold_rows: int
    lowering: LoweringRoute

    @classmethod
    def from_dict(cls, value) -> 'RankedTileEffects':
        required = {'schema_version', 'world_size', 'experts', 'tile_rows',
                    'maximum_chunks_per_rank', 'partial_threshold_rows',
                    'workers', 'controls', 'channels', 'publication',
                    'input_domain', 'steal', 'reset', 'lowering'}
        if (not isinstance(value, Mapping) or set(value) != required
                or type(value['schema_version']) is not int
                or value['schema_version'] != 1):
            raise ValueError('ranked tile effect fields or version differ')
        world = value['world_size']
        experts = value['experts']
        rows = value['tile_rows']
        waves = value['maximum_chunks_per_rank']
        threshold = value['partial_threshold_rows']
        if (any(type(item) is not int for item in
                (world, experts, rows, waves, threshold))
                or world < 2 or experts < world or experts % world
                or rows < 1 or waves < 1 or not 1 <= threshold <= rows):
            raise ValueError('ranked tile geometry, chunk or partial threshold differs')
        if value['workers'] != _WORKERS or value['controls'] != _CONTROLS:
            raise ValueError('ranked tile workers and controls differ')
        channels = value['channels']
        if (not isinstance(channels, Mapping)
                or set(channels) != {'payload', 'bin', 'task', 'return'}
                or channels['payload'] != _PAYLOAD
                or channels['bin'] != _BIN
                or channels['task'] != _TASK
                or channels['return'] != _RETURN):
            raise ValueError('ranked tile channel keys, owners or memory order differ')
        if value['publication'] != _PUBLICATION:
            raise ValueError('ranked tile publication must flush full and terminal bins')
        if value['input_domain'] != _INPUT_DOMAIN:
            raise ValueError('ranked tile input must declare per-item unique expert IDs')
        if value['steal'] != _STEAL:
            raise ValueError('ranked tile steal must borrow the complete tile worker')
        if value['reset'] != _RESET:
            raise ValueError('ranked tile mailboxes require reset before every launch')
        lowering = LoweringRoute.from_dict(value['lowering'],
                                          'ranked tile.lowering')
        return cls(world, experts, rows, waves, threshold, lowering)

    @property
    def document(self) -> dict:
        def copied(value):
            return {**value, **{
                key: list(item) for key, item in value.items()
                if isinstance(item, list)}}
        return {
            'schema_version': 1, 'world_size': self.world_size,
            'experts': self.experts, 'tile_rows': self.tile_rows,
            'maximum_chunks_per_rank': self.maximum_chunks_per_rank,
            'partial_threshold_rows': self.partial_threshold_rows,
            'workers': list(_WORKERS), 'controls': dict(_CONTROLS),
            'channels': {'payload': copied(_PAYLOAD), 'bin': copied(_BIN),
                         'task': copied(_TASK), 'return': copied(_RETURN)},
            'publication': _PUBLICATION, 'input_domain': _INPUT_DOMAIN,
            'steal': dict(_STEAL), 'reset': _RESET,
            'lowering': {'backend': self.lowering.backend.value,
                         'entry_point': self.lowering.entry_point},
        }

    def analyze(self, local: Program, combine: Schedule) -> RankedTileAnalysis:
        """Bound capacity and a stealing CTA's largest sequential stage.

        The bounds assume the declared distinct-expert input domain. This
        structural analysis does not prove a runtime route set, tile readiness,
        peer topology, exact residency or deadlock freedom.
        """
        if Program.from_dict(local.document) != local or local.execution is not None:
            raise ValueError('ranked tile math must be a complete ordered Program')
        if local.target != combine.target or len(local.outputs) != 1:
            raise ValueError('ranked tile local and combine targets or outputs differ')
        local_out = local.tensors[local.outputs[0]]
        if (local_out.dtype is not DType.FP32 or len(local_out.shape) != 2
                or local_out.shape[0] != self.tile_rows):
            raise ValueError('ranked tile Program needs one FP32 [tile, feature] output')
        width = local_out.shape[1]
        if not any(tensor.dtype is DType.BF16
                   and tensor.shape == (self.tile_rows, width)
                   for name, tensor in local.tensors.items()
                   if name in local.inputs):
            raise ValueError('ranked tile Program needs one matching BF16 row input')
        globals_ = [buffer for buffer in combine.buffers
                    if buffer.space is MemorySpace.GLOBAL]
        inputs = [buffer for buffer in globals_ if buffer.mode is BufferMode.INPUT]
        outputs = [buffer for buffer in globals_ if buffer.mode is BufferMode.OUTPUT]
        candidates = [buffer for buffer in inputs
                      if buffer.dtype is DType.FP32 and len(buffer.shape) == 3]
        if (len(globals_) != 3 or len(inputs) != 2 or len(outputs) != 1
                or len(candidates) != 1):
            raise ValueError('ranked tile combine needs contribution, weight and output globals')
        contributions = candidates[0]
        weights = next(buffer for buffer in inputs if buffer is not contributions)
        output = outputs[0]
        if (contributions.shape[2] != width
                or weights.dtype is not DType.FP32
                or weights.shape != contributions.shape[:2]
                or output.dtype is not DType.BF16
                or output.shape != (contributions.shape[0], width)
                or combine.outputs != (output.name,)):
            raise ValueError('ranked tile combine shape, dtype or output differs')
        tokens, routes, _ = contributions.shape
        if (tokens < 1 or routes < 1 or routes > self.experts
                or self.maximum_chunks_per_rank > tokens):
            raise ValueError('ranked tile item, route or chunk extent differs')
        total_routes = self.world_size * tokens * routes
        local_experts = self.experts // self.world_size
        possible_partial_waves = (self.maximum_chunks_per_rank
                                  if self.partial_threshold_rows < self.tile_rows
                                  else 1)
        tile_capacity = min(
            total_routes,
            ceil(total_routes / self.tile_rows)
            + local_experts * possible_partial_waves - 1,
        )
        stages = [stage.schedule for stage in local.stages]
        shared = max(sum(allocation.size_bytes for allocation in stage.allocations
                         if allocation.space is MemorySpace.SHARED)
                     for stage in stages)
        tensor = max(sum(allocation.size_bytes for allocation in stage.allocations
                         if allocation.space is MemorySpace.TENSOR)
                     for stage in stages)
        groups = max(stage.total_execution_group_extent for stage in stages)
        return RankedTileAnalysis(
            self.world_size, tokens, routes, self.experts, local_experts,
            self.tile_rows, width, self.maximum_chunks_per_rank,
            self.partial_threshold_rows,
            (self.world_size - 1) * tokens,
            total_routes, self.world_size * tokens, tile_capacity,
            tokens * routes, groups, shared, tensor,
        )
