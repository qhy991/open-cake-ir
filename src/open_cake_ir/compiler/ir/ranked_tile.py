"""Tile-keyed ranked effects and stage-work-unit capacity/resource analysis.

This contract owns task/return identity and upper bounds. It authorizes no
device emission: the backend must still prove the exact Target, publication,
runtime route domain, CTA residency and progress before a ranked launch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, prod

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
    'reservation': 'returned_old_atomic_system',
    'ready': 'release_acquire_system',
    'row_identity': ['source_rank', 'item', 'route'],
}
_TILE = {
    'owner': 'destination_rank',
    'key': ['destination_rank', 'expert', 'tile_index'],
    'payload': 'expert_bin_rows_with_valid_count',
    'ready': 'release_acquire_system',
}
_TASK = {
    'owner': 'destination_rank',
    'key': ['destination_rank', 'expert', 'tile_index', 'stage', 'subtile'],
    'reservation': 'returned_old_atomic_gpu',
    'ready': 'release_acquire_system',
    'claim': 'atomic_gpu',
    'payload': 'one_stage_program_map_cta_with_valid_rows',
    'precondition': 'all_predecessor_stage_work_units_completed',
}
_RETURN = {
    'owner': 'source_rank',
    'key': ['source_rank', 'item', 'route'],
    'reservation': 'deterministic_index',
    'ready': 'release_acquire_system',
}
_STEAL = {
    'borrower': 'communication',
    'queue': 'stage_task',
    'after': 'dispatch_chunk',
    'before': 'combine',
    'resource_transition': 'maximum_stage_worker',
}
_PUBLICATION = 'full_on_capacity_partial_at_wave_threshold_and_terminal'
_STAGE_WORK_UNIT_ORDER = 'flatten_program_map_axes_xyz'
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
    logical_tile_slots_per_rank: int
    stage_work_units: tuple[tuple[str, int], ...]
    stage_work_units_per_tile: int
    stage_task_slots_per_rank: int
    stage_completion_slots_per_rank: int
    return_slots_per_rank: int
    required_execution_groups: int
    maximum_shared_bytes: int
    maximum_tensor_bytes: int

    def check_plan(self, expert_ids: Sequence, tasks: Sequence,
                   *, source_chunk_tokens: int) -> 'RankedTilePlanCheck':
        """Replay one materialized task plan, without claiming GPU liveness.

        Publication events are a serial witness: `[source_rank, chunk]` for
        a full tile, `["wave_end", chunk]` for an early partial tile, or
        `"all_dispatch_done"` for a terminal partial tile. GPU arrival order
        and release/acquire implementation remain backend responsibilities.
        """
        if (type(source_chunk_tokens) is not int or source_chunk_tokens < 1
                or ceil(self.items_per_rank / source_chunk_tokens)
                > self.maximum_chunks_per_rank):
            raise ValueError('tile plan source chunks exceed the admitted bound')
        if not isinstance(expert_ids, Sequence) or len(expert_ids) != self.world_size:
            raise ValueError('tile plan source rank count differs')
        expected: dict[tuple[int, int, int], int] = {}
        for source_rank, items in enumerate(expert_ids):
            if not isinstance(items, Sequence) or len(items) != self.items_per_rank:
                raise ValueError('tile plan source item extent differs')
            for item, routes in enumerate(items):
                if not isinstance(routes, Sequence) or len(routes) != self.routes_per_item:
                    raise ValueError('tile plan route extent differs')
                if (any(type(expert) is not int or not 0 <= expert < self.experts
                        for expert in routes)
                        or len(set(routes)) != len(routes)):
                    raise ValueError('tile plan expert IDs violate the distinct input domain')
                expected.update({(source_rank, item, route): expert
                                 for route, expert in enumerate(routes)})
        if not isinstance(tasks, Sequence):
            raise ValueError('tile plan tasks must be a sequence')
        seen: set[tuple[int, int, int]] = set()
        next_tile = [0] * self.experts
        expert_rows = [0] * self.experts
        owner_rows = [0] * self.world_size
        owner_tasks = [0] * self.world_size
        full = early = terminal = 0
        waves = ceil(self.items_per_rank / source_chunk_tokens)
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping) or set(task) != {
                    'owner_rank', 'expert_id', 'tile_index', 'valid_rows',
                    'published_after', 'rows'}:
                raise ValueError(f'tile plan task {index} fields differ')
            owner, expert, tile, valid = (task[field] for field in
                                          ('owner_rank', 'expert_id',
                                           'tile_index', 'valid_rows'))
            rows = task['rows']
            if (any(type(value) is not int for value in
                    (owner, expert, tile, valid))
                    or not 0 <= expert < self.experts
                    or owner != expert // self.experts_per_rank
                    or tile != next_tile[expert]
                    or not 1 <= valid <= self.tile_rows
                    or not isinstance(rows, Sequence) or len(rows) != valid):
                raise ValueError(f'tile plan task {index} owner, tile or valid rows differ')
            next_tile[expert] += 1
            owner_tasks[owner] += 1
            owner_rows[owner] += valid
            expert_rows[expert] += valid
            latest_event = (-1, -1)
            latest_wave = -1
            for raw in rows:
                if (not isinstance(raw, Sequence) or len(raw) != 3
                        or any(type(part) is not int for part in raw)):
                    raise ValueError(f'tile plan task {index} route key differs')
                key = tuple(raw)
                if key not in expected or key in seen or expected[key] != expert:
                    raise ValueError(f'tile plan task {index} route is lost, duplicated or misrouted')
                seen.add(key)
                source, item, _ = key
                wave = item // source_chunk_tokens
                latest_event = max(latest_event, (wave, source))
                latest_wave = max(latest_wave, wave)
            publication = task['published_after']
            if valid == self.tile_rows:
                if (not isinstance(publication, (list, tuple))
                        or len(publication) != 2
                        or any(type(part) is not int for part in publication)
                        or not 0 <= publication[0] < self.world_size
                        or not 0 <= publication[1] < waves
                        or latest_event > (publication[1], publication[0])):
                    raise ValueError(f'tile plan task {index} full publication precedes a row')
                full += 1
            elif (isinstance(publication, (list, tuple))
                  and len(publication) == 2
                  and publication[0] == 'wave_end'):
                wave = publication[1]
                if (type(wave) is not int or not 0 <= wave < waves - 1
                        or valid < self.partial_threshold_rows
                        or latest_wave > wave):
                    raise ValueError(f'tile plan task {index} early partial publication differs')
                early += 1
            elif publication == 'all_dispatch_done':
                terminal += 1
            else:
                raise ValueError(f'tile plan task {index} publication kind differs')
        if seen != set(expected):
            raise ValueError('tile plan has missing source return routes')
        if (any(rows > self.rows_per_expert for rows in expert_rows)
                or any(rows > self.packed_route_rows_per_rank for rows in owner_rows)
                or any(count > self.logical_tile_slots_per_rank
                       for count in owner_tasks)):
            raise ValueError('tile plan exceeds an admitted bin or task capacity')
        return RankedTilePlanCheck(
            tuple(owner_tasks), tuple(owner_rows),
            tuple(count * self.stage_work_units_per_tile
                  for count in owner_tasks), full, early, terminal)


@dataclass(frozen=True)
class RankedTilePlanCheck:
    tasks_by_owner: tuple[int, ...]
    rows_by_owner: tuple[int, ...]
    required_stage_work_units_by_owner: tuple[int, ...]
    full_tiles: int
    early_partial_tiles: int
    terminal_partial_tiles: int


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
                    'stage_work_unit_order',
                    'input_domain', 'steal', 'reset', 'lowering'}
        if (not isinstance(value, Mapping) or set(value) != required
                or type(value['schema_version']) is not int
                or value['schema_version'] != 2):
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
                or set(channels) != {'payload', 'bin', 'tile', 'task', 'return'}
                or channels['payload'] != _PAYLOAD
                or channels['bin'] != _BIN
                or channels['tile'] != _TILE
                or channels['task'] != _TASK
                or channels['return'] != _RETURN):
            raise ValueError('ranked tile channel keys, owners or memory order differ')
        if value['publication'] != _PUBLICATION:
            raise ValueError('ranked tile publication must flush full and terminal bins')
        if value['stage_work_unit_order'] != _STAGE_WORK_UNIT_ORDER:
            raise ValueError('ranked tile stage work units need explicit program-axis order')
        if value['input_domain'] != _INPUT_DOMAIN:
            raise ValueError('ranked tile input must declare per-item unique expert IDs')
        if value['steal'] != _STEAL:
            raise ValueError('ranked tile steal must borrow a complete stage worker')
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
            'schema_version': 2, 'world_size': self.world_size,
            'experts': self.experts, 'tile_rows': self.tile_rows,
            'maximum_chunks_per_rank': self.maximum_chunks_per_rank,
            'partial_threshold_rows': self.partial_threshold_rows,
            'workers': list(_WORKERS), 'controls': dict(_CONTROLS),
            'channels': {'payload': copied(_PAYLOAD), 'bin': copied(_BIN),
                         'tile': copied(_TILE), 'task': copied(_TASK),
                         'return': copied(_RETURN)},
            'publication': _PUBLICATION, 'input_domain': _INPUT_DOMAIN,
            'stage_work_unit_order': _STAGE_WORK_UNIT_ORDER,
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
        logical_tile_capacity = min(
            total_routes,
            ceil(total_routes / self.tile_rows)
            + local_experts * possible_partial_waves - 1,
        )
        stages = [stage.schedule for stage in local.stages]
        stage_units: list[tuple[str, int]] = []
        for stage in local.stages:
            schedule = stage.schedule
            mapping = schedule.program_map
            if (mapping is None or mapping.persistent or mapping.cooperative
                    or schedule.grid is not None):
                raise ValueError('ranked tile stage work units need a finite explicit ProgramMap')
            extents = []
            for axis in mapping.axes:
                owner = schedule.buffer(axis.buffer)
                if (owner is None or owner.space is not MemorySpace.GLOBAL
                        or axis.dimension >= len(owner.shape)):
                    raise ValueError('ranked tile stage ProgramMap axis owner differs')
                extents.append(axis.tile_count(owner.shape[axis.dimension]))
            count = prod(extents)
            if count < 1:
                raise ValueError('ranked tile stage needs at least one CTA work unit')
            stage_units.append((stage.name, count))
        work_units_per_tile = sum(count for _, count in stage_units)
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
            total_routes, self.world_size * tokens, logical_tile_capacity,
            tuple(stage_units), work_units_per_tile,
            logical_tile_capacity * work_units_per_tile,
            logical_tile_capacity * len(stages),
            tokens * routes, groups, shared, tensor,
        )
