"""Ranked mailbox effects for a complete local Program and origin Schedule.

This is an explicit, bounded control/effect contract. It does not authorize
emission, assume peer access, or replace either mathematical body with an
opaque operator. The backend must prove device ownership, progress and exact
Target support before a four-rank launch can use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .program import Program
from .schedule import LoweringRoute, Schedule
from .vocabulary import BufferMode, DType, MemorySpace


_PAYLOAD = {
    'owner': 'destination_rank',
    'key': ['source_rank', 'item', 'destination_rank'],
    'reservation': 'returned_old_atomic_system',
    'ready': 'release_acquire_system',
}
_TASK = {
    'owner': 'destination_rank',
    'key': ['source_rank', 'item', 'route'],
    'reservation': 'returned_old_atomic_system',
    'ready': 'release_acquire_system',
    'claim': 'atomic_gpu',
    'payload': 'remote_payload_slot_or_local_input',
}
_RETURN = {
    'owner': 'source_rank',
    'key': ['source_rank', 'item', 'route'],
    'reservation': 'deterministic_index',
    'ready': 'release_acquire_system',
}
_CONTROLS = {
    'communication_ctas': 'rank_local_int32',
    'chunks': 'rank_local_int32',
    'steal_budget': 'rank_local_int32',
}
_STEAL = {
    'borrower': 'communication', 'queue': 'task',
    'after': 'dispatch', 'before': 'combine',
}


@dataclass(frozen=True)
class RankedMailboxAnalysis:
    world_size: int
    items_per_rank: int
    routes_per_item: int
    feature_width: int
    remote_payload_capacity_per_rank: int
    compute_task_capacity_per_rank: int
    return_slots_per_rank: int


@dataclass(frozen=True)
class RankedMailboxEffects:
    world_size: int
    lowering: LoweringRoute

    @classmethod
    def from_dict(cls, value) -> 'RankedMailboxEffects':
        if (not isinstance(value, Mapping)
                or set(value) != {'schema_version', 'world_size', 'workers',
                                  'controls', 'channels', 'steal', 'reset',
                                  'lowering'}
                or type(value['schema_version']) is not int
                or value['schema_version'] != 1):
            raise ValueError('ranked mailbox effect fields or version differ')
        world = value['world_size']
        if type(world) is not int or world < 2:
            raise ValueError('ranked mailbox needs at least two exact ranks')
        if value['workers'] != ['communication', 'computation']:
            raise ValueError('ranked mailbox needs both replicated CTA classes')
        if value['controls'] != _CONTROLS:
            raise ValueError('ranked mailbox controls must be rank-local INT32 values')
        channels = value['channels']
        if (not isinstance(channels, Mapping)
                or set(channels) != {'payload', 'task', 'return'}
                or channels['payload'] != _PAYLOAD
                or channels['task'] != _TASK
                or channels['return'] != _RETURN):
            raise ValueError('ranked mailbox keys, owners, reservation or memory order differ')
        if value['steal'] != _STEAL:
            raise ValueError('ranked mailbox steal must borrow the shared task queue')
        if value['reset'] != 'zero_all_rank_mailboxes_before_launch':
            raise ValueError('ranked mailbox requires a reset before each launch')
        lowering = LoweringRoute.from_dict(value['lowering'],
                                          'ranked mailbox.lowering')
        return cls(world, lowering)

    @property
    def document(self) -> dict:
        return {
            'schema_version': 1, 'world_size': self.world_size,
            'workers': ['communication', 'computation'],
            'controls': dict(_CONTROLS),
            'channels': {'payload': {**_PAYLOAD, 'key': list(_PAYLOAD['key'])},
                         'task': {**_TASK, 'key': list(_TASK['key'])},
                         'return': {**_RETURN, 'key': list(_RETURN['key'])}},
            'steal': dict(_STEAL),
            'reset': 'zero_all_rank_mailboxes_before_launch',
            'lowering': {'backend': self.lowering.backend.value,
                         'entry_point': self.lowering.entry_point},
        }

    def analyze(self, local: Program, combine: Schedule) -> RankedMailboxAnalysis:
        """Derive queue capacities from complete math tensor extents."""
        if Program.from_dict(local.document) != local or local.execution is not None:
            raise ValueError('ranked mailbox local math must be a complete ordered Program')
        if local.target != combine.target or len(local.outputs) != 1:
            raise ValueError('ranked mailbox math targets or local output differ')
        local_out = local.tensors[local.outputs[0]]
        if local_out.dtype is not DType.FP32 or len(local_out.shape) != 1:
            raise ValueError('ranked mailbox local contribution needs one FP32 vector')
        globals_ = [buffer for buffer in combine.buffers
                    if buffer.space is MemorySpace.GLOBAL]
        inputs = [buffer for buffer in globals_ if buffer.mode is BufferMode.INPUT]
        outputs = [buffer for buffer in globals_ if buffer.mode is BufferMode.OUTPUT]
        candidates = [buffer for buffer in inputs
                      if buffer.dtype is DType.FP32 and len(buffer.shape) == 3]
        if len(globals_) != 3 or len(inputs) != 2 or len(outputs) != 1 or len(candidates) != 1:
            raise ValueError('ranked mailbox combine needs contribution, weight and output globals')
        contributions = candidates[0]
        weights = next(buffer for buffer in inputs if buffer is not contributions)
        output = outputs[0]
        if (len(contributions.shape) != 3
                or contributions.shape[2] != local_out.shape[0]
                or weights.dtype is not DType.FP32
                or weights.shape != contributions.shape[:2]
                or output.dtype is not DType.BF16
                or output.shape != (contributions.shape[0], local_out.shape[0])
                or combine.outputs != (output.name,)):
            raise ValueError('ranked mailbox combine shape, dtype or output differs')
        tokens, routes, width = contributions.shape
        if tokens < 1 or routes < 1 or width < 1:
            raise ValueError('ranked mailbox needs positive item, route and feature extents')
        return RankedMailboxAnalysis(
            self.world_size, tokens, routes, width,
            (self.world_size - 1) * tokens,
            self.world_size * tokens * routes,
            tokens * routes,
        )
