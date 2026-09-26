"""Proof of one loop-carried BF16 TMEM tile and its completion phases.

The loop declaration is an effect, not a layout algebra: a single pre-loop store
establishes phase zero, and one store after the loop's readers publishes each next
phase. The native backend remains responsible for realizing those phases.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ir import BarrierMechanism, BufferMode, DType, MemorySpace, OperationKind, Schedule


@dataclass(frozen=True)
class CarriedTmemPair:
    buffer: str
    loop: str
    initializer: str
    updater: str
    barrier: str
    trips: int


@dataclass(frozen=True)
class CarriedTmemIssue:
    code: str
    path: str
    message: str


def analyze_carried_tmem(
    schedule: Schedule,
) -> tuple[dict[str, CarriedTmemPair], tuple[CarriedTmemIssue, ...]]:
    """Resolve the bounded carry proof without trusting malformed references."""
    pairs: dict[str, CarriedTmemPair] = {}
    issues: list[CarriedTmemIssue] = []
    declared: set[str] = set()
    positions = {op.op_id: index for index, op in enumerate(schedule.operations)}
    barriers = {barrier.name: barrier for barrier in schedule.barriers}

    for loop_index, loop in enumerate(schedule.tile_loops):
        for name_index, name in enumerate(loop.carried_buffers):
            path = f"tile_loops[{loop_index}].carried_buffers[{name_index}]"
            def fail(code: str, message: str) -> None:
                issues.append(CarriedTmemIssue(code, path, message))

            if name in declared:
                fail("CARRIED_TMEM_MULTIPLE_LOOPS", f"{name!r} is carried by more than one loop")
                continue
            declared.add(name)
            buffer = schedule.buffer(name)
            if (buffer is None or buffer.space is not MemorySpace.TENSOR
                    or buffer.mode is not BufferMode.SCRATCH or buffer.dtype is not DType.BF16):
                fail("CARRIED_TMEM_BUFFER", "a carried tile must be BF16 TMEM scratch")
                continue
            if loop.stop is not None or schedule.loop_parent().get(loop.name) is not None:
                fail("CARRIED_TMEM_LOOP_SCOPE", "the first carried-TMEM subset uses a static outer loop")
                continue
            if len(loop.carried_buffers) != 1:
                fail("CARRIED_TMEM_LOOP_ARITY", "one loop carries exactly one TMEM tile")
                continue
            owner = schedule.buffer(loop.buffer)
            if owner is None or loop.dimension >= len(owner.shape):
                # The ordinary loop-domain Finding owns this malformed edge.
                continue
            trips = (owner.shape[loop.dimension] + loop.tile - 1) // loop.tile
            if trips < 2:
                # The ordinary single-trip Finding owns the canonical form.
                continue
            writers = [op for op in schedule.operations if name in op.writes]
            if len(writers) != 2 or any(op.kind is not OperationKind.TMEM_STORE for op in writers):
                fail("CARRIED_TMEM_WRITERS", "one root tmem_store initializes and one in-loop tmem_store updates the tile")
                continue
            initializers = [op for op in writers if not schedule.enclosing_loops(op)]
            updaters = [op for op in writers if op.op_id in loop.body]
            if len(initializers) != 1 or len(updaters) != 1:
                fail("CARRIED_TMEM_WRITER_SCOPE", "initializer is outside every loop and updater is direct in this loop")
                continue
            initializer, updater = initializers[0], updaters[0]
            if initializer.role != updater.role:
                fail("CARRIED_TMEM_WRITER_ROLE", "initial and repeated stores use one execution role")
                continue
            body = schedule.loop_operations(loop)
            body_positions = [positions[op.op_id] for op in body]
            if not body_positions or positions[initializer.op_id] >= min(body_positions):
                fail("CARRIED_TMEM_INITIALIZER_ORDER", "initializer precedes the loop body")
                continue
            in_loop_readers = [op for op in body if name in op.reads]
            if not in_loop_readers or any(positions[op.op_id] >= positions[updater.op_id]
                                          for op in in_loop_readers):
                fail("CARRIED_TMEM_UPDATE_ORDER", "each trip reads the current tile before its update")
                continue
            readers = [op for op in schedule.operations if name in op.reads]
            if any(op not in body and positions[op.op_id] <= max(body_positions) for op in readers):
                fail("CARRIED_TMEM_READER_SCOPE", "readers are inside the loop or after its last operation")
                continue
            if (len(initializer.signals) != 1 or initializer.signals != updater.signals):
                fail("CARRIED_TMEM_BARRIER", "initializer and updater publish successive phases of one barrier")
                continue
            barrier_name = initializer.signals[0]
            barrier = barriers.get(barrier_name)
            if (barrier is None or barrier.mechanism is not BarrierMechanism.MBARRIER
                    or barrier.count != 4 or barrier.pipeline is not None
                    or any(barrier_name not in reader.waits for reader in readers)):
                fail("CARRIED_TMEM_BARRIER", "one count-four mbarrier orders every reader of the carried tile")
                continue
            pairs[name] = CarriedTmemPair(
                name, loop.name, initializer.op_id, updater.op_id, barrier_name, trips
            )
    return pairs, tuple(issues)
