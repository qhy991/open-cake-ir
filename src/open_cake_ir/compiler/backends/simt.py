"""Logical lane storage shared by compositional SIMD/SIMT emitters.

These are declared-value slots, never backend register allocation or spill counts.
"""
from ..ir import MemorySpace, Schedule


def slots(buffer, lanes: int = 32):
    return (buffer.elements + lanes - 1) // lanes


def private_values_per_thread(schedule: Schedule, lanes: int = 32) -> int:
    """Peak simultaneously live lane-owned FP32 values, without physical allocation claims.

    Count source and destination at their common operation boundary. No speculative
    in-place aliasing or compiler register reuse is assumed. Shuffle/reduction scalar
    temporaries and backend spills are outside this declared-Buffer domain.
    """
    intervals = []
    for buffer in schedule.buffers:
        if buffer.space is not MemorySpace.REGISTER:
            continue
        uses = [index for index, operation in enumerate(schedule.operations)
                if buffer.name in (*operation.reads, *operation.writes)]
        intervals.append((uses[0], uses[-1], slots(buffer, lanes)) if uses
                         else (0, len(schedule.operations) - 1, slots(buffer, lanes)))
    return max((sum(slots for first, last, slots in intervals if first <= index <= last)
                for index in range(len(schedule.operations))), default=0)

