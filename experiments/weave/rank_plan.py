"""Structural EP4 c/K/steal admission derived from routed work.

The complete Workload route ledger fixes payload and task counts. This plan
adds per-rank CTA partitions and uneven token chunks without pretending to
prove device progress, peer memory order, compiled occupancy or latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .dispatch_ledger import DispatchLedger


@dataclass(frozen=True)
class Chunk:
    first_token: int
    stop_token: int
    expected_contributions: int


@dataclass(frozen=True)
class RankPlan:
    rank: int
    communication_ctas: int
    compute_ctas: int
    steal_budget: int
    chunks: tuple[Chunk, ...]
    inbound_payload_slots: int
    compute_task_slots: int


def _controls(values: Sequence[int], ranks: int, name: str) -> tuple[int, ...]:
    if (not isinstance(values, (list, tuple)) or len(values) != ranks
            or any(type(value) is not int for value in values)):
        raise ValueError(f'{name} must name one INT32 value per rank')
    return tuple(values)


def rank_plans(shape: Mapping, ledger: DispatchLedger, *, sm_count: int,
               communication_ctas: Sequence[int], chunks: Sequence[int],
               steal_budgets: Sequence[int]) -> tuple[RankPlan, ...]:
    """Check c/K/steal domains and derive each origin's completion thresholds."""
    if not isinstance(shape, Mapping) or not {'R', 'T', 'K'} <= set(shape):
        raise ValueError('rank plan needs the Workload route geometry')
    ranks, tokens, top_k = (shape[name] for name in ('R', 'T', 'K'))
    if (any(type(value) is not int for value in (ranks, tokens, top_k, sm_count))
            or ranks < 2 or tokens < 1 or top_k < 1 or sm_count < 2
            or len(ledger.payloads_by_rank) != ranks
            or len(ledger.tasks_by_rank) != ranks):
        raise ValueError('rank plan geometry or ledger rank count differs')
    c = _controls(communication_ctas, ranks, 'communication_ctas')
    k = _controls(chunks, ranks, 'chunks')
    budgets = _controls(steal_budgets, ranks, 'steal_budgets')
    tasks = tuple(task for rank_tasks in ledger.tasks_by_rank for task in rank_tasks)
    return_keys = {(task.source_rank, task.token, task.route) for task in tasks}
    if (len(tasks) != ranks * tokens * top_k
            or len(return_keys) != len(tasks)
            or return_keys != {(source, token, route)
                               for source in range(ranks) for token in range(tokens)
                               for route in range(top_k)}
            or any(not 0 <= task.source_rank < ranks or not 0 <= task.token < tokens
                   or not 0 <= task.route < top_k for task in tasks)):
        raise ValueError('rank plan needs one complete return domain')
    result = []
    for rank in range(ranks):
        if not 0 < c[rank] < sm_count:
            raise ValueError(f'rank {rank} needs nonempty communication and compute CTA classes')
        if not 1 <= k[rank] <= tokens:
            raise ValueError(f'rank {rank} chunk count must cover its token domain')
        if not 0 <= budgets[rank] <= ledger.task_slots[rank]:
            raise ValueError(f'rank {rank} steal budget exceeds its compute tasks')
        base, extras = divmod(tokens, k[rank])
        boundaries = tuple(
            (index * base + min(index, extras),
             (index + 1) * base + min(index + 1, extras))
            for index in range(k[rank]))
        rank_chunks = []
        for first, stop in boundaries:
            expected = sum(task.source_rank == rank and first <= task.token < stop
                           for task in tasks)
            if not first < stop or expected != (stop - first) * top_k:
                raise ValueError(f'rank {rank} chunk has missing or duplicate contributions')
            rank_chunks.append(Chunk(first, stop, expected))
        result.append(RankPlan(rank, c[rank], sm_count - c[rank], budgets[rank],
                               tuple(rank_chunks), ledger.payload_slots[rank],
                               ledger.task_slots[rank]))
    return tuple(result)
