"""Derived EP dispatch ledger: shared remote payloads and per-route tasks.

The Workload's rank/token/expert IDs are the input authority. This pure CPU
projection owns neither Cake IR admission nor a GPU mailbox implementation.
It separates the route count from the smaller deduplicated communication
count that a future ranked-mailbox effect must preserve.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemotePayload:
    source_rank: int
    token: int
    destination_rank: int


@dataclass(frozen=True)
class ExpertTask:
    source_rank: int
    token: int
    route: int
    expert: int
    destination_rank: int
    local_expert: int
    remote_payload_slot: int | None


@dataclass(frozen=True)
class DispatchLedger:
    payloads_by_rank: tuple[tuple[RemotePayload, ...], ...]
    tasks_by_rank: tuple[tuple[ExpertTask, ...], ...]

    @property
    def payload_slots(self) -> tuple[int, ...]:
        return tuple(len(rows) for rows in self.payloads_by_rank)

    @property
    def task_slots(self) -> tuple[int, ...]:
        return tuple(len(rows) for rows in self.tasks_by_rank)


def dispatch_ledger(shape, expert_ids) -> DispatchLedger:
    """Project one valid Workload route array to rank-owned mailbox slots.

    One remote payload is sent per `(source_rank, token, destination_rank)`;
    all expert routes to that destination refer to its slot. Local routes
    read the source rank's hidden tensor and need no remote payload slot.
    Every route still has one compute task and one return contribution.
    """
    if not isinstance(shape, dict) or not {'R', 'T', 'E', 'K'} <= set(shape):
        raise ValueError('EP route geometry is incomplete')
    ranks, tokens, experts, top_k = (shape[name] for name in ('R', 'T', 'E', 'K'))
    if (any(type(value) is not int for value in (ranks, tokens, experts, top_k))
            or ranks < 2 or tokens < 1 or experts < ranks
            or experts % ranks or not 1 <= top_k <= experts):
        raise ValueError('EP route geometry needs positive equal expert shards')
    if (len(expert_ids) != ranks
            or any(len(rank_rows) != tokens for rank_rows in expert_ids)):
        raise ValueError('EP route rank/token dimensions differ')
    experts_per_rank = experts // ranks
    payloads: list[list[RemotePayload]] = [[] for _ in range(ranks)]
    tasks: list[list[ExpertTask]] = [[] for _ in range(ranks)]
    for source, rank_rows in enumerate(expert_ids):
        for token, selected in enumerate(rank_rows):
            if (len(selected) != top_k
                    or any(type(expert) is not int or not 0 <= expert < experts
                           for expert in selected)
                    or len(set(selected)) != top_k):
                raise ValueError('EP expert IDs must be distinct and in range per token')
            payload_slot_by_destination: dict[int, int] = {}
            for route, expert in enumerate(selected):
                destination = expert // experts_per_rank
                slot = None
                if destination != source:
                    if destination not in payload_slot_by_destination:
                        payload_slot_by_destination[destination] = len(payloads[destination])
                        payloads[destination].append(
                            RemotePayload(source, token, destination))
                    slot = payload_slot_by_destination[destination]
                tasks[destination].append(ExpertTask(
                    source, token, route, expert, destination,
                    expert % experts_per_rank, slot))
    result = DispatchLedger(tuple(tuple(rows) for rows in payloads),
                            tuple(tuple(rows) for rows in tasks))
    if (any(count > (ranks - 1) * tokens for count in result.payload_slots)
            or any(count > ranks * tokens * top_k for count in result.task_slots)
            or sum(result.task_slots) != ranks * tokens * top_k):
        raise AssertionError('derived mailbox capacity or task conservation differs')
    return result
