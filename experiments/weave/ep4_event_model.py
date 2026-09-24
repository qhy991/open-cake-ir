"""Bounded fair-interleaving model of the EP4 dispatch/compute/combine DAG.

Each event completes one logical route task or token combine. Publication
sets a payload-ready flag before its dependent task-ready flag. This model
checks unique claims, steal bounds and chunk progress under many orders; it
does not model GPU memory visibility, CTA residency, instruction latency or
prove physical liveness.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from collections.abc import Mapping

from .dispatch_ledger import DispatchLedger, ExpertTask
from .rank_plan import RankPlan


@dataclass(frozen=True)
class Event:
    rank: int
    phase: str
    source_rank: int
    token: int
    route: int | None
    stolen: bool = False


@dataclass(frozen=True)
class EventTrace:
    completed: bool
    events: tuple[Event, ...]
    stolen_by_rank: tuple[int, ...]
    first_combine_compute_done: tuple[int | None, ...]
    computed_tasks: int
    combined_tokens: int
    blocked_reason: str | None


def simulate_ep4(shape: Mapping, ledger: DispatchLedger,
                 plans: tuple[RankPlan, ...], *, seed: int) -> EventTrace:
    """Explore one seeded order while keeping dispatchers and workers fair."""
    if not isinstance(shape, Mapping) or not {'R', 'T', 'K'} <= set(shape):
        raise ValueError('event model needs Workload route geometry')
    ranks, tokens, top_k = (shape[name] for name in ('R', 'T', 'K'))
    if (any(type(value) is not int for value in (ranks, tokens, top_k, seed))
            or ranks < 2 or tokens < 1 or top_k < 1
            or len(plans) != ranks or len(ledger.tasks_by_rank) != ranks
            or len(ledger.payloads_by_rank) != ranks
            or any(plan.rank != index or not plan.chunks
                   for index, plan in enumerate(plans))):
        raise ValueError('event model rank plan or geometry differs')
    task_by_key: dict[tuple[int, int, int], tuple[int, int, ExpertTask]] = {}
    for destination, tasks in enumerate(ledger.tasks_by_rank):
        if (plans[destination].compute_task_slots != len(tasks)
                or plans[destination].inbound_payload_slots
                != len(ledger.payloads_by_rank[destination])):
            raise ValueError('rank plan capacity differs from routed ledger')
        for index, task in enumerate(tasks):
            key = (task.source_rank, task.token, task.route)
            if key in task_by_key or task.destination_rank != destination:
                raise ValueError('event model task identity or destination differs')
            task_by_key[key] = destination, index, task
    if set(task_by_key) != {(source, token, route)
                           for source in range(ranks) for token in range(tokens)
                           for route in range(top_k)}:
        raise ValueError('event model needs every route exactly once')
    if len({plan.communication_ctas + plan.compute_ctas for plan in plans}) != 1:
        raise ValueError('event model rank grids have different CTA extents')
    for plan in plans:
        if (plan.communication_ctas < 1 or plan.compute_ctas < 1
                or plan.steal_budget < 0 or plan.steal_budget > plan.compute_task_slots
                or plan.chunks[0].first_token != 0
                or plan.chunks[-1].stop_token != tokens
                or any(chunk.expected_contributions !=
                       (chunk.stop_token - chunk.first_token) * top_k
                       for chunk in plan.chunks)
                or any(first.stop_token != second.first_token
                       for first, second in zip(plan.chunks, plan.chunks[1:], strict=False))):
            raise ValueError('event model needs a complete bounded rank plan')
    rng = random.Random(seed)
    dispatched = [0] * ranks
    payload_ready: list[set[int]] = [set() for _ in range(ranks)]
    task_ready: list[set[int]] = [set() for _ in range(ranks)]
    claimed: list[set[int]] = [set() for _ in range(ranks)]
    contributions: set[tuple[int, int, int]] = set()
    combined = [0] * ranks
    stolen = [0] * ranks
    comm_done = [False] * ranks
    first_combine: list[int | None] = [None] * ranks
    events: list[Event] = []
    total_routes = ranks * tokens * top_k
    total_combines = ranks * tokens

    def all_dispatch_done() -> bool:
        return all(cursor == tokens * top_k for cursor in dispatched)

    def chunk_ready(source: int, index: int) -> bool:
        chunk = plans[source].chunks[index]
        return sum((source, token, route) in contributions
                   for token in range(chunk.first_token, chunk.stop_token)
                   for route in range(top_k)) == chunk.expected_contributions

    def available(destination: int) -> tuple[int, ...]:
        return tuple(sorted(task_ready[destination] - claimed[destination]))

    while len(events) < 2 * total_routes + total_combines:
        for rank, plan in enumerate(plans):
            if (not comm_done[rank] and dispatched[rank] == tokens * top_k
                    and (stolen[rank] >= plan.steal_budget
                         or chunk_ready(rank, 0)
                         or all_dispatch_done()
                            and len(claimed[rank]) == plan.compute_task_slots)):
                comm_done[rank] = True
        if (len(contributions) == total_routes
                and sum(combined) == total_combines):
            break
        actions: list[tuple[str, int, int | None]] = []
        weights: list[int] = []
        for rank, plan in enumerate(plans):
            if dispatched[rank] < tokens * top_k:
                actions.append(('dispatch', rank, None))
                weights.append(plan.communication_ctas)
            for task_index in available(rank):
                actions.append(('compute', rank, task_index))
                weights.append(plan.compute_ctas)
                if (dispatched[rank] == tokens * top_k and not comm_done[rank]
                        and stolen[rank] < plan.steal_budget
                        and not chunk_ready(rank, 0)):
                    actions.append(('steal', rank, task_index))
                    weights.append(plan.communication_ctas)
            if comm_done[rank] and combined[rank] < tokens:
                token = combined[rank]
                chunk = next((index for index, row in enumerate(plan.chunks)
                              if row.first_token <= token < row.stop_token), None)
                if (chunk is not None and chunk_ready(rank, chunk)
                        and all((rank, token, route) in contributions
                                for route in range(top_k))):
                    actions.append(('combine', rank, None))
                    weights.append(plan.communication_ctas + plan.compute_ctas)
        if not actions:
            return EventTrace(False, tuple(events), tuple(stolen),
                              tuple(first_combine), len(contributions),
                              sum(combined), 'no enabled action before all routes and tokens complete')
        phase, rank, task_index = rng.choices(actions, weights=weights, k=1)[0]
        if phase == 'dispatch':
            cursor = dispatched[rank]
            token, route = divmod(cursor, top_k)
            destination, index, task = task_by_key[(rank, token, route)]
            if task.remote_payload_slot is not None:
                payload = ledger.payloads_by_rank[destination][task.remote_payload_slot]
                if ((payload.source_rank, payload.token, payload.destination_rank)
                        != (rank, token, destination)):
                    raise ValueError('remote payload ownership differs from its task')
                payload_ready[destination].add(task.remote_payload_slot)
            task_ready[destination].add(index)
            dispatched[rank] += 1
            events.append(Event(rank, 'dispatch', rank, token, route))
        elif phase in ('compute', 'steal'):
            assert task_index is not None
            task = ledger.tasks_by_rank[rank][task_index]
            if (task.remote_payload_slot is not None
                    and task.remote_payload_slot not in payload_ready[rank]):
                raise ValueError('compute claim preceded remote payload publication')
            claimed[rank].add(task_index)
            contributions.add((task.source_rank, task.token, task.route))
            if phase == 'steal':
                stolen[rank] += 1
            events.append(Event(rank, phase, task.source_rank,
                                task.token, task.route, phase == 'steal'))
        else:
            if first_combine[rank] is None:
                first_combine[rank] = len(contributions)
            token = combined[rank]
            combined[rank] += 1
            events.append(Event(rank, 'combine', rank, token, None))
    complete = (len(contributions) == total_routes
                and sum(combined) == total_combines)
    return EventTrace(complete, tuple(events), tuple(stolen),
                      tuple(first_combine), len(contributions),
                      sum(combined), None if complete else 'event bound exhausted')
