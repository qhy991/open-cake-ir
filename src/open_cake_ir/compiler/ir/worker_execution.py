"""Typed control plane for a future cooperative Program lowering.

The mathematical stages remain complete Schedules in Program. This descriptor
owns worker/queue/handoff structure only; it does not admit a backend or prove
physical GPU progress. Compiler and Evaluation refuse it until one complete
emission and execution path is qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping, Sequence

from .schedule import LoweringRoute
from .vocabulary import DType


def _object(value, fields: set[str], context: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{context} fields differ")
    return value


def _name(value, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be nonempty text")
    return value


def _names(value, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty list")
    names = tuple(_name(item, context) for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{context} repeats a name")
    return names


@dataclass(frozen=True)
class WorkerClass:
    name: str
    phases: tuple[str, ...]


@dataclass(frozen=True)
class WorkerQueue:
    stage: str
    workers: tuple[str, ...]


class HandoffScope(str, Enum):
    DEVICE = "device"
    SYSTEM = "system"


@dataclass(frozen=True)
class WorkerHandoff:
    payload: str
    producer: str
    consumer: str
    scope: HandoffScope


@dataclass(frozen=True)
class StealWindow:
    borrower: str
    stage: str
    after: str
    before: str


@dataclass(frozen=True)
class RankPlacement:
    world_size: int
    state_rank: int
    worker_ranks: Mapping[str, int]
    tensor_ranks: Mapping[str, int]
    stage_ranks: Mapping[str, int]

    @classmethod
    def from_dict(cls, value, *, workers: tuple[WorkerClass, ...], stages: Sequence,
                  handoffs: tuple[WorkerHandoff, ...], tensors: Mapping,
                  producers: Mapping[str, str], outputs: tuple[str, ...],
                  controls: Mapping[str, str]):
        raw = _object(value, {"world_size", "state_rank", "worker_ranks",
                              "tensor_ranks"}, "worker execution.placement")
        world = raw["world_size"]
        if type(world) is not int or world != 2:
            raise ValueError("rank worker placement currently requires exactly two ranks")
        state_rank = raw["state_rank"]
        if type(state_rank) is not int or not 0 <= state_rank < world:
            raise ValueError("rank worker state owner differs")
        worker_names = {worker.name for worker in workers}
        owner_rows = raw["worker_ranks"]
        if not isinstance(owner_rows, Mapping) or set(owner_rows) != worker_names:
            raise ValueError("rank worker placement must own every CTA class")
        worker_ranks = dict(owner_rows)
        if (any(type(rank) is not int or not 0 <= rank < world
                for rank in worker_ranks.values())
                or set(worker_ranks.values()) != set(range(world))):
            raise ValueError("rank worker classes must populate both valid ranks")
        tensor_rows = raw["tensor_ranks"]
        if not isinstance(tensor_rows, Mapping) or set(tensor_rows) != set(tensors):
            raise ValueError("rank worker placement must own every Program tensor")
        tensor_ranks = dict(tensor_rows)
        if any(type(rank) is not int or not 0 <= rank < world
               for rank in tensor_ranks.values()):
            raise ValueError("rank tensor owner must be an in-range rank")
        if any(tensor_ranks[name] != state_rank for name in controls.values()):
            raise ValueError("rank worker controls must reside with queue state")
        stage_ranks = {}
        for stage in stages:
            owners = {worker_ranks[worker.name] for worker in workers
                      if stage.name in worker.phases}
            if len(owners) != 1:
                raise ValueError(f"stage {stage.name!r} needs one regular worker rank")
            stage_ranks[stage.name] = owners.pop()
        for handoff in handoffs:
            producer = stage_ranks[handoff.producer]
            consumer = stage_ranks[handoff.consumer]
            required_scope = (HandoffScope.SYSTEM if producer != consumer
                              else HandoffScope.DEVICE)
            if handoff.scope is not required_scope:
                raise ValueError(f"handoff {handoff.payload!r} scope differs from rank placement")
            if tensor_ranks[handoff.payload] not in {producer, consumer}:
                raise ValueError(f"handoff {handoff.payload!r} needs an endpoint owner")
        for tensor in outputs:
            if tensor_ranks[tensor] != stage_ranks[producers[tensor]]:
                raise ValueError(f"public output {tensor!r} needs its producing rank")
        return cls(world, state_rank, MappingProxyType(worker_ranks),
                   MappingProxyType(tensor_ranks), MappingProxyType(stage_ranks))

    @property
    def document(self) -> dict:
        return {"world_size": self.world_size, "state_rank": self.state_rank,
                "worker_ranks": dict(self.worker_ranks),
                "tensor_ranks": dict(self.tensor_ranks)}


@dataclass(frozen=True)
class WorkerExecution:
    lowering: LoweringRoute
    controls: Mapping[str, str]
    workers: tuple[WorkerClass, ...]
    queues: tuple[WorkerQueue, ...]
    handoffs: tuple[WorkerHandoff, ...]
    steal: StealWindow
    placement: RankPlacement | None = None

    @classmethod
    def from_dict(cls, value, *, tensors: Mapping, inputs: tuple[str, ...],
                  stages: Sequence, producers: Mapping[str, str],
                  consumers: Mapping[str, set[str]], intermediates: set[str],
                  outputs: tuple[str, ...], version: int):
        raw = _object(value, {"kind", "lowering", "controls", "workers", "queues",
                              "handoffs", "steal"} | ({"placement"} if version == 3
                                                       else set()), "worker execution")
        if raw["kind"] != "cooperative_workers":
            raise ValueError("worker execution kind differs")
        lowering = LoweringRoute.from_dict(raw["lowering"], "worker execution.lowering")
        controls = _object(raw["controls"],
                           {"first_class_ctas", "chunk_count", "steal_budget"},
                           "worker execution.controls")
        selected = {key: _name(name, f"worker execution.controls.{key}")
                    for key, name in controls.items()}
        if len(set(selected.values())) != len(selected):
            raise ValueError("worker execution controls must name distinct tensors")
        for key, name in selected.items():
            tensor = tensors.get(name)
            if (name not in inputs or tensor is None or tensor.dtype is not DType.INT32
                    or tensor.shape != (1,)):
                raise ValueError(f"worker execution control {key!r} needs one public INT32 scalar input")

        stage_names = tuple(stage.name for stage in stages)
        positions = {name: index for index, name in enumerate(stage_names)}
        raw_workers = raw["workers"]
        if not isinstance(raw_workers, list) or len(raw_workers) != 2:
            raise ValueError("worker execution requires exactly two CTA classes")
        workers = []
        for index, item in enumerate(raw_workers):
            row = _object(item, {"name", "phases"}, f"worker execution.workers[{index}]")
            name = _name(row["name"], f"worker execution.workers[{index}].name")
            phases = _names(row["phases"], f"worker execution.workers[{index}].phases")
            if any(phase not in positions for phase in phases):
                raise ValueError(f"worker {name!r} names a stage outside the Program")
            if list(phases) != sorted(phases, key=positions.get):
                raise ValueError(f"worker {name!r} phases must follow Program dataflow order")
            workers.append(WorkerClass(name, phases))
        worker_names = {worker.name for worker in workers}
        if len(worker_names) != 2:
            raise ValueError("worker execution CTA class names must be unique")

        raw_queues = raw["queues"]
        if not isinstance(raw_queues, list) or len(raw_queues) != len(stages):
            raise ValueError("worker execution requires exactly one queue per Program stage")
        queues = []
        for index, item in enumerate(raw_queues):
            row = _object(item, {"stage", "workers"}, f"worker execution.queues[{index}]")
            stage = _name(row["stage"], f"worker execution.queues[{index}].stage")
            eligible = _names(row["workers"], f"worker execution.queues[{index}].workers")
            if stage not in positions or not set(eligible) <= worker_names:
                raise ValueError(f"worker queue {stage!r} names an unknown stage or CTA class")
            if stage != stage_names[index] or eligible != tuple(
                    worker.name for worker in workers if worker.name in eligible):
                raise ValueError("worker queues follow Program stage and CTA class order")
            queues.append(WorkerQueue(stage, eligible))
        if {queue.stage for queue in queues} != set(stage_names):
            raise ValueError("worker queues must name every Program stage exactly once")

        raw_steal = _object(raw["steal"], {"borrower", "stage", "after", "before"},
                            "worker execution.steal")
        steal = StealWindow(*(_name(raw_steal[field], f"worker execution.steal.{field}")
                              for field in ("borrower", "stage", "after", "before")))
        borrower = workers[0]
        if steal.borrower != borrower.name or steal.stage not in positions:
            raise ValueError("steal window must borrow one stage for the first CTA class")
        if (steal.after not in borrower.phases or steal.before not in borrower.phases
                or borrower.phases.index(steal.after) >= borrower.phases.index(steal.before)
                or steal.stage in borrower.phases
                or not positions[steal.after] < positions[steal.stage] < positions[steal.before]):
            raise ValueError("steal window must lie between two ordered borrower phases")
        if not any(steal.stage in worker.phases for worker in workers[1:]):
            raise ValueError("a regular worker must own the borrowed stage")
        queue_by_stage = {queue.stage: queue for queue in queues}
        for stage in stage_names:
            regular = {worker.name for worker in workers if stage in worker.phases}
            eligible = regular | ({borrower.name} if stage == steal.stage else set())
            if set(queue_by_stage[stage].workers) != eligible:
                raise ValueError(f"queue {stage!r} workers differ from regular and stealing owners")

        raw_handoffs = raw["handoffs"]
        if not isinstance(raw_handoffs, list) or len(raw_handoffs) != len(intermediates):
            raise ValueError("worker execution requires one handoff per Program intermediate")
        handoffs = []
        for index, item in enumerate(raw_handoffs):
            row = _object(item, {"payload", "producer", "consumer", "order", "scope"},
                          f"worker execution.handoffs[{index}]")
            payload = _name(row["payload"], f"worker execution.handoffs[{index}].payload")
            producer = _name(row["producer"], f"worker execution.handoffs[{index}].producer")
            consumer = _name(row["consumer"], f"worker execution.handoffs[{index}].consumer")
            try:
                scope = HandoffScope(row["scope"])
            except (TypeError, ValueError):
                raise ValueError(f"handoff {payload!r} requires device or system scope") from None
            if (payload not in intermediates or producers.get(payload) != producer
                    or consumers.get(payload) != {consumer}
                    or row["order"] != "release_acquire"):
                raise ValueError(f"handoff {payload!r} differs from the Program dataflow or memory order")
            handoffs.append(WorkerHandoff(payload, producer, consumer, scope))
        if {handoff.payload for handoff in handoffs} != intermediates:
            raise ValueError("worker handoff payloads must be unique and complete")
        if handoffs != sorted(handoffs, key=lambda item: (positions[item.producer], item.payload)):
            raise ValueError("worker handoffs follow producer stage order")
        placement = (RankPlacement.from_dict(raw["placement"], workers=tuple(workers),
            stages=stages, handoffs=tuple(handoffs), tensors=tensors,
            producers=producers, outputs=outputs, controls=selected)
            if version == 3 else None)
        return cls(lowering, MappingProxyType(selected), tuple(workers), tuple(queues),
                   tuple(handoffs), steal, placement)

    @property
    def document(self) -> dict:
        document = {
            "kind": "cooperative_workers",
            "lowering": {"backend": self.lowering.backend.value,
                         "entry_point": self.lowering.entry_point},
            "controls": dict(self.controls),
            "workers": [{"name": worker.name, "phases": list(worker.phases)}
                        for worker in self.workers],
            "queues": [{"stage": queue.stage, "workers": list(queue.workers)}
                       for queue in self.queues],
            "handoffs": [{"payload": handoff.payload, "producer": handoff.producer,
                          "consumer": handoff.consumer, "order": "release_acquire",
                          "scope": handoff.scope.value} for handoff in self.handoffs],
            "steal": {"borrower": self.steal.borrower, "stage": self.steal.stage,
                      "after": self.steal.after, "before": self.steal.before},
        }
        if self.placement is not None:
            document["placement"] = self.placement.document
        return document
