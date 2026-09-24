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
class WorkerExecution:
    lowering: LoweringRoute
    controls: Mapping[str, str]
    workers: tuple[WorkerClass, ...]
    queues: tuple[WorkerQueue, ...]
    handoffs: tuple[WorkerHandoff, ...]
    steal: StealWindow

    @classmethod
    def from_dict(cls, value, *, tensors: Mapping, inputs: tuple[str, ...],
                  stages: Sequence, producers: Mapping[str, str],
                  consumers: Mapping[str, set[str]], intermediates: set[str]):
        raw = _object(value, {"kind", "lowering", "controls", "workers", "queues",
                              "handoffs", "steal"}, "worker execution")
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
        return cls(lowering, MappingProxyType(selected), tuple(workers), tuple(queues),
                   tuple(handoffs), steal)

    @property
    def document(self) -> dict:
        return {
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
