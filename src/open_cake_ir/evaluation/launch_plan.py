"""Immutable, typed, same-stream composition of independently lowered Cake Schedules.

Evaluation owns tensor allocation and launch order. No task mathematics or task modules
belong here. Each kernel's computation remains visible in its complete Schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from collections.abc import Mapping, Callable
import json
import math

from open_cake_ir.compiler.ir import Schedule, DType, BufferMode, MemorySpace
from open_cake_ir.serialization import canonical_json_bytes


@dataclass(frozen=True)
class PlanTensor:
    shape: tuple[int, ...]
    dtype: DType

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class PlanStage:
    name: str
    schedule_bytes: bytes
    bindings: Mapping[str, str]

    @property
    def schedule(self):
        return Schedule.from_dict(json.loads(self.schedule_bytes))


@dataclass(frozen=True)
class LaunchPlan:
    plan_id: str
    target: str
    tensors: Mapping[str, PlanTensor]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    stages: tuple[PlanStage, ...]
    document_bytes: bytes

    @classmethod
    def from_dict(cls, document):
        fields = {'schema_version','plan_id','target','tensors','inputs','outputs','stages'}
        if not isinstance(document, Mapping) or set(document) != fields or document['schema_version'] != 1:
            raise ValueError('launch plan fields or schema_version differ')
        for field in ('plan_id','target'):
            if not isinstance(document[field], str) or not document[field]:
                raise ValueError(f'launch plan {field} must be nonempty text')
        tensors = {}
        if not isinstance(document['tensors'], Mapping) or not document['tensors']:
            raise ValueError('launch plan tensors must be a nonempty object')
        for name, spec in document['tensors'].items():
            if not isinstance(name, str) or not name or not isinstance(spec, Mapping) or set(spec) != {'shape','dtype'}:
                raise ValueError('launch plan tensor declaration differs')
            shape = spec['shape']
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f'launch plan tensor {name!r} shape differs')
            tensors[name] = PlanTensor(tuple(shape), DType(spec['dtype']))
        io = {}
        for kind in ('inputs','outputs'):
            names = document[kind]
            if not isinstance(names, list) or not names or any(not isinstance(n,str) or n not in tensors for n in names) or len(set(names)) != len(names):
                raise ValueError(f'launch plan {kind} must name unique declared tensors')
            io[kind] = tuple(names)
        if set(io['inputs']) & set(io['outputs']):
            raise ValueError('launch plan public inputs are immutable and cannot be outputs')
        raw_stages = document['stages']
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError('launch plan requires at least one stage')
        available = set(io['inputs'])
        producers = set()
        consumed = set()
        names = set()
        stages = []
        for raw in raw_stages:
            if not isinstance(raw, Mapping) or set(raw) != {'name','schedule','bindings'}:
                raise ValueError('launch plan stage fields differ')
            name = raw['name']
            if not isinstance(name,str) or not name or name in names:
                raise ValueError('launch stage names must be unique')
            names.add(name)
            schedule = Schedule.from_dict(raw['schedule'])
            if schedule.target != document['target']:
                raise ValueError(f'launch stage {name!r} target differs')
            globals_ = {b.name:b for b in schedule.buffers if b.space is MemorySpace.GLOBAL}
            bindings = raw['bindings']
            if not isinstance(bindings, Mapping) or set(bindings) != set(globals_):
                raise ValueError(f'launch stage {name!r} bindings must cover every global')
            if any(not isinstance(value,str) or value not in tensors for value in bindings.values()) or len(set(bindings.values())) != len(bindings):
                raise ValueError(f'launch stage {name!r} bindings cannot alias tensors')
            outputs = {b.name for b in globals_.values() if b.mode is BufferMode.OUTPUT}
            if set(schedule.outputs) != outputs or not outputs:
                raise ValueError(f'launch stage {name!r} must export every written global')
            stage_writes = set()
            for local, buffer in globals_.items():
                tensor_name = bindings[local]
                tensor = tensors[tensor_name]
                if tensor.shape != buffer.shape or tensor.dtype is not buffer.dtype:
                    raise ValueError(f'launch stage {name!r} binding {local!r} shape/dtype differs')
                if buffer.mode is BufferMode.INPUT:
                    if tensor_name not in available:
                        raise ValueError(f'launch stage {name!r} reads {tensor_name!r} before its producer')
                    consumed.add(tensor_name)
                elif buffer.mode is BufferMode.OUTPUT:
                    if tensor_name in available:
                        raise ValueError(f'launch stage {name!r} overwrites {tensor_name!r}; one producer required')
                    stage_writes.add(tensor_name)
                else:
                    raise ValueError('launch plan admits immutable inputs and fresh outputs, not state/scratch globals')
            producers.update(stage_writes)
            available.update(stage_writes)
            stages.append(PlanStage(name, canonical_json_bytes(raw['schedule']), MappingProxyType(dict(bindings))))
        if not set(io['outputs']) <= producers:
            raise ValueError('launch plan has unproduced public outputs')
        if not set(io['inputs']) <= consumed:
            raise ValueError('launch plan ignores a public input')
        intermediates = set(tensors) - set(io['inputs']) - set(io['outputs'])
        if not intermediates <= producers & consumed:
            raise ValueError('launch plan intermediates require a producer and a consumer')
        return cls(document['plan_id'], document['target'], MappingProxyType(tensors),
                   io['inputs'], io['outputs'], tuple(stages), canonical_json_bytes(document))

    @property
    def allocated_bytes(self):
        """Conservative live allocation: output and intermediate tensors are retained."""
        return sum(tensor.nbytes for name,tensor in self.tensors.items() if name not in self.inputs)

    def compile(self, compiler):
        if compiler.commit is None:
            raise ValueError('launch plan compilation requires a clean Compiler commit')
        lowerings = []
        for stage in self.stages:
            assessment = compiler.assess(json.loads(stage.schedule_bytes))
            if not assessment.lowering_eligible:
                raise ValueError(f'launch stage {stage.name!r} refused: {[f.code for f in assessment.findings]}')
            lowerings.append(compiler.lower(assessment))
        return CompiledLaunchPlan(self, lowerings[0].compiler_revision_id, tuple(lowerings))


@dataclass(frozen=True)
class CompiledLaunchPlan:
    plan: LaunchPlan
    compiler_revision_id: str
    lowerings: tuple

    def prepare(self, inputs: Mapping[str, object], *, allocate: Callable,
                load_kernel: Callable, check_tensor: Callable, storage_span: Callable, execution_context: Callable):
        """Bind all storage before timing. The caller supplies platform-owned adapters.

        allocate(name, PlanTensor), check_tensor(tensor, PlanTensor), and
        load_kernel(stage_name, Lowering) perform platform binding; none supplies math.
        Source modules must be loaded from the exact Lowering the Compiler produced.
        """
        if set(inputs) != set(self.plan.inputs):
            raise ValueError('launch plan public input set differs')
        buffers = dict(inputs)
        for name, spec in self.plan.tensors.items():
            if name not in buffers:
                buffers[name] = allocate(name, spec)
            check_tensor(buffers[name], spec)
        spans = []
        for name, tensor in buffers.items():
            device, start, end = storage_span(tensor)
            if type(start) is not int or type(end) is not int or not 0 < start < end:
                raise ValueError('launch plan requires exact nonempty storage byte intervals')
            if any(device == other_device and start < other_end and other_start < end
                   for other_device, other_start, other_end in spans):
                raise ValueError(f'launch plan storage for {name!r} overlaps another tensor')
            spans.append((device, start, end))
        calls = []
        for stage, lowering in zip(self.plan.stages, self.lowerings, strict=True):
            if lowering.target != self.plan.target or not lowering.generated:
                raise ValueError('launch plan stage must bind generated source for its exact target')
            schedule = stage.schedule
            arguments = {b.name:buffers[stage.bindings[b.name]] for b in schedule.buffers
                         if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT}
            outputs = tuple(buffers[stage.bindings[name]] for name in schedule.outputs)
            kernel = load_kernel(stage.name, lowering)
            calls.append((kernel, arguments, outputs[0] if len(outputs)==1 else outputs))
        return PreparedLaunchPlan(tuple(calls), MappingProxyType({name:buffers[name] for name in self.plan.outputs}),
                                  MappingProxyType(buffers), execution_context, execution_context())


@dataclass
class PreparedLaunchPlan:
    calls: tuple
    outputs: Mapping[str, object]
    buffers: Mapping[str, object]
    execution_context: Callable
    bound_context: object
    launch_calls: int = 0
    _lock: object = field(default_factory=Lock, repr=False)

    def run(self):
        """One invocation contains every ordered stage; no task work on the host."""
        with self._lock:
            for kernel, inputs, outputs in self.calls:
                if self.execution_context() != self.bound_context:
                    raise ValueError('launch plan execution device/stream changed')
                kernel(**inputs, out=outputs)
                self.launch_calls += 1
            if self.execution_context() != self.bound_context:
                raise ValueError('launch plan execution device/stream changed')
            return self.outputs
