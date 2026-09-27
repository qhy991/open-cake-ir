"""Complete static same-stream Programs; no runtime, Workload or Lab dependencies.

Global tensors are single-assignment. Stages contain complete Schedules, with exact
bindings and an optional explicit singleton-axis view. Construction proves composition
legality; the Compiler separately assesses each Schedule against its exact Target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
import json
import math

from .schedule import Schedule
from .vocabulary import DType, BufferMode, MemorySpace
from open_cake_ir.serialization import canonical_json_bytes


@dataclass(frozen=True)
class ProgramTensor:
    shape: tuple[int, ...]
    dtype: DType

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class ProgramBinding:
    tensor: str
    singleton_view: bool = False


@dataclass(frozen=True)
class ProgramStage:
    name: str
    schedule_bytes: bytes
    bindings: Mapping[str, ProgramBinding]
    _schedule: Schedule = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Bytes are the stage authority. Replacing a frozen dataclass field must
        # reconstruct the typed view instead of carrying a stale cached Schedule.
        if type(self.schedule_bytes) is not bytes:
            raise TypeError('ProgramStage schedule_bytes must be immutable bytes')
        if not isinstance(self.bindings, Mapping):
            raise TypeError('ProgramStage bindings must be a mapping')
        object.__setattr__(self, 'bindings', MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, '_schedule', Schedule.from_dict(json.loads(self.schedule_bytes)))

    @property
    def schedule(self):
        return self._schedule


@dataclass(frozen=True)
class Program:
    program_id: str
    target: str
    tensors: Mapping[str, ProgramTensor]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    stages: tuple[ProgramStage, ...]

    @classmethod
    def from_dict(cls, document):
        fields = {'schema_version','program_id','target','tensors','inputs','outputs','stages'}
        if not isinstance(document, Mapping) or set(document) != fields or type(document['schema_version']) is not int or document['schema_version'] != 1:
            raise ValueError('program fields or schema_version differ')
        for field in ('program_id','target'):
            if not isinstance(document[field], str) or not document[field]:
                raise ValueError(f'program {field} must be nonempty text')
        tensors = {}
        if not isinstance(document['tensors'], Mapping) or not document['tensors']:
            raise ValueError('program tensors must be a nonempty object')
        for name, spec in document['tensors'].items():
            if not isinstance(name, str) or not name or not isinstance(spec, Mapping) or set(spec) != {'shape','dtype'}:
                raise ValueError('program tensor declaration differs')
            shape = spec['shape']
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f'program tensor {name!r} shape differs')
            tensors[name] = ProgramTensor(tuple(shape), DType(spec['dtype']))
        io = {}
        for kind in ('inputs','outputs'):
            names = document[kind]
            if not isinstance(names, list) or not names or any(not isinstance(n,str) or n not in tensors for n in names) or len(set(names)) != len(names):
                raise ValueError(f'program {kind} must name unique declared tensors')
            io[kind] = tuple(names)
        if set(io['inputs']) & set(io['outputs']):
            raise ValueError('program public inputs are immutable and cannot be outputs')
        raw_stages = document['stages']
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError('program requires at least one stage')
        available = set(io['inputs'])
        producers = set()
        consumed = set()
        names = set()
        stages = []
        for raw in raw_stages:
            if not isinstance(raw, Mapping) or set(raw) != {'name','schedule','bindings'}:
                raise ValueError('program stage fields differ')
            name = raw['name']
            if not isinstance(name,str) or not name or name in names:
                raise ValueError('program stage names must be unique')
            names.add(name)
            schedule = Schedule.from_dict(raw['schedule'])
            if schedule.target != document['target']:
                raise ValueError(f'program stage {name!r} target differs')
            globals_ = {b.name:b for b in schedule.buffers if b.space is MemorySpace.GLOBAL}
            raw_bindings = raw['bindings']
            if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != set(globals_):
                raise ValueError(f'program stage {name!r} bindings must cover every global')
            bindings = {}
            for local, value in raw_bindings.items():
                if isinstance(value, str):
                    binding = ProgramBinding(value)
                elif (isinstance(value, Mapping) and set(value) == {'tensor', 'view'}
                      and value['view'] == 'singleton_axes' and isinstance(value['tensor'], str)):
                    binding = ProgramBinding(value['tensor'], True)
                else:
                    raise ValueError(f'program stage {name!r} binding {local!r} is invalid')
                if binding.tensor not in tensors:
                    raise ValueError(f'program stage {name!r} binding {local!r} names an undeclared tensor')
                bindings[local] = binding
            if len({binding.tensor for binding in bindings.values()}) != len(bindings):
                raise ValueError(f'program stage {name!r} bindings cannot alias tensors')
            outputs = {b.name for b in globals_.values() if b.mode is BufferMode.OUTPUT}
            if set(schedule.outputs) != outputs or not outputs:
                raise ValueError(f'program stage {name!r} must export every written global')
            stage_writes = set()
            for local, buffer in globals_.items():
                binding = bindings[local]
                tensor_name = binding.tensor
                tensor = tensors[tensor_name]
                same_shape = tensor.shape == buffer.shape
                if binding.singleton_view:
                    same_shape = tuple(n for n in tensor.shape if n != 1) == tuple(n for n in buffer.shape if n != 1)
                    if tensor.shape == buffer.shape:
                        raise ValueError(f'program stage {name!r} has an unnecessary singleton view')
                if not same_shape or tensor.dtype is not buffer.dtype:
                    raise ValueError(f'program stage {name!r} binding {local!r} shape/dtype differs')
                if buffer.mode is BufferMode.INPUT:
                    if tensor_name not in available:
                        raise ValueError(f'program stage {name!r} reads {tensor_name!r} before its producer')
                    consumed.add(tensor_name)
                elif buffer.mode is BufferMode.OUTPUT:
                    if tensor_name in available:
                        raise ValueError(f'program stage {name!r} overwrites {tensor_name!r}; one producer required')
                    stage_writes.add(tensor_name)
                else:
                    raise ValueError('program admits immutable inputs and fresh outputs, not state/scratch globals')
            producers.update(stage_writes)
            available.update(stage_writes)
            stages.append(ProgramStage(name, canonical_json_bytes(raw['schedule']), bindings))
        if not set(io['outputs']) <= producers:
            raise ValueError('program has unproduced public outputs')
        if not set(io['inputs']) <= consumed:
            raise ValueError('program ignores a public input')
        intermediates = set(tensors) - set(io['inputs']) - set(io['outputs'])
        if not intermediates <= producers & consumed:
            raise ValueError('program intermediates require a producer and a consumer')
        return cls(document['program_id'], document['target'], MappingProxyType(tensors),
                   io['inputs'], io['outputs'], tuple(stages))

    @property
    def allocated_bytes(self):
        """Conservative live allocation: output and intermediate tensors are retained."""
        return sum(tensor.nbytes for name,tensor in self.tensors.items() if name not in self.inputs)

    @property
    def document(self) -> dict:
        """A fresh projection, not mutable Program authority."""
        return {
            'schema_version': 1, 'program_id': self.program_id, 'target': self.target,
            'tensors': {name: {'shape': list(tensor.shape), 'dtype': tensor.dtype.value}
                        for name, tensor in self.tensors.items()},
            'inputs': list(self.inputs), 'outputs': list(self.outputs),
            'stages': [{'name': stage.name, 'schedule': json.loads(stage.schedule_bytes),
                        'bindings': {name: {'tensor': binding.tensor, 'view': 'singleton_axes'}
                                     if binding.singleton_view else binding.tensor
                                     for name, binding in stage.bindings.items()}}
                       for stage in self.stages],
        }

    @property
    def document_bytes(self) -> bytes:
        return canonical_json_bytes(self.document)

    @classmethod
    def from_schedule(cls, document: Mapping) -> Program:
        """Lift an ordinary immutable-input/fresh-output Schedule without boilerplate."""
        schedule = Schedule.from_dict(document)
        tensors = {b.name: {'shape': list(b.shape), 'dtype': b.dtype.value}
                   for b in schedule.buffers if b.space is MemorySpace.GLOBAL}
        return cls.from_dict({
            'schema_version': 1, 'program_id': schedule.schedule_id,
            'target': schedule.target, 'tensors': tensors,
            'inputs': [b.name for b in schedule.buffers
                       if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT],
            'outputs': list(schedule.outputs),
            'stages': [{'name': schedule.schedule_id, 'schedule': dict(document),
                        'bindings': {name: name for name in tensors}}],
        })

    def consumers(self, tensor: str) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages
                     if any(b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT
                            and stage.bindings[b.name].tensor == tensor for b in stage.schedule.buffers))
