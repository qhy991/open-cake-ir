"""Platform-owned allocation and execution of Compiler-lowered Programs."""
from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from collections.abc import Mapping, Callable
from open_cake_ir.compiler.ir import ProgramTensor, MemorySpace, BufferMode
from open_cake_ir.compiler.program import LoweredProgram


def prepare_program(lowered: LoweredProgram, inputs: Mapping[str, object], *, allocate: Callable,
            load_kernel: Callable, check_tensor: Callable, storage_span: Callable, execution_context: Callable, view_tensor: Callable | None = None):
    """Bind all storage before timing. The caller supplies platform-owned adapters.

    allocate(name, ProgramTensor), check_tensor(tensor, ProgramTensor), and
    load_kernel(stage_name, Lowering) perform platform binding; none supplies math.
    Source modules must be loaded from the exact Lowering the Compiler produced.
    """
    lowered.validate_binding()
    if set(inputs) != set(lowered.program.inputs):
        raise ValueError('launch plan public input set differs')
    bound_context = execution_context()
    buffers = dict(inputs)
    for name, spec in lowered.program.tensors.items():
        if name not in buffers:
            buffers[name] = allocate(name, spec)
        check_tensor(buffers[name], spec)
    spans = []
    for name, tensor in buffers.items():
        device, start, end = storage_span(tensor)
        if type(start) is not int or type(end) is not int or not 0 < start < end:
            raise ValueError('launch plan requires exact nonempty storage byte intervals')
        if spans and device != spans[0][0]:
            raise ValueError('program tensors must share one device')
        if end - start != lowered.program.tensors[name].nbytes:
            raise ValueError('program storage interval differs from its tensor extent')
        if any(device == other_device and start < other_end and other_start < end
               for other_device, other_start, other_end in spans):
            raise ValueError(f'launch plan storage for {name!r} overlaps another tensor')
        spans.append((device, start, end))
    def bound(stage, buffer):
        binding = stage.bindings[buffer.name]
        tensor = buffers[binding.tensor]
        if not binding.singleton_view:
            return tensor
        if view_tensor is None:
            raise ValueError('program singleton view needs a platform view adapter')
        viewed = view_tensor(tensor, buffer.shape)
        check_tensor(viewed, ProgramTensor(buffer.shape, buffer.dtype))
        if storage_span(viewed) != storage_span(tensor):
            raise ValueError('program singleton view must preserve its exact storage interval')
        return viewed
    calls = []
    for stage, lowering in zip(lowered.program.stages, lowered.lowerings, strict=True):
        if lowering.target != lowered.program.target or not lowering.generated:
            raise ValueError('launch plan stage must bind generated source for its exact target')
        schedule = stage.schedule
        arguments = {b.name:bound(stage, b) for b in schedule.buffers
                     if b.space is MemorySpace.GLOBAL and b.mode is BufferMode.INPUT}
        outputs = tuple(bound(stage, schedule.buffer(name)) for name in schedule.outputs)
        kernel = load_kernel(stage.name, lowering)
        calls.append((kernel, arguments, outputs[0] if len(outputs)==1 else outputs))
    if execution_context() != bound_context:
        raise ValueError('program execution device/stream changed during preparation')
    return PreparedProgram(tuple(calls), MappingProxyType({name:buffers[name] for name in lowered.program.outputs}),
                              MappingProxyType(buffers), execution_context, bound_context)

@dataclass
class PreparedProgram:
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
