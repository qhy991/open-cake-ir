"""Platform-owned allocation and execution of Compiler-lowered Programs."""
from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from collections.abc import Mapping, Callable
from open_cake_ir.compiler.ir import HandoffScope, ProgramTensor, MemorySpace, BufferMode
from open_cake_ir.compiler.program import LoweredProgram, LoweredWorkerProgram


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


def prepare_worker_program(lowered: LoweredWorkerProgram, inputs: Mapping[str, object], *,
            allocate: Callable, allocate_state: Callable, load_worker: Callable,
            check_tensor: Callable, storage_span: Callable, execution_context: Callable,
            read_status: Callable, launch_device: object | None = None):
    """Bind a cooperative Program's storage before any device launch.

    The backend declares an internal state extent and a host launch that resets
    it on the bound stream. Evaluation owns allocation, storage isolation and
    the post-launch status read. The supplied loader must use this exact
    LoweredWorkerProgram; no ordered-stage fallback is available.
    """
    lowered.validate_binding()
    program = lowered.program
    if program.execution is not None and program.execution.placement is not None:
        raise ValueError('rank-placed worker Program requires a distributed launch adapter')
    if program.execution is None or set(inputs) != set(program.inputs):
        raise ValueError('worker launch public input set or execution kind differs')
    requirements = lowered.toolchain_requirements
    state_bytes = requirements.get('state_bytes')
    status_offset = requirements.get('state_status_offset_bytes')
    host_abi = requirements.get('host_abi')
    if (type(state_bytes) is not int or state_bytes < 4 or state_bytes % 4
            or type(status_offset) is not int or status_offset < 0
            or status_offset % 4 or status_offset + 4 > state_bytes
            or requirements.get('argument_order') != list(program.tensors)
            or requirements.get('state_reset') != 'zero_before_each_launch_on_launch_stream'
            or not isinstance(host_abi, Mapping)
            or set(host_abi) != {'create', 'launch', 'destroy'}
            or any(not isinstance(name, str) or not name for name in host_abi.values())):
        raise ValueError('worker launch state, argument order or host ABI differs')
    system_payloads = {handoff.payload for handoff in program.execution.handoffs
                       if handoff.scope is HandoffScope.SYSTEM}
    if system_payloads and (launch_device is None
            or requirements.get('peer_payload_runtime_check') is not True):
        raise ValueError('system handoff needs an explicit launch device and peer-aware host ABI')
    bound_context = execution_context()
    buffers = dict(inputs)
    for name, spec in program.tensors.items():
        if name not in buffers:
            buffers[name] = allocate(name, spec)
        check_tensor(buffers[name], spec)
    buffers = {name: buffers[name] for name in program.tensors}
    state = allocate_state(state_bytes)
    spans = []
    entries = [(name, tensor, program.tensors[name].nbytes)
               for name, tensor in buffers.items()]
    entries.append((None, state, state_bytes))
    bound_device = launch_device
    for name, tensor, extent in entries:
        device, start, end = storage_span(tensor)
        if type(start) is not int or type(end) is not int or not 0 < start < end:
            raise ValueError('worker launch requires nonempty storage byte intervals')
        if end - start < extent or name is not None and end - start != extent:
            raise ValueError(f'worker launch storage extent differs for {name or "internal state"!r}')
        if bound_device is None:
            bound_device = device
        if name in system_payloads:
            if device == bound_device:
                raise ValueError(f'system handoff payload {name!r} needs peer storage')
        elif device != bound_device:
            raise ValueError('worker Program local storage must share one device')
        if any(device == other_device and start < other_end and other_start < end
               for other_device, other_start, other_end in spans):
            raise ValueError(f'worker launch storage for {name or "internal state"!r} overlaps another tensor')
        spans.append((device, start, end))
    launcher = load_worker(lowered, MappingProxyType(buffers), state)
    if not callable(launcher):
        raise ValueError('worker loader must return one callable launch')
    if execution_context() != bound_context:
        raise ValueError('worker execution device/stream changed during preparation')
    return PreparedWorkerProgram(launcher, MappingProxyType({name: buffers[name]
        for name in program.outputs}), MappingProxyType(buffers), state,
        status_offset, read_status, execution_context, bound_context)


@dataclass
class PreparedWorkerProgram:
    launcher: Callable
    outputs: Mapping[str, object]
    buffers: Mapping[str, object]
    state: object
    status_offset: int
    read_status: Callable
    execution_context: Callable
    bound_context: object
    launch_calls: int = 0
    _lock: object = field(default_factory=Lock, repr=False)

    def run(self):
        """One cooperative launch; the status adapter synchronizes before reading."""
        with self._lock:
            if self.execution_context() != self.bound_context:
                raise ValueError('worker execution device/stream changed')
            self.launcher(self.bound_context)
            self.launch_calls += 1
            status = self.read_status(self.state, self.status_offset, self.bound_context)
            if type(status) is not int or status != 0:
                raise ValueError(f'worker execution returned status {status!r}')
            if self.execution_context() != self.bound_context:
                raise ValueError('worker execution device/stream changed')
            return self.outputs


def prepare_rank_worker_program(lowered: LoweredWorkerProgram,
            inputs: Mapping[str, object], *, allocate: Callable,
            allocate_state: Callable, load_worker: Callable, check_tensor: Callable,
            storage_span: Callable, execution_context: Callable,
            read_status: Callable):
    """Bind a version-3 worker Program to two exact logical GPU ranks.

    Allocation, device/stream observation and post-launch synchronization stay
    with Evaluation. The backend-owned loader returns one callable that invokes
    both rank kernels from this exact lowered source; it cannot replay stages
    or reinterpret a Program tensor's declared owner.
    """
    lowered.validate_binding()
    program = lowered.program
    placement = program.execution.placement if program.execution else None
    if placement is None or set(inputs) != set(program.inputs):
        raise ValueError('rank worker launch needs version-3 execution and every public input')
    requirements = lowered.toolchain_requirements
    state_bytes = requirements.get('state_bytes')
    status_offset = requirements.get('state_status_offset_bytes')
    host_abi = requirements.get('host_abi')
    grids = requirements.get('grid_per_rank')
    if (requirements.get('world_size') != placement.world_size
            or requirements.get('state_rank') != placement.state_rank
            or requirements.get('tensor_ranks') != dict(placement.tensor_ranks)
            or requirements.get('argument_order') != list(program.tensors)
            or type(state_bytes) is not int or state_bytes < 4 or state_bytes % 4
            or type(status_offset) is not int or status_offset < 0
            or status_offset % 4 or status_offset + 4 > state_bytes
            or requirements.get('state_reset') !=
               f'zero_before_both_launches_on_rank{placement.state_rank}_stream'
            or requirements.get('peer_pair_runtime_check') is not True
            or requirements.get('cooperative_grid') is not True
            or not isinstance(grids, list) or len(grids) != placement.world_size
            or any(not isinstance(grid, list) or len(grid) != 3
                   or any(type(extent) is not int or extent <= 0 for extent in grid)
                   for grid in grids)
            or not isinstance(host_abi, Mapping)
            or set(host_abi) != {'create_rank', 'launch_two', 'destroy_rank'}
            or any(not isinstance(name, str) or not name for name in host_abi.values())):
        raise ValueError('rank worker launch placement, state or host ABI differs')
    contexts = tuple(execution_context(rank) for rank in range(placement.world_size))
    buffers = dict(inputs)
    for name, spec in program.tensors.items():
        if name not in buffers:
            buffers[name] = allocate(name, spec, placement.tensor_ranks[name])
        check_tensor(buffers[name], spec)
    buffers = {name: buffers[name] for name in program.tensors}
    state = allocate_state(state_bytes, placement.state_rank)
    spans = []
    entries = [(name, tensor, program.tensors[name].nbytes,
                placement.tensor_ranks[name]) for name, tensor in buffers.items()]
    entries.append((None, state, state_bytes, placement.state_rank))
    for name, tensor, extent, owner in entries:
        device, start, end = storage_span(tensor)
        if (type(device) is not int or device != owner
                or type(start) is not int or type(end) is not int
                or not 0 < start < end):
            raise ValueError(f'rank worker storage owner or interval differs for {name!r}')
        if end - start < extent or name is not None and end - start != extent:
            raise ValueError(f'rank worker storage extent differs for {name!r}')
        if any(device == other_device and start < other_end and other_start < end
               for other_device, other_start, other_end in spans):
            raise ValueError(f'rank worker storage for {name!r} overlaps another allocation')
        spans.append((device, start, end))
    launcher = load_worker(lowered, MappingProxyType(buffers), state)
    if not callable(launcher):
        raise ValueError('rank worker loader must return one combined launch')
    if tuple(execution_context(rank) for rank in range(placement.world_size)) != contexts:
        raise ValueError('rank worker device/stream changed during preparation')
    return PreparedRankWorkerProgram(launcher,
        MappingProxyType({name: buffers[name] for name in program.outputs}),
        MappingProxyType(buffers), state, status_offset, read_status,
        execution_context, contexts)


@dataclass
class PreparedRankWorkerProgram:
    launcher: Callable
    outputs: Mapping[str, object]
    buffers: Mapping[str, object]
    state: object
    status_offset: int
    read_status: Callable
    execution_context: Callable
    bound_contexts: tuple
    launch_calls: int = 0
    _lock: object = field(default_factory=Lock, repr=False)

    def run(self):
        """One combined call launches both ranks; status read synchronizes both."""
        with self._lock:
            contexts = tuple(self.execution_context(rank)
                             for rank in range(len(self.bound_contexts)))
            if contexts != self.bound_contexts:
                raise ValueError('rank worker device/stream changed')
            self.launcher(contexts)
            self.launch_calls += 1
            status = self.read_status(self.state, self.status_offset, contexts)
            if type(status) is not int or status != 0:
                raise ValueError(f'rank worker execution returned status {status!r}')
            if tuple(self.execution_context(rank)
                     for rank in range(len(self.bound_contexts))) != self.bound_contexts:
                raise ValueError('rank worker device/stream changed')
            return self.outputs
