"""Development admission for a Cake system atomic targeting peer GPU state.

The Compiler owns the Schedule's effect and exact Target instruction. The
platform supplies fresh broker-visible pointer ownership and directed P2P
facts, then enables the peer mapping. This does not admit a payload handoff or
a sealed distributed Program candidate.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from open_cake_ir.compiler.ir import (
    AtomicMemoryScope, BufferMode, DType, LoweringBackend, MemorySpace,
    OperationKind, Schedule,
)
from open_cake_ir.compiler.target import CodeObject, Target


_CONTRACT = 'ptx.atom.relaxed.sys.global.add.s32'


@dataclass(frozen=True)
class PeerCapabilities:
    access: bool
    native_atomic: bool


@dataclass(frozen=True)
class PeerAtomicBinding:
    target: str
    state_buffer: str
    execution_device: int
    owner_device: int


def bind_peer_atomic_state(
    schedule: Schedule, target: Target, buffers: Mapping[str, object], *,
    execution_device: int,
    pointer_owner: Callable[[object], int],
    probe_peer: Callable[[int, int], PeerCapabilities],
    enable_peer: Callable[[int, int], bool],
) -> PeerAtomicBinding:
    """Validate and enable one directed peer-state binding before launch.

    Device ordinals are the broker-visible logical ordinals of this one job.
    ``pointer_owner`` must inspect the actual bound pointer, and ``probe_peer``
    must read both CUDA P2P attributes for this exact ordered device pair.
    ``enable_peer`` confirms the mapping became usable. No Target document is
    used as a substitute for a pairwise runtime observation.
    """
    if (schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA
            or schedule.target != target.target_id
            or target.code_object is not CodeObject.CUBIN
            or _CONTRACT not in target.instruction_contracts):
        raise ValueError('peer atomic binding requires the exact native CUDA system contract')
    expected = {buffer.name for buffer in schedule.buffers
                if buffer.space is MemorySpace.GLOBAL}
    if set(buffers) != expected:
        raise ValueError('peer atomic binding must cover every global Schedule buffer')
    names = {operation.reads[0] for operation in schedule.operations
             if operation.kind is OperationKind.ATOMIC_RMW
             and operation.parameters.scope is AtomicMemoryScope.SYSTEM
             and operation.reads}
    if len(names) != 1:
        raise ValueError('peer atomic binding requires one system-scope state target')
    state_name = next(iter(names))
    state = schedule.buffer(state_name)
    if (state is None or state.space is not MemorySpace.GLOBAL
            or state.mode is not BufferMode.STATE or state.dtype is not DType.INT32):
        raise ValueError('peer atomic target must be global INT32 state')
    if type(execution_device) is not int or execution_device < 0:
        raise ValueError('peer execution device ordinal differs')
    owner = pointer_owner(buffers[state_name])
    if type(owner) is not int or owner < 0 or owner == execution_device:
        raise ValueError('peer state pointer must belong to another observed GPU')
    capability = probe_peer(execution_device, owner)
    if (type(capability) is not PeerCapabilities
            or capability.access is not True or capability.native_atomic is not True):
        raise ValueError('exact directed GPU pair lacks peer access or native atomics')
    if enable_peer(execution_device, owner) is not True:
        raise ValueError('peer access was not enabled for the exact directed pair')
    return PeerAtomicBinding(target.target_id, state_name, execution_device, owner)
