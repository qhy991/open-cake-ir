"""Platform-owned binding for a complete Compiler ranked-mailbox lowering.

The caller supplies exact-device allocation, compiled-source loading,
mailbox reset, synchronization and output views. This development adapter
owns no Workload math, CUDA pointer lookup or GPU allocation policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from collections.abc import Callable, Mapping

from open_cake_ir.compiler.ir import DType, ProgramTensor
from open_cake_ir.compiler.program import LoweredRankedMailbox


@dataclass(frozen=True)
class RankedMailboxExecutable:
    """A platform-loaded exact source and its compiled mailbox layout."""

    mailbox_bytes: int
    status_offset: int
    output_offset: int
    output_bytes: int
    bind: Callable


def _specs(rows) -> dict[str, ProgramTensor]:
    if not isinstance(rows, list) or not rows:
        raise ValueError('ranked launch needs declared per-rank inputs')
    specs = {}
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {'name', 'shape', 'dtype'}
                or not isinstance(row['name'], str) or not row['name']
                or row['name'] in specs or not isinstance(row['shape'], list)
                or not row['shape']
                or any(type(extent) is not int or extent <= 0
                       for extent in row['shape'])):
            raise ValueError('ranked launch input declaration differs')
        try:
            dtype = DType(row['dtype'])
        except (TypeError, ValueError):
            raise ValueError('ranked launch input dtype differs') from None
        specs[row['name']] = ProgramTensor(tuple(row['shape']), dtype)
    return specs


def prepare_ranked_mailbox(lowered: LoweredRankedMailbox,
        inputs: Mapping[int, Mapping[str, object]],
        plans: Mapping[int, Mapping[str, int]], *, load_source: Callable,
        allocate_mailbox: Callable, check_tensor: Callable,
        storage_span: Callable, execution_context: Callable,
        output_view: Callable, reset_mailboxes: Callable,
        read_status: Callable):
    """Bind all rank-local inputs and mailbox storage before the first launch.

    `load_source` must load exactly `lowered.source` and return compiled
    layout queries plus one `bind` callback. `reset_mailboxes` synchronizes
    every rank's reset before launch; `read_status` synchronizes all kernels
    before returning one INT32 status per rank. The broker owns GPU leases.
    """
    lowered.validate_binding()
    req = lowered.toolchain_requirements
    world = lowered.analysis.world_size
    expected_ranks = set(range(world))
    grids = req.get('grid_per_rank')
    abi = req.get('host_abi')
    if (req.get('world_size') != world
            or req.get('tokens_per_rank') != lowered.analysis.items_per_rank
            or req.get('routes_per_token') != lowered.analysis.routes_per_item
            or req.get('feature_width') != lowered.analysis.feature_width
            or req.get('payload_capacity')
            != lowered.analysis.remote_payload_capacity_per_rank
            or req.get('task_capacity')
            != lowered.analysis.compute_task_capacity_per_rank
            or req.get('return_slots') != lowered.analysis.return_slots_per_rank
            or req.get('state_reset') != 'zero_all_rank_mailboxes_before_launch'
            or req.get('peer_pair_runtime_check') is not True
            or req.get('input_domain_runtime_check') is not True
            or req.get('cooperative_grid') is not True
            or req.get('block') != [32, 1, 1]
            or not isinstance(grids, list) or len(grids) != world
            or any(not isinstance(grid, list) or len(grid) != 3
                   or any(type(extent) is not int or extent <= 0 for extent in grid)
                   for grid in grids)
            or not isinstance(abi, Mapping)
            or not {'mailbox_bytes', 'status_offset', 'output_offset',
                    'output_bytes', 'prepare_rank', 'launch'} <= set(abi)
            or any(not isinstance(name, str) or not name for name in abi.values())):
        raise ValueError('ranked launch source, target or mailbox ABI differs')
    specs = _specs(req.get('rank_inputs'))
    if (not isinstance(inputs, Mapping) or set(inputs) != expected_ranks
            or not isinstance(plans, Mapping) or set(plans) != expected_ranks):
        raise ValueError('ranked launch needs every rank input and plan')
    controls = {}
    for rank in range(world):
        row = plans[rank]
        if not isinstance(row, Mapping) or set(row) != {
                'communication_ctas', 'chunks', 'steal_budget'}:
            raise ValueError(f'rank {rank} c/K/steal controls differ')
        c, k, steal = (row[name] for name in
                       ('communication_ctas', 'chunks', 'steal_budget'))
        if (any(type(value) is not int for value in (c, k, steal))
                or not 0 < c < grids[rank][0]
                or not 1 <= k <= lowered.analysis.items_per_rank
                or not 0 <= steal <= lowered.analysis.compute_task_capacity_per_rank):
            raise ValueError(f'rank {rank} c/K/steal values exceed declared bounds')
        controls[rank] = MappingProxyType(dict(row))
    contexts = tuple(execution_context(rank) for rank in range(world))
    buffers = {}
    spans = []
    for rank in range(world):
        row = inputs[rank]
        if not isinstance(row, Mapping) or set(row) != set(specs):
            raise ValueError(f'rank {rank} public input set differs')
        buffers[rank] = MappingProxyType(dict(row))
        for name, tensor in row.items():
            check_tensor(tensor, specs[name])
            device, start, end = storage_span(tensor)
            if (type(device) is not int or device != rank
                    or type(start) is not int or type(end) is not int
                    or not 0 < start < end or end - start != specs[name].nbytes
                    or any(device == other_device and start < other_end
                           and other_start < end
                           for other_device, other_start, other_end in spans)):
                raise ValueError(f'rank {rank} input {name!r} owner, extent or alias differs')
            spans.append((device, start, end))
    executable = load_source(lowered)
    if not isinstance(executable, RankedMailboxExecutable):
        raise ValueError('ranked loader must return compiled mailbox layout and binder')
    size = executable.mailbox_bytes
    status = executable.status_offset
    offset = executable.output_offset
    output_bytes = executable.output_bytes
    expected_output = ProgramTensor(
        (lowered.analysis.items_per_rank, lowered.analysis.feature_width), DType.BF16)
    if (type(size) is not int or size <= 0
            or type(status) is not int or status < 0 or status % 4
            or status + 4 > size
            or type(offset) is not int or offset < 0
            or type(output_bytes) is not int or output_bytes != expected_output.nbytes
            or offset + output_bytes > size or not callable(executable.bind)):
        raise ValueError('compiled mailbox size, status or output view differs')
    mailboxes = {}
    outputs = {}
    for rank in range(world):
        mailbox = allocate_mailbox(rank, size)
        device, start, end = storage_span(mailbox)
        if (type(device) is not int or device != rank
                or type(start) is not int or type(end) is not int
                or not 0 < start < end or end - start != size
                or any(device == other_device and start < other_end
                       and other_start < end
                       for other_device, other_start, other_end in spans)):
            raise ValueError(f'rank {rank} mailbox owner, extent or alias differs')
        spans.append((device, start, end))
        mailboxes[rank] = mailbox
        view = output_view(mailbox, offset, expected_output)
        check_tensor(view, expected_output)
        if storage_span(view) != (rank, start + offset,
                                 start + offset + output_bytes):
            raise ValueError(f'rank {rank} output must be exact mailbox view')
        outputs[rank] = view
    frozen_buffers = MappingProxyType(buffers)
    frozen_mailboxes = MappingProxyType(mailboxes)
    launcher = executable.bind(frozen_buffers, frozen_mailboxes,
                               MappingProxyType(controls), contexts)
    if not callable(launcher):
        raise ValueError('ranked binder must return one combined launch')
    if tuple(execution_context(rank) for rank in range(world)) != contexts:
        raise ValueError('ranked execution contexts changed during preparation')
    return PreparedRankedMailbox(
        launcher, MappingProxyType(outputs), frozen_buffers,
        frozen_mailboxes, status, reset_mailboxes, read_status,
        execution_context, contexts)


@dataclass
class PreparedRankedMailbox:
    launcher: Callable
    outputs: Mapping[int, object]
    inputs: Mapping[int, Mapping[str, object]]
    mailboxes: Mapping[int, object]
    status_offset: int
    reset_mailboxes: Callable
    read_status: Callable
    execution_context: Callable
    bound_contexts: tuple
    launch_calls: int = 0
    _lock: object = field(default_factory=Lock, repr=False)

    def run(self):
        """One reset, one four-rank launch, then synchronized status checks."""
        with self._lock:
            contexts = tuple(self.execution_context(rank)
                             for rank in range(len(self.bound_contexts)))
            if contexts != self.bound_contexts:
                raise ValueError('ranked execution contexts changed')
            self.reset_mailboxes(self.mailboxes, contexts)
            self.launcher(contexts)
            self.launch_calls += 1
            statuses = self.read_status(self.mailboxes, self.status_offset, contexts)
            if (not isinstance(statuses, tuple)
                    or len(statuses) != len(contexts)
                    or any(type(status) is not int or status != 0
                           for status in statuses)):
                raise ValueError(f'ranked worker statuses differ: {statuses!r}')
            if tuple(self.execution_context(rank)
                     for rank in range(len(contexts))) != contexts:
                raise ValueError('ranked execution contexts changed after launch')
            return self.outputs
