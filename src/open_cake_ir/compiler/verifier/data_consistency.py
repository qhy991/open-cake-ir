"""Buffer placement, operation representations and producer/consumer wiring.

Shape and access checks retain their localized hardware and safety findings when
those contracts depend on the same representation proof."""

from __future__ import annotations

from ..ir import (
    AccessIndexKind,
    BufferMode,
    LoadMovement,
    MemorySpace,
    DType,
    ElementwiseOp,
    OperationKind,
    Schedule,
)
from ..ir.operations import elementwise_result_dtype, ELEMENTWISE_FLOAT_DTYPES as _ELEMENTWISE_FLOAT_DTYPES
from ..diagnostics import FindingCategory
from ._collector import _Collector
from .hardware_conformance import _BLOCK_SCALE_MMA_CONTRACT


def _cycle_members(graph: dict[str, tuple[str, ...]]) -> set[str]:
    """Nodes that participate in a dependency cycle (iterative Tarjan-free variant)."""

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    members: set[str] = set()

    for root in graph:
        if colour[root] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        while stack:
            node, index = stack.pop()
            if index == 0:
                if colour[node] != WHITE:
                    continue
                colour[node] = GREY
                path.append(node)
            edges = graph.get(node, ())
            if index < len(edges):
                stack.append((node, index + 1))
                nxt = edges[index]
                if nxt not in graph:
                    continue
                if colour[nxt] == GREY:
                    members.update(path[path.index(nxt):])
                elif colour[nxt] == WHITE:
                    stack.append((nxt, 0))
            else:
                colour[node] = BLACK
                if path and path[-1] == node:
                    path.pop()
    return members



_ARITY = {
    OperationKind.LOAD: (1, 1, "load"),
    OperationKind.MMA: (2, 1, "mma"),
    OperationKind.EPILOGUE: (1, 1, "epilogue"),
    OperationKind.REDUCE_ARGMIN: (1, 1, "reduce_argmin"),
    OperationKind.REDUCE: (1, 1, "reduce"),
    OperationKind.SCAN: (1, 1, "scan"),
    OperationKind.STORE: (1, 1, "store"),
}


def _verify_scale_relations(schedule: Schedule, buffers, out: _Collector) -> None:
    """Hold scale storage to the FP8 tensor relation that gives it meaning."""

    category = FindingCategory.DATA_CONSISTENCY
    for index, scale in enumerate(schedule.buffers):
        relation = scale.scale_of
        if relation is None:
            continue
        path = f"buffers[{index}].scale_of"
        data = buffers.get(relation.buffer)
        if data is None:
            out.add(
                "SCALE_TARGET_UNKNOWN",
                f"{path}.buffer",
                f"scale buffer {scale.name!r} names unknown data buffer "
                f"{relation.buffer!r}",
                category,
            )
            continue
        if data is scale:
            out.add(
                "SCALE_TARGET_SELF",
                f"{path}.buffer",
                f"scale buffer {scale.name!r} cannot scale itself",
                category,
            )
            continue
        if data.dtype is not DType.FP8_E4M3:
            out.add(
                "SCALE_DATA_DTYPE",
                f"{path}.buffer",
                f"block scales describe fp8_e4m3 data, but {data.name!r} is "
                f"{data.dtype.value}",
                category,
            )
        if scale.dtype is not DType.FP32:
            out.add(
                "SCALE_DTYPE",
                f"buffers[{index}].dtype",
                f"block scale {scale.name!r} must be fp32, not {scale.dtype.value}",
                category,
            )
        rank = len(data.shape)
        if len(relation.granularity) != rank:
            out.add(
                "SCALE_GRANULARITY_RANK",
                f"{path}.granularity",
                f"scale granularity has rank {len(relation.granularity)} but "
                f"{data.name!r} has rank {rank}",
                category,
            )
        if tuple(sorted(relation.axis_order)) != tuple(range(rank)):
            out.add(
                "SCALE_AXIS_ORDER",
                f"{path}.axis_order",
                f"scale axis order {list(relation.axis_order)} is not a permutation "
                f"of all {rank} axes of {data.name!r}",
                category,
            )
        if len(relation.granularity) == rank and tuple(
            sorted(relation.axis_order)
        ) == tuple(range(rank)):
            grouped = tuple(
                (extent + granularity - 1) // granularity
                for extent, granularity in zip(data.shape, relation.granularity)
            )
            expected = tuple(grouped[axis] for axis in relation.axis_order)
            if scale.shape != expected:
                out.add(
                    "SCALE_SHAPE_MISMATCH",
                    f"buffers[{index}].shape",
                    f"{scale.name!r} shape {list(scale.shape)} differs from the "
                    f"derived grouped shape {list(expected)} for {data.name!r}",
                    category,
                )

    # A load changes storage, not meaning. The scale load and the data load together
    # establish which staged tile the staged scale describes.
    loads = [
        operation
        for operation in schedule.operations
        if operation.kind is OperationKind.LOAD
    ]
    load_edges = {
        (operation.reads[0], operation.writes[0])
        for operation in loads
        if len(operation.reads) == 1 and len(operation.writes) == 1
    }
    for operation_index, operation in enumerate(schedule.operations):
        if operation.kind is not OperationKind.LOAD:
            continue
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            continue
        source = buffers.get(operation.reads[0])
        destination = buffers.get(operation.writes[0])
        if source is None or destination is None:
            continue
        before = source.scale_of
        after = destination.scale_of
        if (before is None) != (after is None):
            out.add(
                "SCALE_RELATION_DROPPED",
                f"operations[{operation_index}]",
                f"load {operation.op_id!r} changes whether {source.name!r}/"
                f"{destination.name!r} is a scale buffer",
                category,
            )
            continue
        if before is None or after is None:
            continue
        if (
            before.granularity != after.granularity
            or before.axis_order != after.axis_order
            or (before.buffer, after.buffer) not in load_edges
        ):
            out.add(
                "SCALE_RELATION_DRIFT",
                f"operations[{operation_index}]",
                f"load {operation.op_id!r} does not preserve the scale relation and "
                "the corresponding data-load edge",
                category,
            )


def _verify_valid_extents(schedule: Schedule, buffers, out: _Collector) -> None:
    """Type-check the one runtime authority for a padded Buffer's valid prefix."""

    category = FindingCategory.DATA_CONSISTENCY
    for index, data in enumerate(schedule.buffers):
        relation = data.valid_extent
        if relation is None:
            continue
        path = f"buffers[{index}].valid_extent"
        if data.space is not MemorySpace.GLOBAL:
            out.add(
                "VALID_EXTENT_DATA_SPACE",
                path,
                f"runtime valid extents apply to global padded storage, but "
                f"{data.name!r} is {data.space.value}",
                category,
            )
        extent = buffers.get(relation.buffer)
        if extent is None:
            out.add(
                "VALID_EXTENT_BUFFER_UNKNOWN",
                f"{path}.buffer",
                f"{data.name!r} names unknown extent buffer {relation.buffer!r}",
                category,
            )
        elif extent is data:
            out.add(
                "VALID_EXTENT_SELF",
                f"{path}.buffer",
                f"{data.name!r} cannot supply its own runtime extent",
                category,
            )
        else:
            if extent.dtype is not DType.INT32:
                out.add(
                    "VALID_EXTENT_DTYPE",
                    f"buffers[{schedule.buffers.index(extent)}].dtype",
                    f"extent buffer {extent.name!r} must be int32, not "
                    f"{extent.dtype.value}",
                    category,
                )
            if extent.space is not MemorySpace.GLOBAL or extent.mode is not BufferMode.INPUT:
                out.add(
                    "VALID_EXTENT_BUFFER_CONTRACT",
                    f"buffers[{schedule.buffers.index(extent)}]",
                    f"extent buffer {extent.name!r} must be a global input",
                    category,
                )

        rank = len(data.shape)
        axes_valid = True
        if relation.dimension >= rank:
            axes_valid = False
            out.add(
                "VALID_EXTENT_DIMENSION",
                f"{path}.dimension",
                f"valid dimension {relation.dimension} is outside rank-{rank} "
                f"buffer {data.name!r}",
                category,
            )
        for position, axis in enumerate(relation.indexed_by):
            if axis >= rank:
                axes_valid = False
                out.add(
                    "VALID_EXTENT_INDEX_AXIS",
                    f"{path}.indexed_by[{position}]",
                    f"index axis {axis} is outside rank-{rank} buffer {data.name!r}",
                    category,
                )
        if relation.dimension in relation.indexed_by:
            axes_valid = False
            out.add(
                "VALID_EXTENT_INDEX_AXIS",
                f"{path}.indexed_by",
                f"valid dimension {relation.dimension} cannot also index its lengths",
                category,
            )
        if extent is not None and extent is not data and axes_valid:
            expected = tuple(data.shape[axis] for axis in relation.indexed_by)
            if extent.shape != expected:
                out.add(
                    "VALID_EXTENT_SHAPE_MISMATCH",
                    f"buffers[{schedule.buffers.index(extent)}].shape",
                    f"extent buffer {extent.name!r} shape {list(extent.shape)} differs "
                    f"from indexed data extents {list(expected)}",
                    category,
                )


def verify(schedule: Schedule, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY

    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    allocations = {item.name: item for item in schedule.allocations}
    roles = {role.name for role in schedule.roles}
    pipelines = {pipeline.name for pipeline in schedule.pipelines}

    _verify_scale_relations(schedule, buffers, out)
    _verify_valid_extents(schedule, buffers, out)

    # ---- buffer placement -------------------------------------------------
    for index, buffer in enumerate(schedule.buffers):
        path = f"buffers[{index}]"
        if (
            buffer.mode in (BufferMode.INPUT, BufferMode.OUTPUT, BufferMode.STATE)
            and buffer.space is not MemorySpace.GLOBAL
        ):
            out.add(
                "BUFFER_IO_SPACE",
                f"{path}.space",
                f"{buffer.mode.value} buffer {buffer.name!r} must live in global memory",
                category,
            )
        if buffer.space is MemorySpace.GLOBAL and buffer.allocation is not None:
            out.add(
                "BUFFER_GLOBAL_ALLOCATION",
                f"{path}.allocation",
                f"global buffer {buffer.name!r} must not claim a CTA Allocation",
                category,
            )
        if buffer.space is MemorySpace.REGISTER and buffer.allocation is not None:
            out.add(
                "BUFFER_REGISTER_ALLOCATION",
                f"{path}.allocation",
                f"register buffer {buffer.name!r} must not claim a CTA Allocation",
                category,
            )
        if (
            buffer.space in (MemorySpace.SHARED, MemorySpace.TENSOR)
            and buffer.allocation is None
        ):
            out.add(
                "BUFFER_ALLOCATION_MISSING",
                f"{path}.allocation",
                f"{buffer.space.value} buffer {buffer.name!r} must name its Allocation",
                category,
            )
        if buffer.allocation is not None:
            if buffer.allocation not in allocations:
                out.add(
                    "BUFFER_ALLOCATION_UNKNOWN",
                    f"{path}.allocation",
                    f"unknown Allocation {buffer.allocation!r}",
                    category,
                )
            else:
                owner = allocations[buffer.allocation]
                if (
                    buffer.space in (MemorySpace.SHARED, MemorySpace.TENSOR)
                    and owner.space is not buffer.space
                ):
                    out.add(
                        "BUFFER_SPACE_MISMATCH",
                        f"{path}.allocation",
                        f"buffer {buffer.name!r} is {buffer.space.value} but Allocation "
                        f"{owner.name!r} is {owner.space.value}",
                        category,
                    )
                # Allocation existence and byte extent are independent of the
                # memory-space relation. One typed rule owns the full reference
                # domain, including a misplaced global/register buffer.
                if buffer.byte_extent[1] > owner.size_bytes:
                    out.add(
                        "BUFFER_ALLOCATION_OVERFLOW",
                        f"{path}.byte_offset",
                        f"buffer {buffer.name!r} ends at byte {buffer.byte_extent[1]}, "
                        f"past the {owner.size_bytes}-byte Allocation {owner.name!r}",
                        category,
                    )

    # ---- aliasing inside one allocation -----------------------------------
    # Two buffers sharing an Allocation must not overlap unless the Schedule says so.
    # Nothing else in the pipeline detects this: the offsets are hand-authored and a
    # silent overlap corrupts one of the two tiles at runtime.
    by_allocation: dict[str, list[tuple[int, object]]] = {}
    for index, buffer in enumerate(schedule.buffers):
        if buffer.allocation in allocations:
            by_allocation.setdefault(buffer.allocation, []).append((index, buffer))
    for name, entries in sorted(by_allocation.items()):
        ordered = sorted(entries, key=lambda item: item[1].byte_extent)
        for (_, earlier), (later_index, later) in zip(ordered, ordered[1:]):
            if later.byte_extent[0] < earlier.byte_extent[1]:
                out.add(
                    "BUFFER_VIEW_OVERLAP",
                    f"buffers[{later_index}].byte_offset",
                    f"buffer {later.name!r} occupies bytes "
                    f"[{later.byte_extent[0]}, {later.byte_extent[1]}) of Allocation "
                    f"{name!r}, overlapping {earlier.name!r} at "
                    f"[{earlier.byte_extent[0]}, {earlier.byte_extent[1]})",
                    category,
                )

    # ---- operation wiring --------------------------------------------------
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}]"
        if operation.role not in roles:
            out.add(
                "OP_ROLE_UNKNOWN",
                f"{path}.role",
                f"unknown role {operation.role!r}",
                category,
            )
        if operation.pipeline is not None and operation.pipeline not in pipelines:
            out.add(
                "OP_PIPELINE_UNKNOWN",
                f"{path}.pipeline",
                f"unknown pipeline {operation.pipeline!r}",
                category,
            )
        for name in operation.reads:
            if name not in buffers:
                out.add(
                    "OP_BUFFER_UNKNOWN",
                    f"{path}.reads",
                    f"unknown buffer {name!r}",
                    category,
                )
            else:
                readers.setdefault(name, []).append(operation.op_id)
        for name in operation.writes:
            if name not in buffers:
                out.add(
                    "OP_BUFFER_UNKNOWN",
                    f"{path}.writes",
                    f"unknown buffer {name!r}",
                    category,
                )
            else:
                writers.setdefault(name, []).append(operation.op_id)
        overlap = sorted(set(operation.reads) & set(operation.writes))
        for name in overlap:
            buffer = buffers.get(name)
            if (
                operation.kind is OperationKind.ATOMIC_RMW
                and operation.reads
                and operation.writes
                and name == operation.reads[0] == operation.writes[0]
                and buffer is not None
                and buffer.mode is BufferMode.STATE
            ):
                # Reading and writing one state location is the defined effect of RMW,
                # not an accidental alias. Every other overlap remains illegal.
                continue
            out.add(
                "OP_SELF_READ_WRITE",
                path,
                f"operation {operation.op_id!r} both reads and writes {name!r}",
                category,
            )
        _verify_operation_shape(
            operation,
            path,
            buffers,
            out,
            backend=schedule.lowering.backend,
        )

    # ---- loop-carried lifetime --------------------------------------------
    # A register value produced inside a tile loop does not survive it: registers hold
    # the loop body's live values and the next trip overwrites them. Shared and tensor
    # memory are allocations that outlive the loop, which is exactly how a warp-
    # specialized Schedule hands a staged tile from a producer role to a consumer that
    # reads it outside the loop, so the rule follows the declared space rather than the
    # nesting alone. The exception in registers is a reduction's result, which the loop
    # carries by construction.
    for loop in schedule.tile_loops:
        body = {op.op_id for op in schedule.loop_operations(loop)}
        for op_id in loop.body:
            operation = schedule.operation(op_id)
            if (
                operation is None
                or operation.kind is not OperationKind.TOP_K
                or not operation.parameters.across_loop
                or not operation.reads
            ):
                continue
            source = buffers.get(operation.reads[0])
            if source is not None and source.shape != (loop.tile,):
                out.add(
                    "TOP_K_LOOP_TILE_MISMATCH",
                    f"operations[{schedule.operations.index(operation)}].reads",
                    f"loop-carried top_k consumes one score per {loop.name!r} position, "
                    f"so {source.name!r} must have shape [{loop.tile}], got "
                    f"{list(source.shape)}",
                    category,
                )
            if operation.parameters.source_tiles_per_merge != 2:
                continue

            for reader_id in sorted(
                {
                    reader
                    for result in operation.writes
                    for reader in readers.get(result, ())
                    if reader in body
                }
            ):
                reader = schedule.operation(reader_id)
                if reader is None:
                    continue
                observed = sorted(set(reader.reads) & set(operation.writes))
                out.add(
                    "TOP_K_MERGE_CADENCE_OUTPUT_READ_IN_LOOP",
                    f"operations[{schedule.operations.index(reader)}].reads",
                    f"{reader.op_id!r} reads delayed top_k result(s) "
                    f"{', '.join(repr(name) for name in observed)} inside {loop.name!r}; "
                    "a two-source-tile merge exposes only the finalized state after "
                    "the loop",
                    category,
                )

        for op_id in loop.body:
            operation = schedule.operation(op_id)
            if operation is None or operation.kind is not OperationKind.ONLINE_SOFTMAX:
                continue
            if operation.reads:
                logits = buffers.get(operation.reads[0])
                values = buffers.get(operation.reads[1]) if len(operation.reads) > 1 else None
                if (
                    logits is not None
                    and len(logits.shape) == 2
                    and logits.shape[1] != loop.tile
                ) or (
                    values is not None
                    and len(values.shape) == 2
                    and values.shape[0] != loop.tile
                ):
                    out.add(
                        "ONLINE_SOFTMAX_LOOP_TILE_MISMATCH",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"online_softmax selected extent must equal {loop.name!r} tile "
                        f"{loop.tile}",
                        category,
                    )
            if len(operation.writes) == 4:
                normalized = operation.writes[3]
                in_loop_readers = sorted(set(readers.get(normalized, ())) & body)
                if in_loop_readers:
                    out.add(
                        "ONLINE_SOFTMAX_FINAL_READ_IN_LOOP",
                        f"operations[{schedule.operations.index(operation)}].writes",
                        f"normalized output is finalized after {loop.name!r}, but is read "
                        f"inside it by {', '.join(in_loop_readers)}",
                        category,
                    )
        carried = {
            name
            for op_id in loop.body
            if (op := schedule.operation(op_id)) is not None
            and (
                (
                    op.kind is OperationKind.REDUCE
                    and op.parameters.across_loop
                )
                or op.kind is OperationKind.REDUCE_ARGMIN
                or (
                    op.kind is OperationKind.TOP_K
                    and op.parameters.across_loop
                )
                or op.kind is OperationKind.ONLINE_SOFTMAX
                # A contraction whose loop walks K is a reduction too, and its result is
                # carried by the same construction. Which loops do that is derived from
                # the operands rather than declared, so a Schedule that tiles the output
                # axis instead still hits the rule -- and should, because that shape has
                # no accumulator to carry.
                or (
                    op.kind is OperationKind.MMA
                    and schedule.mma_accumulates_over(op, loop)
                )
            )
            for name in op.writes
        }
        for name, producers in sorted(writers.items()):
            if name in carried or not set(producers) <= body:
                continue
            if buffers[name].space is not MemorySpace.REGISTER:
                continue
            escaping = sorted(set(readers.get(name, ())) - body)
            if escaping:
                # A contraction accumulated across the loop is the case an author is
                # most likely to expect to work, because that is what a GEMM is. Saying
                # only "a reduction result" sends them looking for a rule they broke
                # rather than telling them a register accumulator is not one -- an
                # accumulator that outlives its loop has to live in a space that does.
                contraction = any(
                    (op := schedule.operation(op_id)) is not None
                    and op.kind is OperationKind.MMA
                    for op_id in producers
                )
                reason = (
                    "a contraction accumulated across a loop needs an accumulator in a "
                    "space that outlives it, and a register buffer does not"
                    if contraction
                    else "only a reduction result is carried out of a loop"
                )
                out.add(
                    "BUFFER_ESCAPES_LOOP",
                    f"buffers[{schedule.buffers.index(buffers[name])}]",
                    f"buffer {name!r} is written only inside {loop.name!r} but read by "
                    f"{', '.join(escaping)} outside it; {reason}",
                    category,
                )

    loop_bodies = {op_id for loop in schedule.tile_loops for op_id in loop.body}
    for operation in schedule.operations:
        if (
            operation.kind is OperationKind.TOP_K
            and operation.parameters.across_loop
            and operation.op_id not in loop_bodies
        ):
            out.add(
                "TOP_K_LOOP_REQUIRED",
                f"operations[{schedule.operations.index(operation)}].parameters.across_loop",
                "loop-carried top_k must be declared inside one tile loop",
                category,
            )
        if (
            operation.kind is OperationKind.ONLINE_SOFTMAX
            and operation.op_id not in loop_bodies
        ):
            out.add(
                "ONLINE_SOFTMAX_LOOP_REQUIRED",
                f"operations[{schedule.operations.index(operation)}]",
                "online_softmax must be declared inside one selected-token tile loop",
                category,
            )

    # ---- buffer roles ------------------------------------------------------
    # An input buffer that is written is the static form of the candidate mutating its
    # own inputs; downstream correctness compares against a reference recomputed from
    # those same buffers, so a writer here is a silent oracle bypass.
    for name, ops in sorted(writers.items()):
        buffer = buffers[name]
        if buffer.mode is BufferMode.INPUT:
            out.add(
                "INPUT_WRITTEN",
                f"buffers[{schedule.buffers.index(buffer)}].mode",
                f"input buffer {name!r} is written by {', '.join(sorted(ops))}; inputs "
                "are read-only for the whole Schedule",
                category,
            )
        if len(ops) > 1:
            out.add(
                "BUFFER_MULTIPLE_WRITERS",
                f"buffers[{schedule.buffers.index(buffer)}]",
                f"buffer {name!r} is written by {', '.join(sorted(ops))}",
                category,
            )
    for name, ops in sorted(readers.items()):
        buffer = buffers[name]
        if buffer.mode is BufferMode.SCRATCH and name not in writers:
            out.add(
                "BUFFER_UNPRODUCED",
                f"buffers[{schedule.buffers.index(buffer)}]",
                f"scratch buffer {name!r} is read by {', '.join(sorted(ops))} but never "
                "written",
                category,
            )
    for buffer in schedule.buffers:
        if buffer.mode is not BufferMode.STATE:
            continue
        position = schedule.buffers.index(buffer)
        if buffer.name not in readers:
            out.add(
                "STATE_NOT_READ",
                f"buffers[{position}].mode",
                f"state buffer {buffer.name!r} is never read; use output for a "
                "write-only result",
                category,
            )
        if buffer.name not in writers:
            out.add(
                "STATE_NOT_WRITTEN",
                f"buffers[{position}].mode",
                f"state buffer {buffer.name!r} is never written; use input for "
                "read-only data",
                category,
            )

    if not schedule.outputs and not any(
        buffer.mode is BufferMode.STATE for buffer in schedule.buffers
    ):
        out.add(
            "OUTPUTS_EMPTY_WITHOUT_STATE",
            "outputs",
            "a Schedule with no returned output must update caller-owned state",
            category,
        )

    for position, name in enumerate(schedule.outputs):
        if name in schedule.outputs[:position]:
            # `outputs` is the order the caller receives its tensors in, so a name
            # appearing twice asks for one buffer in two slots. The host wrapper can
            # only allocate it once, and the count it reports would not match.
            out.add(
                "OUTPUT_DUPLICATE",
                f"outputs[{position}]",
                f"buffer {name!r} is exported more than once",
                category,
            )
        if name not in buffers:
            out.add(
                "OUTPUT_UNKNOWN", f"outputs[{position}]", f"unknown buffer {name!r}", category
            )
            continue
        if buffers[name].mode is not BufferMode.OUTPUT:
            out.add(
                "OUTPUT_MODE",
                f"outputs[{position}]",
                f"buffer {name!r} is declared {buffers[name].mode.value}, not output",
                category,
            )
        if name not in writers:
            out.add(
                "OUTPUT_UNWRITTEN",
                f"outputs[{position}]",
                f"declared output {name!r} is never written",
                category,
            )
    for index, buffer in enumerate(schedule.buffers):
        if buffer.mode is BufferMode.OUTPUT and buffer.name not in schedule.outputs:
            out.add(
                "OUTPUT_NOT_EXPORTED",
                f"buffers[{index}].mode",
                f"buffer {buffer.name!r} is an output but is missing from outputs",
                category,
            )

    # ---- dependency graph ---------------------------------------------------
    op_ids = {operation.op_id for operation in schedule.operations}
    graph: dict[str, tuple[str, ...]] = {}
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}].depends_on"
        edges = []
        for name in operation.depends_on:
            if name == operation.op_id:
                out.add(
                    "OP_SELF_DEPENDENCY", path, "operation depends on itself", category
                )
                continue
            if name not in op_ids:
                out.add(
                    "OP_DEPENDENCY_UNKNOWN", path, f"unknown operation {name!r}", category
                )
                continue
            edges.append(name)
        graph[operation.op_id] = tuple(edges)
    # One cycle is one defect. Reporting a finding per member gave an author six blocking
    # findings for one mistake, none of them pointing anywhere: the path was `operations`
    # every time. The paper asks feedback to be localized diagnostics, and a set of names
    # with no index is the other way to fail that -- noise without a place to look.
    members = sorted(_cycle_members(graph))
    if members:
        position = {op.op_id: index for index, op in enumerate(schedule.operations)}
        first = min(members, key=lambda name: position.get(name, 0))
        out.add(
            "OP_DEPENDENCY_CYCLE",
            f"operations[{position[first]}].depends_on",
            f"dependency cycle through {', '.join(repr(name) for name in members)}",
            category,
        )

    _verify_access_maps(schedule, buffers, out)


def _verify_block_scaled_mma(operation, path: str, buffers, out: _Collector) -> None:
    """Prove scale association, then gate the first backend's exact static subset."""

    if len(operation.reads) != 4:
        return
    a, b, scale_a, scale_b = (buffers.get(name) for name in operation.reads)
    if any(buffer is None for buffer in (a, b, scale_a, scale_b)):
        return
    association_ok = True
    for position, (scale, data) in enumerate(((scale_a, a), (scale_b, b)), start=2):
        relation = scale.scale_of
        if relation is None or relation.buffer != data.name:
            association_ok = False
            out.add(
                "MMA_SCALE_ASSOCIATION",
                f"{path}.reads[{position}]",
                f"scale operand {scale.name!r} does not declare that it scales "
                f"data operand {data.name!r}",
                FindingCategory.DATA_CONSISTENCY,
            )
    if not association_ok:
        return

    relation_a = scale_a.scale_of
    relation_b = scale_b.scale_of
    assert relation_a is not None and relation_b is not None
    tile = operation.parameters.tile_shape
    supported = (
        len(a.shape) == len(b.shape) == 2
        and a.space is not MemorySpace.GLOBAL
        and b.space is not MemorySpace.GLOBAL
        and scale_a.space is not MemorySpace.GLOBAL
        and scale_b.space is not MemorySpace.GLOBAL
        and a.shape[1] == b.shape[1]
        and len(relation_a.granularity) == len(relation_b.granularity) == 2
        and relation_a.granularity[0] == 1
        and relation_a.axis_order == (1, 0)
        and relation_b.axis_order == (0, 1)
        and relation_b.granularity[0] == b.shape[0]
        and relation_a.granularity[1] == relation_b.granularity[1]
        and a.shape[1] % relation_a.granularity[1] == 0
        and a.shape[1] // relation_a.granularity[1] == 2
        and tile == (a.shape[0], b.shape[0], a.shape[1])
    )
    if not supported:
        out.add(
            "MMA_BLOCK_SCALE_UNLOWERABLE",
            f"{path}.parameters",
            "the Triton block-scale lowering requires rank-2 staged A/B, A "
            "granularity [1, block_k] stored [K-block, M], B granularity "
            "[N, block_k] stored [N-block, K-block], equal divisible K blocks, "
            "exactly two K blocks, and a tile equal to the staged contraction",
            FindingCategory.HARDWARE_CONFORMANCE,
        )



def _verify_operation_shape(
    operation,
    path: str,
    buffers,
    out: _Collector,
    *,
    backend=None,
) -> None:
    category = FindingCategory.DATA_CONSISTENCY
    if operation.kind is OperationKind.REDUCE_ARGMIN:
        # `reduce_argmin` is not a generic numeric reduction with an incidental
        # output type.  The admitted operation compares fp32 distances and returns
        # int32 source positions; making that contract explicit prevents a backend
        # gaining a new storage dtype from silently widening every argmin use.
        if operation.reads:
            source = buffers.get(operation.reads[0])
            if source is not None and source.dtype is not DType.FP32:
                out.add(
                    "REDUCE_ARGMIN_VALUE_DTYPE",
                    f"{path}.reads",
                    f"reduce_argmin compares fp32 values, but {source.name!r} is "
                    f"{source.dtype.value}",
                    category,
                )
        if operation.writes:
            indices = buffers.get(operation.writes[0])
            if indices is not None and indices.dtype is not DType.INT32:
                out.add(
                    "REDUCE_ARGMIN_INDEX_DTYPE",
                    f"{path}.writes",
                    f"reduce_argmin returns int32 source positions, but "
                    f"{indices.name!r} is {indices.dtype.value}",
                    category,
                )
    if operation.kind is OperationKind.CAST:
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            out.add(
                "CAST_ARITY",
                path,
                "cast reads one numeric tile and writes one converted tile",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            output = buffers.get(operation.writes[0])
            if source is not None and output is not None:
                if source.shape != output.shape:
                    out.add(
                        "CAST_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"cast preserves shape {list(source.shape)}, got "
                        f"{list(output.shape)}",
                        category,
                    )
                allowed = {DType.BF16, DType.FP16, DType.FP32}
                if source.dtype not in allowed or output.dtype not in allowed:
                    out.add(
                        "CAST_DTYPE_UNSUPPORTED",
                        path,
                        "the admitted cast converts among bf16, fp16, and fp32",
                        category,
                    )
                if output.dtype is not operation.parameters.to:
                    out.add(
                        "CAST_RESULT_DTYPE",
                        f"{path}.writes",
                        f"cast(to={operation.parameters.to.value}) writes "
                        f"{output.dtype.value}",
                        category,
                    )
                if source.dtype is operation.parameters.to:
                    out.add(
                        "CAST_IDENTITY",
                        f"{path}.parameters.to",
                        "an identity cast is a second spelling of the unchanged tile",
                        category,
                    )
                if (
                    source.space is not MemorySpace.REGISTER
                    or output.space is not MemorySpace.REGISTER
                ):
                    out.add(
                        "CAST_SPACE",
                        path,
                        "cast operands must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
    if operation.kind is OperationKind.INDEX_EXPAND:
        if len(operation.reads) != 1 or len(operation.writes) != 1:
            out.add(
                "INDEX_EXPAND_ARITY",
                path,
                "index_expand reads selected int32 indices and writes one flat int32 run",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            output = buffers.get(operation.writes[0])
            if source is not None:
                if len(source.shape) != 1:
                    out.add(
                        "INDEX_EXPAND_SOURCE_SHAPE",
                        f"{path}.reads",
                        "index_expand consumes one rank-one selected-index tile",
                        category,
                    )
                if source.dtype is not DType.INT32:
                    out.add(
                        "INDEX_EXPAND_DTYPE",
                        f"{path}.reads",
                        f"index_expand input must be int32, got {source.dtype.value}",
                        category,
                    )
                if source.space is not MemorySpace.REGISTER:
                    out.add(
                        "INDEX_EXPAND_SPACE",
                        f"{path}.reads",
                        "index_expand input must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if len(source.shape) == 1 and source.shape[0] & (source.shape[0] - 1):
                    out.add(
                        "INDEX_EXPAND_SOURCE_UNLOWERABLE",
                        f"{path}.reads",
                        "the Triton index_expand input extent must be a power of two",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if output is not None:
                expected = (
                    source.shape[0] * operation.parameters.extent
                    if source is not None and len(source.shape) == 1
                    else None
                )
                if expected is not None and output.shape != (expected,):
                    out.add(
                        "INDEX_EXPAND_RESULT_SHAPE",
                        f"{path}.writes",
                        f"index_expand must write [{expected}], got {list(output.shape)}",
                        category,
                    )
                if output.dtype is not DType.INT32:
                    out.add(
                        "INDEX_EXPAND_DTYPE",
                        f"{path}.writes",
                        f"index_expand output must be int32, got {output.dtype.value}",
                        category,
                    )
                if output.space is not MemorySpace.REGISTER:
                    out.add(
                        "INDEX_EXPAND_SPACE",
                        f"{path}.writes",
                        "index_expand output must stay in registers before store",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            extent = operation.parameters.extent
            if extent & (extent - 1):
                out.add(
                    "INDEX_EXPAND_EXTENT_UNLOWERABLE",
                    f"{path}.parameters.extent",
                    "the Triton index_expand extent must be a power of two",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
            if not -(1 << 31) <= operation.parameters.sentinel < (1 << 31):
                out.add(
                    "INDEX_EXPAND_SENTINEL_RANGE",
                    f"{path}.parameters.sentinel",
                    "index_expand sentinel must fit signed int32",
                    category,
                )
    if operation.kind is OperationKind.ONLINE_SOFTMAX:
        if len(operation.reads) not in {2, 3} or len(operation.writes) != 4:
            out.add(
                "ONLINE_SOFTMAX_ARITY",
                path,
                "online_softmax reads logits, values, optional validity indices and "
                "writes running maximum, "
                "normalizer, weighted accumulator, and normalized output",
                category,
            )
        else:
            logits = buffers.get(operation.reads[0])
            values = buffers.get(operation.reads[1])
            validity = (
                buffers.get(operation.reads[2]) if len(operation.reads) == 3 else None
            )
            maximum = buffers.get(operation.writes[0])
            normalizer = buffers.get(operation.writes[1])
            accumulator = buffers.get(operation.writes[2])
            normalized = buffers.get(operation.writes[3])
            axis = operation.parameters.axis
            if logits is not None:
                if len(logits.shape) != 2 or axis != 1:
                    out.add(
                        "ONLINE_SOFTMAX_LOGITS_SHAPE",
                        f"{path}.reads",
                        "the admitted online_softmax lowering consumes rank-two "
                        "[rows, selected] logits and reduces axis 1",
                        category,
                    )
                if logits.dtype is not DType.FP32:
                    out.add(
                        "ONLINE_SOFTMAX_LOGITS_DTYPE",
                        f"{path}.reads",
                        f"online_softmax logits must be fp32, got {logits.dtype.value}",
                        category,
                    )
                if logits.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.reads",
                        "online_softmax logits must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if values is not None:
                if len(values.shape) != 2:
                    out.add(
                        "ONLINE_SOFTMAX_VALUE_SHAPE",
                        f"{path}.reads",
                        "online_softmax values must have shape [selected, value_dim]",
                        category,
                    )
                if values.dtype not in {DType.BF16, DType.FP16, DType.FP32}:
                    out.add(
                        "ONLINE_SOFTMAX_VALUE_DTYPE",
                        f"{path}.reads",
                        f"online_softmax values must be floating, got {values.dtype.value}",
                        category,
                    )
                if values.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.reads",
                        "online_softmax values must be resident in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if len(operation.reads) == 3:
                if operation.parameters.sentinel is None:
                    out.add(
                        "ONLINE_SOFTMAX_SENTINEL_REQUIRED",
                        f"{path}.parameters.sentinel",
                        "validity indices require one explicit sentinel",
                        category,
                    )
                if validity is not None:
                    selected = (
                        logits.shape[1]
                        if logits is not None and len(logits.shape) == 2
                        else None
                    )
                    if selected is not None and validity.shape != (selected,):
                        out.add(
                            "ONLINE_SOFTMAX_VALIDITY_SHAPE",
                            f"{path}.reads",
                            f"validity indices must have shape [{selected}], got "
                            f"{list(validity.shape)}",
                            category,
                        )
                    if validity.dtype is not DType.INT32:
                        out.add(
                            "ONLINE_SOFTMAX_VALIDITY_DTYPE",
                            f"{path}.reads",
                            f"validity indices must be int32, got {validity.dtype.value}",
                            category,
                        )
                    if validity.space is not MemorySpace.REGISTER:
                        out.add(
                            "ONLINE_SOFTMAX_SPACE",
                            f"{path}.reads",
                            "validity indices must be resident in registers",
                            FindingCategory.HARDWARE_CONFORMANCE,
                        )
            elif operation.parameters.sentinel is not None:
                out.add(
                    "ONLINE_SOFTMAX_SENTINEL_UNUSED",
                    f"{path}.parameters.sentinel",
                    "online_softmax sentinel has no validity-index input",
                    category,
                )
            if operation.parameters.sentinel is not None and not (
                -(1 << 31) <= operation.parameters.sentinel < (1 << 31)
            ):
                out.add(
                    "ONLINE_SOFTMAX_SENTINEL_RANGE",
                    f"{path}.parameters.sentinel",
                    "online_softmax sentinel must fit signed int32",
                    category,
                )
            if (
                logits is not None
                and values is not None
                and len(logits.shape) == 2
                and len(values.shape) == 2
                and logits.shape[1] != values.shape[0]
            ):
                out.add(
                    "ONLINE_SOFTMAX_SELECTED_MISMATCH",
                    f"{path}.reads",
                    f"logits select {logits.shape[1]} values but the value tile has "
                    f"{values.shape[0]} rows",
                    category,
                )
            rows = logits.shape[0] if logits is not None and len(logits.shape) == 2 else None
            columns = (
                values.shape[1] if values is not None and len(values.shape) == 2 else None
            )
            expected = ((rows,), (rows,), (rows, columns), (rows, columns))
            for buffer, shape, label in zip(
                (maximum, normalizer, accumulator, normalized),
                expected,
                ("maximum", "normalizer", "accumulator", "normalized output"),
            ):
                if buffer is None or rows is None or columns is None:
                    continue
                if buffer.shape != shape:
                    out.add(
                        "ONLINE_SOFTMAX_STATE_SHAPE",
                        f"{path}.writes",
                        f"online_softmax {label} must have shape {list(shape)}, got "
                        f"{list(buffer.shape)}",
                        category,
                    )
                if buffer.dtype is not DType.FP32:
                    out.add(
                        "ONLINE_SOFTMAX_STATE_DTYPE",
                        f"{path}.writes",
                        f"online_softmax {label} must be fp32, got {buffer.dtype.value}",
                        category,
                    )
                if buffer.space is not MemorySpace.REGISTER:
                    out.add(
                        "ONLINE_SOFTMAX_SPACE",
                        f"{path}.writes",
                        f"online_softmax {label} must stay in registers",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
    if operation.kind is OperationKind.TOP_K:
        if (
            operation.parameters.source_tiles_per_merge == 2
            and not operation.parameters.across_loop
        ):
            out.add(
                "TOP_K_MERGE_CADENCE_REQUIRES_ACROSS_LOOP",
                f"{path}.parameters.source_tiles_per_merge",
                "source_tiles_per_merge=2 changes a loop-carried merge cadence, but "
                "this top_k does not declare across_loop=true",
                category,
            )
        if len(operation.reads) != 1 or len(operation.writes) != 2:
            out.add(
                "TOP_K_ARITY",
                path,
                "top_k reads exactly one score tile and writes values then int32 "
                f"indices, got {len(operation.reads)} read(s) and "
                f"{len(operation.writes)} write(s)",
                category,
            )
        else:
            source = buffers.get(operation.reads[0])
            values = buffers.get(operation.writes[0])
            indices = buffers.get(operation.writes[1])
            k = operation.parameters.k
            if source is not None:
                if len(source.shape) != 1:
                    out.add(
                        "TOP_K_SOURCE_RANK",
                        f"{path}.reads",
                        f"top_k selects from one resident rank-one tile, but "
                        f"{source.name!r} has shape {list(source.shape)}",
                        category,
                    )
                elif k > source.shape[0] and not operation.parameters.across_loop:
                    out.add(
                        "TOP_K_K_OUT_OF_RANGE",
                        f"{path}.parameters.k",
                        f"top_k asks for {k} values from {source.name!r} with extent "
                        f"{source.shape[0]}",
                        category,
                    )
                if len(source.shape) == 1 and source.shape[0] & (source.shape[0] - 1):
                    out.add(
                        "TOP_K_SOURCE_UNLOWERABLE",
                        f"{path}.reads",
                        "the admitted SM100 Triton top_k merge uses power-of-two "
                        f"resident vectors, but {source.name!r} has extent "
                        f"{source.shape[0]}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if source.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_SOURCE_SPACE",
                        f"{path}.reads",
                        f"top_k reduces a resident register tile, but {source.name!r} "
                        f"is in {source.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if source.dtype not in {DType.FP32, DType.INT32}:
                    out.add(
                        "TOP_K_VALUE_DTYPE",
                        f"{path}.reads",
                        f"the admitted top_k lowering orders fp32 or int32 values, but "
                        f"{source.name!r} is {source.dtype.value}",
                        category,
                    )
                if (
                    source.dtype is DType.INT32
                    and operation.parameters.across_loop
                ):
                    out.add(
                        "TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE",
                        f"{path}.parameters.across_loop",
                        "the admitted signed-int32 top_k lowering orders one resident "
                        "tile; loop-carried int32 state is not implemented",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if values is not None:
                if values.shape != (k,):
                    out.add(
                        "TOP_K_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"top_k with k={k} writes values[{k}], but {values.name!r} has "
                        f"shape {list(values.shape)}",
                        category,
                    )
                if source is not None and values.dtype is not source.dtype:
                    out.add(
                        "TOP_K_VALUE_DTYPE",
                        f"{path}.writes",
                        f"top_k values preserve {source.name!r}'s {source.dtype.value} "
                        f"dtype, but {values.name!r} is {values.dtype.value}",
                        category,
                    )
                if values.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_RESULT_SPACE",
                        f"{path}.writes",
                        f"top_k values stay in registers before an explicit store, but "
                        f"{values.name!r} is in {values.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if indices is not None:
                if indices.shape != (k,):
                    out.add(
                        "TOP_K_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"top_k with k={k} writes indices[{k}], but {indices.name!r} "
                        f"has shape {list(indices.shape)}",
                        category,
                    )
                if indices.dtype is not DType.INT32:
                    out.add(
                        "TOP_K_INDEX_DTYPE",
                        f"{path}.writes",
                        f"top_k source positions are int32, but {indices.name!r} is "
                        f"{indices.dtype.value}",
                        category,
                    )
                if indices.space is not MemorySpace.REGISTER:
                    out.add(
                        "TOP_K_RESULT_SPACE",
                        f"{path}.writes",
                        f"top_k indices stay in registers before an explicit store, but "
                        f"{indices.name!r} is in {indices.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
            if k & (k - 1):
                out.add(
                    "TOP_K_K_UNLOWERABLE",
                    f"{path}.parameters.k",
                    f"the admitted SM100 Triton top_k lowering requires power-of-two k, "
                    f"but k is {k}",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
    if operation.kind is OperationKind.SCAN and operation.reads and operation.writes:
        source = buffers.get(operation.reads[0])
        result = buffers.get(operation.writes[0])
        axis = operation.parameters.axis
        if source is not None and result is not None:
            if (
                source.dtype not in _ELEMENTWISE_FLOAT_DTYPES
                or result.dtype is not DType.FP32
            ):
                out.add(
                    "SCAN_DTYPE_MISMATCH",
                    f"{path}.writes",
                    f"a {operation.parameters.op.value} scan accumulates bf16/fp16/fp32 "
                    f"into fp32, but {source.name!r} is {source.dtype.value} and "
                    f"{result.name!r} is {result.dtype.value}",
                    category,
                )
            if axis >= len(source.shape):
                out.add(
                    "SCAN_AXIS_OUT_OF_RANGE",
                    f"{path}.parameters.axis",
                    f"axis {axis} is outside {source.name!r}, which has "
                    f"{len(source.shape)} dimension(s)",
                    category,
                )
            elif tuple(result.shape) != tuple(source.shape):
                out.add(
                    "SCAN_SHAPE_MISMATCH",
                    f"{path}.writes",
                    f"scanning axis {axis} of {source.name!r} {list(source.shape)} "
                    f"keeps that shape, but {result.name!r} is {list(result.shape)}",
                    category,
                )

    expected = _ARITY.get(operation.kind)
    if expected is not None:
        reads, writes, label = expected
        if len(operation.reads) < reads or len(operation.writes) < writes:
            out.add(
                "OP_ARITY",
                path,
                f"{label} requires at least {reads} read(s) and {writes} write(s), got "
                f"{len(operation.reads)} and {len(operation.writes)}",
                category,
            )
    if operation.kind is OperationKind.LOAD:
        # The first read is the data source. Later reads are permitted only as
        # AccessMap-owned runtime coordinates; _verify_access_maps proves their exact
        # set, placement and dtype. Treating every read as a second data source made the
        # operation graph unable to name those real dependencies.
        for name in operation.reads[:1]:
            buffer = buffers.get(name)
            if buffer is not None and buffer.space is not (
                MemorySpace.TENSOR if operation.parameters.movement is LoadMovement.TMEM
                else MemorySpace.GLOBAL
            ):
                out.add(
                    "OP_LOAD_SOURCE",
                    f"{path}.reads",
                    f"load source {name!r} is {buffer.space.value}; a load moves from "
                    f"{'tensor' if operation.parameters.movement is LoadMovement.TMEM else 'global'} memory",
                    category,
                )
    if operation.kind is OperationKind.LOAD and operation.parameters.movement is LoadMovement.TMEM:
        source = buffers.get(operation.reads[0]) if len(operation.reads) == 1 else None
        dest = buffers.get(operation.writes[0]) if len(operation.writes) == 1 else None
        atom = operation.parameters.source_atom
        if (source is None or dest is None or source.space is not MemorySpace.TENSOR
                or dest.space is not MemorySpace.REGISTER or source.dtype is not DType.FP32
                or dest.dtype is not DType.FP32 or source.shape != dest.shape):
            out.add("TMEM_LOAD_CONTRACT", path,
                    "tmem load moves one FP32 tensor tile into an identical register tile",
                    FindingCategory.DATA_CONSISTENCY)
        if atom is None or atom.op != "tcgen05.Ld32x32b" or atom.repetition not in (1,2,4,8,16,32,64,128):
            out.add("TMEM_LOAD_ATOM", f"{path}.parameters.source_atom",
                    "tmem load requires an explicit 32x32b power-of-two repetition in [1,128]",
                    FindingCategory.HARDWARE_CONFORMANCE)
    if operation.kind is OperationKind.ATOMIC_RMW:
        if not -(1 << 31) <= operation.parameters.value < (1 << 31):
            out.add(
                "ATOMIC_VALUE_RANGE",
                f"{path}.parameters.value",
                "the admitted int32 atomic add scalar must be in signed 32-bit range",
                FindingCategory.HARDWARE_CONFORMANCE,
            )
        if len(operation.reads) != 2 or len(operation.writes) != 2:
            out.add(
                "ATOMIC_EDGE_COUNT",
                path,
                "the admitted atomic_rmw reads target then one runtime index and "
                "writes the same target then one returned-old-value buffer",
                category,
            )
        if operation.reads and operation.writes:
            target = buffers.get(operation.reads[0])
            result = buffers.get(operation.writes[-1])
            if operation.writes[0] != operation.reads[0]:
                out.add(
                    "ATOMIC_TARGET_EDGE",
                    f"{path}.writes",
                    "atomic_rmw must write the same state buffer it reads first",
                    category,
                )
            if target is not None:
                if target.mode is not BufferMode.STATE:
                    out.add(
                        "ATOMIC_TARGET_MODE",
                        f"{path}.reads",
                        f"atomic target {target.name!r} is {target.mode.value}, not state",
                        category,
                    )
                if target.dtype is not DType.INT32:
                    out.add(
                        "ATOMIC_TARGET_DTYPE",
                        f"{path}.reads",
                        f"the admitted atomic add targets int32, but {target.name!r} is "
                        f"{target.dtype.value}",
                        category,
                    )
            if result is not None:
                if result.space is not MemorySpace.REGISTER:
                    out.add(
                        "ATOMIC_RESULT_SPACE",
                        f"{path}.writes",
                        f"atomic old values stay in registers before an explicit store, "
                        f"but {result.name!r} is in {result.space.value}",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                if result.dtype is not DType.INT32:
                    out.add(
                        "ATOMIC_RESULT_DTYPE",
                        f"{path}.writes",
                        f"int32 atomic add returns int32, but {result.name!r} is "
                        f"{result.dtype.value}",
                        category,
                    )
    if operation.kind is OperationKind.STORE:
        if len(operation.writes) != 1:
            out.add(
                "STORE_EDGE_COUNT",
                f"{path}.writes",
                f"store declares exactly one destination edge, got "
                f"{len(operation.writes)}",
                category,
            )
        for name in operation.writes:
            buffer = buffers.get(name)
            if buffer is not None and buffer.mode not in {
                BufferMode.OUTPUT,
                BufferMode.STATE,
            }:
                out.add(
                    "OP_STORE_DESTINATION",
                    f"{path}.writes",
                    f"store destination {name!r} is {buffer.mode.value}, not output/state",
                    category,
                )
        if operation.reads and operation.writes:
            source = buffers.get(operation.reads[0])
            destination = buffers.get(operation.writes[0])
            if source is not None and destination is not None:
                supported = source.dtype is destination.dtype or (
                    source.dtype is DType.FP32
                    and destination.dtype in {DType.BF16, DType.FP16}
                )
                if not supported:
                    out.add(
                        "STORE_DTYPE_UNSUPPORTED",
                        f"{path}.writes",
                        f"store cannot convert {source.name!r} from "
                        f"{source.dtype.value} to {destination.name!r} "
                        f"{destination.dtype.value}; the admitted conversions are "
                        "identity and fp32 to bf16/fp16",
                        category,
                    )
    # An ordinary contraction reads two data operands. The one admitted block-scale
    # contract reads the same pair followed by their two related scales; the contract,
    # rather than a flag, is the single owner of that arity and meaning.
    if operation.kind is OperationKind.MMA:
        instruction = operation.parameters.instruction
        contract = instruction.contract if instruction is not None else None
        expected_reads = 4 if contract == _BLOCK_SCALE_MMA_CONTRACT else 2
        staged = [
            name
            for name in operation.reads
            if (buffer := buffers.get(name)) is not None
            and buffer.space is not MemorySpace.GLOBAL
        ]
        if len(operation.reads) != expected_reads or len(staged) != expected_reads:
            out.add(
                "MMA_OPERAND_COUNT",
                f"{path}.reads",
                f"contract {contract!r} reads exactly {expected_reads} staged "
                f"operands, got {list(operation.reads)}",
                category,
            )
        elif contract == _BLOCK_SCALE_MMA_CONTRACT:
            _verify_block_scaled_mma(operation, path, buffers, out)
        if len(operation.writes) != 1:
            out.add(
                "MMA_RESULT_COUNT", f"{path}.writes",
                "one contraction writes exactly one explicit result", category,
            )
        tile = operation.parameters.tile_shape
        if tile is not None and backend is not None and backend.value in {"native_cuda", "triton"}:
            operands = [buffers.get(name) for name in operation.reads[:2]]
            if (len(operands) != 2
                    or operands[0] is None or operands[1] is None
                    or operands[0].shape != (tile[0], tile[2])
                    or operands[1].shape != (tile[1], tile[2])):
                out.add(
                    "MMA_INPUT_TILE_DOMAIN", f"{path}.parameters.tile_shape",
                    "a contraction uses the full rank-two input tile domain "
                    "A[M,K] and B[N,K]; tile_shape must describe that input domain", category,
                )

    # An arithmetic primitive takes what its op says it takes. A binary op reads two
    # buffers, or one buffer and a declared scalar; anything else is a Schedule asking
    # for arithmetic whose operands are not all named.
    if operation.kind is OperationKind.ELEMENTWISE:
        parameters = operation.parameters
        if parameters.op is ElementwiseOp.FMA:
            if len(operation.writes) != 1:
                out.add(
                    "ELEMENTWISE_FMA_RESULT_COUNT", f"{path}.writes",
                    "fma writes exactly one FP32 register result", category,
                )
            for name in (*operation.reads, *operation.writes):
                buffer = buffers.get(name)
                if buffer is not None and buffer.space is not MemorySpace.REGISTER:
                    out.add(
                        "ELEMENTWISE_FMA_SPACE", path,
                        f"fma operand/result {name!r} must be register-resident",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
        supplied = len(operation.reads) + (parameters.scalar is not None)
        if supplied != parameters.arity_needed:
            out.add(
                "ELEMENTWISE_ARITY",
                f"{path}.reads",
                f"{parameters.op.value} takes {parameters.arity_needed} operand(s); "
                f"{len(operation.reads)} read(s) and "
                f"{'a' if parameters.scalar is not None else 'no'} scalar were given",
                category,
            )
        elif len(operation.writes) == 1:
            result = buffers.get(operation.writes[0])
            reads = [buffers[name] for name in operation.reads if name in buffers]
            if result is not None and len(reads) == len(operation.reads):
                inferred_dtype = elementwise_result_dtype(read.dtype for read in reads)
                if inferred_dtype is None:
                    out.add(
                        "ELEMENTWISE_DTYPE_UNSUPPORTED",
                        f"{path}.reads",
                        f"{parameters.op.value} has no admitted promotion for "
                        f"{[read.dtype.value for read in reads]}",
                        category,
                    )
                elif result.dtype is not inferred_dtype:
                    out.add(
                        "ELEMENTWISE_RESULT_DTYPE",
                        f"{path}.writes",
                        f"{parameters.op.value} over "
                        f"{[read.dtype.value for read in reads]} produces "
                        f"{inferred_dtype.value}, but {result.name!r} is "
                        f"{result.dtype.value}",
                        category,
                    )
                # A canonical scalar [1] broadcasts without selecting a result axis.
                # Other singleton dimensions keep their existing explicit shape rules;
                # the instruction-specific FMA contract still requires equal shapes.
                shape_reads = [r for r in reads if not r.is_scalar] if parameters.op is not ElementwiseOp.FMA else reads
                widest = max((r.shape for r in (shape_reads or reads)), key=len, default=())
                if tuple(result.shape) != tuple(widest):
                    out.add(
                        "ELEMENTWISE_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"{parameters.op.value} over "
                        f"{[list(r.shape) for r in reads]} yields {list(widest)}, but "
                        f"{result.name!r} is {list(result.shape)}",
                        category,
                    )
                for read in reads:
                    if read.is_scalar and parameters.op is not ElementwiseOp.FMA:
                        axis = parameters.broadcast_axis
                        if axis is not None and (axis >= len(widest) or widest[axis] != 1):
                            out.add(
                                "ELEMENTWISE_BROADCAST", f"{path}.parameters.broadcast_axis",
                                f"scalar operand {read.name!r} needs no broadcast axis; "
                                f"declared axis {axis} does not have extent one", category,
                            )
                        continue
                    if len(read.shape) == len(widest):
                        if tuple(read.shape) != tuple(widest):
                            out.add(
                                "ELEMENTWISE_SHAPE_MISMATCH",
                                f"{path}.reads",
                                f"operand {read.name!r} {list(read.shape)} differs from "
                                f"{list(widest)} and is not narrower",
                                category,
                            )
                        continue
                    axis = parameters.broadcast_axis
                    if axis is None:
                        out.add(
                            "ELEMENTWISE_BROADCAST_UNDECLARED",
                            f"{path}.parameters.broadcast_axis",
                            f"operand {read.name!r} {list(read.shape)} is narrower than "
                            f"{list(widest)}; the axis it spans must be declared",
                            category,
                        )
                    elif len(read.shape) != 1 or axis >= len(widest) or read.shape[0] != widest[axis]:
                        out.add(
                            "ELEMENTWISE_BROADCAST",
                            f"{path}.parameters.broadcast_axis",
                            f"operand {read.name!r} {list(read.shape)} does not span "
                            f"axis {axis} of {list(widest)}",
                            category,
                        )

    # A sum collapses one axis, so its result is its input with that axis dropped. The
    # extent used to be restated in the operation, which put the same fact in two places
    # and left the pair uncheckable; deriving it from the buffers makes disagreement
    # between them a Finding instead of a kernel that reduces the wrong number of values.
    if operation.kind is OperationKind.REDUCE and operation.reads and operation.writes:
        source = buffers.get(operation.reads[0])
        result = buffers.get(operation.writes[0])
        axis = operation.parameters.axis
        if source is not None and result is not None:
            if (
                source.dtype not in _ELEMENTWISE_FLOAT_DTYPES
                or result.dtype is not DType.FP32
            ):
                out.add(
                    "REDUCE_DTYPE_MISMATCH",
                    f"{path}.writes",
                    f"{operation.parameters.op.value} reduces bf16/fp16/fp32 into "
                    f"fp32, but {source.name!r} is {source.dtype.value} and "
                    f"{result.name!r} is {result.dtype.value}",
                    category,
                )
            if axis >= len(source.shape):
                out.add(
                    "REDUCE_AXIS_OUT_OF_RANGE",
                    f"{path}.parameters.axis",
                    f"axis {axis} is outside {source.name!r}, which has "
                    f"{len(source.shape)} dimension(s)",
                    category,
                )
            else:
                # Buffer scalars use [1], as scalar loads and the Python frontend do;
                # a rank-zero Buffer is not part of the IR.
                collapsed = source.shape[:axis] + source.shape[axis + 1 :] or (1,)
                if tuple(result.shape) != tuple(collapsed):
                    out.add(
                        "REDUCE_SHAPE_MISMATCH",
                        f"{path}.writes",
                        f"summing axis {axis} of {source.name!r} {list(source.shape)} "
                        f"yields {list(collapsed)}, but {result.name!r} is "
                        f"{list(result.shape)}",
                        category,
                    )


def _verify_access_maps(schedule: Schedule, buffers, out: _Collector) -> None:
    category = FindingCategory.DATA_CONSISTENCY
    op_by_id = {operation.op_id: operation for operation in schedule.operations}
    axis_names = (
        {axis.name for axis in schedule.program_map.axes}
        if schedule.program_map is not None
        else set()
    )
    tiled_axes = (
        {axis.name for axis in schedule.program_map.axes if axis.is_tiled}
        if schedule.program_map is not None
        else set()
    )
    loop_iterators = {loop.iterator for loop in schedule.tile_loops}

    for index, access in enumerate(schedule.access_maps):
        path = f"access_maps[{index}]"
        operation = op_by_id.get(access.operation)
        if operation is None:
            out.add(
                "ACCESS_OPERATION_UNKNOWN",
                f"{path}.operation",
                f"unknown operation {access.operation!r}",
                category,
            )
            continue
        buffer = buffers.get(access.buffer)
        if buffer is None:
            out.add(
                "ACCESS_BUFFER_UNKNOWN",
                f"{path}.buffer",
                f"unknown buffer {access.buffer!r}",
                category,
            )
            continue
        if access.buffer not in set(operation.reads) | set(operation.writes):
            out.add(
                "ACCESS_EDGE_UNKNOWN",
                f"{path}.buffer",
                f"operation {access.operation!r} neither reads nor writes "
                f"{access.buffer!r}",
                category,
            )
        if buffer.space is not MemorySpace.GLOBAL:
            out.add(
                "ACCESS_BUFFER_LOCAL",
                f"{path}.buffer",
                f"buffer {access.buffer!r} is {buffer.space.value}; access maps address "
                "global memory",
                category,
            )
        if len(access.indices) != len(buffer.shape):
            out.add(
                "ACCESS_RANK",
                f"{path}.indices",
                f"{len(access.indices)} index components for rank-{len(buffer.shape)} "
                f"buffer {access.buffer!r}",
                category,
            )
        for position, component in enumerate(access.indices):
            component_path = f"{path}.indices[{position}]"
            if component.source is AccessIndexKind.DIMENSION:
                if component.dimension is not None and component.dimension >= len(buffer.shape):
                    out.add(
                        "ACCESS_DIMENSION_MISMATCH",
                        component_path,
                        f"dimension {component.dimension} is outside rank-"
                        f"{len(buffer.shape)} buffer {access.buffer!r}",
                        category,
                    )
                elif component.dimension is not None:
                    # A sub-range that runs past the axis would read whatever follows it
                    # in the allocation, so the bound is checked here rather than left
                    # for the emitter to mask. An offset at or past the end selects
                    # nothing, which is a declaration no operation can have meant.
                    size = buffer.shape[component.dimension]
                    if component.offset >= size or component.offset + component.span(size) > size:
                        out.add(
                            "ACCESS_DIMENSION_SUBRANGE",
                            component_path,
                            f"sub-range [{component.offset}, "
                            f"{component.offset + component.span(size)}) leaves dimension "
                            f"{component.dimension} of size {size} in buffer "
                            f"{access.buffer!r}",
                            category,
                        )
                    elif component.extent == size - component.offset:
                        # A range reaching the end of the axis is written by omitting
                        # `extent`. Spelling it out names the same elements a second way,
                        # and the two spellings lower to different source -- so the same
                        # access would pin two different digests.
                        out.add(
                            "ACCESS_SUBRANGE_NONCANONICAL",
                            component_path,
                            f"extent {component.extent} reaches the end of dimension "
                            f"{component.dimension}; omit it to name the same range",
                            category,
                        )
                    if (
                        position < len(buffer.shape)
                        and component.dimension != position
                        and component.offset < size
                        and component.offset + component.span(size) <= size
                        and component.offset + component.span(size) > buffer.shape[position]
                    ):
                        out.add(
                            "ACCESS_DIMENSION_COORDINATE_RANGE",
                            component_path,
                            f"dimension vector reaches {component.offset + component.span(size)}, "
                            f"but addressed dimension {position} of {buffer.name!r} "
                            f"has extent {buffer.shape[position]}",
                            category,
                        )
            elif component.source is AccessIndexKind.LOOP_TILE:
                if component.name not in loop_iterators:
                    out.add(
                        "ACCESS_LOOP_UNKNOWN",
                        component_path,
                        f"unknown loop iterator {component.name!r}",
                        category,
                    )
                elif component.name not in {
                    loop.iterator for loop in schedule.enclosing_loops(operation)
                }:
                    out.add(
                        "ACCESS_LOOP_SCOPE",
                        component_path,
                        f"loop coordinate {component.name!r} is not visible in "
                        f"operation {operation.op_id!r}'s lexical scope",
                        category,
                    )
            elif component.source is AccessIndexKind.BUFFER:
                index_buffer = buffers.get(component.name)
                if index_buffer is None:
                    out.add(
                        "ACCESS_INDEX_BUFFER_UNKNOWN",
                        component_path,
                        f"unknown runtime index buffer {component.name!r}",
                        category,
                    )
                else:
                    if index_buffer.space is not MemorySpace.REGISTER:
                        out.add(
                            "ACCESS_INDEX_BUFFER_SPACE",
                            component_path,
                            f"runtime index buffer {component.name!r} is in "
                            f"{index_buffer.space.value}, not registers",
                            category,
                        )
                    if index_buffer.dtype is not DType.INT32:
                        out.add(
                            "ACCESS_INDEX_BUFFER_DTYPE",
                            component_path,
                            f"runtime index buffer {component.name!r} is "
                            f"{index_buffer.dtype.value}, not int32",
                            category,
                        )
            else:
                if component.name not in axis_names:
                    out.add(
                        "ACCESS_PROGRAM_AXIS_UNKNOWN",
                        component_path,
                        f"unknown program axis {component.name!r}",
                        category,
                    )
                elif (
                    component.source is AccessIndexKind.PROGRAM_TILE
                    and component.name not in tiled_axes
                ):
                    out.add(
                        "ACCESS_PROGRAM_AXIS_UNTILED",
                        component_path,
                        f"program axis {component.name!r} has tile 1 and carries no "
                        "offset vector; use source 'program'",
                        category,
                    )
                elif component.source is AccessIndexKind.PROGRAM:
                    axis = schedule.program_map.axis(component.name)
                    owner = buffers.get(axis.buffer) if axis is not None else None
                    if (
                        axis is not None
                        and owner is not None
                        and axis.dimension < len(owner.shape)
                        and position < len(buffer.shape)
                    ):
                        program_extent = axis.tile_count(owner.shape[axis.dimension])
                        if program_extent > buffer.shape[position]:
                            out.add(
                                "ACCESS_PROGRAM_EXTENT_MISMATCH",
                                component_path,
                                f"program axis {component.name!r} spans "
                                f"{program_extent} coordinate(s), but dimension "
                                f"{position} of {access.buffer!r} has extent "
                                f"{buffer.shape[position]}",
                                category,
                            )

        indirect = [
            component
            for component in access.indices
            if component.source is AccessIndexKind.BUFFER
        ]
        # AccessMap, not the ordering of Operation.reads, owns the global source.
        # Keep one dtype relation and one Finding spelling for direct and runtime-
        # indexed loads alike.
        if operation.kind is OperationKind.LOAD and len(operation.writes) == 1:
            staged = buffers.get(operation.writes[0])
            if staged is not None and staged.dtype is not buffer.dtype:
                out.add(
                    "LOAD_DTYPE_MISMATCH",
                    f"operations[{schedule.operations.index(operation)}].writes",
                    f"load preserves {buffer.name!r}'s {buffer.dtype.value}, but "
                    f"{staged.name!r} is {staged.dtype.value}",
                    category,
                )
        if (
            operation.kind is OperationKind.STORE
            and not indirect
            and len(operation.reads) != 1
        ):
            out.add(
                "STORE_EDGE_COUNT",
                f"operations[{schedule.operations.index(operation)}].reads",
                f"direct store declares exactly one value edge, got "
                f"{len(operation.reads)}",
                category,
            )
        if indirect:
            index_names = tuple(dict.fromkeys(component.name for component in indirect))
            if operation.kind not in {
                OperationKind.LOAD,
                OperationKind.ATOMIC_RMW,
                OperationKind.STORE,
            }:
                out.add(
                    "ACCESS_INDEXED_OPERATION_UNLOWERABLE",
                    path,
                    "the admitted runtime-indexed subset applies to global loads and "
                    "reservation-owned stores and atomic_rmw only",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )
            elif operation.kind is OperationKind.LOAD:
                if operation.parameters.movement is not LoadMovement.GLOBAL:
                    out.add(
                        "ACCESS_INDEXED_MOVEMENT_UNLOWERABLE",
                        path,
                        "runtime-indexed loads use direct global movement; TMA transfer "
                        "semantics are not admitted",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                expected_reads = (access.buffer,) + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "ACCESS_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"runtime-indexed load reads data then its first-use ordered "
                        f"index buffers {list(expected_reads)}, got "
                        f"{list(operation.reads)}",
                        category,
                    )
            elif operation.kind is OperationKind.ATOMIC_RMW:
                if len(access.indices) != 1 or len(indirect) != 1:
                    out.add(
                        "ATOMIC_INDEX_FORM",
                        f"{path}.indices",
                        "the admitted atomic target is rank one and has exactly one "
                        "runtime INT32 index",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                expected_reads = (access.buffer,) + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "ATOMIC_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        f"atomic_rmw reads target then its first-use ordered index "
                        f"buffers {list(expected_reads)}, got {list(operation.reads)}",
                        category,
                    )
            else:
                expected_reads = operation.reads[:1] + index_names
                if operation.reads != expected_reads:
                    out.add(
                        "STORE_INDEX_BUFFER_READS",
                        f"operations[{schedule.operations.index(operation)}].reads",
                        "runtime-indexed store reads its value then its first-use "
                        f"ordered index buffers {list(expected_reads)}, got "
                        f"{list(operation.reads)}",
                        category,
                    )

                canonical_form = (
                    len(indirect) == 2
                    and len(access.indices) >= 2
                    and all(
                        component.source is AccessIndexKind.BUFFER
                        for component in access.indices[:2]
                    )
                    and all(
                        component.source is not AccessIndexKind.BUFFER
                        for component in access.indices[2:]
                    )
                )
                if not canonical_form:
                    out.add(
                        "STORE_INDEX_RESERVATION_FORM",
                        f"{path}.indices",
                        "the admitted indexed store begins with exactly two runtime "
                        "coordinates: atomic target index then returned old value",
                        FindingCategory.PROGRAM_SAFETY,
                    )
                else:
                    state_index = access.indices[0].name
                    position = access.indices[1].name
                    producers = [
                        candidate
                        for candidate in schedule.operations
                        if candidate.kind is OperationKind.ATOMIC_RMW
                        and len(candidate.writes) == 2
                        and candidate.writes[1] == position
                    ]
                    if len(producers) != 1:
                        out.add(
                            "STORE_INDEX_RESERVATION_UNPROVEN",
                            f"{path}.indices[1]",
                            f"runtime position {position!r} is not the unique returned "
                            "old value of one atomic_rmw",
                            FindingCategory.PROGRAM_SAFETY,
                        )
                    else:
                        reservation = producers[0]
                        if reservation.role != operation.role:
                            out.add(
                                "STORE_INDEX_RESERVATION_ROLE",
                                f"operations[{schedule.operations.index(operation)}].role",
                                "the atomic result is register-resident, so reservation "
                                "and indexed store must execute in the same role",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        target_access = (
                            schedule.access_map(
                                reservation.op_id, reservation.reads[0]
                            )
                            if reservation.reads
                            else None
                        )
                        coordinates_match = (
                            target_access is not None
                            and len(target_access.indices) == 1
                            and target_access.indices[0].source
                            is AccessIndexKind.BUFFER
                            and target_access.indices[0].name == state_index
                        )
                        if not coordinates_match:
                            out.add(
                                "STORE_INDEX_RESERVATION_COORDINATES",
                                f"{path}.indices",
                                "the indexed store must pair the atomic target's exact "
                                "runtime index with that atomic's returned old value",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        target = (
                            buffers.get(reservation.reads[0])
                            if reservation.reads
                            else None
                        )
                        if (
                            target is not None
                            and target.shape
                            and buffer.shape
                            and target.shape[0] != buffer.shape[0]
                        ):
                            out.add(
                                "STORE_INDEX_RESERVATION_DOMAIN",
                                f"{path}.indices[0]",
                                "the store's reservation coordinate must have the same "
                                "extent as the atomic state target so their masks agree",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        if reservation.parameters.value != 1:
                            out.add(
                                "STORE_INDEX_RESERVATION_INCREMENT",
                                f"operations[{schedule.operations.index(reservation)}]."
                                "parameters.value",
                                "only atomic increment by one proves distinct returned "
                                "positions for an ordinary indexed store",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        reservation_index = schedule.operations.index(reservation)
                        loop_bodies = {
                            op_id for loop in schedule.tile_loops for op_id in loop.body
                        }
                        if (
                            reservation.op_id in loop_bodies
                            or operation.op_id in loop_bodies
                        ):
                            out.add(
                                "STORE_INDEX_RESERVATION_LOOP",
                                f"operations[{reservation_index}]",
                                "the first reservation-owned store subset executes its "
                                "atomic and store exactly once per program, outside tile "
                                "loops",
                                FindingCategory.PROGRAM_SAFETY,
                            )
                        program_instances = 1
                        program_domain_known = schedule.program_map is not None
                        if schedule.program_map is not None:
                            for axis in schedule.program_map.axes:
                                owner = buffers.get(axis.buffer)
                                if owner is None or axis.dimension >= len(owner.shape):
                                    program_domain_known = False
                                    break
                                program_instances *= axis.tile_count(
                                    owner.shape[axis.dimension]
                                )
                        result = buffers.get(position)
                        if result is None:
                            program_domain_known = False
                        else:
                            for extent in result.shape:
                                program_instances *= extent
                        if program_domain_known and program_instances > 1 << 32:
                            out.add(
                                "STORE_INDEX_RESERVATION_WRAP",
                                f"operations[{reservation_index}]",
                                "the launch may execute more than 2^32 reservations, "
                                "so an int32 returned position can repeat after wrap",
                                FindingCategory.PROGRAM_SAFETY,
                            )

            index_buffers = [buffers.get(name) for name in index_names]
            known = [item for item in index_buffers if item is not None]
            if len(known) == len(index_buffers):
                shapes = {item.shape for item in known}
                if any(len(item.shape) != 1 for item in known):
                    out.add(
                        "ACCESS_INDEX_DOMAIN_RANK",
                        path,
                        "the admitted runtime index domain is rank one",
                        FindingCategory.HARDWARE_CONFORMANCE,
                    )
                elif len(shapes) != 1:
                    out.add(
                        "ACCESS_INDEX_DOMAIN_MISMATCH",
                        path,
                        f"runtime index buffers must share one zipped domain, got "
                        f"{[list(item.shape) for item in known]}",
                        category,
                    )

            # The local value has the zipped index domain once, plus every independent
            # tile/full-dimension domain in access order. That is the exact shape the
            # Triton address branch emits for a load result, atomic result or store
            # value.
            result_name = None
            if operation.kind is OperationKind.LOAD and len(operation.writes) == 1:
                result_name = operation.writes[0]
            elif operation.kind is OperationKind.ATOMIC_RMW and len(operation.writes) == 2:
                result_name = operation.writes[1]
            elif operation.kind is OperationKind.STORE and operation.reads:
                result_name = operation.reads[0]
            if result_name is not None:
                staged = buffers.get(result_name)
                source = buffer
                expected_shape: list[int] = []
                added_index_domain = False
                shape_known = len(known) == len(index_buffers) and bool(known)
                for component in access.indices:
                    if component.source is AccessIndexKind.PROGRAM:
                        continue
                    if component.source is AccessIndexKind.BUFFER:
                        if not added_index_domain and shape_known:
                            expected_shape.extend(known[0].shape)
                            added_index_domain = True
                        continue
                    if component.source is AccessIndexKind.PROGRAM_TILE:
                        axis = (
                            schedule.program_map.axis(component.name)
                            if schedule.program_map is not None
                            else None
                        )
                        if axis is not None:
                            expected_shape.append(axis.tile)
                    elif component.source is AccessIndexKind.LOOP_TILE:
                        loop = next(
                            (
                                item
                                for item in schedule.tile_loops
                                if item.iterator == component.name
                            ),
                            None,
                        )
                        if loop is not None:
                            expected_shape.append(loop.tile)
                    elif (
                        component.dimension is not None
                        and component.dimension < len(source.shape)
                    ):
                        # The value domain is what the access covers. Reading the axis
                        # size here demanded a staged buffer sized for elements a
                        # sub-range never addresses, which blocked a gather the emitter
                        # already lowered correctly.
                        expected_shape.append(
                            component.span(source.shape[component.dimension])
                        )
                if staged is not None and shape_known:
                    if staged.shape != tuple(expected_shape):
                        out.add(
                            "ACCESS_INDEXED_VALUE_SHAPE",
                            f"operations[{schedule.operations.index(operation)}]."
                            + (
                                "writes"
                                if operation.kind is not OperationKind.STORE
                                else "reads"
                            ),
                            f"runtime-indexed access has value domain {expected_shape}, but "
                            f"{staged.name!r} has shape {list(staged.shape)}",
                            category,
                        )

        relation = buffer.valid_extent
        if relation is not None and relation.indexed_by:
            supported = (
                len(relation.indexed_by) == 1
                and relation.indexed_by[0] < len(access.indices)
                and access.indices[relation.indexed_by[0]].source
                is AccessIndexKind.PROGRAM
            )
            if not supported:
                out.add(
                    "VALID_EXTENT_ACCESS_UNLOWERABLE",
                    path,
                    "the Triton valid-extent lowering requires one extent axis "
                    "indexed by one scalar program axis",
                    FindingCategory.HARDWARE_CONFORMANCE,
                )

    # A tile axis produces a staged extent. If the axis says 128 and the buffer it
    # stages into says 256, the two disagree about the same tile and one of them is
    # wrong; nothing downstream can tell which.
    for index, access in enumerate(schedule.access_maps):
        operation = op_by_id.get(access.operation)
        if operation is None or not operation.writes:
            continue
        # The staged side, not the written side. A load writes its tile and a store reads
        # it, so taking `writes[0]` compared a store's *global* output against the tile
        # axes. Every store in the corpus escaped that by a rank coincidence -- the global
        # buffer had one more dimension than the access map had vector components, so the
        # check below skipped it -- and the first rank-2 output made it a false positive
        # on a Schedule that was correct.
        staged = next(
            (
                buffer
                for name in list(operation.writes) + list(operation.reads)
                if (buffer := buffers.get(name)) is not None
                and buffer.space is not MemorySpace.GLOBAL
            ),
            None,
        )
        source = buffers.get(access.buffer)
        if staged is None or source is None or source.space is not MemorySpace.GLOBAL:
            continue
        # Buffer-valued accesses have one zipped domain no matter how many coordinates
        # it supplies. The dedicated check above derives that shape; the legacy loop
        # below intentionally remains byte-for-byte the path for all existing cases.
        if any(
            component.source is AccessIndexKind.BUFFER
            for component in access.indices
        ):
            continue
        vectors = []
        for component_index, component in enumerate(access.indices):
            if not component.is_vector:
                continue
            if component.source is AccessIndexKind.PROGRAM_TILE:
                axis = (
                    schedule.program_map.axis(component.name)
                    if schedule.program_map is not None
                    else None
                )
                expected = axis.tile if axis is not None else None
            elif component.source is AccessIndexKind.LOOP_TILE:
                loop = next(
                    (l for l in schedule.tile_loops if l.iterator == component.name), None
                )
                expected = loop.tile if loop is not None else None
            else:
                # A sub-range stages what it covers, not the whole axis. Reading the
                # axis size here would demand a tile the access never addresses.
                expected = (
                    component.span(source.shape[component.dimension])
                    if component.dimension is not None and component.dimension < len(source.shape)
                    else None
                )
            vectors.append((component_index, component, expected))
        # A direct register load produces exactly the vector axes in its address.
        # Scalar addresses use the canonical one-value register shape (1,).
        # A rank mismatch must not bypass this relation: otherwise a scalar can be
        # mislabeled as a vector and silently broadcast by a later FMA.
        shape_known = all(extent is not None for _, _, extent in vectors)
        expected_shape = tuple(extent for _, _, extent in vectors) or (1,)
        if (
            operation.kind is OperationKind.LOAD
            and operation.parameters.movement is LoadMovement.GLOBAL
            and staged.space is MemorySpace.REGISTER
            and shape_known
            and (not vectors or len(staged.shape) != len(vectors))
            and staged.shape != expected_shape
        ):
            out.add(
                "LOAD_ACCESS_SHAPE_MISMATCH",
                f"operations[{schedule.operations.index(operation)}].writes[0]",
                f"load address produces register shape {list(expected_shape)}, but "
                f"{staged.name!r} declares {list(staged.shape)}; a load does not splat or reshape",
                category,
            )
        if (operation.kind is OperationKind.STORE and staged.space is MemorySpace.REGISTER
                and shape_known and (not vectors or len(staged.shape) != len(vectors))
                and staged.shape != expected_shape):
            out.add("STORE_ACCESS_SHAPE_MISMATCH",
                f"operations[{schedule.operations.index(operation)}].reads[0]",
                f"store address requires register shape {list(expected_shape)}, but "
                f"{staged.name!r} declares {list(staged.shape)}; a store does not splat or reshape",
                category)
        if len(staged.shape) != len(vectors):
            continue
        for staged_dimension, (component_index, component, expected) in enumerate(vectors):
            if expected is not None and staged.shape[staged_dimension] != expected:
                out.add(
                    "ACCESS_TILE_MISMATCH",
                    f"access_maps[{index}].indices[{component_index}]",
                    f"tile axis {component.name or component.dimension} carries "
                    f"{expected} but {staged.name!r} stages {staged.shape[staged_dimension]}",
                    category,
                )

    # Every global buffer an operation touches needs an addressing rule, otherwise
    # lowering has to invent one.
    if schedule.access_maps:
        declared = {(item.operation, item.buffer) for item in schedule.access_maps}
        for index, operation in enumerate(schedule.operations):
            for name in sorted(set(operation.reads) | set(operation.writes)):
                buffer = buffers.get(name)
                if buffer is None or buffer.space is not MemorySpace.GLOBAL:
                    continue
                if (operation.op_id, name) not in declared:
                    out.add(
                        "ACCESS_MAP_MISSING",
                        f"operations[{index}]",
                        f"operation {operation.op_id!r} touches global buffer {name!r} "
                        "without an access map",
                        category,
                    )
