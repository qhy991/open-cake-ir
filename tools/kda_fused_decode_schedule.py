"""Lab-owned Open-CAKE Schedule seed for the complete Kimi-K3 decode boundary.

This module is deliberately outside the Compiler.  It owns the frozen nine-cell
rank-local workload projection and the zero-copy logical views that adapt the public
17-argument KDA ABI to one Schedule.  The Schedule composes only public IR primitives;
it contains no workload-named Compiler branch and no copied CUDA, PTX, SASS, or emitted
implementation strategy.

The first seed gives one CTA one graph row and one local head.  That CTA owns the full
128x128 V-first recurrent-state tile and performs the complete boundary:

    width-4 causal convolution -> KDA recurrence -> sigmoid-gated RMSNorm

L2 and RMS reductions produce rank-zero register scratch.  The typed elementwise rule
broadcasts that runtime scalar without inventing a data axis; ``outer`` is consequently
reserved for the real ``delta x k`` recurrent-state update.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from open_cake_ir.compiler.ir import Schedule

HEAD_DIM = 128
CONV_WIDTH = 4
CONV_HISTORY = CONV_WIDTH - 1
STATE_SLOT_PAD = 256
LOWER_BOUND = -5.0
SCALE = 1.0 / math.sqrt(HEAD_DIM)
ONORM_EPS = 1e-6
QK_L2_EPS = 1e-6

ABI_ARGUMENTS = (
    "mixed_qkv",
    "a",
    "b",
    "conv_states",
    "w_q_t",
    "w_k_t",
    "w_v_t",
    "conv_bias",
    "A_log",
    "dt_bias",
    "onorm_g",
    "onorm_weight",
    "ssm_states",
    "cache_indices",
    "scale",
    "onorm_eps",
    "lower_bound",
)


@dataclass(frozen=True)
class KdaFusedDecodeCell:
    """One frozen rank-local operator cell; active rows are runtime index data."""

    heads: int
    batch_size: int
    active_rows: int
    role: str

    @property
    def slots(self) -> int:
        return self.batch_size + 4

    @property
    def cell_id(self) -> str:
        return f"h{self.heads}-m{self.batch_size}-a{self.active_rows}"


FROZEN_CELLS = (
    KdaFusedDecodeCell(12, 1, 1, "small_guardrail"),
    KdaFusedDecodeCell(12, 4, 3, "small_guardrail"),
    KdaFusedDecodeCell(12, 8, 8, "primary"),
    KdaFusedDecodeCell(12, 16, 13, "primary"),
    KdaFusedDecodeCell(12, 32, 32, "primary"),
    KdaFusedDecodeCell(12, 64, 51, "primary"),
    KdaFusedDecodeCell(12, 128, 128, "primary"),
    KdaFusedDecodeCell(6, 32, 25, "deployment_guardrail"),
    KdaFusedDecodeCell(3, 64, 51, "deployment_guardrail"),
)
COMPARISON_CELLS_V5 = (
    *FROZEN_CELLS[:6],
    KdaFusedDecodeCell(12, 64, 64, "primary"),
    *FROZEN_CELLS[6:],
)


def frozen_cell(heads: int, batch_size: int) -> KdaFusedDecodeCell:
    """Return the one declared cell, refusing workload expansion by convenience."""

    matches = [
        cell
        for cell in FROZEN_CELLS
        if cell.heads == heads and cell.batch_size == batch_size
    ]
    if len(matches) != 1:
        raise ValueError(
            f"H={heads}, M={batch_size} is outside the frozen kda_fused_decode matrix"
        )
    return matches[0]


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    result: list[int] = []
    for extent in reversed(shape):
        result.insert(0, running)
        running *= extent
    return tuple(result)


@dataclass(frozen=True)
class TensorViewSpec:
    """One zero-copy tensor view constructed by the Lab ABI adapter.

    ``storage_offset`` and ``strides`` are in elements.  ``source_argument=None`` is
    reserved for the exact-sized output allocation; every input/state view names one of
    the tensor arguments in :data:`ABI_ARGUMENTS` and never authorizes a copy.
    """

    name: str
    source_argument: str | None
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int
    mode: str

    @property
    def schedule_strides(self) -> tuple[int, ...] | None:
        """Concrete strides only when the view is not canonical dense row-major."""

        return None if self.strides == _dense_strides(self.shape) else self.strides


@dataclass(frozen=True)
class KdaFusedDecodeAdapterContract:
    """Pure Lab contract from the public ABI to the generated Schedule ABI."""

    cell: KdaFusedDecodeCell
    abi_arguments: tuple[str, ...]
    views: tuple[TensorViewSpec, ...]
    scalar_constants: Mapping[str, float]
    returned_shape: tuple[int, ...]
    uses_contiguous_copy: bool = False

    def view(self, name: str) -> TensorViewSpec:
        matches = [view for view in self.views if view.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


def kda_fused_decode_adapter_contract(
    heads: int, batch_size: int
) -> KdaFusedDecodeAdapterContract:
    """Describe every logical view without materializing or copying a tensor."""

    cell = frozen_cell(heads, batch_size)
    h, m, d, slots = cell.heads, cell.batch_size, HEAD_DIM, cell.slots
    segment = h * d
    qkv_row = 3 * segment
    conv_slot = 3 * qkv_row

    views: list[TensorViewSpec] = []

    def view(
        name: str,
        source: str | None,
        dtype: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        *,
        offset: int = 0,
        mode: str = "input",
    ) -> None:
        views.append(TensorViewSpec(name, source, dtype, shape, strides, offset, mode))

    for channel_index, channel in enumerate(("q", "k", "v")):
        view(
            f"{channel}_current",
            "mixed_qkv",
            "bf16",
            (m, h, d),
            (qkv_row, d, 1),
            offset=channel_index * segment,
        )
    view("gate_a", "a", "bf16", (m, h, d), (segment, d, 1))
    view("beta", "b", "bf16", (m, h, d), (h, 1, 0))

    for channel_index, channel in enumerate(("q", "k", "v")):
        for history in range(CONV_HISTORY):
            view(
                f"{channel}_history_{history}",
                "conv_states",
                "bf16",
                (slots, h, d),
                (conv_slot, d, 1),
                offset=history * qkv_row + channel_index * segment,
                mode="state",
            )

    for channel in ("q", "k", "v"):
        for tap in range(CONV_WIDTH):
            view(
                f"{channel}_weight_{tap}",
                f"w_{channel}_t",
                "fp32",
                (h, d),
                (d, 1),
                offset=tap * segment,
            )
    for channel_index, channel in enumerate(("q", "k", "v")):
        view(
            f"{channel}_bias",
            "conv_bias",
            "fp32",
            (h, d),
            (d, 1),
            offset=channel_index * segment,
        )

    view("a_log", "A_log", "fp32", (h, d), (1, 0))
    view("dt_bias", "dt_bias", "fp32", (h, d), (d, 1))
    view("onorm_gate", "onorm_g", "bf16", (m, h, d), (segment, d, 1))
    view("onorm_weight", "onorm_weight", "fp32", (d,), (1,))
    view(
        "ssm_state",
        "ssm_states",
        "fp32",
        (slots, h, d, d),
        (h * d * d + STATE_SLOT_PAD, d * d, d, 1),
        mode="state",
    )
    view("cache_indices", "cache_indices", "int32", (m,), (1,))
    view("output", None, "bf16", (m, h, d), (segment, d, 1), mode="output")

    return KdaFusedDecodeAdapterContract(
        cell=cell,
        abi_arguments=ABI_ARGUMENTS,
        views=tuple(views),
        scalar_constants=MappingProxyType(
            {"scale": SCALE, "onorm_eps": ONORM_EPS, "lower_bound": LOWER_BOUND}
        ),
        returned_shape=(1, m, h, d),
    )


class _ScheduleBuilder:
    """Small SSA-oriented builder over the public Schedule dictionary schema."""

    def __init__(self, contract: KdaFusedDecodeAdapterContract) -> None:
        self.contract = contract
        self.buffers: list[dict[str, object]] = []
        self.operations: list[dict[str, object]] = []
        self.access_maps: list[dict[str, object]] = []
        self._buffer_names: set[str] = set()
        self._producer: dict[str, str] = {}
        self._add_global_buffers()

    def _add_global_buffers(self) -> None:
        for spec in self.contract.views:
            document: dict[str, object] = {
                "name": spec.name,
                "space": "global",
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "mode": spec.mode,
            }
            strides = spec.schedule_strides
            if strides is not None:
                document["strides"] = list(strides)
            if spec.name == "cache_indices":
                document["unique_index"] = {
                    "buffer": "ssm_state",
                    "dimension": 0,
                    "sentinel": -1,
                }
            self._append_buffer(document)

    def _append_buffer(self, document: dict[str, object]) -> None:
        name = str(document["name"])
        if name in self._buffer_names:
            raise ValueError(f"duplicate Schedule buffer {name!r}")
        self._buffer_names.add(name)
        self.buffers.append(document)

    def scratch(self, name: str, dtype: str, shape: tuple[int, ...]) -> str:
        self._append_buffer(
            {
                "name": name,
                "space": "register",
                "dtype": dtype,
                "shape": list(shape),
                "mode": "scratch",
            }
        )
        return name

    def _dependencies(self, reads: tuple[str, ...]) -> list[str]:
        return list(
            dict.fromkeys(
                self._producer[name] for name in reads if name in self._producer
            )
        )

    def _operation(
        self,
        op_id: str,
        kind: str,
        reads: tuple[str, ...],
        writes: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> None:
        document: dict[str, object] = {
            "id": op_id,
            "kind": kind,
            "role": "compute",
            "reads": list(reads),
            "writes": list(writes),
            "parameters": dict(parameters),
        }
        depends_on = self._dependencies(reads)
        if depends_on:
            document["depends_on"] = depends_on
        self.operations.append(document)
        for name in writes:
            if name in self._buffer_names and any(
                buffer["name"] == name and buffer["space"] == "register"
                for buffer in self.buffers
            ):
                self._producer[name] = op_id

    def load(
        self,
        source: str,
        result: str,
        dtype: str,
        shape: tuple[int, ...],
        indices: tuple[Mapping[str, object], ...],
    ) -> str:
        self.scratch(result, dtype, shape)
        coordinates = tuple(
            str(index["name"]) for index in indices if index["source"] == "buffer"
        )
        self._operation(
            f"load_{source}",
            "load",
            (source, *coordinates),
            (result,),
            {"movement": "global", "reuse": "streamed"},
        )
        self.access_maps.append(
            {
                "operation": f"load_{source}",
                "buffer": source,
                "indices": [dict(index) for index in indices],
                "boundary": "mask_tiled_axes",
            }
        )
        return result

    def elementwise(
        self,
        op_id: str,
        op: str,
        reads: tuple[str, ...],
        result: str,
        dtype: str,
        shape: tuple[int, ...],
        *,
        scalar: float | None = None,
        broadcast_axis: int | None = None,
        instruction: str | None = None,
    ) -> str:
        self.scratch(result, dtype, shape)
        parameters: dict[str, object] = {"op": op}
        if scalar is not None:
            parameters["scalar"] = scalar
        if broadcast_axis is not None:
            parameters["broadcast_axis"] = broadcast_axis
        if instruction is not None:
            parameters["instruction"] = {"contract": instruction}
        self._operation(op_id, "elementwise", reads, (result,), parameters)
        return result

    def cast(
        self,
        op_id: str,
        source: str,
        result: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> str:
        self.scratch(result, dtype, shape)
        self._operation(op_id, "cast", (source,), (result,), {"to": dtype})
        return result

    def reduce_sum(
        self,
        op_id: str,
        source: str,
        result: str,
        shape: tuple[int, ...],
        axis: int,
    ) -> str:
        self.scratch(result, "fp32", shape)
        self._operation(
            op_id,
            "reduce",
            (source,),
            (result,),
            {"op": "sum", "axis": axis, "scope": "cta"},
        )
        return result

    def outer(self, op_id: str, left: str, right: str, result: str) -> str:
        self.scratch(result, "fp32", (HEAD_DIM, HEAD_DIM))
        self._operation(op_id, "outer", (left, right), (result,), {})
        return result

    def store(
        self,
        op_id: str,
        source: str,
        destination: str,
        indices: tuple[Mapping[str, object], ...],
        *,
        inactive: str,
        coalesced: bool,
    ) -> None:
        self._operation(
            op_id,
            "store",
            (source, "slot"),
            (destination,),
            {
                "coalesced": coalesced,
                "valid_if": "slot",
                "inactive": inactive,
            },
        )
        self.access_maps.append(
            {
                "operation": op_id,
                "buffer": destination,
                "indices": [dict(index) for index in indices],
                "boundary": "mask_tiled_axes",
            }
        )


_ROW = {"source": "program", "name": "row"}
_HEAD = {"source": "program", "name": "head"}
_SLOT = {"source": "buffer", "name": "slot"}
_D0 = {"source": "dimension", "dimension": 0}
_D1 = {"source": "dimension", "dimension": 1}
_D2 = {"source": "dimension", "dimension": 2}
_D3 = {"source": "dimension", "dimension": 3}


def _sigmoid(builder: _ScheduleBuilder, prefix: str, source: str) -> str:
    half = builder.elementwise(
        f"{prefix}_half",
        "mul",
        (source,),
        f"{prefix}_half_value",
        "fp32",
        (HEAD_DIM,),
        scalar=0.5,
    )
    tanh = builder.elementwise(
        f"{prefix}_tanh",
        "tanh",
        (half,),
        f"{prefix}_tanh_value",
        "fp32",
        (HEAD_DIM,),
        instruction="libdevice.tanh.f32",
    )
    shifted = builder.elementwise(
        f"{prefix}_shift",
        "add",
        (tanh,),
        f"{prefix}_shifted",
        "fp32",
        (HEAD_DIM,),
        scalar=1.0,
    )
    return builder.elementwise(
        f"{prefix}_sigmoid",
        "mul",
        (shifted,),
        f"{prefix}_sigmoid_value",
        "fp32",
        (HEAD_DIM,),
        scalar=0.5,
    )


def _convolution_channel(builder: _ScheduleBuilder, channel: str) -> str:
    current = builder.load(
        f"{channel}_current",
        f"{channel}_current_tile",
        "bf16",
        (HEAD_DIM,),
        (_ROW, _HEAD, _D2),
    )
    histories = tuple(
        builder.load(
            f"{channel}_history_{history}",
            f"{channel}_history_{history}_tile",
            "bf16",
            (HEAD_DIM,),
            (_SLOT, _HEAD, _D2),
        )
        for history in range(CONV_HISTORY)
    )
    weights = tuple(
        builder.load(
            f"{channel}_weight_{tap}",
            f"{channel}_weight_{tap}_tile",
            "fp32",
            (HEAD_DIM,),
            (_HEAD, _D1),
        )
        for tap in range(CONV_WIDTH)
    )
    bias = builder.load(
        f"{channel}_bias", f"{channel}_bias_tile", "fp32", (HEAD_DIM,), (_HEAD, _D1)
    )

    products = tuple(
        builder.elementwise(
            f"{channel}_tap_{tap}",
            "mul",
            (source, weights[tap]),
            f"{channel}_tap_{tap}_product",
            "fp32",
            (HEAD_DIM,),
        )
        for tap, source in enumerate((*histories, current))
    )
    running = products[0]
    for tap, product in enumerate(products[1:], start=1):
        running = builder.elementwise(
            f"{channel}_sum_through_{tap}",
            "add",
            (running, product),
            f"{channel}_sum_through_{tap}_value",
            "fp32",
            (HEAD_DIM,),
        )
    preactivation = builder.elementwise(
        f"{channel}_add_bias",
        "add",
        (running, bias),
        f"{channel}_preactivation",
        "fp32",
        (HEAD_DIM,),
    )
    sigmoid = _sigmoid(builder, f"{channel}_silu", preactivation)
    silu = builder.elementwise(
        f"{channel}_silu_product",
        "mul",
        (preactivation, sigmoid),
        f"{channel}_silu_fp32",
        "fp32",
        (HEAD_DIM,),
    )
    seam = builder.cast(
        f"{channel}_bf16_seam", silu, f"{channel}_bf16", "bf16", (HEAD_DIM,)
    )

    state_indices = (_SLOT, _HEAD, _D2)
    builder.store(
        f"store_{channel}_history_0",
        histories[1],
        f"{channel}_history_0",
        state_indices,
        inactive="no_effect",
        coalesced=True,
    )
    builder.store(
        f"store_{channel}_history_1",
        histories[2],
        f"{channel}_history_1",
        state_indices,
        inactive="no_effect",
        coalesced=True,
    )
    builder.store(
        f"store_{channel}_history_2",
        current,
        f"{channel}_history_2",
        state_indices,
        inactive="no_effect",
        coalesced=True,
    )
    return seam


def _norm_sum(builder: _ScheduleBuilder, prefix: str, vector: str) -> str:
    square = builder.elementwise(
        f"{prefix}_square",
        "square",
        (vector,),
        f"{prefix}_square_value",
        "fp32",
        (HEAD_DIM,),
    )
    return builder.reduce_sum(
        f"{prefix}_sum", square, f"{prefix}_sum_scalar", (), axis=0
    )


def _l2_normalize(builder: _ScheduleBuilder, prefix: str, vector: str) -> str:
    norm_sum = _norm_sum(builder, prefix, vector)
    stabilized = builder.elementwise(
        f"{prefix}_epsilon",
        "add",
        (norm_sum,),
        f"{prefix}_sum_epsilon",
        "fp32",
        (),
        scalar=QK_L2_EPS,
    )
    inverse = builder.elementwise(
        f"{prefix}_rsqrt",
        "rsqrt",
        (stabilized,),
        f"{prefix}_inv_norm",
        "fp32",
        (),
    )
    return builder.elementwise(
        f"{prefix}_normalize",
        "mul",
        (vector, inverse),
        f"{prefix}_normalized",
        "fp32",
        (HEAD_DIM,),
    )


def kda_fused_decode_schedule_document(
    heads: int, batch_size: int
) -> dict[str, object]:
    """Build one complete Schedule document for an exact frozen operator cell."""

    contract = kda_fused_decode_adapter_contract(heads, batch_size)
    cell = contract.cell
    builder = _ScheduleBuilder(contract)

    builder.load("cache_indices", "slot", "int32", (), (_ROW,))

    q_bf16 = _convolution_channel(builder, "q")
    k_bf16 = _convolution_channel(builder, "k")
    v_bf16 = _convolution_channel(builder, "v")
    q_fp32 = builder.cast("q_recurrent_fp32", q_bf16, "q_fp32", "fp32", (HEAD_DIM,))
    k_fp32 = builder.cast("k_recurrent_fp32", k_bf16, "k_fp32", "fp32", (HEAD_DIM,))
    v_fp32 = builder.cast("v_recurrent_fp32", v_bf16, "v_fp32", "fp32", (HEAD_DIM,))

    q_norm = _l2_normalize(builder, "q_l2", q_fp32)
    k_norm = _l2_normalize(builder, "k_l2", k_fp32)

    gate_a_bf16 = builder.load(
        "gate_a", "gate_a_bf16_tile", "bf16", (HEAD_DIM,), (_ROW, _HEAD, _D2)
    )
    gate_a = builder.cast(
        "gate_a_fp32", gate_a_bf16, "gate_a_fp32_tile", "fp32", (HEAD_DIM,)
    )
    dt_bias = builder.load("dt_bias", "dt_bias_tile", "fp32", (HEAD_DIM,), (_HEAD, _D1))
    gate_shifted = builder.elementwise(
        "gate_add_dt_bias",
        "add",
        (gate_a, dt_bias),
        "gate_a_plus_dt_bias",
        "fp32",
        (HEAD_DIM,),
    )
    a_log = builder.load("a_log", "a_log_tile", "fp32", (HEAD_DIM,), (_HEAD, _D1))
    a_scale = builder.elementwise(
        "gate_exp_a_log", "exp", (a_log,), "gate_a_scale", "fp32", (HEAD_DIM,)
    )
    gate_logits = builder.elementwise(
        "gate_scale_logits",
        "mul",
        (a_scale, gate_shifted),
        "gate_logits",
        "fp32",
        (HEAD_DIM,),
    )
    bounded_gate_sigmoid = _sigmoid(builder, "safe_gate", gate_logits)
    bounded_gate = builder.elementwise(
        "safe_gate_lower_bound",
        "mul",
        (bounded_gate_sigmoid,),
        "bounded_gate",
        "fp32",
        (HEAD_DIM,),
        scalar=LOWER_BOUND,
    )
    decay = builder.elementwise(
        "state_decay_exp", "exp", (bounded_gate,), "state_decay", "fp32", (HEAD_DIM,)
    )

    state = builder.load(
        "ssm_state",
        "ssm_state_tile",
        "fp32",
        (HEAD_DIM, HEAD_DIM),
        (_SLOT, _HEAD, _D2, _D3),
    )
    decayed_state = builder.elementwise(
        "decay_state_k_channel",
        "mul",
        (state, decay),
        "decayed_state",
        "fp32",
        (HEAD_DIM, HEAD_DIM),
        broadcast_axis=1,
    )
    state_times_k = builder.elementwise(
        "prediction_weight_state",
        "mul",
        (decayed_state, k_norm),
        "state_times_k",
        "fp32",
        (HEAD_DIM, HEAD_DIM),
        broadcast_axis=1,
    )
    prediction = builder.reduce_sum(
        "prediction_sum", state_times_k, "prediction", (HEAD_DIM,), axis=1
    )

    beta_bf16 = builder.load(
        "beta", "beta_bf16_tile", "bf16", (HEAD_DIM,), (_ROW, _HEAD, _D2)
    )
    beta_fp32 = builder.cast(
        "beta_fp32", beta_bf16, "beta_fp32_tile", "fp32", (HEAD_DIM,)
    )
    beta = _sigmoid(builder, "beta", beta_fp32)
    residual = builder.elementwise(
        "delta_residual",
        "sub",
        (v_fp32, prediction),
        "delta_residual_value",
        "fp32",
        (HEAD_DIM,),
    )
    delta = builder.elementwise(
        "delta_apply_beta",
        "mul",
        (beta, residual),
        "delta",
        "fp32",
        (HEAD_DIM,),
    )
    rank_one = builder.outer("delta_outer_k", delta, k_norm, "delta_k_outer")
    next_state = builder.elementwise(
        "state_rank_one_update",
        "add",
        (decayed_state, rank_one),
        "next_ssm_state",
        "fp32",
        (HEAD_DIM, HEAD_DIM),
    )
    builder.store(
        "store_ssm_state",
        next_state,
        "ssm_state",
        (_SLOT, _HEAD, _D2, _D3),
        inactive="no_effect",
        coalesced=True,
    )

    scaled_q = builder.elementwise(
        "scale_recurrent_q",
        "mul",
        (q_norm,),
        "scaled_q",
        "fp32",
        (HEAD_DIM,),
        scalar=SCALE,
    )
    state_times_q = builder.elementwise(
        "recurrent_weight_state",
        "mul",
        (next_state, scaled_q),
        "state_times_q",
        "fp32",
        (HEAD_DIM, HEAD_DIM),
        broadcast_axis=1,
    )
    recurrent = builder.reduce_sum(
        "recurrent_output_sum", state_times_q, "recurrent_output", (HEAD_DIM,), axis=1
    )

    rms_sum = _norm_sum(builder, "output_rms", recurrent)
    rms_mean = builder.elementwise(
        "output_rms_mean",
        "mul",
        (rms_sum,),
        "output_mean_square",
        "fp32",
        (),
        scalar=1.0 / HEAD_DIM,
    )
    rms_shifted = builder.elementwise(
        "output_rms_epsilon",
        "add",
        (rms_mean,),
        "output_mean_square_eps",
        "fp32",
        (),
        scalar=ONORM_EPS,
    )
    inv_rms = builder.elementwise(
        "output_rms_rsqrt",
        "rsqrt",
        (rms_shifted,),
        "output_inv_rms",
        "fp32",
        (),
    )
    normalized = builder.elementwise(
        "output_rms_normalize",
        "mul",
        (recurrent, inv_rms),
        "normalized_output",
        "fp32",
        (HEAD_DIM,),
    )
    onorm_weight = builder.load(
        "onorm_weight", "onorm_weight_tile", "fp32", (HEAD_DIM,), (_D0,)
    )
    weighted = builder.elementwise(
        "output_rms_weight",
        "mul",
        (normalized, onorm_weight),
        "weighted_output",
        "fp32",
        (HEAD_DIM,),
    )
    onorm_gate_bf16 = builder.load(
        "onorm_gate",
        "onorm_gate_bf16_tile",
        "bf16",
        (HEAD_DIM,),
        (_ROW, _HEAD, _D2),
    )
    onorm_gate_fp32 = builder.cast(
        "onorm_gate_fp32",
        onorm_gate_bf16,
        "onorm_gate_fp32_tile",
        "fp32",
        (HEAD_DIM,),
    )
    onorm_gate = _sigmoid(builder, "onorm_gate", onorm_gate_fp32)
    gated = builder.elementwise(
        "output_apply_gate",
        "mul",
        (weighted, onorm_gate),
        "gated_output_fp32",
        "fp32",
        (HEAD_DIM,),
    )
    output = builder.cast("output_bf16_seam", gated, "output_bf16", "bf16", (HEAD_DIM,))
    builder.store(
        "store_output",
        output,
        "output",
        (_ROW, _HEAD, _D2),
        inactive="write_zero",
        coalesced=True,
    )

    return {
        "schema_version": 1,
        "schedule_id": f"kda-fused-decode-{cell.cell_id}-full-state-v1",
        "target": "sm_100a",
        "lowering": {
            "backend": "triton",
            "entry_point": f"cake_kda_fused_decode_h{cell.heads}_m{cell.batch_size}",
        },
        "roles": [{"name": "compute", "warps": list(range(8))}],
        "allocations": [],
        "buffers": builder.buffers,
        "pipelines": [],
        "barriers": [],
        "operations": builder.operations,
        "outputs": ["output"],
        "metadata": {},
        "program_map": {
            "axes": [
                {
                    "name": "row",
                    "axis": 0,
                    "buffer": "q_current",
                    "dimension": 0,
                    "tile": 1,
                },
                {
                    "name": "head",
                    "axis": 1,
                    "buffer": "q_current",
                    "dimension": 1,
                    "tile": 1,
                },
            ]
        },
        "tile_loops": [],
        "access_maps": builder.access_maps,
    }


def build_kda_fused_decode_schedule(heads: int, batch_size: int) -> Schedule:
    """Parse one Lab-owned complete-boundary seed through the canonical IR parser."""

    return Schedule.from_dict(kda_fused_decode_schedule_document(heads, batch_size))


def build_frozen_kda_fused_decode_schedules() -> tuple[Schedule, ...]:
    """Build the exact nine-cell matrix without treating it as broad generalization."""

    return tuple(
        build_kda_fused_decode_schedule(cell.heads, cell.batch_size)
        for cell in FROZEN_CELLS
    )


def build_comparison_kda_fused_decode_schedules_v5() -> tuple[Schedule, ...]:
    """Build the ten-cell comparison successor; retain the nine-cell Study."""

    return tuple(
        build_kda_fused_decode_schedule(cell.heads, cell.batch_size)
        for cell in COMPARISON_CELLS_V5
    )
