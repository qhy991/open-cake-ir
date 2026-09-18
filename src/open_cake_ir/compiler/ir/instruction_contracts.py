"""The one owner of what an instruction or synchronization contract means.

A Target names the contracts it admits (`instruction_contracts`,
`synchronization_contracts`); this table is what those names mean. Typing stays with the
instruction and not with the Target (ADR 0022): sm_100a and sm_103a admit the same
tcgen05 contract and it reads the same operands on both. Before this module the same
facts lived in four shared tables keyed by the mnemonic string -- the verifier's dtype and
elementwise tables, the IR's placed-contract set and the barrier vocabulary -- so one new
contract was two to four shared edits and the two verifier tables disagreed on what an
unmodeled name meant. A contract is one record here; a backend keeps only its own
emission spelling.

A name a Target declares that has no record here is refused when the Revision loads. A
name the verifier meets that has no record is a refusal too, never a silent skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .vocabulary import BarrierMechanism, DType, ElementwiseOp


class ContractKind(str, Enum):
    MMA = "mma"
    ELEMENTWISE = "elementwise"
    ATOMIC = "atomic"
    SYNCHRONIZATION = "synchronization"


@dataclass(frozen=True)
class InstructionContract:
    name: str
    kind: ContractKind
    # MMA: the operand dtypes the instruction reads and the dtype it accumulates in.
    operand_dtypes: frozenset[DType] = frozenset()
    accumulator: DType | None = None
    # MMA: whether the atom places its operands (shape, cta_group, operand source and
    # major-ness are then legal, and required, on the Schedule's placement).
    places_operands: bool = False
    # Elementwise: the operation and dtype the contract realizes.
    elementwise_op: ElementwiseOp | None = None
    elementwise_dtype: DType | None = None
    # Synchronization: the barrier mechanism the contract realizes, or None when the
    # contract is an ordering guarantee with no barrier object (Triton program order).
    realizes: BarrierMechanism | None = None


def _mma(name: str, operands: set[DType], *, places: bool = False) -> InstructionContract:
    return InstructionContract(name, ContractKind.MMA, frozenset(operands), DType.FP32,
                               places_operands=places)


def _elementwise(name: str, op: ElementwiseOp, dtype: DType) -> InstructionContract:
    return InstructionContract(name, ContractKind.ELEMENTWISE,
                               elementwise_op=op, elementwise_dtype=dtype)


def _sync(name: str, realizes: BarrierMechanism | None) -> InstructionContract:
    return InstructionContract(name, ContractKind.SYNCHRONIZATION, realizes=realizes)


_RECORDS = (
    # NVIDIA tensor-core atoms that place their operands.
    _mma("tcgen05.mma.cta_group::1.kind::f16", {DType.BF16, DType.FP16}, places=True),
    _mma("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", {DType.BF16}, places=True),
    # Triton `tl.dot` routes; the emitter chooses input precision by name.
    _mma("triton.dot.bf16_fp32", {DType.BF16}),
    _mma("triton.dot.fp32_ieee", {DType.FP32}),
    _mma("triton.dot.fp32_tf32", {DType.FP32}),
    _mma("triton.dot.fp8e4m3_block_scale_fp32", {DType.FP8_E4M3}),
    # gfx938's two, measured on a BW1101 at 64x64x64 against a torch oracle at matched
    # input precision: fp16 max_abs_error 1.14441e-05, fp8e4m3 exactly 0. Both accumulate
    # in fp32; neither carries a scale operand.
    _mma("triton.dot.fp16_fp32", {DType.FP16}),
    _mma("triton.dot.fp8e4m3_fp32", {DType.FP8_E4M3}),
    InstructionContract("triton.atomic_add.i32.relaxed.gpu", ContractKind.ATOMIC),
    _elementwise("libdevice.tanh.f32", ElementwiseOp.TANH, DType.FP32),
    # Metal's own named-precision spelling; the fast:: namespace is a different function.
    _elementwise("metal.precise.tanh.f32", ElementwiseOp.TANH, DType.FP32),
    _elementwise("metal.fma.f32", ElementwiseOp.FMA, DType.FP32),
    _elementwise("ptx.fma.rn.f32", ElementwiseOp.FMA, DType.FP32),
    _sync("mbarrier", BarrierMechanism.MBARRIER),
    _sync("barrier.sync", BarrierMechanism.NAMED),
    _sync("triton_program_order", None),
)

CONTRACTS: Mapping[str, InstructionContract] = MappingProxyType(
    {record.name: record for record in _RECORDS}
)


def contract(name: str) -> InstructionContract | None:
    """The record for one declared name, or None when no record owns it."""
    return CONTRACTS.get(name)


def contracts_of(kind: ContractKind) -> frozenset[str]:
    return frozenset(name for name, record in CONTRACTS.items() if record.kind is kind)


PLACED_CONTRACTS: frozenset[str] = frozenset(
    name for name, record in CONTRACTS.items() if record.places_operands
)
