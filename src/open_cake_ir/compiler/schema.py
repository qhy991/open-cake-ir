"""Project the typed IR into a JSON Schema.

`schedule.schema.json` is generated into every Turn's reference bundle, so an agent
authoring a Schedule reads it rather than the Python. It is a projection, not a second
checked-in authority: `lab.compose` calls `schedule_schema_bytes()` when it builds the
bundle. Closed vocabularies come from the IR enums, and a contract test holds the
projection against every Compiler Corpus document.
"""

from __future__ import annotations

import json
from typing import Any

from .ir import (
    PLACED_CONTRACT_PREFIXES,
    PLACEMENT_FIELDS,
    AccessIndexKind,
    AtomicMemoryOrder,
    AtomicMemoryScope,
    AtomicOp,
    BarrierMechanism,
    BoundaryPolicy,
    BufferMode,
    DType,
    ElementwiseOp,
    EpilogueFormula,
    IndexTieBreak,
    LoadMovement,
    LoadReuse,
    LoweringBackend,
    MemorySpace,
    NaNPolicy,
    OperandMajorMode,
    OperandSource,
    OperationKind,
    ReduceOp,
    ScanDirection,
    ScanOp,
    ReductionScope,
    Swizzle,
)

_NAME = {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}
_NAMES = {"type": "array", "items": _NAME}
_POSITIVE = {"type": "integer", "minimum": 1}
_NONNEGATIVE = {"type": "integer", "minimum": 0}
_SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}


def _values(enum_type: type) -> list[str]:
    return sorted(member.value for member in enum_type)


def _enum(enum_type: type) -> dict[str, Any]:
    return {"enum": _values(enum_type)}


def _mnk(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": _POSITIVE,
        "description": description,
    }


def _permitted_only_when(
    obj: dict[str, Any],
    sibling: str,
    permitting: dict[str, Any],
    fields: tuple[str, ...],
    description: str,
) -> dict[str, Any]:
    """Forbid `fields` unless `sibling` satisfies `permitting`.

    Some fields are legal only for particular values of a neighbour. The Verifier already
    refuses the other combinations, but only after a Turn has been spent writing one. The
    authoring Schema is the agent's one machine-readable guide, so it carries the same
    condition instead of leaving the rule to be discovered from a blocking Finding.
    """

    return {
        **obj,
        "allOf": [
            {
                "if": {
                    "required": [sibling],
                    "properties": {sibling: {"not": permitting}},
                },
                "then": {
                    "not": {"anyOf": [{"required": [name]} for name in fields]},
                    "description": description,
                },
            }
        ],
    }


def _placed_only(instruction: dict[str, Any]) -> dict[str, Any]:
    escaped = "|".join(prefix.replace(".", r"\.") for prefix in PLACED_CONTRACT_PREFIXES)
    return _permitted_only_when(
        instruction,
        "contract",
        {"pattern": "^(?:%s)" % escaped},
        PLACEMENT_FIELDS,
        "Only %s place their operands; every other contract leaves %s to the backend."
        % (", ".join(PLACED_CONTRACT_PREFIXES), ", ".join(PLACEMENT_FIELDS)),
    )


def _object(
    required: dict[str, Any], optional: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(required),
        "properties": {**required, **(optional or {})},
    }


_PARAMETERS = {
    OperationKind.LOAD: _permitted_only_when(
        _object(
            {"movement": _enum(LoadMovement)},
            {
                "reuse": _enum(LoadReuse),
                "source_atom": _object({"op": {"const": "tcgen05.Ld32x32b"}, "repetition": {"enum": [1,2,4,8,16,32,64,128]}}),
                "descriptor_box": {
                    "type": "array",
                    "minItems": 1,
                    "items": _POSITIVE,
                    "description": "TMA descriptor box extents; tma movement only.",
                }
            },
        ),
        "movement",
        {"const": LoadMovement.TMA.value},
        ("descriptor_box",),
        "Only tma movement addresses a descriptor box; a global load has none.",
    ),
    OperationKind.MMA: _object(
        {"accumulator": {"const": DType.FP32.value}},
        {
            "tile_shape": _mnk("The complete input tile domain, as M, N and K; k_ranges selects contributions within K."),
            "k_ranges": {
                "type": "array", "minItems": 1,
                "items": {"type": "array", "minItems": 2, "maxItems": 2,
                          "items": {"type": "integer", "minimum": 0}},
                "description": "Ordered disjoint half-open input K contributions; requires tile_shape. "
                    "Each start < end <= tile_shape.K. Adjacent intervals merge; full coverage is omitted.",
            },
            "instruction": _placed_only(
                _object(
                    {"contract": {
                        "type": "string",
                        "description": "Must be admitted by the Target.",
                    }},
                    {
                        "shape": _mnk("The atom's M, N and K, which is not the tile's."),
                        "cta_group": {"enum": [1, 2]},
                        "operand_source": _enum(OperandSource),
                        "operand_major": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": _enum(OperandMajorMode),
                            "description": "Major mode for operand A and operand B.",
                        },
                    }
                )
            ),
        },
    ),
    OperationKind.EPILOGUE: _object(
        {"formula": _enum(EpilogueFormula), "coalesced": {"type": "boolean"}},
        {
            "subtile": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": _POSITIVE,
            },
            "source_atom": _object(
                {"op": {"type": "string"}, "repetition": _POSITIVE}
            ),
        },
    ),
    OperationKind.REDUCE_ARGMIN: _object(
        {"tie_break": _enum(IndexTieBreak), "nan_policy": _enum(NaNPolicy)},
        {"across_loop": {"type": "boolean"}},
    ),
    OperationKind.REDUCE: _object(
        {
            "op": _enum(ReduceOp),
            "axis": _NONNEGATIVE,
            "scope": _enum(ReductionScope),
        },
        {"across_loop": {"const": False}},
    ),
    OperationKind.SCAN: _object(
        {"op": _enum(ScanOp), "axis": _NONNEGATIVE},
        {"direction": _enum(ScanDirection)},
    ),
    OperationKind.TOP_K: _object(
        {
            "k": _POSITIVE,
            "tie_break": _enum(IndexTieBreak),
            "nan_policy": _enum(NaNPolicy),
        },
        {
            "across_loop": {"type": "boolean"},
            "source_tiles_per_merge": {
                "const": 2,
                "description": (
                    "Batch exactly two FP32 source tiles per loop-carried merge; "
                    "omit for the canonical one-tile cadence."
                ),
            },
        },
    ),
    OperationKind.ONLINE_SOFTMAX: _object(
        {
            "axis": _NONNEGATIVE,
            "scope": _enum(ReductionScope),
        },
        {"sentinel": {"type": "integer"}},
    ),
    OperationKind.INDEX_EXPAND: _object(
        {
            "scale": _POSITIVE,
            "extent": _POSITIVE,
            "sentinel": {"type": "integer"},
        }
    ),
    OperationKind.CAST: _object({"to": _enum(DType)}),
    OperationKind.ATOMIC_RMW: _object(
        {
            "op": _enum(AtomicOp),
            "value": {"type": "integer"},
            "order": _enum(AtomicMemoryOrder),
            "scope": _enum(AtomicMemoryScope),
        }
    ),
    OperationKind.ELEMENTWISE: {
        "oneOf": [
            _object(
                {
                    "op": {"const": ElementwiseOp.FMA.value},
                    "instruction": _object({"contract": {"type": "string"}}),
                }
            ),
            _object(
                {
                    "op": {"const": ElementwiseOp.TANH.value},
                    "instruction": _object({"contract": {"type": "string"}}),
                },
                {
                    "scalar": {"type": "number"},
                    "broadcast_axis": _NONNEGATIVE,
                },
            ),
            _object(
                {
                    "op": {
                        "enum": [
                            member.value
                            for member in ElementwiseOp
                            if member not in (ElementwiseOp.TANH, ElementwiseOp.FMA)
                        ]
                    }
                },
                {
                    "scalar": {"type": "number"},
                    "broadcast_axis": _NONNEGATIVE,
                },
            ),
        ]
    },
    OperationKind.STORE: _object({"coalesced": {"type": "boolean"}}),
}


_PARAMETERS[OperationKind.MMA]["dependentRequired"] = {"k_ranges": ["tile_shape"]}

_PARAMETERS[OperationKind.LOAD].setdefault("allOf", []).extend([
    {"if": {"properties": {"movement": {"const": "tmem"}}},
     "then": {"required": ["source_atom"], "not": {"required": ["reuse"]}},
     "else": {"not": {"required": ["source_atom"]}}}
])


def schedule_schema() -> dict[str, Any]:
    """The Schedule JSON Schema, derived from the typed IR."""

    # `additionalProperties` has to live in each branch: at the top level it sees no
    # declared properties, because they are declared inside the alternatives.
    access_index = {
        "type": "object",
        "oneOf": [
            {
                "additionalProperties": False,
                "properties": {
                    "source": {"const": AccessIndexKind.DIMENSION.value},
                    "dimension": _NONNEGATIVE,
                    # Omitting both covers the whole axis, which is why they are optional
                    # rather than defaulted here: one spelling of "all of it", not two.
                    # `offset` starts at 1 for the same reason -- writing 0 would be a
                    # second spelling of omitting it. The Verifier owns the matching
                    # rule for `extent`, which needs the axis size to state.
                    "offset": _POSITIVE,
                    "extent": _POSITIVE,
                },
                "required": ["source", "dimension"],
            },
            {
                "additionalProperties": False,
                "properties": {
                    "source": {
                        "enum": [
                            kind.value
                            for kind in AccessIndexKind
                            if kind is not AccessIndexKind.DIMENSION
                        ]
                    },
                    "name": _NAME,
                },
                "required": ["source", "name"],
            },
        ],
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://open-cake-ir.local/schema/schedule-v1",
        "title": "Open Cake Schedule v1",
        "description": (
            "Generated from open_cake_ir.compiler.ir by "
            "open_cake_ir.compiler.schema.schedule_schema_bytes()."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "allocations",
            "barriers",
            "buffers",
            "lowering",
            "metadata",
            "operations",
            "outputs",
            "pipelines",
            "roles",
            "schedule_id",
            "schema_version",
            "target",
        ],
        "oneOf": [
            {"required": ["grid"], "not": {"required": ["program_map"]}},
            {"required": ["program_map"], "not": {"required": ["grid"]}},
        ],
        "properties": {
            "schema_version": {"const": 1},
            "schedule_id": {"type": "string", "minLength": 1},
            "target": {"type": "string", "minLength": 1},
            "lowering": _object(
                {
                    "backend": _enum(LoweringBackend),
                    "entry_point": _NAME,
                }
            ),
            "grid": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": _POSITIVE,
            },
            "residency": _object(
                {},
                {
                    "ctas_per_multiprocessor": _POSITIVE,
                    "registers_per_thread": dict(_POSITIVE, description="Compile-time cap; bounded by the selected Target's maximum_registers_per_thread, not setmaxnreg immediates."),
                },
            ),
            "program_map": _object(
                {
                    "axes": {
                        "type": "array",
                        "minItems": 1,
                        "items": _object(
                            {
                                "name": _NAME,
                                "axis": _NONNEGATIVE,
                                "buffer": _NAME,
                                "dimension": _NONNEGATIVE,
                                "tile": _POSITIVE,
                            }
                        ),
                    }
                },
                {
                    "persistent": {"type": "boolean"},
                    "traversal": _NAMES,
                },
            ),
            "roles": {
                "type": "array",
                "minItems": 1,
                "items": _object(
                    {
                        "name": _NAME,
                        "warps": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": _NONNEGATIVE,
                        },
                    },
                    {
                        "registers_per_thread": {
                            "type": "integer",
                            "minimum": 24,
                            "maximum": 256,
                            "multipleOf": 8,
                        }
                    },
                ),
            },
            "allocations": {
                "type": "array",
                "items": _object(
                    {
                        "name": _NAME,
                        "space": _enum(MemorySpace),
                        "size_bytes": _POSITIVE,
                    },
                    {
                        "tensor_columns": dict(
                            _POSITIVE,
                            description="Tensor-memory column range: power-of-two in [32,512], consistent with size_bytes and Target capacity.",
                        ),
                        "allocating_role": _NAME,
                    },
                ),
            },
            "buffers": {
                "type": "array",
                "minItems": 1,
                "items": _object(
                    {
                        "name": _NAME,
                        "space": _enum(MemorySpace),
                        "dtype": _enum(DType),
                        "shape": {
                            "type": "array",
                            "minItems": 1,
                            "items": _POSITIVE,
                        },
                        "mode": _enum(BufferMode),
                    },
                    {
                        "allocation": _NAME,
                        "byte_offset": _NONNEGATIVE,
                        "stages": _POSITIVE,
                        "swizzle": _enum(Swizzle),
                        "scale_of": _object(
                            {
                                "buffer": _NAME,
                                "granularity": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": _POSITIVE,
                                },
                                "axis_order": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": _NONNEGATIVE,
                                },
                            }
                        ),
                        "valid_extent": _object(
                            {
                                "dimension": _NONNEGATIVE,
                                "buffer": _NAME,
                                "indexed_by": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": _NONNEGATIVE,
                                },
                            }
                        ),
                    },
                ),
            },
            "pipelines": {
                "type": "array",
                "items": _object({"name": _NAME, "stages": _POSITIVE}),
            },
            "barriers": {
                "type": "array",
                "items": _object(
                    {
                        "name": _NAME,
                        "count": _POSITIVE,
                        "producers": _NAMES,
                        "consumers": _NAMES,
                    },
                    {"pipeline": _NAME, "mechanism": _enum(BarrierMechanism)},
                ),
            },
            "tile_loops": {
                "type": "array",
                "items": _object(
                    {
                        "name": _NAME,
                        "iterator": _NAME,
                        "buffer": _NAME,
                        "dimension": _NONNEGATIVE,
                        "tile": _POSITIVE,
                        "body": dict(
                            _NAMES,
                            minItems=1,
                            description=(
                                "Operations and nested loops inside this loop, in order."
                            ),
                        ),
                        "range_options": _object(
                            {
                                "num_stages": _POSITIVE,
                                "loop_unroll_factor": _POSITIVE,
                                "flatten": {"type": "boolean"},
                                "warp_specialize": {"type": "boolean"},
                                "disallow_acc_multi_buffer": {"type": "boolean"},
                                "disable_licm": {"type": "boolean"},
                            }
                        ),
                    },
                    {
                        "stop": _object(
                            {
                                "program": _NAME,
                                "add": {"type": "integer"},
                                "floor_div": _POSITIVE,
                            }
                        )
                    },
                ),
            },
            "access_maps": {
                "type": "array",
                "items": _object(
                    {
                        "operation": _NAME,
                        "buffer": _NAME,
                        "indices": {
                            "type": "array",
                            "minItems": 1,
                            "items": access_index,
                        },
                        "boundary": _enum(BoundaryPolicy),
                    }
                ),
            },
            "operations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "kind", "role", "reads", "writes", "parameters"],
                    "properties": {
                        "id": _NAME,
                        "kind": _enum(OperationKind),
                        "role": _NAME,
                        "reads": _NAMES,
                        "writes": _NAMES,
                        "waits": _NAMES,
                        "signals": _NAMES,
                        "depends_on": _NAMES,
                        "pipeline": _NAME,
                        "parameters": {"type": "object"},
                    },
                    # Index the complete IR enum: an unprojected new kind must fail
                    # while this schema is built, not silently accept unconstrained
                    # parameters for authoring.
                    "allOf": [
                        {
                            "if": {"properties": {"kind": {"const": kind.value}}},
                            "then": {
                                "properties": {"parameters": _PARAMETERS[kind]}
                            },
                        }
                        for kind in OperationKind
                    ],
                },
            },
            "outputs": _NAMES,
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "workload_contract_sha256": _SHA256,
                    "legacy_source": _object(
                        {
                            "revision": {"type": "string", "minLength": 1},
                            "path": {"type": "string", "minLength": 1},
                            "canonical_json_sha256": _SHA256,
                        }
                    ),
                },
            },
        },
    }


def _without_prose(node: Any) -> Any:
    """Drop descriptions. These bytes go into every Turn prompt, and the prose the
    agent needs is in compiler/AUTHORING_CONTRACT.md, which is bundled beside them."""

    if isinstance(node, dict):
        return {k: _without_prose(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_prose(item) for item in node]
    return node


def schedule_schema_bytes() -> bytes:
    schema = _without_prose(schedule_schema())
    return (json.dumps(schema, separators=(",", ":"), sort_keys=True) + "\n").encode()
