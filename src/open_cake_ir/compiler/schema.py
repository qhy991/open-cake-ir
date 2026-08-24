"""Project the typed IR into a JSON Schema.

`compiler/schedule.schema.json` is embedded in every Turn prompt, so an agent authoring
a Schedule reads it rather than the Python. It used to be maintained by hand beside
`ir.py`, which made it a second authority for the same facts -- and it had already
drifted: the checked-in schema restricted an Allocation to shared or tensor memory while
the parser accepted every space the Target admits.

The file is now a projection with a refresh path. Closed vocabularies come from the
enums and field sets from the dataclasses, so a vocabulary cannot diverge from the code
that enforces it. `tools/refresh_schedule_schema.py` rewrites it; a contract test fails
if the checked-in bytes fall behind.
"""

from __future__ import annotations

import json
from typing import Any

from .ir import (
    AccessIndexKind,
    ArgminTieBreak,
    BarrierMechanism,
    BoundaryPolicy,
    BufferMode,
    DType,
    EpilogueFormula,
    LoadMovement,
    LoadReuse,
    MemorySpace,
    OperandMajorMode,
    OperandSource,
    OperationKind,
    ReductionScope,
    Swizzle,
)

_NAME = {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}
_NAMES = {"type": "array", "items": _NAME}
_POSITIVE = {"type": "integer", "minimum": 1}
_NONNEGATIVE = {"type": "integer", "minimum": 0}


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
    OperationKind.LOAD: _object(
        {"movement": _enum(LoadMovement)},
        {
            "reuse": _enum(LoadReuse),
            "descriptor_box": {
                "type": "array",
                "minItems": 1,
                "items": _POSITIVE,
                "description": "TMA descriptor box extents; tma movement only.",
            }
        },
    ),
    OperationKind.MMA: _object(
        {"accumulator": {"const": DType.FP32.value}},
        {
            "tile_shape": _mnk("The tile this operation walks, as M, N and K."),
            "instruction": _object(
                {
                    "contract": {
                        "type": "string",
                        "description": "Must be admitted by the Target.",
                    },
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
        {"tie_break": _enum(ArgminTieBreak), "nan_policy": {"enum": ["reject_input"]}},
        {"across_loop": {"type": "boolean"}},
    ),
    OperationKind.REDUCE: _object(
        {"axis": _NONNEGATIVE, "scope": _enum(ReductionScope)}
    ),
    OperationKind.STORE: _object({"coalesced": {"type": "boolean"}}),
}


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
            "Generated from src/open_cake_ir/compiler/ir.py by "
            "tools/refresh_schedule_schema.py. Do not edit by hand."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "allocations",
            "barriers",
            "buffers",
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
                    "registers_per_thread": _POSITIVE,
                    "allow_spill": {"type": "boolean"},
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
                }
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
                            description="Tensor-memory column range; must agree with size_bytes.",
                        )
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
                    }
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
                    "allOf": [
                        {
                            "if": {"properties": {"kind": {"const": kind.value}}},
                            "then": {"properties": {"parameters": parameters}},
                        }
                        for kind, parameters in _PARAMETERS.items()
                    ],
                },
            },
            "outputs": _NAMES,
            "metadata": {
                "type": "object",
                "required": ["profile"],
                "properties": {"profile": {"type": "string", "minLength": 1}},
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
