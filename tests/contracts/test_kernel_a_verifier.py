"""Verifier contracts for the first explicit Q8_1 producer primitives."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/gfx1151.json")
RAW_Q8 = ROOT / "corpus/schedules/packed-q8_1-record-copy-gfx1151.json"


def _buffer(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    space: str = "register",
    mode: str = "scratch",
    packed: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "name": name,
        "space": space,
        "dtype": dtype,
        "shape": shape,
        "mode": mode,
    }
    if packed is not None:
        value["packed_block"] = {
            "format": packed,
            "record_axis": len(shape) - 1,
        }
    return value


def _document(
    buffers: list[dict[str, object]],
    *,
    kind: str,
    reads: list[str],
    writes: list[str],
    parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "schedule_id": "kernel-a-verifier-test",
        "target": "gfx1151",
        "lowering": {"backend": "triton", "entry_point": "kernel_a_test"},
        "grid": [1, 1, 1],
        "roles": [{"name": "compute", "warps": [0]}],
        "allocations": [],
        "buffers": buffers,
        "pipelines": [],
        "barriers": [],
        "operations": [
            {
                "id": "subject",
                "kind": kind,
                "role": "compute",
                "reads": reads,
                "writes": writes,
                "parameters": parameters,
            }
        ],
        "outputs": [
            value["name"] for value in buffers if value["mode"] == "output"
        ],
        "metadata": {},
        "tile_loops": [],
        "access_maps": [],
    }


def _codes(document: dict[str, object], prefix: str) -> set[str]:
    return {
        finding.code
        for finding in verify(Schedule.from_dict(document), TARGET)
        if finding.code.startswith(prefix)
    }


def _reshape_document() -> dict[str, object]:
    return _document(
        [
            _buffer("flat", "fp32", [512]),
            _buffer("blocks", "fp32", [16, 32]),
        ],
        kind="reshape",
        reads=["flat"],
        writes=["blocks"],
        parameters={},
    )


def _cast_document(
    destination_dtype: str,
    rounding: str,
    overflow: str,
) -> dict[str, object]:
    return _document(
        [
            _buffer("source", "fp32", [16, 32]),
            _buffer("result", destination_dtype, [16, 32]),
        ],
        kind="cast",
        reads=["source"],
        writes=["result"],
        parameters={"to": destination_dtype, "rounding": rounding, "overflow": overflow},
    )


def _xor_document() -> dict[str, object]:
    return _document(
        [
            _buffer("source", "fp32", [16, 32]),
            _buffer("result", "fp32", [16]),
        ],
        kind="reduce",
        reads=["source"],
        writes=["result"],
        parameters={
            "op": "sum",
            "axis": 1,
            "scope": "cta",
            "algorithm": "xor_tree_32",
        },
    )


def _q8_store_document() -> dict[str, object]:
    document = _document(
        [
            _buffer("d", "fp16", [16]),
            _buffer("s", "fp16", [16]),
            _buffer("qs", "int8", [16, 32]),
            _buffer(
                "workspace",
                "uint8",
                [16, 36],
                space="global",
                mode="output",
                packed="ggml_q8_1_v1",
            ),
        ],
        kind="store",
        reads=["d", "s", "qs"],
        writes=["workspace"],
        parameters={"coalesced": True},
    )
    document["access_maps"] = [
        {
            "operation": "subject",
            "buffer": "workspace",
            "indices": [{"source": "dimension", "dimension": 0}],
            "boundary": "mask_tiled_axes",
        }
    ]
    return document


class ReshapeVerifierTests(unittest.TestCase):
    def test_register_view_preserves_dtype_and_element_count(self) -> None:
        self.assertEqual(_codes(_reshape_document(), "RESHAPE_"), set())

    def test_each_reshape_contract_has_one_stable_finding(self) -> None:
        cases = (
            (
                "edge_count",
                lambda document: document["operations"][0]["reads"].append("flat"),
                "RESHAPE_EDGE_COUNT",
            ),
            (
                "register_only",
                lambda document: document["buffers"][0].update(space="global"),
                "RESHAPE_REGISTER_ONLY",
            ),
            (
                "dtype",
                lambda document: document["buffers"][1].update(dtype="fp16"),
                "RESHAPE_DTYPE_MISMATCH",
            ),
            (
                "elements",
                lambda document: document["buffers"][1].update(shape=[16, 31]),
                "RESHAPE_ELEMENT_COUNT",
            ),
            (
                "packed",
                lambda document: document["buffers"][0].update(
                    dtype="uint8",
                    shape=[512, 18],
                    packed_block={"format": "ggml_q4_0_v1", "record_axis": 1},
                )
                or document["buffers"][1].update(dtype="uint8", shape=[16, 576]),
                "RESHAPE_PACKED_BLOCK_UNSUPPORTED",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(case=label):
                document = _reshape_document()
                mutate(document)
                self.assertEqual(_codes(document, "RESHAPE_"), {expected})


class ElementwiseAndCastVerifierTests(unittest.TestCase):
    def test_abs_divide_no_nan_and_round_keep_float_promotion_and_shape(self) -> None:
        cases = (
            _document(
                [_buffer("x", "fp16", [16, 32]), _buffer("y", "fp16", [16, 32])],
                kind="elementwise",
                reads=["x"],
                writes=["y"],
                parameters={"op": "abs"},
            ),
            _document(
                [
                    _buffer("x", "fp32", [16, 32]),
                    _buffer("d", "fp16", [16]),
                    _buffer("y", "fp32", [16, 32]),
                ],
                kind="elementwise",
                reads=["x", "d"],
                writes=["y"],
                parameters={"op": "divide_no_nan", "broadcast_axis": 0},
            ),
            _document(
                [_buffer("x", "fp32", [16, 32]), _buffer("y", "fp32", [16, 32])],
                kind="elementwise",
                reads=["x"],
                writes=["y"],
                parameters={"op": "round", "rounding": "nearest_away_from_zero"},
            ),
        )
        for document in cases:
            with self.subTest(op=document["operations"][0]["parameters"]["op"]):
                self.assertEqual(_codes(document, "ELEMENTWISE_"), set())

    def test_cast_admits_only_the_two_explicit_policies(self) -> None:
        admitted = (
            _cast_document("int8", "toward_zero", "forbid"),
            _cast_document("fp16", "nearest_even", "ieee"),
        )
        for document in admitted:
            with self.subTest(dtype=document["buffers"][1]["dtype"]):
                self.assertEqual(_codes(document, "CAST_"), set())

        cases = (
            (
                "dtype",
                lambda document: document["buffers"][0].update(dtype="bf16"),
                "CAST_DTYPE_UNSUPPORTED",
            ),
            (
                "rounding",
                lambda document: document["operations"][0]["parameters"].update(
                    rounding="nearest_away_from_zero"
                ),
                "CAST_ROUNDING_UNSUPPORTED",
            ),
            (
                "overflow",
                lambda document: document["operations"][0]["parameters"].update(
                    overflow="forbid"
                ),
                "CAST_OVERFLOW_UNSUPPORTED",
            ),
            (
                "shape",
                lambda document: document["buffers"][1].update(shape=[16, 31]),
                "CAST_SHAPE_MISMATCH",
            ),
            (
                "result_space",
                lambda document: document["buffers"][1].update(
                    space="global", mode="output"
                ),
                "CAST_SPACE",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(case=label):
                document = _cast_document("fp16", "nearest_even", "ieee")
                mutate(document)
                self.assertEqual(_codes(document, "CAST_"), {expected})


class XorReductionVerifierTests(unittest.TestCase):
    def test_exact_xor_tree_and_historical_backend_are_both_admitted(self) -> None:
        self.assertEqual(_codes(_xor_document(), "REDUCE_"), set())

        backend = _xor_document()
        backend["operations"][0]["parameters"].pop("algorithm")
        backend["buffers"][0]["dtype"] = "fp16"
        self.assertEqual(_codes(backend, "REDUCE_"), set())

    def test_xor_tree_rejects_one_contract_drift_at_a_time(self) -> None:
        cases = (
            (
                "dtype",
                lambda document: document["buffers"][0].update(dtype="fp16"),
                "REDUCE_XOR_DTYPE",
            ),
            (
                "axis",
                lambda document: document["operations"][0]["parameters"].update(axis=0)
                or document["buffers"][0].update(shape=[32, 16])
                or document["buffers"][1].update(shape=[16]),
                "REDUCE_XOR_AXIS",
            ),
            (
                "extent",
                lambda document: document["buffers"][0].update(shape=[16, 31]),
                "REDUCE_XOR_EXTENT",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(case=label):
                document = _xor_document()
                mutate(document)
                self.assertEqual(_codes(document, "REDUCE_XOR_"), {expected})

    def test_xor_tree_refuses_non_cta_and_loop_carried_forms(self) -> None:
        schedule = Schedule.from_dict(_xor_document())
        operation = schedule.operations[0]
        non_cta = replace(
            schedule,
            operations=(
                replace(operation, parameters=replace(operation.parameters, scope="wave")),
            ),
        )
        self.assertEqual(
            {
                finding.code
                for finding in verify(non_cta, TARGET)
                if finding.code.startswith("REDUCE_XOR_")
            },
            {"REDUCE_XOR_SCOPE"},
        )

        looped = _xor_document()
        looped["tile_loops"] = [
            {
                "name": "block_loop",
                "iterator": "block",
                "buffer": "source",
                "dimension": 0,
                "tile": 8,
                "body": ["subject"],
                "range_options": {
                    "num_stages": 1,
                    "loop_unroll_factor": 1,
                    "flatten": False,
                    "warp_specialize": False,
                    "disallow_acc_multi_buffer": True,
                    "disable_licm": False,
                },
            }
        ]
        self.assertEqual(
            _codes(looped, "REDUCE_XOR_"), {"REDUCE_XOR_ACROSS_LOOP"}
        )


class PackedStoreVerifierTests(unittest.TestCase):
    def test_raw_uint8_copy_and_typed_q8_registry_order_are_admitted(self) -> None:
        raw_codes = {
            finding.code
            for finding in verify(Schedule.load(RAW_Q8), TARGET)
            if finding.code.startswith("PACKED_STORE_")
        }
        self.assertEqual(raw_codes, set())
        self.assertEqual(_codes(_q8_store_document(), "PACKED_STORE_"), set())

        prefix_eight = _q8_store_document()
        prefix_eight["buffers"][0]["shape"] = [8]
        prefix_eight["buffers"][1]["shape"] = [8]
        prefix_eight["buffers"][2]["shape"] = [8, 32]
        prefix_eight["buffers"][3]["shape"] = [8, 36]
        self.assertEqual(_codes(prefix_eight, "PACKED_STORE_"), set())

    def test_typed_q8_fields_follow_registry_dtype_and_shape(self) -> None:
        cases = (
            (
                "raw_dtype",
                lambda document: document["operations"][0].update(reads=["qs"]),
                "PACKED_STORE_RAW_DTYPE",
            ),
            (
                "arity",
                lambda document: document["operations"][0].update(reads=["d", "s"]),
                "PACKED_STORE_ARITY",
            ),
            (
                "field_dtype",
                lambda document: document["buffers"][1].update(dtype="int8"),
                "PACKED_STORE_FIELD_DTYPE",
            ),
            (
                "field_space",
                lambda document: document["buffers"][0].update(
                    space="global", mode="input"
                ),
                "PACKED_STORE_FIELD_SPACE",
            ),
            (
                "field_shape",
                lambda document: document["buffers"][2].update(shape=[16, 31]),
                "PACKED_STORE_FIELD_SHAPE",
            ),
            (
                "record_count",
                lambda document: document["buffers"][3].update(shape=[8, 36]),
                "PACKED_BLOCK_STORE_RECORD_COUNT",
            ),
            (
                "access_map",
                lambda document: document.update(access_maps=[]),
                "PACKED_STORE_ACCESS_MAP",
            ),
            (
                "layout",
                lambda document: document["buffers"][3].update(
                    shape=[1, 16, 36],
                    packed_block={"format": "ggml_q8_1_v1", "record_axis": 2},
                ),
                "PACKED_STORE_LAYOUT_UNSUPPORTED",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(case=label):
                document = _q8_store_document()
                mutate(document)
                prefix = (
                    "PACKED_BLOCK_"
                    if expected.startswith("PACKED_BLOCK_")
                    else "PACKED_STORE_"
                )
                self.assertEqual(_codes(document, prefix), {expected})

    def test_full_prefix_typed_store_cannot_repeat_across_programs(self):
        document = _q8_store_document()
        document["grid"] = [2, 1, 1]
        self.assertEqual(_codes(document, "PACKED_STORE_"), {"PACKED_STORE_PROGRAM_OWNERSHIP"})

    def test_q4_typed_encode_is_explicitly_unsupported(self) -> None:
        document = _document(
            [
                _buffer("d", "fp16", [16]),
                _buffer("qs", "uint8", [16, 16]),
                _buffer(
                    "workspace",
                    "uint8",
                    [16, 18],
                    space="global",
                    mode="output",
                    packed="ggml_q4_0_v1",
                ),
            ],
            kind="store",
            reads=["d", "qs"],
            writes=["workspace"],
            parameters={"coalesced": True},
        )
        self.assertEqual(
            _codes(document, "PACKED_STORE_"),
            {"PACKED_STORE_Q4_TYPED_UNSUPPORTED"},
        )


if __name__ == "__main__":
    unittest.main()
