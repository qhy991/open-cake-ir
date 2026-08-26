"""Workload-owned contracts for the packed-Q8 producer Schedule.

These tests freeze the DAG and Workload semantics separately from Compiler findings and
keep their exact Workload relation separate from Compiler findings.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.evaluation import WorkloadContract


ROOT = Path(__file__).resolve().parents[2]
POSITIVE = ROOT / "corpus/schedules/packed-q8_1-producer-gfx1151.json"
RECORD_COUNT_DRIFT = (
    ROOT
    / "corpus/schedules/packed-q8_1-producer-gfx1151-record-count-drift.json"
)
WORKLOAD = ROOT / "contracts/workloads/llama-q4_0-q8_1-mmvq-f32-v1.json"
WORKLOAD_SHA256 = "41388c3cae8032d1c8281495eae2b071fac59a82d59bd292e0195b3b451dd629"


def _document(path: Path = POSITIVE) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("packed Q8 producer Schedule must be an object")
    return value


def _named(values: object, key: str) -> dict[str, dict[str, object]]:
    if not isinstance(values, list):
        raise TypeError(f"{key} must be a list")
    return {str(value[key]): value for value in values}


def _conformance_codes(document: dict[str, object]) -> tuple[str, ...]:
    """Workload diagnostics, deliberately separate from Compiler Findings."""

    buffers = _named(document["buffers"], "name")
    operations = _named(document["operations"], "id")
    codes: list[str] = []

    if (
        buffers["activation"]["shape"] != [32]
        or buffers["x_flat"]["shape"] != [512]
        or buffers["x_blocks"]["shape"] != [16, 32]
    ):
        codes.append("Q8_ACTIVATION_SHAPE")
    if operations["scale_activation"]["reads"] != ["x_blocks", "d_fp32"]:
        codes.append("Q8_QUANT_SCALE_SOURCE")
    if operations["reduce_sum"]["parameters"].get("algorithm") != "xor_tree_32":
        codes.append("Q8_SUM_REDUCTION_ALGORITHM")
    if (
        operations["round_q"]["parameters"].get("rounding")
        != "nearest_away_from_zero"
    ):
        codes.append("Q8_QUANT_ROUNDING")

    workspace = buffers["q8_workspace"]
    if workspace.get("mode") != "output":
        codes.append("Q8_WORKSPACE_OWNER")
    workspace_shape = workspace.get("shape")
    if not isinstance(workspace_shape, list) or len(workspace_shape) != 2:
        codes.append("Q8_WORKSPACE_SHAPE")
        workspace_records = None
    else:
        workspace_records = workspace_shape[0]
        if workspace_shape[1] != 36:
            codes.append("Q8_WORKSPACE_SHAPE")

    field_records = tuple(
        buffers[name]["shape"][0] for name in ("d_fp16", "s_fp16", "q_i8")
    )
    if workspace_records is not None and any(
        records != workspace_records for records in field_records
    ):
        codes.append("PACKED_BLOCK_STORE_RECORD_COUNT")

    if any(
        operations[name]["parameters"]
        != {"op": "cast", "rounding": "nearest_even", "overflow": "ieee"}
        for name in ("cast_d_fp16", "cast_s_fp16")
    ):
        codes.append("Q8_STORAGE_ROUNDING")
    return tuple(codes)


class PackedQ8ProducerContractTests(unittest.TestCase):
    def test_positive_owns_the_exact_workload_shape_and_576_byte_output(self) -> None:
        document = _document()
        workload = WorkloadContract.load(WORKLOAD)
        buffers = _named(document["buffers"], "name")
        workspace = buffers["q8_workspace"]

        self.assertEqual(workload.canonical_sha256, WORKLOAD_SHA256)
        self.assertEqual(
            document["metadata"]["workload_contract_sha256"], WORKLOAD_SHA256
        )
        self.assertEqual(document["target"], "gfx1151")
        self.assertEqual(document["roles"], [{"name": "compute", "warps": list(range(8))}])
        self.assertEqual(buffers["activation"]["shape"], [32])
        self.assertEqual(buffers["x_flat"]["shape"], [512])
        self.assertEqual(buffers["x_blocks"]["shape"], [16, 32])
        self.assertEqual(workspace["shape"], [16, 36])
        self.assertEqual(workspace["dtype"], "uint8")
        self.assertEqual(
            workspace["packed_block"],
            {"format": "ggml_q8_1_v1", "record_axis": 1},
        )
        self.assertEqual(workspace["shape"][0] * workspace["shape"][1], 576)
        self.assertEqual(_conformance_codes(document), ())

    def test_dag_keeps_fp32_scale_source_xor_sum_and_relation_derived_store(self) -> None:
        document = _document()
        operations = _named(document["operations"], "id")
        buffers = _named(document["buffers"], "name")

        self.assertEqual(
            [operation["id"] for operation in document["operations"]],
            [
                "load_activation",
                "reshape_blocks",
                "abs_activation",
                "reduce_amax",
                "reduce_sum",
                "make_d_fp32",
                "cast_d_fp16",
                "cast_s_fp16",
                "scale_activation",
                "round_q",
                "cast_q_int8",
                "store_q8_workspace",
            ],
        )
        self.assertEqual(operations["reshape_blocks"]["kind"], "reshape")
        self.assertEqual(operations["reshape_blocks"]["reads"], ["x_flat"])
        self.assertEqual(operations["reshape_blocks"]["writes"], ["x_blocks"])
        self.assertEqual(operations["reshape_blocks"]["parameters"], {})
        self.assertEqual(operations["abs_activation"]["parameters"], {"op": "abs"})
        self.assertEqual(operations["reduce_amax"]["reads"], ["abs_x"])
        self.assertEqual(operations["reduce_sum"]["reads"], ["x_blocks"])
        for name in ("reduce_amax", "reduce_sum"):
            self.assertEqual(
                operations[name]["parameters"]["algorithm"], "xor_tree_32"
            )
        self.assertEqual(
            operations["make_d_fp32"]["parameters"],
            {"op": "divide_no_nan", "scalar": 127.0},
        )
        self.assertEqual(
            operations["scale_activation"]["reads"], ["x_blocks", "d_fp32"]
        )
        self.assertEqual(
            operations["scale_activation"]["parameters"],
            {"op": "divide_no_nan", "broadcast_axis": 0},
        )
        self.assertEqual(
            operations["round_q"]["parameters"],
            {"op": "round", "rounding": "nearest_away_from_zero"},
        )
        self.assertEqual(
            operations["cast_q_int8"]["parameters"],
            {"op": "cast", "rounding": "toward_zero", "overflow": "forbid"},
        )
        for name in ("cast_d_fp16", "cast_s_fp16"):
            self.assertEqual(
                operations[name]["parameters"],
                {"op": "cast", "rounding": "nearest_even", "overflow": "ieee"},
            )
        self.assertEqual(
            operations["store_q8_workspace"]["reads"],
            ["d_fp16", "s_fp16", "q_i8"],
        )
        self.assertEqual(operations["store_q8_workspace"]["writes"], ["q8_workspace"])
        self.assertEqual(buffers["d_fp16"]["shape"], [16])
        self.assertEqual(buffers["s_fp16"]["shape"], [16])
        self.assertEqual(buffers["q_i8"]["shape"], [16, 32])
        self.assertEqual(
            document["program_map"]["axes"],
            [
                {
                    "name": "activation_tile",
                    "axis": 0,
                    "buffer": "activation",
                    "dimension": 0,
                    "tile": 512,
                }
            ],
        )
        store_access = next(
            access
            for access in document["access_maps"]
            if access["operation"] == "store_q8_workspace"
        )
        self.assertEqual(
            store_access["indices"],
            [{"source": "dimension", "dimension": 0}],
        )
        lowered = json.dumps(document, sort_keys=True).lower()
        for forbidden in ("q4_0", "dot", "mmvq"):
            self.assertNotIn(forbidden, lowered)

    def test_record_count_negative_changes_only_output_prefix_and_identity(self) -> None:
        positive = _document()
        negative = _document(RECORD_COUNT_DRIFT)
        expected = copy.deepcopy(positive)
        expected["schedule_id"] = (
            "packed-q8_1-producer-gfx1151-record-count-drift-v1"
        )
        _named(expected["buffers"], "name")["q8_workspace"]["shape"] = [8, 36]
        expected["lowering"]["entry_point"] = (
            "cake_packed_q8_1_producer_gfx1151_record_count_drift"
        )

        self.assertEqual(negative, expected)
        buffers = _named(negative["buffers"], "name")
        self.assertEqual(buffers["q8_workspace"]["shape"], [8, 36])
        self.assertEqual(buffers["d_fp16"]["shape"], [16])
        self.assertEqual(buffers["s_fp16"]["shape"], [16])
        self.assertEqual(buffers["q_i8"]["shape"], [16, 32])
        self.assertEqual(
            _conformance_codes(negative), ("PACKED_BLOCK_STORE_RECORD_COUNT",)
        )

    def test_each_workload_semantic_drift_has_one_owned_diagnostic(self) -> None:
        mutations = {
            "stored_fp16_d": (
                lambda value: _named(value["operations"], "id")[
                    "scale_activation"
                ].__setitem__("reads", ["x_blocks", "d_fp16"]),
                "Q8_QUANT_SCALE_SOURCE",
            ),
            "sequential_sum": (
                lambda value: _named(value["operations"], "id")["reduce_sum"][
                    "parameters"
                ].__setitem__("algorithm", "sequential"),
                "Q8_SUM_REDUCTION_ALGORITHM",
            ),
            "ties_even_q": (
                lambda value: _named(value["operations"], "id")["round_q"][
                    "parameters"
                ].__setitem__("rounding", "ties_to_even"),
                "Q8_QUANT_ROUNDING",
            ),
            "caller_q8": (
                lambda value: _named(value["buffers"], "name")[
                    "q8_workspace"
                ].__setitem__("mode", "input"),
                "Q8_WORKSPACE_OWNER",
            ),
            "activation_shape": (
                lambda value: _named(value["buffers"], "name")["activation"].__setitem__(
                    "shape", [31]
                ),
                "Q8_ACTIVATION_SHAPE",
            ),
            "workspace_shape": (
                lambda value: _named(value["buffers"], "name")[
                    "q8_workspace"
                ].__setitem__("shape", [16, 35]),
                "Q8_WORKSPACE_SHAPE",
            ),
            "storage_rounding": (
                lambda value: _named(value["operations"], "id")["cast_d_fp16"][
                    "parameters"
                ].__setitem__("rounding", "toward_zero"),
                "Q8_STORAGE_ROUNDING",
            ),
        }
        for label, (mutate, expected) in mutations.items():
            with self.subTest(label=label):
                document = copy.deepcopy(_document())
                mutate(document)
                self.assertEqual(_conformance_codes(document), (expected,))

    def test_schedules_use_current_parser_syntax_and_close_the_47_case_gate(self) -> None:
        schedule = Schedule.from_dict(_document())
        self.assertIs(schedule.operation("reshape_blocks").kind, OperationKind.RESHAPE)
        self.assertEqual(
            schedule.operation("round_q").parameters.rounding.value,
            "nearest_away_from_zero",
        )
        self.assertEqual(
            schedule.operation("cast_q_int8").parameters.overflow.value,
            "forbid",
        )

        manifest = json.loads(
            (ROOT / "corpus/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["cases"]), 47)
        self.assertEqual(
            [case["case_id"] for case in manifest["cases"][-2:]],
            [
                "packed-q8_1-producer-gfx1151-accepted",
                "packed-q8_1-producer-gfx1151-record-count-drift",
            ],
        )
        self.assertEqual(
            [case["schedule"] for case in manifest["cases"][-2:]],
            [
                POSITIVE.relative_to(ROOT).as_posix(),
                RECORD_COUNT_DRIFT.relative_to(ROOT).as_posix(),
            ],
        )
        gate = Compiler.load(ROOT, ROOT / "compiler/revision.json").check_corpus()
        self.assertTrue(gate.passed)
        self.assertEqual(gate.case_count, 47)


if __name__ == "__main__":
    unittest.main()
