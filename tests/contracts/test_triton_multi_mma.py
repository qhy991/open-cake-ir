"""Generic multi-MMA Triton lowering and per-operation contract boundaries."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.analysis import logical_register_pressure_per_thread
from open_cake_ir.compiler.emit import EmitError
from open_cake_ir.compiler.emit_triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.profile_model import profile_envelope
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.work import work_bound
from open_cake_ir.evaluation import ProgramContract

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")
QSA = ROOT / "corpus/schedules/qsa-score-topk-t32768.json"
PROGRAM = ROOT / "contracts/programs/qsa-prefill-t32768-v2.json"
FROZEN_QSA_LOWERING = "d530fcac401ef45e69e77b951209836db12a59579c6b63984d5ee8985d3395e3"


def _two_k_slice_document() -> dict:
    return {
        "schema_version": 1,
        "schedule_id": "triton-fp32-two-k-slice-mma-v1",
        "target": "sm_100a",
        "roles": [{"name": "compute", "warps": [0, 1, 2, 3]}],
        "allocations": [],
        "pipelines": [],
        "barriers": [],
        "buffers": [
            {"name": "a", "space": "global", "dtype": "fp32", "shape": [16, 128], "mode": "input"},
            {"name": "b", "space": "global", "dtype": "fp32", "shape": [16, 128], "mode": "input"},
            {"name": "c", "space": "global", "dtype": "fp32", "shape": [16, 16], "mode": "output"},
            {
                "name": "a_k0",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 64],
                "mode": "scratch",
            },
            {
                "name": "b_k0",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 64],
                "mode": "scratch",
            },
            {
                "name": "a_k1",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 64],
                "mode": "scratch",
            },
            {
                "name": "b_k1",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 64],
                "mode": "scratch",
            },
            {
                "name": "partial_k0",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 16],
                "mode": "scratch",
            },
            {
                "name": "partial_k1",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 16],
                "mode": "scratch",
            },
            {
                "name": "accumulator",
                "space": "register",
                "dtype": "fp32",
                "shape": [16, 16],
                "mode": "scratch",
            },
        ],
        "operations": [
            {
                "id": "load_a_k0",
                "kind": "load",
                "role": "compute",
                "reads": ["a"],
                "writes": ["a_k0"],
                "parameters": {"movement": "global", "reuse": "reused"},
            },
            {
                "id": "load_b_k0",
                "kind": "load",
                "role": "compute",
                "reads": ["b"],
                "writes": ["b_k0"],
                "parameters": {"movement": "global", "reuse": "reused"},
            },
            {
                "id": "load_a_k1",
                "kind": "load",
                "role": "compute",
                "reads": ["a"],
                "writes": ["a_k1"],
                "parameters": {"movement": "global", "reuse": "reused"},
            },
            {
                "id": "load_b_k1",
                "kind": "load",
                "role": "compute",
                "reads": ["b"],
                "writes": ["b_k1"],
                "parameters": {"movement": "global", "reuse": "reused"},
            },
            {
                "id": "mma_k0",
                "kind": "mma",
                "role": "compute",
                "reads": ["a_k0", "b_k0"],
                "writes": ["partial_k0"],
                "depends_on": ["load_a_k0", "load_b_k0"],
                "parameters": {
                    "accumulator": "fp32",
                    "instruction": {"contract": "triton.dot.fp32_ieee"},
                    "tile_shape": [16, 16, 64],
                },
            },
            {
                "id": "mma_k1",
                "kind": "mma",
                "role": "compute",
                "reads": ["a_k1", "b_k1"],
                "writes": ["partial_k1"],
                "depends_on": ["load_a_k1", "load_b_k1"],
                "parameters": {
                    "accumulator": "fp32",
                    "instruction": {"contract": "triton.dot.fp32_ieee"},
                    "tile_shape": [16, 16, 64],
                },
            },
            {
                "id": "add_partials",
                "kind": "elementwise",
                "role": "compute",
                "reads": ["partial_k0", "partial_k1"],
                "writes": ["accumulator"],
                "depends_on": ["mma_k0", "mma_k1"],
                "parameters": {"op": "add"},
            },
            {
                "id": "store_c",
                "kind": "store",
                "role": "compute",
                "reads": ["accumulator"],
                "writes": ["c"],
                "depends_on": ["add_partials"],
                "parameters": {"coalesced": True},
            },
        ],
        "outputs": ["c"],
        "program_map": {
            "axes": [
                {"name": "row_block", "axis": 0, "buffer": "a", "dimension": 0, "tile": 16}
            ]
        },
        "tile_loops": [],
        "access_maps": [
            {
                "operation": "load_a_k0",
                "buffer": "a",
                "indices": [
                    {"source": "program_tile", "name": "row_block"},
                    {"source": "dimension", "dimension": 1, "extent": 64},
                ],
                "boundary": "mask_tiled_axes",
            },
            {
                "operation": "load_b_k0",
                "buffer": "b",
                "indices": [
                    {"source": "dimension", "dimension": 0},
                    {"source": "dimension", "dimension": 1, "extent": 64},
                ],
                "boundary": "mask_tiled_axes",
            },
            {
                "operation": "load_a_k1",
                "buffer": "a",
                "indices": [
                    {"source": "program_tile", "name": "row_block"},
                    {"source": "dimension", "dimension": 1, "offset": 64},
                ],
                "boundary": "mask_tiled_axes",
            },
            {
                "operation": "load_b_k1",
                "buffer": "b",
                "indices": [
                    {"source": "dimension", "dimension": 0},
                    {"source": "dimension", "dimension": 1, "offset": 64},
                ],
                "boundary": "mask_tiled_axes",
            },
            {
                "operation": "store_c",
                "buffer": "c",
                "indices": [
                    {"source": "program_tile", "name": "row_block"},
                    {"source": "dimension", "dimension": 1},
                ],
                "boundary": "mask_tiled_axes",
            },
        ],
        "metadata": {"workload_contract_sha256": "0" * 64},
        "residency": {"ctas_per_multiprocessor": 4, "registers_per_thread": 64},
        "lowering": {"backend": "triton", "entry_point": "cake_triton_fp32_two_k_slice_mma"},
    }


class TritonMultiMmaTest(unittest.TestCase):
    def test_two_k_slices_lower_in_declared_dag_order_and_profile_both_dots(self) -> None:
        schedule = Schedule.from_dict(_two_k_slice_document())
        findings = verify(schedule, TARGET)

        self.assertFalse([finding for finding in findings if finding.blocks_lowering])
        self.assertEqual(preflight(schedule, TARGET), ())
        lowering = emit(schedule, TARGET)
        ast.parse(lowering.source)
        source = lowering.source
        order = [
            source.index(f"# CAKE_OP:{operation}")
            for operation in ("mma_k0", "mma_k1", "add_partials", "store_c")
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(source.count('input_precision="ieee"'), 2)
        self.assertIn("partial_k0 = tl.dot(", source)
        self.assertIn("partial_k1 = tl.dot(", source)
        self.assertIn("accumulator = partial_k0 + partial_k1", source)

        profile = profile_envelope(
            schedule,
            TARGET,
            lowered_source=source,
        ).as_dict()
        self.assertEqual(profile["lowering"]["triton_dot_count"], 2)
        bound = work_bound(schedule)
        assert bound is not None
        self.assertEqual(bound.mma_flops, 2 * 2 * 16 * 16 * 64)
        self.assertEqual(bound.flops, bound.mma_flops + 16 * 16)
        self.assertEqual(
            bound.arithmetic_contracts,
            ("triton.dot.fp32_ieee", "triton.dot.fp32_ieee"),
        )
        self.assertEqual(bound.contended_contract, "triton.dot.fp32_ieee")
        self.assertEqual(profile["work"]["mma_flops"], bound.mma_flops)
        self.assertEqual(profile["work"]["flops"], bound.flops)
        self.assertEqual(
            logical_register_pressure_per_thread(schedule, TARGET),
            34,
        )
        registers = next(
            metric
            for metric in profile["ncu_metrics"]
            if metric["metric"] == "launch__registers_per_thread"
        )
        self.assertEqual(
            (registers["estimate_kind"], registers["value"]),
            ("unknown", None),
        )

    def test_each_mma_keeps_its_own_backend_contract_boundary(self) -> None:
        document = _two_k_slice_document()
        second = document["operations"][5]["parameters"]["instruction"]
        second["contract"] = "tcgen05.mma.cta_group::1.kind::f16"
        schedule = Schedule.from_dict(document)

        findings = preflight(schedule, TARGET)

        self.assertEqual(
            [(finding.code, finding.path) for finding in findings],
            [
                (
                    "TRITON_MMA_INSTRUCTION_UNSUPPORTED",
                    "operations[5].parameters.instruction.contract",
                )
            ],
        )

    def test_each_mma_must_be_admitted_by_the_target(self) -> None:
        document = _two_k_slice_document()
        document["operations"][5]["parameters"]["instruction"]["contract"] = (
            "triton.dot.fp32_unadmitted"
        )
        schedule = Schedule.from_dict(document)

        target_finding = next(
            finding
            for finding in verify(schedule, TARGET)
            if finding.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        )
        self.assertEqual(
            target_finding.path,
            "operations[5].parameters.instruction.contract",
        )
        with self.assertRaisesRegex(EmitError, "not admitted"):
            emit(schedule, TARGET)

    def test_each_mma_must_declare_its_own_instruction_contract(self) -> None:
        document = _two_k_slice_document()
        document["operations"][5]["parameters"].pop("instruction")

        findings = preflight(Schedule.from_dict(document), TARGET)

        self.assertEqual(
            [(finding.code, finding.path) for finding in findings],
            [
                (
                    "BACKEND_MMA_INSTRUCTION_REQUIRED",
                    "operations[5].parameters.instruction",
                )
            ],
        )

    def test_two_mmas_cannot_hide_an_implicit_shared_accumulator(self) -> None:
        document = _two_k_slice_document()
        document["operations"][5]["writes"] = ["partial_k0"]

        findings = verify(Schedule.from_dict(document), TARGET)

        multiple = next(
            finding for finding in findings if finding.code == "BUFFER_MULTIPLE_WRITERS"
        )
        self.assertEqual(multiple.path, "buffers[7]")

    def test_single_mma_qsa_and_frozen_program_v2_remain_byte_identical(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess_file(QSA)

        self.assertEqual(compiler.lower(assessment).source_sha256, FROZEN_QSA_LOWERING)
        program = ProgramContract.load(ROOT, PROGRAM, compiler)
        self.assertEqual(program.program_id, "qsa-prefill-t32768-cake-port-v2")
        self.assertEqual(len(program.nodes), 5)


if __name__ == "__main__":
    unittest.main()
