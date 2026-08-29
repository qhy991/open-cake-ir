"""Static contract for the complete Lab-owned KDA fused-decode Schedule seed."""

from __future__ import annotations

import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import (
    BufferMode,
    OperationKind,
    StoreInactive,
)
from open_cake_ir.compiler.profile_model import profile_envelope
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.work import work_bound
from tools.kda_fused_decode_schedule import (
    ABI_ARGUMENTS,
    FROZEN_CELLS,
    HEAD_DIM,
    LOWER_BOUND,
    ONORM_EPS,
    QK_L2_EPS,
    SCALE,
    build_kda_fused_decode_schedule,
    frozen_cell,
    kda_fused_decode_adapter_contract,
    kda_fused_decode_schedule_document,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _blocking(schedule) -> list[str]:
    return [
        finding.code for finding in verify(schedule, TARGET) if finding.blocks_lowering
    ]


class FrozenMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_exact_nine_cells_parse_verify_and_lower(self) -> None:
        self.assertEqual(
            [
                (12, 1),
                (12, 4),
                (12, 8),
                (12, 16),
                (12, 32),
                (12, 64),
                (12, 128),
                (6, 32),
                (3, 64),
            ],
            [(cell.heads, cell.batch_size) for cell in FROZEN_CELLS],
        )
        self.assertEqual(
            [8, 16, 32, 64, 128],
            [cell.batch_size for cell in FROZEN_CELLS if cell.role == "primary"],
        )

        for cell in FROZEN_CELLS:
            with self.subTest(cell=cell.cell_id):
                schedule = build_kda_fused_decode_schedule(cell.heads, cell.batch_size)
                document = kda_fused_decode_schedule_document(
                    cell.heads, cell.batch_size
                )
                Draft202012Validator(schedule_schema()).validate(document)
                self.assertEqual([], _blocking(schedule))
                self.assertEqual(
                    [("row", 0, 1), ("head", 1, 1)],
                    [
                        (axis.name, axis.axis, axis.tile)
                        for axis in schedule.program_map.axes
                    ],
                )
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.accepted)
                self.assertTrue(assessment.lowering_eligible)
                lowered = self.compiler.lower(assessment)
                self.assertTrue(lowered.generated)
                compile(lowered.source, schedule.schedule_id, "exec")

    def test_builder_refuses_an_unfrozen_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the frozen"):
            frozen_cell(12, 2)


class AdapterViewContractTest(unittest.TestCase):
    def test_public_abi_and_zero_copy_views_are_single_owned(self) -> None:
        contract = kda_fused_decode_adapter_contract(12, 8)
        self.assertEqual(17, len(contract.abi_arguments))
        self.assertEqual(ABI_ARGUMENTS, contract.abi_arguments)
        self.assertFalse(contract.uses_contiguous_copy)
        self.assertEqual((1, 8, 12, HEAD_DIM), contract.returned_shape)
        self.assertEqual(
            {"scale": SCALE, "onorm_eps": ONORM_EPS, "lower_bound": LOWER_BOUND},
            dict(contract.scalar_constants),
        )
        self.assertEqual(
            ["output"],
            [view.name for view in contract.views if view.source_argument is None],
        )

    def test_mixed_qkv_broadcast_and_envelope_strides_are_exact(self) -> None:
        contract = kda_fused_decode_adapter_contract(12, 8)
        segment = 12 * HEAD_DIM
        qkv_row = 3 * segment
        for channel, offset in zip(("q", "k", "v"), (0, segment, 2 * segment)):
            current = contract.view(f"{channel}_current")
            self.assertEqual((qkv_row, HEAD_DIM, 1), current.strides)
            self.assertEqual(offset, current.storage_offset)

        self.assertEqual((12, 1, 0), contract.view("beta").strides)
        self.assertEqual((1, 0), contract.view("a_log").strides)
        self.assertEqual(
            (12 * HEAD_DIM * HEAD_DIM + 256, 16384, 128, 1),
            contract.view("ssm_state").strides,
        )

        histories = [view for view in contract.views if "_history_" in view.name]
        self.assertEqual(9, len(histories))
        self.assertEqual(
            {(9 * 12 * HEAD_DIM, HEAD_DIM, 1)},
            {view.strides for view in histories},
        )
        self.assertTrue(all(view.mode == "state" for view in histories))

        dense_weight_or_bias = [
            view
            for view in contract.views
            if "_weight_" in view.name or view.name in {"q_bias", "k_bias", "v_bias"}
        ]
        self.assertEqual(15, len(dense_weight_or_bias))
        self.assertTrue(
            all(view.schedule_strides is None for view in dense_weight_or_bias)
        )


class CompleteBoundarySemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = build_kda_fused_decode_schedule(12, 8)
        cls.operations = {
            operation.op_id: operation for operation in cls.schedule.operations
        }
        cls.source = emit(cls.schedule, TARGET).source

    def test_output_and_all_mutable_state_have_padding_effects(self) -> None:
        relation = self.schedule.buffer("cache_indices").unique_index
        self.assertEqual(
            ("ssm_state", 0, -1),
            (
                relation.buffer,
                relation.dimension,
                relation.sentinel,
            ),
        )
        states = [
            buffer
            for buffer in self.schedule.buffers
            if buffer.mode is BufferMode.STATE
        ]
        self.assertEqual(10, len(states))
        state_stores = [
            operation
            for operation in self.schedule.operations
            if operation.kind is OperationKind.STORE
            and self.schedule.buffer(operation.writes[0]).mode is BufferMode.STATE
        ]
        self.assertEqual(10, len(state_stores))
        self.assertTrue(
            all(store.parameters.valid_if == "slot" for store in state_stores)
        )
        self.assertTrue(
            all(
                store.parameters.inactive is StoreInactive.NO_EFFECT
                for store in state_stores
            )
        )

        output = self.operations["store_output"]
        self.assertEqual(("output",), output.writes)
        self.assertEqual("slot", output.parameters.valid_if)
        self.assertIs(output.parameters.inactive, StoreInactive.WRITE_ZERO)
        self.assertIn("tl.where((slot >= 0), output_bf16, 0.0)", self.source)

    def test_conv_recurrence_and_onorm_math_are_explicit(self) -> None:
        for channel in ("q", "k", "v"):
            seam = self.operations[f"{channel}_bf16_seam"]
            self.assertIs(seam.kind, OperationKind.CAST)
            self.assertEqual("bf16", seam.parameters.to.value)

        self.assertEqual(
            LOWER_BOUND,
            self.operations["safe_gate_lower_bound"].parameters.scalar,
        )
        self.assertEqual(SCALE, self.operations["scale_recurrent_q"].parameters.scalar)
        self.assertEqual(
            ONORM_EPS,
            self.operations["output_rms_epsilon"].parameters.scalar,
        )
        self.assertEqual(
            QK_L2_EPS,
            self.operations["q_l2_epsilon"].parameters.scalar,
        )
        self.assertEqual(
            QK_L2_EPS,
            self.operations["k_l2_epsilon"].parameters.scalar,
        )
        self.assertEqual(
            ("delta", "k_l2_normalized"),
            self.operations["delta_outer_k"].reads,
        )
        self.assertEqual(
            ("decayed_state", "delta_k_outer"),
            self.operations["state_rank_one_update"].reads,
        )
        self.assertEqual(
            1,
            sum(
                operation.kind is OperationKind.OUTER
                for operation in self.schedule.operations
            ),
        )

    def test_rank_local_state_and_logical_view_strides_are_preserved(self) -> None:
        self.assertEqual(
            (12, 128, 128),
            self.schedule.buffer("ssm_state").shape[1:],
        )
        self.assertEqual(
            (12 * 128 * 128 + 256, 16384, 128, 1),
            self.schedule.buffer("ssm_state").strides,
        )
        self.assertEqual(
            (9 * 12 * 128, 128, 1),
            self.schedule.buffer("q_history_0").strides,
        )
        self.assertEqual(
            (3 * 12 * 128, 128, 1),
            self.schedule.buffer("q_current").strides,
        )


class WorkAndProfileTest(unittest.TestCase):
    def test_primary_seed_has_static_work_and_profile_without_a_time_claim(
        self,
    ) -> None:
        schedule = build_kda_fused_decode_schedule(12, 8)
        source = emit(schedule, TARGET).source
        bound = work_bound(schedule)
        self.assertIsNotNone(bound)
        self.assertEqual(8 * 12, bound.program_tiles)
        self.assertGreater(bound.flops, 0)
        self.assertFalse(bound.flops_exact)
        self.assertIn("ssm_state", bound.partially_addressed)
        self.assertIn("output", bound.partially_addressed)

        profile = profile_envelope(schedule, TARGET, lowered_source=source)
        self.assertEqual(1, profile.lowering["outer_operations"])
        self.assertEqual(35, profile.lowering["global_load_operations"])
        self.assertEqual(11, profile.lowering["global_store_operations"])
        self.assertEqual(
            ["a_log", "beta"], profile.lowering["broadcast_global_buffers"]
        )
        self.assertEqual(10, len(profile.lowering["runtime_indexed_buffers"]))
        throughput = next(
            metric
            for metric in profile.ncu_metrics
            if metric.metric == "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        self.assertEqual("unknown", throughput.estimate_kind)
        self.assertIsNone(throughput.value)


if __name__ == "__main__":
    unittest.main()
