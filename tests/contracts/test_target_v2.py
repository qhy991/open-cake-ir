"""Contract tests for the vendor-neutral Target schema."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import DType, MemorySpace, OperationKind
from open_cake_ir.compiler.target import (
    InstructionPlacement,
    InstructionShape,
    Target,
    TargetParseError,
)


ROOT = Path(__file__).resolve().parents[2]
SM100A = ROOT / "compiler" / "targets" / "sm_100a.json"


def _apple_target() -> dict[str, object]:
    return {
        "schema_version": 2,
        "target_id": "apple_gpu_family9",
        "architecture": "apple_gpu_family9",
        "execution_group_width": 32,
        "memory_spaces": ["global", "shared", "register"],
        "operation_kinds": ["load", "elementwise", "reduce", "store"],
        "resource_limits": {
            "maximum_threads_per_workgroup": 1024,
            "maximum_threadgroup_memory_bytes": 32768,
        },
        "instruction_contracts": [],
        "synchronization_contracts": [],
        "citations": [
            {
                "kind": "vendor_documentation",
                "source": "Apple Metal feature set tables",
            }
        ],
    }


class SchemaV1CompatibilityTest(unittest.TestCase):
    def test_current_target_retains_every_admitted_contract(self) -> None:
        target = Target.load(SM100A)

        self.assertEqual(target.schema_version, 1)
        self.assertEqual(target.warp_size, 32)
        self.assertEqual(target.warps_per_warpgroup, 4)
        self.assertEqual(
            target.instruction_contracts,
            frozenset(
                {
                    "tcgen05.mma.cta_group::1.kind::f16",
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                    "triton.dot.bf16_fp32",
                    "triton.dot.fp8e4m3_block_scale_fp32",
                    "triton.atomic_add.i32.relaxed.gpu",
                    "libdevice.tanh.f32",
                }
            ),
        )
        self.assertEqual(target.resource_limits.maximum_threads_per_cta, 1024)
        self.assertEqual(target.resource_limits.maximum_warps_per_cta, 32)
        self.assertEqual(target.resource_limits.maximum_shared_memory_bytes, 232448)

    def test_schema_v1_contract_names_do_not_depend_on_a_parser_whitelist(self) -> None:
        document = json.loads(SM100A.read_text(encoding="utf-8"))
        document["instruction_contracts"].append("future.backend.contract")

        target = Target.from_dict(document)

        contract = target.instruction("future.backend.contract")
        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.name, "future.backend.contract")
        self.assertIsNone(contract.operand_dtypes)


class SchemaV2Test(unittest.TestCase):
    def test_empty_instruction_and_synchronization_sets_are_valid(self) -> None:
        target = Target.from_dict(_apple_target())

        self.assertEqual(target.schema_version, 2)
        self.assertEqual(target.target_id, "apple_gpu_family9")
        self.assertEqual(target.execution_group_width, 32)
        self.assertEqual(target.warp_size, 32)
        self.assertIsNone(target.warps_per_warpgroup)
        self.assertEqual(target.instruction_contracts, frozenset())
        self.assertEqual(target.synchronization_contracts, frozenset())
        self.assertEqual(
            target.memory_spaces,
            frozenset({MemorySpace.GLOBAL, MemorySpace.SHARED, MemorySpace.REGISTER}),
        )
        self.assertEqual(
            target.operation_kinds,
            frozenset(
                {
                    OperationKind.LOAD,
                    OperationKind.ELEMENTWISE,
                    OperationKind.REDUCE,
                    OperationKind.STORE,
                }
            ),
        )

        limits = target.resource_limits
        self.assertEqual(limits.maximum_threads_per_workgroup, 1024)
        self.assertEqual(limits.maximum_execution_groups_per_workgroup, 32)
        self.assertEqual(limits.maximum_threadgroup_memory_bytes, 32768)
        self.assertEqual(limits.maximum_threads_per_cta, 1024)
        self.assertEqual(limits.maximum_warps_per_cta, 32)
        self.assertEqual(limits.maximum_shared_memory_bytes, 32768)
        self.assertIsNone(limits.maximum_tensor_memory_bytes)
        self.assertIsNone(limits.maximum_grid)

    def test_optional_resource_budgets_remain_exact_when_declared(self) -> None:
        document = _apple_target()
        limits = copy.deepcopy(document["resource_limits"])
        assert isinstance(limits, dict)
        limits["maximum_tensor_memory_bytes"] = 65536
        limits["maximum_grid"] = {"x": 4096, "y": 2048, "z": 1024}
        document["resource_limits"] = limits

        target = Target.from_dict(document)

        self.assertEqual(target.resource_limits.maximum_tensor_memory_bytes, 65536)
        self.assertEqual(target.resource_limits.maximum_grid, (4096, 2048, 1024))

    def test_structured_instruction_contract_is_available_to_an_adapter(self) -> None:
        document = _apple_target()
        document["instruction_contracts"] = [
            {
                "name": "metal.simdgroup_mma.m8n8k8.bf16_bf16_fp32",
                "operand_dtypes": ["bf16"],
                "accumulator_dtype": "fp32",
                "atom_shapes": [[8, 8, 8]],
                "placement": "backend",
                "shape": "explicit",
            }
        ]

        target = Target.from_dict(document)

        contract = target.instruction("metal.simdgroup_mma.m8n8k8.bf16_bf16_fp32")
        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.operand_dtypes, frozenset({DType.BF16}))
        self.assertEqual(contract.accumulator_dtype, DType.FP32)
        self.assertEqual(contract.atom_shapes, ((8, 8, 8),))
        self.assertEqual(contract.placement, InstructionPlacement.BACKEND)
        self.assertEqual(contract.shape, InstructionShape.EXPLICIT)

    def test_workgroup_limit_must_contain_whole_execution_groups(self) -> None:
        document = _apple_target()
        limits = copy.deepcopy(document["resource_limits"])
        assert isinstance(limits, dict)
        limits["maximum_threads_per_workgroup"] = 1000
        document["resource_limits"] = limits

        with self.assertRaisesRegex(
            TargetParseError, "must contain whole execution groups"
        ):
            Target.from_dict(document)


if __name__ == "__main__":
    unittest.main()
