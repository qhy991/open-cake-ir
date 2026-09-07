"""Source-level lowering contract for the first live-Q8 producer slice."""

from __future__ import annotations

import unittest
import json
from unittest.mock import MagicMock
import math
import struct
from types import SimpleNamespace
from pathlib import Path

from open_cake_ir.compiler import Compiler, emit_triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import FindingSeverity, verify


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = (
    ROOT / "corpus/schedules/packed-q8_1-producer-gfx1151.json"
)
TARGET = Target.load(ROOT / "compiler/targets/gfx1151.json")


class PackedQ8ProducerTritonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = Schedule.load(SCHEDULE)
        cls.emission = emit_triton.emit(cls.schedule, TARGET)

    def test_schedule_is_verified_and_lowerable_before_a_gpu_claim(self) -> None:
        findings = verify(self.schedule, TARGET)

        self.assertEqual(
            [finding.code for finding in findings],
            ["RESIDENCY_TARGET_UNMODELED"],
        )
        self.assertTrue(
            all(finding.severity is FindingSeverity.REPORT for finding in findings)
        )
        self.assertEqual(emit_triton.preflight(self.schedule, TARGET), ())
        self.assertEqual(self.emission.toolchain["grid"], [1, 1, 1])
        self.assertEqual(
            self.emission.toolchain["compile_options"], {"num_warps": 8}
        )
        self.assertEqual(
            self.emission.toolchain["signature"],
            {"activation": "*fp32", "q8_workspace": "*u8"},
        )

    def test_source_preserves_padding_shape_and_the_explicit_xor_tree(self) -> None:
        source = self.emission.source

        self.assertIn("tl.arange(0, BLOCK_ACTIVATION_TILE)", source)
        self.assertIn("mask=activation_tile_offsets < N_ACTIVATION_TILE,", source)
        self.assertIn("other=0.0,", source)
        self.assertIn("x_blocks = tl.reshape(x_flat, (16, 32))", source)
        for operation in ("reduce_amax", "reduce_sum"):
            for half in (16, 8, 4, 2, 1):
                self.assertIn(f"{operation}_xor_{half}_shaped", source)
                self.assertIn(f"{operation}_xor_{half}_paired", source)
                self.assertIn(f"{operation}_xor_{half}_lower", source)
                self.assertIn(f"{operation}_xor_{half}_upper", source)
        self.assertNotIn("tl.sum(", source)
        self.assertNotIn("tl.max(", source)

    def test_source_uses_precise_division_and_declared_rounding_conversions(self) -> None:
        source = self.emission.source

        self.assertEqual(source.count("tl.div_rn("), 2)
        self.assertIn("d_fp32 = tl.where(127.0 == 0.0, 0.0", source)
        self.assertIn("d_fp32[:, None] == 0.0", source)
        self.assertIn("libdevice.copysign(tl.floor(tl.abs(scaled))", source)
        self.assertIn(
            'd_fp32.to(tl.float16, fp_downcast_rounding="rtne")', source
        )
        self.assertIn(
            'sum_x.to(tl.float16, fp_downcast_rounding="rtne")', source
        )
        self.assertIn("q_i8 = rounded.to(tl.int8)", source)

    def test_emitted_rounding_preserves_tie_neighbors_large_integers_and_signed_zero(self):
        # Each scalar operation is rounded back to FP32, like the emitted tensor
        # expression. The oracle performs the rounding decision in FP64 instead.
        class FP32(float):
            def __new__(cls, value):
                return super().__new__(cls, struct.unpack("f", struct.pack("f", value))[0])
            def __add__(self, other):
                return FP32(float(self) + other)
            def __sub__(self, other):
                return FP32(float(self) - other)
        tl = SimpleNamespace(abs=lambda x: FP32(abs(x)), floor=lambda x: FP32(math.floor(x)))
        libdevice = SimpleNamespace(copysign=lambda x, y: FP32(math.copysign(x, y)))
        line = next(line for line in self.emission.source.splitlines() if line.strip().startswith("rounded = "))
        expression = line.split(" = ", 1)[1]
        values = [FP32(0.0), FP32(8388609.0)]
        for tie in (0.5, 1.5, 2.5, 127.5):
            bits = struct.unpack("I", struct.pack("f", tie))[0]
            values.extend(FP32(struct.unpack("f", struct.pack("I", bits + delta))[0]) for delta in (-1, 0, 1))
        values += [FP32(-value) for value in values]
        for value in values:
            with self.subTest(value=value):
                observed = eval(expression, {"__builtins__": {}, "tl": tl, "libdevice": libdevice, "scaled": value})
                expected = math.copysign(math.floor(abs(float(value)) + 0.5), value)
                self.assertEqual(observed, expected)
                self.assertEqual(math.copysign(1, observed), math.copysign(1, expected))

    def test_packed_store_derives_five_little_endian_writes_without_arange_36(self) -> None:
        source = self.emission.source

        self.assertIn("d_fp16.to(tl.uint16, bitcast=True)", source)
        self.assertIn("s_fp16.to(tl.uint16, bitcast=True)", source)
        for offset in range(4):
            self.assertIn(f"record_ptrs + {offset}", source)
        self.assertIn("q_offsets = tl.arange(0, 32)", source)
        self.assertIn("record_ptrs[:, None] + 4 +", source)
        self.assertIn("q_i8.to(tl.uint8, bitcast=True)", source)
        self.assertNotIn("tl.arange(0, 36)", source)
        self.assertEqual(source.count("tl.store("), 5)
        self.assertIn(
            "out = torch.empty((16, 36), dtype=torch.uint8", source
        )

    def test_generated_source_is_valid_python(self) -> None:
        compile(self.emission.source, "<packed-q8-producer>", "exec")


class ScratchNameCollisionTests(unittest.TestCase):
    @staticmethod
    def rename(value, old, new):
        if isinstance(value, str):
            return new if value == old else value
        if isinstance(value, list):
            return [ScratchNameCollisionTests.rename(item, old, new) for item in value]
        if isinstance(value, dict):
            return {key: ScratchNameCollisionTests.rename(item, old, new) for key, item in value.items()}
        return value

    def lower(self, renames):
        document = json.loads(SCHEDULE.read_text())
        for old, new in renames:
            document = self.rename(document, old, new)
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        source = compiler.lower(assessment).source
        self.assertEqual(source, compiler.lower(assessment).source)
        return source

    @staticmethod
    def region(source, start_op, end_marker):
        lines = source.splitlines()
        start = next(index for index, line in enumerate(lines) if f"# CAKE_OP:{start_op}" in line)
        end = next(index for index, line in enumerate(lines[start:], start) if end_marker in line)
        return "\n".join(line.strip() for line in lines[start:end])

    def test_xor_scratch_preserves_a_renamed_live_buffer_through_its_next_reduction(self):
        class Tensor:
            def __init__(self, shape, origin="activation"):
                self.shape, self.origin = tuple(shape), origin
            def __add__(self, other):
                if self.shape != other.shape:
                    raise ValueError("addition shapes differ")
                return Tensor(self.shape, self.origin)

        def reshape(value, shape):
            if math.prod(value.shape) != math.prod(shape):
                raise ValueError(f"cannot reshape {value.shape} into {shape}")
            return Tensor(shape, value.origin)

        tl = SimpleNamespace(
            reshape=reshape,
            abs=lambda value: Tensor(value.shape, "absolute"),
            maximum=lambda left, right: left + right,
            permute=lambda value, axes: Tensor(tuple(value.shape[axis] for axis in axes), value.origin),
            split=lambda value: (Tensor(value.shape[:-1], value.origin), Tensor(value.shape[:-1], value.origin)),
        )
        collision = "reduce_amax_xor_16_lower"
        for occupy_first_suffix in (False, True):
            with self.subTest(occupy_first_suffix=occupy_first_suffix):
                renames = [("x_blocks", collision)]
                if occupy_first_suffix:
                    renames.append(("abs_x", collision + "_1"))
                source = self.lower(renames)
                scope = {"tl": tl, "x_flat": Tensor((512,))}
                exec(self.region(source, "reshape_blocks", "# CAKE_OP:make_d_fp32"), {"__builtins__": {}}, scope)
                self.assertEqual(scope[collision].shape, (16, 32))
                self.assertEqual(scope[collision].origin, "activation")
                self.assertEqual(scope["sum_x"].shape, (16,))
                self.assertEqual(scope["sum_x"].origin, "activation")
                if occupy_first_suffix:
                    self.assertEqual(scope[collision + "_1"].shape, (16, 32))
                    self.assertEqual(scope[collision + "_1"].origin, "absolute")

    def test_packed_scratch_preserves_renamed_qs_until_the_payload_store(self):
        collision = "store_q8_workspace_d_bits"
        for occupy_first_suffix in (False, True):
            with self.subTest(occupy_first_suffix=occupy_first_suffix):
                d_name = collision + "_1" if occupy_first_suffix else "d_fp16"
                renames = [("q_i8", collision)]
                if occupy_first_suffix:
                    renames.append(("d_fp16", d_name))
                source = self.lower(renames)
                qs, d, s = MagicMock(name="qs"), MagicMock(name="d"), MagicMock(name="s")
                tl = MagicMock(name="tl")
                scope = {
                    "tl": tl, collision: qs, d_name: d, "s_fp16": s,
                    "q8_workspace": MagicMock(name="workspace"),
                    "q8_workspace_d0_offsets": MagicMock(name="record_offsets"),
                    "D_Q8_WORKSPACE_1": 36,
                }
                exec(self.region(source, "store_q8_workspace", "# CAKE_KERNEL_END"), {"__builtins__": {}}, scope)
                self.assertIs(scope[collision], qs)
                self.assertIs(scope[d_name], d)
                self.assertEqual(tl.store.call_count, 5)
                self.assertIs(tl.store.call_args.args[1], qs.to.return_value)
                qs.to.assert_called_once_with(tl.uint8, bitcast=True)


if __name__ == "__main__":
    unittest.main()
