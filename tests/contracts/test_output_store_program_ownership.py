"""Ordinary output stores must distinguish varying direct program coordinates."""
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend

ROOT = Path(__file__).resolve().parents[2]
CODE = "OUTPUT_STORE_PROGRAM_AXIS_COLLISION"


def moments_source(rows=8, row_tile=4, target="gfx938"):
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="moments-owner", target="{target}", backend="triton")
def candidate(lm, x: cake.Tensor(({rows}, 16), "fp32"),
              mean: cake.Tensor((16,), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    column = lm.program(x, axis=0, dimension=1, tile=4)
    row_part = lm.program(x, axis=1, dimension=0, tile={row_tile})
    with compute:
        values = lm.load(x[row_part, column], id="load_x")
        total = lm.reduce(values, op="sum", axis=0, scope="cta", id="sum_x")
        lm.store(mean[column], total / {float(rows)}, id="store_mean")
'''


class OutputStoreProgramOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def assess(self, source):
        return self.compiler.assess(frontend.parse(source).document)

    def test_dcu_partial_reduction_is_refused_by_the_ownership_rule(self):
        result = self.assess(moments_source())
        conflicts = [f for f in result.findings if f.code == CODE]
        self.assertFalse(result.accepted)
        self.assertFalse(result.lowering_eligible)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("row_part", conflicts[0].message)
        self.assertIn("mean", conflicts[0].message)
        self.assertEqual(conflicts[0].path, "operations[3]")
        self.assertEqual([f.code for f in result.findings if f.blocks_lowering], [CODE])

    def test_ownership_is_shared_by_other_triton_targets(self):
        for target in ("sm_100a", "gfx1151", "xcore1002"):
            with self.subTest(target=target):
                result = self.assess(moments_source(target=target))
                self.assertIn(CODE, [f.code for f in result.findings])

    def test_tail_tile_is_still_a_second_writer(self):
        result = self.assess(moments_source(rows=7))
        self.assertIn(CODE, [f.code for f in result.findings])

    def test_an_unused_singleton_program_axis_is_safe(self):
        result = self.assess(moments_source(row_tile=8))
        self.assertTrue(result.accepted)
        self.assertTrue(result.lowering_eligible)
        self.assertIn("mean + column_offsets", self.compiler.lower(result).source)

    def test_input_anchored_two_axis_output_is_disjoint(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="two-axis-copy", target="gfx938", backend="triton")
def candidate(lm, x: cake.Tensor((8, 16), "fp32"),
              out: cake.Tensor((8, 16), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=4)
    column = lm.program(x, axis=1, dimension=1, tile=4)
    with compute:
        values = lm.load(x[row, column], id="load_x")
        lm.store(out[row, column], values, id="store_out")
'''
        result = self.assess(source)
        self.assertTrue(result.accepted)
        self.assertTrue(result.lowering_eligible)
        self.assertIn("out + row_offsets", self.compiler.lower(result).source)

    def test_repeated_axis_use_is_not_a_missing_axis(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="diagonal-copy", target="gfx938", backend="triton")
def candidate(lm, x: cake.Tensor((8, 8), "fp32"),
              out: cake.Tensor((8, 8), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, row], id="load_x")
        lm.store(out[row, row], value, id="store_out")
'''
        result = self.assess(source)
        self.assertNotIn(CODE, [f.code for f in result.findings])
        # Coverage of the whole Task output is still the external oracle's job.
        self.assertTrue(result.lowering_eligible)

    def test_persistent_walk_needs_a_separate_physical_writer_proof(self):
        doc = frontend.parse(moments_source()).document
        doc["program_map"]["persistent"] = True
        result = self.compiler.assess(doc)
        self.assertNotIn(CODE, [f.code for f in result.findings])


if __name__ == "__main__":
    unittest.main()
