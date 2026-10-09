"""Exact concrete output partitions, and the boundaries of the admitted proof."""
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.verifier.output_partitions import partition_refusal

ROOT = Path(__file__).resolve().parents[2]


def partition_source(target="xcore1002", backend="triton", row_tile=1):
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="static-output-partition", target="{target}", backend="{backend}")
def candidate(lm, x: cake.Tensor((8, 16), "fp32"),
              out: cake.Tensor((8, 16), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile={row_tile})
    with compute:
        left = lm.load(x[row, :8], id="load_left")
        right = lm.load(x[row, 8:], id="load_right")
        lm.store(out[row, :8], left, id="store_left")
        lm.store(out[row, 8:], right, id="store_right")
'''


class StaticOutputPartitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def document(self):
        return frontend.parse(partition_source()).document

    def refusal(self, document, code, text=""):
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.accepted)
        findings = [f for f in assessment.findings if f.code == code]
        self.assertTrue(findings, [(f.code, f.message) for f in assessment.findings])
        self.assertIn(text, findings[0].message)

    def test_partitioned_stores_preserve_all_emitted_effects_and_dependencies(self):
        for target in ("xcore1002", "gfx938", "gfx1151", "sm_100a"):
            with self.subTest(target=target):
                document = frontend.parse(partition_source(target)).document
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                source = self.compiler.lower(assessment).source
                self.assertEqual(source.count("tl.store("), 2)
                self.assertIn("# CAKE_OP:store_left", source)
                self.assertIn("# CAKE_OP:store_right", source)
                self.assertIn("store_left", document["operations"][-1]["depends_on"])

    def test_complete_tiled_program_axis_has_injective_ownership(self):
        document = frontend.parse(partition_source(row_tile=2)).document
        result = self.compiler.assess(document)
        self.assertTrue(result.lowering_eligible, result.findings)

    def test_two_static_dimensions_prove_rectangular_union(self):
        document = self.document()
        for buffer in document["buffers"]:
            buffer["shape"].append(4)
        for access in document["access_maps"]:
            access["indices"].append(dict(source="dimension", dimension=2))
        self.assertIsNone(partition_refusal(Schedule.from_dict(document), "out"))

    def test_overlap_is_refused_by_partition_proof(self):
        document = self.document()
        access = next(a for a in document["access_maps"] if a["operation"] == "store_right")
        access["indices"][1]["offset"] = 4
        self.refusal(document, "OUTPUT_PARTITION_OVERLAP")

    def test_hole_is_refused_by_partition_proof(self):
        document = self.document()
        access = next(a for a in document["access_maps"] if a["operation"] == "store_right")
        access["indices"][1]["offset"] = 12
        self.refusal(document, "OUTPUT_PARTITION_COVERAGE", "12 of 16")

    def test_out_of_bounds_is_refused_by_partition_proof(self):
        document = self.document()
        access = next(a for a in document["access_maps"] if a["operation"] == "store_right")
        access["indices"][1]["extent"] = 16
        self.refusal(document, "OUTPUT_PARTITION_BOUNDS")

    def test_an_aliasing_output_stays_outside_the_subset(self):
        document = self.document()
        output = next(b for b in document["buffers"] if b["name"] == "out")
        output["allocation"] = "shared_owner"
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "nonaliasing")

    def test_two_roles_are_not_one_writer(self):
        document = self.document()
        document["roles"].append(dict(name="other", execution_groups=[1]))
        document["operations"][-1]["role"] = "other"
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "one role")

    def test_output_readback_requires_all_producers_to_execute(self):
        document = self.document()
        # The old single-producer map overwrote store_left with store_right,
        # hiding the cross-role hazard in this invalid readback Schedule.
        document["roles"].append(dict(name="other", execution_groups=[1]))
        document["operations"][-2]["role"] = "other"
        read = deepcopy(document["operations"][0])
        read.update(id="readback", reads=["out"], writes=["loaded_output"])
        document["operations"].append(read)
        value = deepcopy(next(b for b in document["buffers"] if b["name"] == "left"))
        value["name"] = "loaded_output"
        document["buffers"].append(value)
        access = deepcopy(document["access_maps"][0])
        access.update(operation="readback", buffer="out")
        document["access_maps"].append(access)
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "readback")
        self.refusal(document, "OP_CROSS_ROLE_RACE", "store_left")

    def test_changed_program_owner_is_refused(self):
        document = self.document()
        document["program_map"]["axes"].append(dict(name="extra", axis=1, buffer="x", dimension=0, tile=1))
        self.refusal(document, "OUTPUT_PARTITION_OWNERSHIP", "exactly once")

    def test_duplicate_program_coordinates_are_refused(self):
        document = self.document()
        for access in document["access_maps"]:
            if access["buffer"] == "out":
                access["indices"][1] = dict(source="program", name="row")
        self.refusal(document, "OUTPUT_PARTITION_OWNERSHIP")

    def test_different_store_mapping_is_refused(self):
        document = self.document()
        for buffer in document["buffers"]:
            if buffer["name"] in ("x", "out"):
                buffer["shape"] = [16, 16]
        access = next(a for a in document["access_maps"] if a["operation"] == "store_right")
        access["indices"] = [dict(source="dimension", dimension=0, offset=8), dict(source="program", name="row")]
        self.refusal(document, "OUTPUT_PARTITION_OWNERSHIP", "identical")

    def test_partial_program_tile_and_persistent_walk_stay_unproven(self):
        document = frontend.parse(partition_source(row_tile=4).replace("(8, 16)", "(7, 16)")).document
        self.refusal(document, "OUTPUT_PARTITION_OWNERSHIP", "masked tail")
        document = self.document()
        document["program_map"]["persistent"] = True
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "nonpersistent")

    def test_loop_and_mutable_state_stay_outside_subset(self):
        document = self.document()
        document["tile_loops"] = [dict(name="column_loop", iterator="c", buffer="x", dimension=1, tile=8,
            body=["load_left", "store_left"], range_options=dict(num_stages=1,
                loop_unroll_factor=1, flatten=False, warp_specialize=False,
                disallow_acc_multi_buffer=False, disable_licm=False))]
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "loops")
        document = self.document()
        next(b for b in document["buffers"] if b["name"] == "x")["mode"] = "state"
        self.refusal(document, "BUFFER_MULTIPLE_WRITERS", "state")


if __name__ == "__main__":
    unittest.main()
