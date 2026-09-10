from __future__ import annotations

import copy
import importlib.abc
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.compiler import Compiler, CompilerError, EmpiricalCostModel


class EmpiricalCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = cls.compiler.assess_file(ROOT / "corpus/schedules/fma-b8-smoke.json")
        cls.revision_id = assessment.compiler_revision_id
        cls.revision_sha256 = assessment.compiler_revision_sha256

    def schedule(self, filename, extent=None):
        document = json.loads((ROOT / "corpus/schedules" / filename).read_text())
        names = {"a", "b", "c", "y"} if filename.startswith("fma-") else {"a", "c"}
        if extent is not None:
            for buffer in document["buffers"]:
                if buffer["name"] in names:buffer["shape"][0] = extent
        return document

    def model_document(self):
        curves = []
        for name, bindings, extents in [
            ("fma-b8-smoke.json", ["a", "b", "c", "y"], [8, 16, 32]),
            ("gemm-bias-b1-smoke.json", ["a", "c"], [512, 1024, 2048]),
        ]:
            curves.append({"template": self.schedule(name),
                           "varying_dimensions": [{"buffer": buffer, "dimension": 0} for buffer in bindings],
                           "extent_multiple": extents[0],
                           "points": [{"extent": extent, "kernel_us": duration} for extent, duration in zip(extents, [10, 18, 34])],
                           "relative_error_envelope": .1})
        return {"schema_version": 2, "model_id": "synthetic-api-contract-fixture",
                "compiler_revision_id": self.revision_id, "compiler_revision_sha256": self.revision_sha256, "target": "sm_100a",
                "context": {"timer": "synthetic; no measurement", "cache_protocol": "synthetic",
                            "runtime": {"compiler_version": "synthetic"}, "input_scope": "unit fixture only"},
                "reported_evidence": {"kind": "synthetic; no calibration qualification"}, "curves": curves}

    def profile(self, schedule, model=None):
        assessment = self.compiler.assess(schedule)
        self.assertTrue(assessment.lowering_eligible)
        return self.compiler.profile(assessment, cost_model=model).as_dict()

    def test_one_model_representation_handles_fma_and_gemm(self):
        document = self.model_document()
        # Interior aligned points exercise interpolation, rather than lookup replay.
        for curve in document["curves"]:curve["extent_multiple"] //= 2
        model = EmpiricalCostModel(document)
        for name, extent in [("fma-b8-smoke.json", 12), ("gemm-bias-b1-smoke.json", 768)]:
            cost = self.profile(self.schedule(name, extent), model)["empirical_cost"]
            self.assertTrue(cost["covered"])
            self.assertEqual(cost["predicted_kernel_us"], 14)
            self.assertEqual(cost["empirical_range_us"], [12.6, 15.400000000000002])

    def test_single_point_covers_only_its_exact_extent(self):
        document = self.model_document()
        for curve in document["curves"]:
            # The template's varying extent remains a placeholder, not a point.
            curve["points"] = curve["points"][1:2]
        model = EmpiricalCostModel(document)
        for name, extent, multiple in [("fma-b8-smoke.json", 16, 8), ("gemm-bias-b1-smoke.json", 1024, 512)]:
            with self.subTest(schedule=name):
                schedule = self.schedule(name, extent)
                schedule["schedule_id"] = "display-name-only"
                base = self.profile(schedule)
                modeled = self.profile(schedule, model)
                cost = modeled.pop("empirical_cost")
                self.assertEqual(modeled, base)
                self.assertTrue(cost["covered"])
                self.assertEqual(cost["predicted_kernel_us"], 18)
                self.assertEqual(cost["empirical_range_us"], [16.2, 19.8])
                for outside in [extent - multiple, extent - multiple // 2, extent + multiple]:
                    cost = self.profile(self.schedule(name, outside), model)["empirical_cost"]
                    self.assertFalse(cost["covered"])
                    self.assertIsNone(cost["predicted_kernel_us"])

    def test_default_profile_and_qualified_rank_are_unchanged(self):
        schedule = self.schedule("fma-b8-smoke.json")
        base = self.profile(schedule)
        self.assertNotIn("empirical_cost", base)
        modeled = self.profile(schedule, EmpiricalCostModel(self.model_document()))
        del modeled["empirical_cost"]
        self.assertEqual(modeled, base)
        assessment = self.compiler.assess(schedule)
        self.assertEqual(self.compiler.rank([assessment]), ((), (assessment.schedule_id,)))

    def test_mma_canonical_ranges_match_but_partial_work_never_inherits_a_full_curve(self):
        query = self.schedule('gemm-bias-b1-smoke.json')
        op = next(op for op in query['operations'] if op['kind'] == 'mma')
        k = op['parameters']['tile_shape'][2]
        model = EmpiricalCostModel(self.model_document())
        identity = dict(compiler_revision_id=self.revision_id,
                        compiler_revision_sha256=self.revision_sha256, target='sm_100a')
        original = model.estimate(query, **identity)
        self.assertTrue(original['covered'])
        op['parameters']['k_ranges'] = [[0, k // 2], [k // 2, k]]
        self.assertEqual(model.estimate(query, **identity), original)
        # The public model boundary must also normalize its stored template.
        model_doc = self.model_document(); model_doc['curves'][1]['template'] = copy.deepcopy(query)
        canonical_model = EmpiricalCostModel(model_doc)
        self.assertEqual(canonical_model.estimate(self.schedule('gemm-bias-b1-smoke.json'), **identity), original)
        op['parameters']['k_ranges'] = [[0, k // 2]]
        partial = model.estimate(query, **identity)
        self.assertFalse(partial['covered'])
        self.assertIsNone(partial['predicted_kernel_us'])

    def test_wrong_revision_and_target_abstain(self):
        for field, value in [("compiler_revision_id", "other-revision"), ("compiler_revision_sha256", "0" * 64), ("target", "different-target")]:
            model = EmpiricalCostModel(self.model_document())
            arguments = {"compiler_revision_id": self.revision_id, "compiler_revision_sha256": self.revision_sha256, "target": "sm_100a", field: value}
            cost = model.estimate(self.schedule("fma-b8-smoke.json"), **arguments)
            self.assertFalse(cost["covered"])
            self.assertIsNone(cost["predicted_kernel_us"])

    def test_observed_backend_version_drift_abstains(self):
        model = EmpiricalCostModel(self.model_document())
        cost = model.estimate(self.schedule("fma-b8-smoke.json"), compiler_revision_id=self.revision_id, compiler_revision_sha256=self.revision_sha256,
                              target="sm_100a", compiled_compiler_version="other-toolchain")
        self.assertFalse(cost["covered"])
        self.assertIn("compiler version", cost["reason"])

    def test_extent_alignment_extrapolation_and_template_drift_abstain(self):
        model = EmpiricalCostModel(self.model_document())
        for extent in [4, 12, 40]:
            cost = self.profile(self.schedule("fma-b8-smoke.json", extent), model)["empirical_cost"]
            self.assertFalse(cost["covered"])
        changed = self.schedule("fma-b8-smoke.json")
        changed["residency"]["registers_per_thread"] = 128
        self.assertFalse(self.profile(changed, model)["empirical_cost"]["covered"])
        changed = self.schedule("fma-b8-smoke.json")
        changed["schedule_id"] = "display-name-only"
        self.assertTrue(self.profile(changed, model)["empirical_cost"]["covered"])

    def test_overlapping_curves_abstain_without_picking_an_id(self):
        for point_count in (1, 3):
            document = self.model_document()
            document["curves"][0]["points"] = document["curves"][0]["points"][:point_count]
            document["curves"].append(copy.deepcopy(document["curves"][0]))
            with self.subTest(point_count=point_count):
                cost = self.profile(self.schedule("fma-b8-smoke.json"), EmpiricalCostModel(document))["empirical_cost"]
                self.assertFalse(cost["covered"])
                self.assertIn("ambiguous", cost["reason"])

    def test_equal_numbers_with_different_json_types_do_not_match(self):
        model = EmpiricalCostModel(self.model_document())
        for version in (True, 1.0):
            schedule = self.schedule("fma-b8-smoke.json")
            schedule["schema_version"] = version
            with self.subTest(version=version):
                cost = model.estimate(schedule, compiler_revision_id=self.revision_id,
                                      compiler_revision_sha256=self.revision_sha256, target="sm_100a")
                self.assertFalse(cost["covered"])
                self.assertIsNone(cost["predicted_kernel_us"])
                self.assertEqual(cost["reason"], "no curve covers this exact Schedule and extent")

    def test_assessment_tampering_cannot_reach_cost_prediction(self):
        assessment = self.compiler.assess(self.schedule("fma-b8-smoke.json"))
        with self.assertRaises(CompilerError):
            self.compiler.profile(replace(assessment, schedule_id="forged"), cost_model=EmpiricalCostModel(self.model_document()))

    def test_model_is_detached_and_immutable(self):
        document = self.model_document()
        model = EmpiricalCostModel(document)
        document["curves"][0]["points"][0]["kernel_us"] = 999
        document["context"]["timer"] = "changed"
        result = self.profile(self.schedule("fma-b8-smoke.json"), model)["empirical_cost"]
        self.assertEqual(result["predicted_kernel_us"], 10)
        self.assertEqual(result["context"]["timer"], "synthetic; no measurement")
        with self.assertRaises(FrozenInstanceError):model.model_id = "changed"

    def test_malformed_model_values_are_rejected(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(schema_version=1),
            lambda d: d.update(compiler_revision_sha256="not-an-identity"),
            lambda d: d.update(context={}),
            lambda d: d.update(reported_evidence={}),
            lambda d: d["curves"][0].update(extent_multiple=True),
            lambda d: d["curves"][0].update(points=[]),
            lambda d: d["curves"][0]["points"][0].update(kernel_us=float("nan")),
            lambda d: d["curves"][0]["points"][0].update(kernel_us=10**400),
            lambda d: d["curves"][0]["points"][0].update(kernel_us=True),
            lambda d: d["curves"][0]["points"][1].update(extent=8),
            lambda d: d["curves"][0]["varying_dimensions"].append({"buffer": "a", "dimension": 0}),
            lambda d: d["curves"][0]["varying_dimensions"][0].update(dimension=-1),
            lambda d: d["curves"][0]["varying_dimensions"][0].update(buffer="absent"),
            lambda d: d["curves"][0].update(relative_error_envelope=1),
        ]
        for mutation in mutations:
            document = self.model_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):EmpiricalCostModel(document)

    def test_cli_json_and_python_agent_feedback_accept_explicit_model_without_gpu(self):
        document = self.model_document()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            model_path.write_text(json.dumps(document))
            guard = """import importlib.abc,runpy,sys
class NoGPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','triton','cuda','cupti','nvidia'}:raise AssertionError(fullname)
sys.meta_path.insert(0,NoGPU())
script=sys.argv.pop(1)
runpy.run_path(script,run_name='__main__')
"""
            args = [sys.executable, "-c", guard, str(ROOT / "tools/report_schedule_profile.py"), "--revision", str(ROOT / "compiler/revision.json"), "--json", "--cost-model", str(model_path), str(ROOT / "corpus/schedules/fma-b8-smoke.json")]
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)
            profile = json.loads(result.stdout)["rows"][0]["profile"]
            self.assertEqual(profile["empirical_cost"]["predicted_kernel_us"], 10)
            from open_cake_ir.tasks.qsa.feedback import qsa_compiler_feedback
            assessment = self.compiler.assess(self.schedule("fma-b8-smoke.json"))
            feedback = qsa_compiler_feedback(assessment, static_profile=profile)
            self.assertIn("empirical_cost", str(feedback))

    def test_metal_report_abstains_without_fabricated_nvidia_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            model.write_text(json.dumps(self.model_document()))
            command = [sys.executable, str(ROOT / "tools/report_schedule_profile.py"), "--revision", str(ROOT / "compiler/revision.json"), "--cost-model", str(model), str(ROOT / "corpus/schedules/metal-elementwise-odd.json")]
            table = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("est kernel us", table.stdout)
            data = subprocess.run([*command, "--json"], cwd=ROOT, capture_output=True, text=True, check=True)
            profile = json.loads(data.stdout)["rows"][0]["profile"]
            self.assertEqual(profile["ncu_metrics"], [])
            self.assertFalse(profile["empirical_cost"]["covered"])
            self.assertIsNone(profile["empirical_cost"]["predicted_kernel_us"])


if __name__ == "__main__":unittest.main()
