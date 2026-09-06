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
        cls.revision_id = cls.compiler.assess_file(ROOT / "corpus/schedules/fma-b8-smoke.json").compiler_revision_id

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
        return {"schema_version": 1, "model_id": "synthetic-api-contract-fixture",
                "compiler_revision_id": self.revision_id, "target": "sm_100a",
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

    def test_default_profile_and_qualified_rank_are_unchanged(self):
        schedule = self.schedule("fma-b8-smoke.json")
        base = self.profile(schedule)
        self.assertNotIn("empirical_cost", base)
        modeled = self.profile(schedule, EmpiricalCostModel(self.model_document()))
        del modeled["empirical_cost"]
        self.assertEqual(modeled, base)
        assessment = self.compiler.assess(schedule)
        self.assertEqual(self.compiler.rank([assessment]), ((), (assessment.schedule_id,)))

    def test_wrong_revision_and_target_abstain(self):
        for field, value in [("compiler_revision_id", "other-revision"), ("target", "different-target")]:
            model = EmpiricalCostModel(self.model_document())
            arguments = {"compiler_revision_id": self.revision_id, "target": "sm_100a", field: value}
            cost = model.estimate(self.schedule("fma-b8-smoke.json"), **arguments)
            self.assertFalse(cost["covered"])
            self.assertIsNone(cost["predicted_kernel_us"])

    def test_observed_backend_version_drift_abstains(self):
        model = EmpiricalCostModel(self.model_document())
        cost = model.estimate(self.schedule("fma-b8-smoke.json"), compiler_revision_id=self.revision_id,
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
        document = self.model_document()
        document["curves"].append(copy.deepcopy(document["curves"][0]))
        cost = self.profile(self.schedule("fma-b8-smoke.json"), EmpiricalCostModel(document))["empirical_cost"]
        self.assertFalse(cost["covered"])
        self.assertIn("ambiguous", cost["reason"])

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
            from open_cake_ir.lab import qsa_compiler_feedback
            assessment = self.compiler.assess(self.schedule("fma-b8-smoke.json"))
            feedback = qsa_compiler_feedback(assessment, static_profile=profile)
            self.assertIn("empirical_cost", str(feedback))


if __name__ == "__main__":unittest.main()
