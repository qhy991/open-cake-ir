"""CPU-only synthetic artifact relations; none of these fixtures is a GPU result."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("empirical_fitter", ROOT / "tools/calibrate_empirical_cost.py")
instrument = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instrument)
from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.compiled_resources import CompiledResources


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


class FitterBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def fixture(self, root):
        run = root / "run"
        base = json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())
        cases, curves, rows = [], [], []
        collector = Path(instrument.__file__).read_bytes()
        assessment = self.compiler.assess(base)
        revision = assessment.compiler_revision_id
        revision_sha256 = assessment.compiler_revision_sha256
        for variant, cap in enumerate((64, 96, 128)):
            curve_id = f"synthetic-r{cap}"
            curves.append({"id": curve_id, "extent_multiple": 8, "varying_dimensions": [{"buffer": name, "dimension": 0} for name in ("a", "b", "c", "y")]})
            for extent, split in ((8, "fit"), (32, "fit"), (16, "calibration"), (24, "audit")):
                document = copy.deepcopy(base)
                document["residency"]["registers_per_thread"] = cap
                document["schedule_id"] = f"{curve_id}-x{extent}"
                for buffer in document["buffers"]:
                    if buffer["space"] == "global":buffer["shape"][0] = extent
                case = {"id": document["schedule_id"], "curve_id": curve_id, "extent": extent, "split": split, "schedule": f"schedules/{document['schedule_id']}.json", "family": "fma", "workload_id": f"synthetic-x{extent}"}
                cases.append(case)
                write(run / "candidate" / case["schedule"], document)
                assessment = self.compiler.assess(document)
                lowering = self.compiler.lower(assessment)
                # This synthetic ELF-marked blob tests identity checks, not execution.
                cubin = b"\x7fELF-SYNTHETIC-NOT-EXECUTABLE-" + case["id"].encode()
                resource = CompiledResources(lowering.source_sha256, sha256(cubin).hexdigest(), "sm_100a", lowering.toolchain_requirements["kernel_entry_point"], 128, 16, 0, 0, 0, 0, "synthetic", "synthetic")
                duration = 10 + variant + extent / 16
                row = {**case, "grid": list(lowering.toolchain_requirements["grid"]), "profile": {"compiled_resources": resource.as_dict()}, "correct": True, "inputs_unchanged": True, "quality_passed": True, "samples_us": [[duration] * 2 for _ in range(2)]}
                rows.append(row)
                for phase in ("correctness", "collection"):
                    path = run / "stages" / phase / f"{len(rows)-1:04d}"
                    write(path / "schedule.json", json.loads(assessment.schedule_bytes))
                    (path / "lowered.py").write_bytes(lowering.source.encode())
                    (path / "kernel.cubin").write_bytes(cubin)
        plan = {"state": "frozen", "collector_sha256": sha256(collector).hexdigest(), "compiler_revision_id": revision, "compiler_revision_sha256": revision_sha256, "model_id": "synthetic-boundary-test", "input_scope": "SYNTHETIC CPU CONTRACT TEST ONLY; no measured performance", "cases": cases, "curves": curves, "sampling": {"rounds": 2, "repetitions": 2, "l2_flush_bytes": 268435456}, "acceptance": {"maximum_cohort_cv": .05, "maximum_repeat_median_ratio": 1.05}, "model_acceptance": {"maximum_mape": .1, "maximum_relative_error": .2, "maximum_top2_regret_ratio": 1.05, "envelope_allowance": .05}}
        write(run / "candidate/plan.json", plan)
        stages = []
        for phase in ("correctness", "collection"):
            path = run / "stages" / phase
            (path / "collector.py").write_bytes(collector)
            write(path / "plan.json", plan)
            write(path / "observations.json", {"schema_version": 1, "runtime": {"compiler_version": "synthetic"}, "quality_passed": True, "rows": rows})
            write(path / "receipt.json", {"execution": "broker", "exit_code": 0, "judge_result_valid": True, "broker_job_id": "synthetic-not-an-actual-job"})
            stages.append({"id": phase, "status": "passed", "validity": "valid"})
        write(run / "result.json", {"outcome": "completed", "validity": "valid", "run_id": "SYNTHETIC-NOT-A-GPU-RUN", "stages": stages})
        for repetition in range(2):
            order = []
            for round_index in range(2):
                indices = [(round_index + repetition * 7 + offset) % len(rows) for offset in range(len(rows))]
                order.extend(indices[::-1] if repetition else indices)
            events = []
            for position, index in enumerate(order):
                row = rows[index]
                common = {"device": 0, "context": 1, "stream": 7}
                events.extend([
                    {"cat": "kernel", "name": "FillFunctor<unsigned char>", "ts": position * 100, "dur": 1, "args": common},
                    {"cat": "kernel", "name": row["profile"]["compiled_resources"]["entry_point"], "ts": position * 100 + 2, "dur": row["samples_us"][repetition][0], "args": {**common, "grid": row["grid"], "block": [128, 1, 1], "correlation": position + 1}},
                    {"cat": "cuda_driver", "name": "cuLaunchKernel", "args": {"correlation": position + 1}},
                ])
            path = run / "stages/collection"
            write(path / f"cupti-trace-{repetition}.json", {"traceEvents": events})
            write(path / f"launch-order-{repetition}.json", [rows[index]["id"] for index in order])
        return run

    def test_positive_complete_fitter_contract(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run = self.fixture(root)
            self.assertEqual(instrument._fit(run, root / "output"), 0)
            model = json.loads((root / "output/model.json").read_text())
            self.assertEqual(len(model["curves"]), 3)
            self.assertEqual(model["reported_evidence"]["validation"]["max_top2_regret_ratio"], 1)

    def test_changed_stage_templates_cannot_relabel_original_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            for path in (run / "stages/collection").glob("*/schedule.json"):
                value = json.loads(path.read_text())
                value["residency"]["registers_per_thread"] = 128
                write(path, value)
            with self.assertRaisesRegex(ValueError, "Schedule differs"):
                instrument._fit(run, root / "output")
            self.assertFalse((root / "output").exists())

    def test_missing_or_corrupt_artifacts_refuse_model_generation(self):
        for filename, action in [("kernel.cubin", "delete"), ("kernel.cubin", "corrupt"), ("lowered.py", "corrupt")]:
            with self.subTest(filename=filename, action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self.fixture(root)
                path = run / "stages/collection/0000" / filename
                if action == "delete":path.unlink()
                else:path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaises((ValueError, OSError)):instrument._fit(run, root / "output")
                self.assertFalse((root / "output").exists())

    def test_different_correctness_binary_with_updated_identity_still_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            path = run / "stages/correctness/0000/kernel.cubin"
            changed = path.read_bytes() + b"different-correctness-binary"
            path.write_bytes(changed)
            observation = run / "stages/correctness/observations.json"
            value = json.loads(observation.read_text())
            value["rows"][0]["profile"]["compiled_resources"]["cubin_sha256"] = sha256(changed).hexdigest()
            write(observation, value)
            with self.assertRaisesRegex(ValueError, "compiled artifacts differ"):
                instrument._fit(run, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":unittest.main()
