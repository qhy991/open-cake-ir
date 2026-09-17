from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/gpu/aka_qualified_ir_copy4_b200_canary"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AkaIrLabAdmissionTests(unittest.TestCase):
    def test_static_survivor_gate_preserves_evidence_boundaries(self) -> None:
        tool = _module(
            ROOT / "tools/prepare_aka_ir_lab_admission.py", "aka_lab_admission"
        )
        accepted = {
            "schema": "open-cake.aka-qualified-ir-result.v1",
            "status": "reviewed",
            "owner": "schedule",
            "current_ir_expressibility": "expressible",
            "verification": {
                "status": "accepted",
                "parse": {"status": "passed"},
                "validate": {"status": "passed"},
                "lower": {"status": "passed"},
            },
            "eligibility": {
                "gpu": {"eligible": True},
                "training": {"eligible": False},
            },
        }
        self.assertTrue(tool._accepted_schedule(accepted))
        for path, replacement in (
            (("owner",), "ir_gap"),
            (("current_ir_expressibility",), "not_expressible"),
            (("verification", "lower", "status"), "failed"),
            (("eligibility", "gpu", "eligible"), False),
            (("eligibility", "training", "eligible"), True),
        ):
            candidate = json.loads(json.dumps(accepted))
            owner = candidate
            for name in path[:-1]:
                owner = owner[name]
            owner[path[-1]] = replacement
            self.assertFalse(tool._accepted_schedule(candidate), path)

    def test_canary_preserves_virtualenv_entry_and_stage_modes(self) -> None:
        prepare = _module(EXAMPLE / "prepare.py", "aka_copy4_prepare")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = root / "admission"
            case = admission / "cases" / prepare.CASE_ID / "candidate"
            case.mkdir(parents=True)
            (case / "kernel.py").write_text(
                "def copy4_contiguous_fp32_n1024(src, out=None):\n    return out\n",
                encoding="utf-8",
            )
            (case / "provenance.json").write_text("{}\n", encoding="utf-8")
            (admission / "admission-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "open-cake.aka-ir-lab-admission.v1",
                        "canary_case_id": prepare.CASE_ID,
                        "policy": {"max_in_flight_after_canary": 5},
                        "entries": [
                            {
                                "case_id": prepare.CASE_ID,
                                "canary": True,
                                "entry_point": "copy4_contiguous_fp32_n1024",
                                "target": "sm_100a",
                                "workload_id": "n1024",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "python-venv-entry"
            runtime.symlink_to(Path(sys.executable))
            judge_cwd = root / "judge"
            judge_cwd.mkdir()
            output = root / "output"
            result = prepare.prepare(
                admission_root=admission,
                output_root=output,
                python=runtime,
                judge_cwd=judge_cwd,
            )
            task = json.loads((output / "task.json").read_text(encoding="utf-8"))
            self.assertEqual(task["task_id"], prepare.TASK_ID)
            self.assertEqual(task["stages"][0]["judge"]["command"][0], str(runtime))
            self.assertEqual(
                [(stage["kind"], stage["resources"]["mode"]) for stage in task["stages"]],
                [
                    ("correctness", "shared"),
                    ("sanitize", "exclusive"),
                    ("sanitize", "exclusive"),
                ],
            )
            self.assertEqual(result["max_in_flight_after_canary"], 5)

    def test_canary_judge_is_torch_only_and_checks_complete_outputs(self) -> None:
        source = (EXAMPLE / "evaluate.py").read_text(encoding="utf-8")
        compile(source, str(EXAMPLE / "evaluate.py"), "exec")
        self.assertNotIn("numpy", source)
        self.assertIn('"source_before"', source)
        self.assertIn('"source_after"', source)
        self.assertIn('"expected_output"', source)
        self.assertIn('"actual_output"', source)
        self.assertIn("output_mismatch_count", source)
        self.assertIn("source_mismatch_count", source)
        self.assertIn('tool="memcheck"', source)
        self.assertIn('tool="racecheck"', source)


if __name__ == "__main__":
    unittest.main()
