from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/gpu/state_store_b200_correctness"
sys.path.insert(0, str(ROOT / "src"))


def _generator():
    path = EXAMPLE / "prepare_candidate.py"
    spec = importlib.util.spec_from_file_location("state_store_b200_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load state-store B200 generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StateStoreB200CorrectnessContractTests(unittest.TestCase):
    def test_create_only_bundle_uses_the_released_compiler_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            preflight = _generator().prepare(output)
            source = (output / "candidate/kernel.py").read_text(encoding="utf-8")
            provenance = json.loads(
                (output / "candidate/provenance.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                preflight["compiler_revision_id"], "open-cake-ir-sm100a-v40"
            )
            self.assertTrue(preflight["positive"]["accepted"])
            self.assertTrue(preflight["positive"]["lowering_eligible"])
            self.assertEqual(
                preflight["negative"]["state-store-b8-smoke-owner-drift.json"][
                    "decisive_findings"
                ],
                [("STATE_STORE_PROGRAM_OWNER", "access_maps[2].indices[0]")],
            )
            self.assertEqual(
                preflight["negative"]["state-store-b8-smoke-axis-drift.json"][
                    "decisive_findings"
                ],
                [("STATE_STORE_PROGRAM_AXIS_COVERAGE", "program_map.axes[1]")],
            )
            compile(source, "<generated-state-store>", "exec")
            self.assertIn(
                "tl.store(\n        state + batch * D_STATE_1 + state_d1_offsets,",
                source,
            )
            self.assertNotIn("torch.empty((8, 128)", source)
            self.assertIn("return out", source)
            self.assertEqual(provenance["shape"], [8, 128])
            self.assertIn("no arbitrary-n", provenance["claim_boundary"])

    def test_frozen_task_is_one_broker_correctness_stage(self) -> None:
        task = json.loads((EXAMPLE / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(
            task["task_id"],
            "open-cake-state-store-b8x128-b200-correctness-v4",
        )
        self.assertIn("Broker-result-readable successor", task["description"])
        self.assertIn("not a retry", task["description"])
        self.assertEqual(len(task["stages"]), 1)
        stage = task["stages"][0]
        self.assertEqual(stage["kind"], "correctness")
        self.assertEqual(stage.get("execution", "broker"), "broker")
        self.assertEqual(stage["resources"]["mode"], "shared")
        self.assertEqual(stage["resources"]["gpu_count"], 1)
        self.assertNotIn("benchmark", json.dumps(task))
        self.assertNotIn("profile", json.dumps(task))

    def test_catalog_is_fixed_to_the_declared_b200_endpoint(self) -> None:
        catalog = json.loads((EXAMPLE / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(len(catalog["nodes"]), 1)
        node = catalog["nodes"][0]
        self.assertEqual(node["ssh"], "verda-b200x4")
        self.assertEqual(
            node["socket"],
            "/tmp/kernelinfra-open-cake-state-store-b200-v4.sock",
        )
        self.assertIn("state-store-b200-v4/gpu-infra", node["kernelctl"])
        self.assertEqual(set(node["capabilities"]), {"b200", "cuda", "sm100"})

    def test_judge_is_syntax_valid_and_declares_four_full_outputs(self) -> None:
        source = (EXAMPLE / "evaluate.py").read_text(encoding="utf-8")
        compile(source, str(EXAMPLE / "evaluate.py"), "exec")
        self.assertIn('"state_before"', source)
        self.assertIn('"update_before"', source)
        self.assertIn('"actual_state_after"', source)
        self.assertIn("state_mismatch_count", source)
        self.assertIn("update_mismatch_count", source)
        self.assertIn("state_data_ptr_unchanged", source)
        self.assertIn("wrapper_returned_empty_tuple", source)


if __name__ == "__main__":
    unittest.main()
