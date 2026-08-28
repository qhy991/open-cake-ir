from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.plan_aka_expressibility_queue import (  # noqa: E402
    PlanError,
    build_plan,
    selected_entries,
    write_plan,
)


class AkaExpressibilityPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "AKA"
        self.dataset = self.source / "datasets/curated/cuda_kernel_dataset_v1"
        self.campaign = root / "campaign"
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.name", "Plan Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.source),
                "config",
                "user.email",
                "plan@test.invalid",
            ],
            check=True,
        )
        rows = (
            (
                "categories/data_movement_and_layout/copy_tile/generation.jsonl",
                {
                    "instruction": "Generate one copy kernel.",
                    "input": "",
                    "reasoning": "One visible kernel.",
                    "output": "__global__ void copy(float *x) { x[threadIdx.x] = 0; }",
                },
            ),
            (
                "categories/systems_and_frameworks/communication/analysis.jsonl",
                {
                    "instruction": "Analyze a two-kernel program.",
                    "input": (
                        "__global__ void a(float *x) { x[0] = 0; }\n"
                        "__global__ void b(float *x) { x[1] = 0; }"
                    ),
                    "reasoning": "Two visible kernels.",
                    "output": "Program analysis.",
                },
            ),
        )
        for relative, value in rows:
            path = self.dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.source), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "commit", "-q", "-m", "freeze"],
            check=True,
        )
        self.revision = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        cases = (
            (
                "data_movement_and_layout__copy_tile__generation__l000001_b200_v1",
                "end-to-end/copy/tests/result.json",
                "qualified",
                "valid",
            ),
            (
                "systems_and_frameworks__communication__analysis__l000001_b200_v1",
                "end-to-end/communication/tests/result.json",
                "invalid",
                "invalid",
            ),
        )
        ledger = []
        for case_id, relative, parent_status, validity in cases:
            result = self.campaign / relative
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(
                json.dumps(
                    {
                        "schema": "aka.mechanism-augmentation-result.v1",
                        "parent_status": parent_status,
                        "validity": validity,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ledger.append(
                {
                    "case_id": case_id,
                    "result_path": relative,
                    "validity": validity,
                }
            )
        (self.campaign / "final-verified-ledger.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_plan_separates_historical_parent_and_ir_claim_qualification(self) -> None:
        plan = build_plan(
            dataset_root=self.dataset,
            source_revision=self.revision,
            campaign_root=self.campaign,
        )

        self.assertEqual(plan["counts"]["parent_qualified_terminal_valid_schedule"], 1)
        self.assertEqual(plan["counts"]["parent_invalid"], 1)
        self.assertEqual(plan["claim_boundary"]["qualified_for_ir_claim"], 0)
        first = selected_entries(
            plan, ["parent_qualified_terminal_valid_schedule"]
        )[0]
        self.assertEqual(first["case_id"], "case-000001")
        self.assertFalse(first["qualified_for_ir_claim"])
        self.assertEqual(first["next_gate"], "canonical_complete_kernel_parent_bridge")

    def test_plan_is_create_only_and_rejects_ledger_result_drift(self) -> None:
        plan = build_plan(
            dataset_root=self.dataset,
            source_revision=self.revision,
            campaign_root=self.campaign,
        )
        output = Path(self.temporary.name) / "plan.json"
        write_plan(output, plan)
        with self.assertRaises(PlanError):
            write_plan(output, plan)

        ledger = self.campaign / "final-verified-ledger.jsonl"
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        rows[0]["validity"] = "unknown"
        ledger.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with self.assertRaisesRegex(PlanError, "validity differs"):
            build_plan(
                dataset_root=self.dataset,
                source_revision=self.revision,
                campaign_root=self.campaign,
            )


if __name__ == "__main__":
    unittest.main()
