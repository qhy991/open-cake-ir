from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.contracts._parent_validator_fixture import write_parent_validator_fixture


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.review_aka_expressibility import materialize  # noqa: E402
from tools.run_aka_expressibility_codex import (  # noqa: E402
    MODEL,
    REASONING_EFFORT,
    RunnerError,
    run_queue,
)


class AkaExpressibilityCodexRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_temporary = tempfile.TemporaryDirectory()
        self.work_temporary = tempfile.TemporaryDirectory()
        self.parent_validator = write_parent_validator_fixture(
            Path(self.work_temporary.name) / "parent-validator-fixture.py"
        )
        self.source_root = Path(self.source_temporary.name) / "source"
        self.dataset = self.source_root / "cuda_kernel_dataset_test"
        self.work_root = Path(self.work_temporary.name) / "work"
        subprocess.run(["git", "init", "-q", str(self.source_root)], check=True)
        subprocess.run(
            ["git", "-C", str(self.source_root), "config", "user.name", "Runner Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.source_root),
                "config",
                "user.email",
                "runner@test.invalid",
            ],
            check=True,
        )
        shard = self.dataset / "categories/movement/copy/analysis.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text(
            json.dumps(
                {
                    "instruction": "Review one bounded copy.",
                    "input": "__global__ void copy(float *out) { out[threadIdx.x] = 0; }",
                    "reasoning": "Visible-source reasoning only.",
                    "output": "One bounded review.",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(self.source_root), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.source_root), "commit", "-q", "-m", "freeze"],
            check=True,
        )
        self.revision = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        self.fake_codex = Path(self.work_temporary.name) / "fake-codex"
        self.fake_codex.write_text(
            """#!/usr/bin/env python3
import json
from pathlib import Path
import sys
if '--version' in sys.argv:
    print('codex-cli 0.test')
    raise SystemExit(0)
arguments = sys.argv[1:]
output = Path(arguments[arguments.index('--output-last-message') + 1])
review = json.loads(Path('review.template.json').read_text())
review['parent_contract'] = {
    'status': 'missing',
    'missing_facts': ['The launch and independent reference contracts are absent.'],
    'completion_path': None,
}
output.write_text(json.dumps(review) + '\\n')
print(json.dumps({'type': 'turn.completed', 'argv': arguments}))
""",
            encoding="utf-8",
        )
        self.fake_codex.chmod(
            self.fake_codex.stat().st_mode | stat.S_IXUSR
        )
        checker_commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        self.checker_patch = mock.patch(
            "tools.review_aka_expressibility._checker_identity",
            return_value={
                "git_commit": checker_commit,
                "paths": [
                    "tools/review_aka_expressibility.py",
                    "tools/run_aka_expressibility_codex.py",
                    "tools/audit_aka_corpus.py",
                ],
            },
        )
        self.checker_patch.start()

    def tearDown(self) -> None:
        self.checker_patch.stop()
        self.source_temporary.cleanup()
        self.work_temporary.cleanup()

    def run_campaign(self, *, limit: int = 1) -> dict[str, object]:
        return run_queue(
            work_root=self.work_root,
            dataset_root=self.dataset,
            source_revision=self.revision,
            record_format="aka_v1_operator_sft",
            dataset_label="test",
            limit=limit,
            timeout_seconds=60,
            codex_bin=self.fake_codex,
                   parent_validator=self.parent_validator,
        )

    def test_fixed_sol_max_treatment_runs_one_case_and_checks_it(self) -> None:
        result = self.run_campaign()

        self.assertEqual(result["runner"]["model"], MODEL)
        self.assertEqual(result["runner"]["reasoning_effort"], REASONING_EFFORT)
        self.assertEqual(result["processed"][0]["primary_class"], "contract_missing")
        self.assertEqual(result["status"]["checked"], 1)
        case_root = self.work_root / "cases/case-000001"
        self.assertTrue((case_root / "review.json").is_file())
        self.assertTrue((case_root / "checked.json").is_file())
        event = json.loads(
            (self.work_root / "runs/case-000001/codex.events.jsonl").read_text()
        )
        argv = event["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="max"', argv)
        self.assertIn("multi_agent", argv)
        self.assertIn("--ephemeral", argv)
        output_schema = json.loads(
            (
                self.work_root
                / "runs/case-000001/review.schema.json"
            ).read_text(encoding="utf-8")
        )
        source_properties = output_schema["properties"]["source_ref"]["properties"]
        template_source = json.loads(
            (case_root / "review.template.json").read_text(encoding="utf-8")
        )["source_ref"]
        self.assertEqual(
            {field: schema["const"] for field, schema in source_properties.items()},
            template_source,
        )

    def test_runner_refuses_rebinding_the_parent_validator(self) -> None:
        self.run_campaign()
        self.parent_validator = write_parent_validator_fixture(
            Path(self.work_temporary.name) / "different-validator.py"
        )
        with self.assertRaisesRegex(RunnerError, "different parent validator"):
            self.run_campaign()

    def test_runner_refuses_to_adopt_a_preexisting_review_queue(self) -> None:
        self.run_campaign(limit=1)
        second_root = Path(self.work_temporary.name) / "second-work"
        self.work_root = second_root
        # Initialize and materialize through a zero-work call is deliberately unsupported;
        # create the queue with a normal run that is stopped before Codex by pre-materializing.
        from tools.review_aka_expressibility import initialize

        initialize(
            dataset_root=self.dataset,
            work_root=self.work_root,
            source_revision=self.revision,
            record_format="aka_v1_operator_sft",
            parent_validator=self.parent_validator,
        )
        materialize(self.work_root, "case-000001")
        case_root = self.work_root / "cases/case-000001"
        (case_root / "review.json").write_bytes(
            (case_root / "review.template.json").read_bytes()
        )

        with self.assertRaisesRegex(RunnerError, "fresh review queue"):
            self.run_campaign()


if __name__ == "__main__":
    unittest.main()
