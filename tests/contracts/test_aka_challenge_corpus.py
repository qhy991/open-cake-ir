from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.audit_aka_corpus import (  # noqa: E402
    CorpusAuditError,
    build_summary,
    load_records,
    projected_case,
    verify_git_snapshot,
)


FIELDS = {
    "instruction": "Do the bounded CUDA task.",
    "input": "",
    "reasoning": "Visible-source reasoning only.",
    "output": "Finished output.",
}


class AkaChallengeCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name) / "cuda_kernel_dataset_v-test"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_record(
        self,
        category: str,
        operator: str,
        task: str,
        *,
        input_source: str = "",
        output_source: str = "Finished output.",
        extra: dict[str, object] | None = None,
    ) -> None:
        path = self.dataset / "categories" / category / operator / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = dict(FIELDS)
        record["input"] = input_source
        record["output"] = output_source
        record.update(extra or {})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def test_projection_preserves_locator_and_routes_scope_without_copying_views(self) -> None:
        single = "__global__ void copy(float *x) { x[threadIdx.x] = 0; }"
        multi = (
            "__global__ void first(float *x) { x[threadIdx.x] = 0; }\n"
            "__global__ void second(float *x) { x[threadIdx.x] = 1; }"
        )
        self.write_record("movement", "copy", "analysis", input_source=single)
        self.write_record("movement", "copy", "generation", output_source=multi)
        self.write_record(
            "framework",
            "wrapper",
            "debug",
            input_source="void launch() { cub::DeviceScan::InclusiveSum(); }",
        )

        records = load_records(self.dataset)
        self.assertEqual(
            [record.scope_signal for record in records],
            ["fragment_or_library", "single_kernel", "multi_kernel_or_launch"],
        )
        by_task = {record.task: record for record in records}
        self.assertEqual(by_task["generation"].primary_field, "output")
        self.assertEqual(by_task["debug"].relation, "repair")

        case = projected_case(
            by_task["analysis"], dataset_id=self.dataset.name, source_revision="abc123"
        )
        self.assertEqual(
            case["source_ref"],
            {
                "dataset_id": self.dataset.name,
                "revision": "abc123",
                "path": "categories/movement/copy/analysis.jsonl",
                "line": 1,
                "primary_field": "input",
            },
        )
        self.assertEqual(case["artifact_fields"], {"source": "input"})
        debug_case = projected_case(
            by_task["debug"], dataset_id=self.dataset.name, source_revision="abc123"
        )
        self.assertEqual(
            debug_case["artifact_fields"], {"broken": "input", "repaired": "output"}
        )
        self.assertEqual(case["complete_parent_expressibility"], "unknown")
        self.assertEqual(case["delta_expressibility"], "unknown")
        self.assertNotIn("instruction", case)
        self.assertNotIn("input", case)
        self.assertNotIn("output", case)

    def test_summary_keeps_challenge_evidence_separate_from_compiler_eligibility(self) -> None:
        source = "__global__ void shared(float *x) { x[threadIdx.x] = 0; }"
        self.write_record("movement", "copy", "analysis", input_source=source)
        self.write_record(
            "normalization",
            "norm",
            "analysis",
            input_source=source,
            extra={"reasoning": "A distinct question over the same source parent."},
        )
        self.write_record(
            "movement",
            "copy",
            "generation",
            output_source=(
                "__global__ void generated(float *x) { "
                "atomicAdd(x + threadIdx.x, 1.0f); }"
            ),
        )

        summary = build_summary(
            load_records(self.dataset),
            dataset_id=self.dataset.name,
            source_revision="abc123",
        )
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["duplicate_full_record_groups"], 0)
        self.assertEqual(summary["duplicate_primary_source_groups"], 1)
        self.assertEqual(summary["duplicate_primary_source_records"], 2)
        self.assertEqual(summary["cross_category_primary_source_groups"], 1)
        self.assertEqual(summary["lexical_signals"]["thread_mapping"], 3)
        self.assertEqual(summary["lexical_signals"]["atomic"], 1)
        self.assertFalse(summary["authority_boundary"]["compiler_corpus_eligible"])
        self.assertEqual(
            summary["authority_boundary"]["allowed_claim"],
            "external_challenge_and_gap_discovery_only",
        )

    def test_malformed_record_fails_closed_and_excluded_is_not_read(self) -> None:
        self.write_record(
            "movement",
            "copy",
            "analysis",
            input_source="__global__ void copy() {}",
            extra={"verdict": "positive"},
        )
        excluded = self.dataset / "excluded" / "preserved" / "bad.jsonl"
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_text("not json\n", encoding="utf-8")

        with self.assertRaisesRegex(CorpusAuditError, "fields are"):
            load_records(self.dataset)

    def test_generation_requires_its_output_primary_but_may_have_empty_input(self) -> None:
        self.write_record("movement", "copy", "generation", output_source="")
        with self.assertRaisesRegex(CorpusAuditError, "empty primary field 'output'"):
            load_records(self.dataset)

    def test_git_locator_must_match_the_exact_clean_dataset_tree(self) -> None:
        self.write_record(
            "movement",
            "copy",
            "analysis",
            input_source="__global__ void copy() {}",
        )
        repository = self.dataset.parent
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Corpus Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.email", "corpus@test.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(repository), "add", self.dataset.name], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-q", "-m", "freeze corpus"],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

        verify_git_snapshot(self.dataset, revision)

        tracked = self.dataset / "categories" / "movement" / "copy" / "analysis.jsonl"
        untracked = self.dataset / "categories" / "movement" / "new" / "analysis.jsonl"
        untracked.parent.mkdir(parents=True, exist_ok=True)
        untracked.write_text(json.dumps(FIELDS, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusAuditError, "contains untracked files"):
            verify_git_snapshot(self.dataset, revision)

        with tracked.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(FIELDS, sort_keys=True) + "\n")
        with self.assertRaisesRegex(CorpusAuditError, "tracked bytes differ"):
            verify_git_snapshot(self.dataset, revision)

        with self.assertRaisesRegex(CorpusAuditError, "exact lowercase 40-hex"):
            verify_git_snapshot(self.dataset, "HEAD")


if __name__ == "__main__":
    unittest.main()
