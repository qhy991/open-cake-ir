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
    RECORD_FORMATS,
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
V1_FORMAT = "aka_v1_operator_sft"
V2_FORMAT = "aka_v2_review_projection"


class AkaChallengeCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name) / "cuda_kernel_dataset_v-test"
        self.repository = self.dataset.parent
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.name", "Corpus Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "config",
                "user.email",
                "corpus@test.invalid",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "remote",
                "add",
                "origin",
                "https://account:credential@example.invalid/AKA.git",
            ],
            check=True,
        )

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

    def commit(self) -> str:
        subprocess.run(["git", "-C", str(self.repository), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-q", "-m", "freeze corpus"],
            check=True,
        )
        return subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def snapshot(self):
        return verify_git_snapshot(self.dataset, self.commit())

    def records(self, snapshot, *, record_format: str = V1_FORMAT):
        return load_records(snapshot, record_format=record_format)

    def test_cli_requires_one_of_the_two_record_formats(self) -> None:
        self.assertEqual(RECORD_FORMATS, (V1_FORMAT, V2_FORMAT))
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/audit_aka_corpus.py"),
                str(self.dataset),
                "--source-revision",
                "0" * 40,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--record-format", completed.stderr)

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

        snapshot = self.snapshot()
        records = self.records(snapshot)
        self.assertEqual(
            [
                next(iter(record.artifact_signals.values()))["scope_signal"]
                for record in records
            ],
            ["fragment_or_library", "single_kernel", "multi_kernel_or_launch"],
        )
        by_task = {record.task: record for record in records}
        self.assertEqual(by_task["generation"].primary_field, "output")
        self.assertEqual(by_task["debug"].relation, "repair")

        case = projected_case(
            by_task["analysis"],
            snapshot=snapshot,
            reported_dataset_label="reported-v-test",
        )
        self.assertEqual(case["schema"], "open-cake.aka-challenge-corpus-case.v2")
        self.assertEqual(case["record_format"], V1_FORMAT)
        self.assertEqual(
            case["source_ref"],
            {
                "repository_remote_observed_sanitized": (
                    "https://example.invalid/AKA.git"
                ),
                "revision": snapshot.revision,
                "dataset_path": self.dataset.name,
                "reported_dataset_label": "reported-v-test",
                "path": (
                    f"{self.dataset.name}/categories/movement/copy/analysis.jsonl"
                ),
                "line": 1,
                "primary_field": "input",
            },
        )
        self.assertEqual(case["artifact_fields"], {"source": "input"})
        debug_case = projected_case(
            by_task["debug"],
            snapshot=snapshot,
            reported_dataset_label="reported-v-test",
        )
        self.assertEqual(
            debug_case["artifact_fields"], {"broken": "input", "repaired": "output"}
        )
        self.assertEqual(case["complete_parent_expressibility"], "unknown")
        self.assertEqual(case["delta_expressibility"], "unknown")
        self.assertNotIn("instruction", case)
        self.assertNotIn("input", case)
        self.assertNotIn("output", case)
        self.assertNotIn("scope_signal", case)
        self.assertNotIn("lexical_signals", case)
        self.assertEqual(case["artifact_signals"]["source"]["scope_signal"], "single_kernel")

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

        snapshot = self.snapshot()
        summary = build_summary(
            self.records(snapshot),
            snapshot=snapshot,
            reported_dataset_label="reported-v-test",
        )
        self.assertEqual(summary["schema"], "open-cake.aka-challenge-corpus-audit.v2")
        self.assertEqual(summary["record_format"], V1_FORMAT)
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["duplicate_full_record_groups"], 0)
        self.assertEqual(summary["duplicate_primary_source_groups"], 1)
        self.assertEqual(summary["duplicate_primary_source_records"], 2)
        self.assertEqual(summary["cross_category_primary_source_groups"], 1)
        self.assertEqual(summary["artifact_lexical_signals"]["source"]["thread_mapping"], 2)
        self.assertEqual(summary["artifact_lexical_signals"]["generated"]["thread_mapping"], 1)
        self.assertEqual(summary["artifact_lexical_signals"]["generated"]["atomic"], 1)
        self.assertEqual(
            summary["source"],
            {
                "repository_remote_observed_sanitized": (
                    "https://example.invalid/AKA.git"
                ),
                "revision": snapshot.revision,
                "dataset_path": self.dataset.name,
                "reported_dataset_label": "reported-v-test",
            },
        )
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
            self.records(self.snapshot())

    def test_generation_requires_its_output_primary_but_may_have_empty_input(self) -> None:
        self.write_record("movement", "copy", "generation", output_source="")
        with self.assertRaisesRegex(CorpusAuditError, "empty primary field 'output'"):
            self.records(self.snapshot())

    def test_empty_shard_fails_closed(self) -> None:
        shard = self.dataset / "categories" / "movement" / "copy" / "analysis.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(CorpusAuditError, "contains no records"):
            self.records(self.snapshot())

    def test_git_snapshot_reads_commit_blobs_not_ignored_or_modified_worktree_files(self) -> None:
        (self.repository / ".gitignore").write_text(
            "optimization_negative.jsonl\n", encoding="utf-8"
        )
        self.write_record(
            "movement",
            "copy",
            "analysis",
            input_source="__global__ void copy() {}",
        )
        revision = self.commit()

        tracked = self.dataset / "categories" / "movement" / "copy" / "analysis.jsonl"
        ignored = (
            self.dataset
            / "categories"
            / "movement"
            / "new"
            / "optimization_negative.jsonl"
        )
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text(json.dumps(FIELDS, sort_keys=True) + "\n", encoding="utf-8")

        with tracked.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(FIELDS, sort_keys=True) + "\n")

        snapshot = verify_git_snapshot(self.dataset, revision)
        records = self.records(snapshot)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].task, "analysis")

        with self.assertRaisesRegex(CorpusAuditError, "exact lowercase 40-hex"):
            verify_git_snapshot(self.dataset, "HEAD")

    def test_git_snapshot_rejects_a_symlink_shard(self) -> None:
        outside = self.repository / "outside.jsonl"
        outside.write_text(json.dumps(FIELDS, sort_keys=True) + "\n", encoding="utf-8")
        shard = self.dataset / "categories" / "movement" / "copy" / "analysis.jsonl"
        shard.parent.mkdir(parents=True)
        shard.symlink_to("../../../../outside.jsonl")

        revision = self.commit()
        with self.assertRaisesRegex(CorpusAuditError, "not a regular Git blob"):
            verify_git_snapshot(self.dataset, revision)

    def test_git_snapshot_rejects_a_nested_shard_path(self) -> None:
        nested = (
            self.dataset
            / "categories"
            / "movement"
            / "copy"
            / "nested"
            / "analysis.jsonl"
        )
        nested.parent.mkdir(parents=True)
        nested.write_text(json.dumps(FIELDS, sort_keys=True) + "\n", encoding="utf-8")

        revision = self.commit()
        with self.assertRaisesRegex(CorpusAuditError, "<dataset>/categories"):
            verify_git_snapshot(self.dataset, revision)

    def test_candidate_signals_are_separate_from_the_baseline(self) -> None:
        baseline = "__global__ void baseline(float *x) { x[threadIdx.x] = 0; }"
        candidate = (
            "__global__ void first(float *x) { atomicAdd(x + threadIdx.x, 1.0f); }\n"
            "__global__ void second(float *x) { x[threadIdx.x] = 2; }"
        )
        self.write_record(
            "movement",
            "copy",
            "optimization_positive",
            input_source=baseline,
            output_source=candidate,
        )
        snapshot = self.snapshot()
        record = self.records(snapshot)[0]

        case = projected_case(
            record,
            snapshot=snapshot,
            reported_dataset_label="reported-v-test",
        )
        self.assertEqual(
            case["artifact_signals"]["baseline"]["scope_signal"], "single_kernel"
        )
        self.assertEqual(
            case["artifact_signals"]["candidate"]["scope_signal"],
            "multi_kernel_or_launch",
        )
        self.assertNotIn("atomic", case["artifact_signals"]["baseline"]["lexical_signals"])
        self.assertIn("atomic", case["artifact_signals"]["candidate"]["lexical_signals"])
        self.assertEqual(case["delta_expressibility"], "unknown")

        summary = build_summary(
            [record], snapshot=snapshot, reported_dataset_label="reported-v-test"
        )
        self.assertEqual(
            summary["artifact_scope_signals"]["baseline"], {"single_kernel": 1}
        )
        self.assertEqual(
            summary["artifact_scope_signals"]["candidate"],
            {"multi_kernel_or_launch": 1},
        )

    def test_git_snapshot_disables_commit_replacement_objects(self) -> None:
        self.write_record(
            "movement",
            "copy",
            "analysis",
            input_source="__global__ void selected_commit() {}",
        )
        selected_revision = self.commit()
        shard = self.dataset / "categories" / "movement" / "copy" / "analysis.jsonl"
        replacement = dict(FIELDS)
        replacement["input"] = "__global__ void replacement_commit() {}"
        shard.write_text(json.dumps(replacement, sort_keys=True) + "\n", encoding="utf-8")
        replacement_revision = self.commit()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "replace",
                selected_revision,
                replacement_revision,
            ],
            check=True,
        )

        snapshot = verify_git_snapshot(self.dataset, selected_revision)
        records = self.records(snapshot)
        self.assertEqual(snapshot.revision, selected_revision)
        self.assertIn("selected_commit", records[0].fields["input"])
        self.assertNotIn("replacement_commit", records[0].fields["input"])

    def test_v2_reviewer_rows_emit_roles_but_no_code_signals(self) -> None:
        self.write_record(
            "review",
            "candidate",
            "optimization_neutral",
            input_source="__global__ void context() { atomicAdd(nullptr, 1); }",
            output_source="__global__ void prose_shaped_review() {}",
        )
        snapshot = self.snapshot()
        record = self.records(snapshot, record_format=V2_FORMAT)[0]

        case = projected_case(
            record,
            snapshot=snapshot,
            reported_dataset_label="reported-v2-test",
        )
        self.assertEqual(case["record_format"], V2_FORMAT)
        self.assertEqual(
            case["artifact_fields"], {"review_context": "input", "review": "output"}
        )
        self.assertEqual(case["artifact_signals"], {})
        self.assertEqual(case["delta_expressibility"], "unknown")

        summary = build_summary(
            [record], snapshot=snapshot, reported_dataset_label="reported-v2-test"
        )
        self.assertEqual(summary["record_format"], V2_FORMAT)
        self.assertEqual(summary["artifact_scope_signals"], {})
        self.assertEqual(summary["artifact_lexical_signals"], {})


if __name__ == "__main__":
    unittest.main()
