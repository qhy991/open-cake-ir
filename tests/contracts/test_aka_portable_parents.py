from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_aka_portable_parents_codex import (  # noqa: E402
    build_portable_plan,
    materialize_portable_completion,
    select_review_entries,
)


class AkaPortableParentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        from tests.contracts._parent_validator_fixture import write_parent_validator_fixture
        self.parent_validator = write_parent_validator_fixture(base / "parent-validator-fixture.py")
        self.repository = base / "AKA"
        self.source = self.repository / "datasets/curated/cuda_kernel_dataset_v1"
        self.portable = (
            self.repository
            / "datasets/curated/cuda_kernel_parent_completions_v1"
        )
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.name", "Portable Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "config",
                "user.email",
                "portable@test.invalid",
            ],
            check=True,
        )
        shard = self.source / "categories/data_movement_and_layout/copy/analysis.jsonl"
        shard.parent.mkdir(parents=True)
        source_rows = []
        portable_rows = []
        for index in range(1, 51):
            source_record = {
                "instruction": f"Analyze bounded copy {index}.",
                "input": (
                    "__global__ void copy(float *out, const float *in) "
                    "{ out[threadIdx.x] = in[threadIdx.x]; }"
                ),
                "reasoning": "Visible-source reasoning only.",
                "output": "One bounded review.",
            }
            source_rows.append(source_record)
            bundle = f"bundles/copy-{index:02d}"
            artifact_paths = {}
            for role in ("baseline", "reference", "harness"):
                relative = f"{bundle}/sources/{role}/{role}.cu"
                path = self.portable / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"// Portable {role}\n", encoding="utf-8")
                artifact_paths[role] = [relative]
            input_relative = f"{bundle}/source/input.json"
            input_path = self.portable / input_relative
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(
                json.dumps({"record": source_record}) + "\n", encoding="utf-8"
            )
            portable_rows.append(
                {
                    "schema": "aka.portable-kernel-parent.v1",
                    "case_id": f"copy-{index:02d}-parent-v1",
                    "original_parent": {
                        "case_path": (
                            "categories/data_movement_and_layout/copy/analysis.jsonl:"
                            f"{index}"
                        ),
                        "record_field": "input",
                    },
                    "recovery_mode": "contract_narrowed",
                    "source": {
                        "provenance": "visible_record",
                        "repository": None,
                        "revision": None,
                        "path": None,
                        "symbol": "copy",
                    },
                    "taxonomy": {
                        "category": "data_movement_and_layout",
                        "operator": "copy",
                    },
                    "derived_parent_id": f"copy-contiguous-fp32-{index:02d}-v1",
                    "semantics": {
                        "inputs": ["one contiguous float32 input"],
                        "outputs": ["one contiguous float32 output"],
                        "computation": "Copy every input element to the output.",
                        "valid_domain": ["the element count is positive"],
                        "material_unknowns": ["framework behavior is not claimed"],
                    },
                    "contract": {
                        "api": "Copy one contiguous float32 array.",
                        "dtypes": ["float32"],
                        "index_types": ["int32"],
                        "layouts": ["contiguous"],
                        "optional_inputs": [],
                        "invariants": ["complete output matches the reference"],
                        "exclusions": ["framework equivalence is not claimed"],
                        "launch_policy": "block 256 with a bounded grid",
                    },
                    "optimization_handoff": {
                        "mechanism": "vectorized_copy",
                        "hypothesis": "Aligned vector movement may reduce instructions.",
                        "eligibility": ["aligned float32 arrays"],
                        "anti_conditions": ["unaligned tails"],
                    },
                    "artifacts": artifact_paths,
                    "bundle": {
                        "path": bundle,
                        "documents": {},
                        "source_files": {"input.json": input_relative},
                        "omitted_runtime_roles": ["task"],
                        "omitted_nonportable_files": {},
                    },
                    "provenance": {
                        "source_selection": {
                            "path": "categories/data_movement_and_layout/copy/analysis.jsonl",
                            "line": index,
                            "parent_field": "input",
                        }
                    },
                    "qualification": {
                        "authority": "node-owned evidence summarized by portable export",
                        "locator": {
                            "node_id": "b200-local",
                            "run_id": f"copy-{index:02d}-run",
                        },
                        "lifecycle": "completed",
                        "validity": "valid",
                        "stages": {
                            "compile": {
                                "status": "passed",
                                "validity": "valid",
                                "summary": "native compile passed",
                            },
                            "correctness": {
                                "status": "passed",
                                "validity": "valid",
                                "summary": "complete-output correctness passed",
                                "workloads": [
                                    {"id": "small", "correct": True},
                                    {"id": "tail", "correct": True},
                                ],
                            },
                            "sanitize": {
                                "status": "passed",
                                "validity": "valid",
                                "summary": "memcheck and racecheck passed",
                            },
                        },
                    },
                    "outcome": "qualified",
                    "missing_facts": [],
                    "training_route": "augmentation_parent",
                    "training_eligibility": False,
                    "next_action": "Start a fresh mechanism augmentation case.",
                }
            )
        shard.write_text(
            "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
        )
        self.portable.mkdir(parents=True, exist_ok=True)
        (self.portable / "records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in portable_rows), encoding="utf-8"
        )
        subprocess.run(["git", "-C", str(self.repository), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-q", "-m", "freeze"],
            check=True,
        )
        self.revision = subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_plan_maps_fifty_commit_bound_parents_and_materializes_runnable_case(self) -> None:
        plan, entries = build_portable_plan(
            source_dataset_root=self.source,
            portable_dataset_root=self.portable,
            source_revision=self.revision,
        )

        self.assertEqual(plan["portable_dataset"]["records"], 50)
        self.assertEqual(len(entries), 50)
        self.assertEqual(entries[0].queue_case_id, "case-000001")
        self.assertEqual(
            plan["claim_boundary"]["parent_completion_outcome"],
            "runnable_unqualified",
        )

        completion_root = Path(self.temporary.name) / "completion"
        (completion_root / "cases").mkdir(parents=True)
        (completion_root / "runs").mkdir()
        completion = materialize_portable_completion(
            completion_root=completion_root,
            repository=self.repository,
            revision=self.revision,
            portable_dataset=PurePosixPath(
                "datasets/curated/cuda_kernel_parent_completions_v1"
            ),
            source_dataset=PurePosixPath(
                "datasets/curated/cuda_kernel_dataset_v1"
            ),
            entry=entries[0],
            parent_validator=self.parent_validator,
        )
        document = json.loads(completion.read_text(encoding="utf-8"))
        marker = json.loads(
            (completion.parent / "PARENT_DONE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(document["outcome"], "runnable_unqualified")
        self.assertIsNone(document["qualification"])
        self.assertTrue(marker["counts_as_runnable_bundle"])
        self.assertFalse(marker["counts_as_executable_parent"])
        self.assertEqual(
            document["original_parent"]["case_path"],
            (
                f"{self.revision}:datasets/curated/cuda_kernel_dataset_v1/"
                "categories/data_movement_and_layout/copy/analysis.jsonl:1"
            ),
        )

        selected = select_review_entries(
            entries,
            limit=2,
            start_after=entries[0].portable_record["case_id"],
        )
        self.assertEqual(
            [entry.queue_case_id for entry in selected],
            ["case-000002", "case-000003"],
        )


if __name__ == "__main__":
    unittest.main()
