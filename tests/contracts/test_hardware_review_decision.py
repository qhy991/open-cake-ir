"""End-to-end contracts for read-only external Stage-3 review intake."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Iterator


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "operator_library"
EXTRACTION = ROOT / "abstraction_extraction"
DESIGN = ROOT / "hardware_informed_design"
TARGET = ROOT / "compiler" / "targets" / "apple_gpu_family9.json"
TOOL = ROOT / "tools" / "validate_hardware_informed_design.py"

REVIEW_REQUEST_PATH = (
    "hardware_informed_design/review_requests/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json"
)
REVIEWED_ARTIFACT_PATHS = (
    "hardware_informed_design/schema.json",
    "hardware_informed_design/manifest.json",
    "hardware_informed_design/evidence/apple-metal-hierarchical-reduction-v1.json",
    "hardware_informed_design/proposals/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json",
    REVIEW_REQUEST_PATH,
)
APPROVAL_OPTION = "preimplementation-standalone-metal-prototypes"
SUCCESSOR_OPTION = "amend-gate-to-port-acceptance-after-bounded-implementation"
RESOURCE_ITEM = "resource-accounting-and-runtime-gates"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _git(project: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(project), *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr}"
        )
    return result.stdout.strip()


def _git_bytes(project: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(project), *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(arguments)} failed ({result.returncode}): "
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def _object_store_snapshot(project: Path) -> dict[str, bytes]:
    objects = project / ".git" / "objects"
    return {
        str(path.relative_to(objects)): path.read_bytes()
        for path in objects.rglob("*")
        if path.is_file()
    }


@dataclass(frozen=True)
class _Fixture:
    root: Path
    project: Path
    library: Path
    extraction: Path
    design: Path
    target: Path
    decision: Path
    reviewed_revision: str


@contextmanager
def _temporary_git_project() -> Iterator[_Fixture]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        project = root / "project"
        library = project / "operator_library"
        extraction = project / "abstraction_extraction"
        design = project / "hardware_informed_design"
        target = project / "compiler" / "targets" / "apple_gpu_family9.json"
        project.mkdir()
        shutil.copytree(LIBRARY, library)
        shutil.copytree(EXTRACTION, extraction)
        shutil.copytree(DESIGN, design)
        target.parent.mkdir(parents=True)
        shutil.copy2(TARGET, target)
        _git(project, "init", "--quiet")
        _git(project, "config", "user.name", "Hardware Review Contract Test")
        _git(project, "config", "user.email", "hardware-review-test@example.invalid")
        _git(project, "add", ".")
        _git(project, "commit", "--quiet", "-m", "reviewed stage three closure")
        reviewed_revision = _git(project, "rev-parse", "HEAD")
        yield _Fixture(
            root=root,
            project=project,
            library=library,
            extraction=extraction,
            design=design,
            target=target,
            decision=root / "external-hardware-review-decision.json",
            reviewed_revision=reviewed_revision,
        )


def _decision(
    fixture: _Fixture,
    *,
    disposition: str = "approved",
    option: str = APPROVAL_OPTION,
    verdicts: dict[str, str] | None = None,
    reviewed_revision: str | None = None,
) -> dict[str, object]:
    request = _read(fixture.project / REVIEW_REQUEST_PATH)
    overrides = verdicts or {}
    return {
        "$schema": (
            "https://open-cake-ir.local/hardware-informed-design/"
            "schema-v1.json#/$defs/hardware_review_decision"
        ),
        "schema_version": 1,
        "decision_id": "external-human-hardware-review-apple-family9-v1",
        "kind": "external_human_hardware_review_decision",
        "authority": "external_human_hardware_reviewer_only",
        "binding": {
            "design_stage_id": "apple-metal-hardware-informed-design-v1",
            "request_id": request["review_request_id"],
            "path": REVIEW_REQUEST_PATH,
            "reviewed_git_revision": (reviewed_revision or fixture.reviewed_revision),
            "reviewed_artifact_paths": list(REVIEWED_ARTIFACT_PATHS),
        },
        "reviewer_attestation": {
            "reviewer_identity": "外部人工硬件评审者",
            "authorship_attestation": "external-human-outside-automation",
            "decision_basis": "逐项检查固定的 Stage-3 Git 闭包与硬件映射边界。",
        },
        "global_disposition": disposition,
        "item_verdicts": [
            {
                "review_item_id": item["review_item_id"],
                "verdict": overrides.get(item["review_item_id"], "approved"),
                "localized_reason": (
                    "该项已按固定审查问题、引用闭包与非声明边界逐项检查："
                    f"{item['review_item_id']}。"
                ),
            }
            for item in request["review_items"]
        ],
        "ambiguity_resolution": {
            "ambiguity_id": "resource-accounting-phase-order",
            "selected_option_id": option,
            "localized_reason": (
                "此选择只解决资源阶段顺序，不解决资源 hypothesis，"
                "也不授权 Compiler 实现。"
            ),
        },
        "scope_acknowledgements": {
            "approval_scope": "stage3_hardware_review_only",
            "principle_driven_iteration_required": True,
            "resource_accounting_hypothesis_resolved": False,
            "compiler_change_authorized": False,
            "implementation_authorized": False,
            "evaluation_authorized": False,
            "performance_claim_authorized": False,
            "scientific_claim_authorized": False,
            "promotion_authorized": False,
        },
    }


def _run(
    fixture: _Fixture,
    *,
    decision: Path | str | None = None,
    output_format: str = "json",
    require_clear: bool = False,
    check_reviewable_head: bool = False,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(TOOL),
        "--library-root",
        str(fixture.library),
        "--extraction-root",
        str(fixture.extraction),
        "--design-root",
        str(fixture.design),
        "--target-path",
        str(fixture.target),
        "--repository-root",
        str(fixture.project),
        "--format",
        output_format,
    ]
    if decision is not None:
        command.extend(("--review-decision", str(decision)))
    if check_reviewable_head:
        command.append("--check-reviewable-head")
    if require_clear:
        command.append("--require-hardware-review-clear")
    return subprocess.run(
        command,
        cwd=fixture.project,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(environment or {})},
        check=False,
    )


def _write_and_run(
    fixture: _Fixture,
    decision: dict[str, object],
    *,
    require_clear: bool = False,
    output_format: str = "json",
) -> subprocess.CompletedProcess[str]:
    _write(fixture.decision, decision)
    return _run(
        fixture,
        decision=fixture.decision,
        require_clear=require_clear,
        output_format=output_format,
    )


def _combined(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout + result.stderr).lower()


class HardwareReviewDecisionIntakeTests(unittest.TestCase):
    def assertInvalid(
        self,
        fixture: _Fixture,
        decision: dict[str, object],
        expected: str,
    ) -> None:
        result = _write_and_run(fixture, decision)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(expected.lower(), _combined(result))

    def test_no_decision_preserves_the_review_pending_summary(self) -> None:
        with _temporary_git_project() as fixture:
            result = _run(fixture)
            gate = _run(fixture, require_clear=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(gate.returncode, 3, gate.stderr)
        self.assertTrue(json.loads(gate.stdout)["valid"])
        summary = json.loads(result.stdout)
        self.assertTrue(summary["valid"])
        self.assertIs(summary["hardware_review_decision_present"], False)
        self.assertIsNone(summary["hardware_review_decision_id"])
        self.assertIsNone(summary["hardware_review_decision_disposition"])
        self.assertIsNone(summary["hardware_reviewed_git_revision"])
        self.assertIs(summary["hardware_reviewed_revision_bound"], False)
        self.assertIs(summary["reviewable_head_check_performed"], False)
        self.assertIsNone(summary["reviewable_head_git_revision"])
        self.assertIs(summary["reviewable_head_fixed_closure_verified"], False)
        self.assertEqual(summary["non_approved_review_item_ids"], [])
        self.assertIsNone(summary["resource_phase_option_id"])
        self.assertIs(summary["successor_stage3_proposal_required"], False)
        self.assertIs(summary["resource_accounting_hypothesis_resolved"], False)
        self.assertIs(summary["hardware_review_complete"], False)
        self.assertIs(summary["ready_for_principle_review"], False)
        self.assertIs(summary["stage_transition_applied"], False)
        self.assertEqual(
            summary["reviewer_authorship_assurance"],
            "no_external_decision_supplied",
        )

    def test_reviewable_head_preflight_binds_only_the_fixed_live_closure(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            before_files = {
                str(path.relative_to(fixture.project)): path.read_bytes()
                for path in fixture.project.rglob("*")
                if path.is_file() and ".git" not in path.parts
            }
            before_objects = _object_store_snapshot(fixture.project)
            head = _git(fixture.project, "rev-parse", "HEAD")
            result = _run(fixture, check_reviewable_head=True)
            gate = _run(
                fixture,
                check_reviewable_head=True,
                require_clear=True,
            )
            after_files = {
                str(path.relative_to(fixture.project)): path.read_bytes()
                for path in fixture.project.rglob("*")
                if path.is_file() and ".git" not in path.parts
            }
            after_objects = _object_store_snapshot(fixture.project)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(gate.returncode, 3, gate.stdout + gate.stderr)
        summary = json.loads(result.stdout)
        self.assertIs(summary["reviewable_head_check_performed"], True)
        self.assertEqual(summary["reviewable_head_git_revision"], head)
        self.assertIs(summary["reviewable_head_fixed_closure_verified"], True)
        self.assertEqual(
            summary["verification_scope"],
            "offline_metadata_derivation_and_reviewable_head_binding_validation",
        )
        self.assertIs(summary["hardware_review_decision_present"], False)
        self.assertIs(summary["hardware_reviewed_revision_bound"], False)
        self.assertIs(summary["hardware_review_complete"], False)
        self.assertIs(summary["ready_for_principle_review"], False)
        self.assertEqual(
            summary["reviewer_authorship_assurance"],
            "no_external_decision_supplied",
        )
        self.assertEqual(
            summary["next_action"], "obtain_external_human_hardware_review"
        )
        self.assertEqual(after_files, before_files)
        self.assertEqual(after_objects, before_objects)

    def test_reviewable_head_rejects_fixed_byte_drift_but_ignores_other_files(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            manifest = fixture.design / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            default_result = _run(fixture)
            result = _run(fixture, check_reviewable_head=True)
            strict = _run(
                fixture,
                check_reviewable_head=True,
                require_clear=True,
            )
            text_failure = _run(
                fixture,
                check_reviewable_head=True,
                output_format="text",
            )
            markdown_failure = _run(
                fixture,
                check_reviewable_head=True,
                output_format="markdown",
            )

        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)
        self.assertIn("reviewable-head.git-revision", _combined(result))
        self.assertIn("differs byte-for-byte", _combined(result))
        self.assertNotIn("review-decision", _combined(result))
        self.assertEqual(json.loads(result.stdout)["valid"], False)
        self.assertEqual(result.stderr, "")
        for failure in (text_failure, markdown_failure):
            self.assertEqual(failure.returncode, 1, failure.stdout + failure.stderr)
            self.assertEqual(failure.stdout, "")
            self.assertIn("reviewable-head.git-revision", failure.stderr)

        with _temporary_git_project() as fixture:
            dirty_head = _git(fixture.project, "rev-parse", "HEAD")
            readme = fixture.design / "README.md"
            readme.write_text(
                readme.read_text(encoding="utf-8") + "\nUnbound handoff note.\n",
                encoding="utf-8",
            )
            untracked = fixture.project / "untracked-handoff-note.txt"
            untracked.write_text(
                "untracked and outside fixed closure\n", encoding="utf-8"
            )
            dirty_status = _git(
                fixture.project, "status", "--short", "--untracked-files=all"
            )
            dirty_result = _run(fixture, check_reviewable_head=True)
            _git(
                fixture.project,
                "add",
                str(readme.relative_to(fixture.project)),
                untracked.name,
            )
            _git(fixture.project, "commit", "--quiet", "-m", "unbound readme note")
            head = _git(fixture.project, "rev-parse", "HEAD")
            result = _run(fixture, check_reviewable_head=True)

        self.assertIn("M hardware_informed_design/README.md", dirty_status)
        self.assertIn("?? untracked-handoff-note.txt", dirty_status)
        self.assertEqual(
            dirty_result.returncode, 0, dirty_result.stdout + dirty_result.stderr
        )
        self.assertEqual(
            json.loads(dirty_result.stdout)["reviewable_head_git_revision"],
            dirty_head,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["reviewable_head_git_revision"], head
        )

    def test_reviewable_head_preflight_is_visible_in_every_cli_format(self) -> None:
        markers = {
            "text": "reviewable HEAD fixed closure verified: yes",
            "markdown": "| Reviewable HEAD fixed closure verified | Yes |",
            "json": '"reviewable_head_fixed_closure_verified": true',
        }
        with _temporary_git_project() as fixture:
            for output_format, marker in markers.items():
                with self.subTest(output_format=output_format):
                    result = _run(
                        fixture,
                        check_reviewable_head=True,
                        output_format=output_format,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn(marker, result.stdout)
                    self.assertIn(fixture.reviewed_revision, result.stdout)

    def test_reviewable_head_rejects_missing_or_non_blob_fixed_paths(self) -> None:
        with _temporary_git_project() as fixture:
            path = fixture.project / REVIEW_REQUEST_PATH
            original = path.read_bytes()
            path.unlink()
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "missing review request")
            path.write_bytes(original)
            result = _run(fixture, check_reviewable_head=True)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("missing required closure paths", _combined(result))
        self.assertNotIn("review-decision", _combined(result))

        with _temporary_git_project() as fixture:
            path = fixture.project / REVIEW_REQUEST_PATH
            original = path.read_bytes()
            path.unlink()
            path.symlink_to("../manifest.json")
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "symlink review request")
            path.unlink()
            path.write_bytes(original)
            result = _run(fixture, check_reviewable_head=True)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("reviewable-head.artifact-paths", _combined(result))
        self.assertIn("must be a 100644 blob", _combined(result))
        self.assertNotIn("review-decision", _combined(result))

    def test_reviewable_head_rejects_an_unborn_head(self) -> None:
        with _temporary_git_project() as fixture:
            _git(fixture.project, "update-ref", "-d", "HEAD")
            result = _run(fixture, check_reviewable_head=True)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("reviewable-head.git-revision", _combined(result))
        self.assertIn("head must resolve to an existing full commit", _combined(result))
        self.assertNotIn("review-decision", _combined(result))

    def test_reviewable_head_and_external_decision_are_independent_bindings(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            main_branch = _git(fixture.project, "branch", "--show-current")
            _git(fixture.project, "switch", "--quiet", "-c", "review-side")
            side_note = fixture.project / "review-side-note.txt"
            side_note.write_text("nonancestor review commit\n", encoding="utf-8")
            _git(fixture.project, "add", side_note.name)
            _git(fixture.project, "commit", "--quiet", "-m", "side review note")
            side_revision = _git(fixture.project, "rev-parse", "HEAD")
            _git(fixture.project, "switch", "--quiet", main_branch)
            main_note = fixture.project / "main-handoff-note.txt"
            main_note.write_text("independent reviewable HEAD\n", encoding="utf-8")
            _git(fixture.project, "add", main_note.name)
            _git(fixture.project, "commit", "--quiet", "-m", "advance main head")
            head = _git(fixture.project, "rev-parse", "HEAD")
            ancestry = subprocess.run(
                (
                    "git",
                    "-C",
                    str(fixture.project),
                    "merge-base",
                    "--is-ancestor",
                    side_revision,
                    head,
                ),
                check=False,
            )
            self.assertEqual(ancestry.returncode, 1)
            decision = _decision(fixture, reviewed_revision=side_revision)
            _write(fixture.decision, decision)
            result = _run(
                fixture,
                decision=fixture.decision,
                check_reviewable_head=True,
                require_clear=True,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["hardware_reviewed_git_revision"], side_revision)
        self.assertEqual(summary["reviewable_head_git_revision"], head)
        self.assertNotEqual(
            summary["hardware_reviewed_git_revision"],
            summary["reviewable_head_git_revision"],
        )
        self.assertIs(summary["hardware_review_complete"], True)
        self.assertIs(summary["reviewable_head_fixed_closure_verified"], True)
        self.assertEqual(
            summary["verification_scope"],
            "offline_metadata_derivation_reviewable_head_and_external_decision_binding_validation",
        )

    def test_combined_invalid_decision_and_head_drift_report_both_failures(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            decision = _decision(fixture)
            decision["authority"] = "automation"
            _write(fixture.decision, decision)
            manifest = fixture.design / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            result = _run(
                fixture,
                decision=fixture.decision,
                check_reviewable_head=True,
                require_clear=True,
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        errors = json.loads(result.stdout)["errors"]
        self.assertTrue(any("review-decision.authority" in error for error in errors))
        self.assertTrue(
            any("reviewable-head.git-revision" in error for error in errors)
        )
        self.assertTrue(any("differs byte-for-byte" in error for error in errors))

    def test_approved_option_one_clears_only_stage_three_in_memory(self) -> None:
        with _temporary_git_project() as fixture:
            decision = _decision(fixture)
            result = _write_and_run(fixture, decision)
            gate = _write_and_run(fixture, decision, require_clear=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(gate.returncode, 0, gate.stderr)
        summary = json.loads(result.stdout)
        self.assertIs(summary["hardware_review_decision_present"], True)
        self.assertEqual(
            summary["hardware_review_decision_id"], decision["decision_id"]
        )
        self.assertEqual(summary["hardware_review_decision_disposition"], "approved")
        self.assertEqual(
            summary["hardware_reviewed_git_revision"],
            decision["binding"]["reviewed_git_revision"],
        )
        self.assertIs(summary["hardware_reviewed_revision_bound"], True)
        self.assertEqual(summary["non_approved_review_item_ids"], [])
        self.assertEqual(summary["resource_phase_option_id"], APPROVAL_OPTION)
        self.assertIs(summary["successor_stage3_proposal_required"], False)
        self.assertIs(summary["hardware_review_complete"], True)
        self.assertIs(summary["ready_for_principle_review"], True)
        self.assertIs(summary["stage_transition_applied"], False)
        self.assertEqual(summary["next_action"], "begin_principle_driven_iteration")
        self.assertEqual(
            summary["reviewer_authorship_assurance"],
            "external_human_process_attested_not_cryptographically_verified",
        )
        for field in (
            "resource_accounting_hypothesis_resolved",
            "hardware_semantics_verified",
            "compiler_change_authorized",
            "implementation_authorized",
            "evaluation_authorized",
            "performance_claim_authorized",
            "scientific_claim_authorized",
            "promotion_authorized",
        ):
            with self.subTest(field=field):
                self.assertIs(summary[field], False)

    def test_well_formed_negative_decisions_are_valid_but_do_not_clear(self) -> None:
        cases = (
            (
                "option1-changes",
                APPROVAL_OPTION,
                "changes_requested",
                {"vendor-fact-applicability": "changes_requested"},
                False,
            ),
            (
                "option1-rejected",
                APPROVAL_OPTION,
                "rejected",
                {"vendor-fact-applicability": "rejected"},
                False,
            ),
            (
                "option2-changes",
                SUCCESSOR_OPTION,
                "changes_requested",
                {RESOURCE_ITEM: "changes_requested"},
                True,
            ),
            (
                "option2-rejected",
                SUCCESSOR_OPTION,
                "rejected",
                {RESOURCE_ITEM: "rejected"},
                True,
            ),
        )
        with _temporary_git_project() as fixture:
            for name, option, disposition, verdicts, successor_required in cases:
                with self.subTest(name=name):
                    decision = _decision(
                        fixture,
                        disposition=disposition,
                        option=option,
                        verdicts=verdicts,
                    )
                    result = _write_and_run(fixture, decision)
                    gate = _write_and_run(fixture, decision, require_clear=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(gate.returncode, 3, gate.stderr)
                    summary = json.loads(result.stdout)
                    self.assertIs(summary["hardware_review_decision_present"], True)
                    self.assertEqual(
                        summary["hardware_review_decision_disposition"], disposition
                    )
                    self.assertEqual(
                        summary["non_approved_review_item_ids"], list(verdicts)
                    )
                    self.assertIs(
                        summary["successor_stage3_proposal_required"],
                        successor_required,
                    )
                    self.assertIs(summary["hardware_review_complete"], False)
                    self.assertIs(summary["ready_for_principle_review"], False)
                    self.assertIs(summary["stage_transition_applied"], False)

    def test_disposition_and_option_contradictions_are_rejected(self) -> None:
        with _temporary_git_project() as fixture:
            cases: tuple[tuple[str, Callable[[dict[str, object]], None], str], ...] = (
                (
                    "approved-with-local-change",
                    lambda value: value["item_verdicts"][0].__setitem__(
                        "verdict", "changes_requested"
                    ),
                    "maximum review-item verdict severity",
                ),
                (
                    "changes-with-local-reject",
                    lambda value: (
                        value.__setitem__("global_disposition", "changes_requested"),
                        value["item_verdicts"][0].__setitem__("verdict", "rejected"),
                    ),
                    "maximum review-item verdict severity",
                ),
                (
                    "rejected-without-local-reject",
                    lambda value: (
                        value.__setitem__("global_disposition", "rejected"),
                        value["item_verdicts"][0].__setitem__(
                            "verdict", "changes_requested"
                        ),
                    ),
                    "maximum review-item verdict severity",
                ),
                (
                    "option2-approved",
                    lambda value: value["ambiguity_resolution"].__setitem__(
                        "selected_option_id", SUCCESSOR_OPTION
                    ),
                    "approval-compatible option",
                ),
                (
                    "option2-resource-approved",
                    lambda value: (
                        value.__setitem__("global_disposition", "changes_requested"),
                        value["ambiguity_resolution"].__setitem__(
                            "selected_option_id", SUCCESSOR_OPTION
                        ),
                        value["item_verdicts"][0].__setitem__(
                            "verdict", "changes_requested"
                        ),
                    ),
                    "requires this review item to be non-approved",
                ),
            )
            for name, mutate, diagnostic in cases:
                with self.subTest(name=name):
                    decision = _decision(fixture)
                    mutate(decision)
                    self.assertInvalid(fixture, decision, diagnostic)

    def test_review_item_closure_is_exact_unique_and_ordered(self) -> None:
        with _temporary_git_project() as fixture:
            cases: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
                ("missing", lambda value: value["item_verdicts"].pop()),
                (
                    "duplicate",
                    lambda value: value["item_verdicts"][1].__setitem__(
                        "review_item_id",
                        value["item_verdicts"][0]["review_item_id"],
                    ),
                ),
                (
                    "unknown",
                    lambda value: value["item_verdicts"][0].__setitem__(
                        "review_item_id", "unknown-review-item"
                    ),
                ),
                (
                    "reordered",
                    lambda value: value["item_verdicts"].__setitem__(
                        slice(0, 2), reversed(value["item_verdicts"][:2])
                    ),
                ),
            )
            for name, mutate in cases:
                with self.subTest(name=name):
                    decision = _decision(fixture)
                    mutate(decision)
                    self.assertInvalid(fixture, decision, "item_verdicts")

    def test_human_authored_text_must_be_nonblank_bounded_and_control_free(
        self,
    ) -> None:
        def reviewer(value: dict[str, object], text: str) -> None:
            value["reviewer_attestation"]["reviewer_identity"] = text

        def basis(value: dict[str, object], text: str) -> None:
            value["reviewer_attestation"]["decision_basis"] = text

        def item_reason(value: dict[str, object], text: str) -> None:
            value["item_verdicts"][0]["localized_reason"] = text

        def ambiguity_reason(value: dict[str, object], text: str) -> None:
            value["ambiguity_resolution"]["localized_reason"] = text

        with _temporary_git_project() as fixture:
            cases = (
                ("blank-reviewer", reviewer, " \t "),
                ("blank-basis", basis, "   "),
                ("blank-item-reason", item_reason, "\t"),
                ("blank-ambiguity-reason", ambiguity_reason, "  "),
                ("control", item_reason, "contains\u0000control"),
                ("too-long", basis, "x" * 4097),
            )
            for name, mutate, text in cases:
                with self.subTest(name=name):
                    decision = _decision(fixture)
                    mutate(decision, text)
                    self.assertInvalid(fixture, decision, "review-decision")

    def test_decision_contract_rejects_authority_binding_and_scope_drift(self) -> None:
        with _temporary_git_project() as fixture:
            cases: tuple[tuple[str, Callable[[dict[str, object]], None], str], ...] = (
                (
                    "unknown-field",
                    lambda value: value.__setitem__("ready", True),
                    "unknown fields: ready",
                ),
                (
                    "automation-authority",
                    lambda value: value.__setitem__("authority", "automation"),
                    "authority",
                ),
                (
                    "automation-attestation",
                    lambda value: value["reviewer_attestation"].__setitem__(
                        "authorship_attestation", "automation-authored"
                    ),
                    "authorship_attestation",
                ),
                (
                    "request-drift",
                    lambda value: value["binding"].__setitem__(
                        "request_id", "different-request"
                    ),
                    "request_id",
                ),
                (
                    "path-drift",
                    lambda value: value["binding"].__setitem__(
                        "path", "hardware_informed_design/manifest.json"
                    ),
                    "binding.path",
                ),
                (
                    "closure-drift",
                    lambda value: value["binding"]["reviewed_artifact_paths"].pop(),
                    "reviewed_artifact_paths",
                ),
                (
                    "unknown-option",
                    lambda value: value["ambiguity_resolution"].__setitem__(
                        "selected_option_id", "automation-selected-option"
                    ),
                    "selected_option_id",
                ),
                (
                    "implementation-authority",
                    lambda value: value["scope_acknowledgements"].__setitem__(
                        "implementation_authorized", True
                    ),
                    "implementation_authorized",
                ),
            )
            for name, mutate, diagnostic in cases:
                with self.subTest(name=name):
                    decision = _decision(fixture)
                    mutate(decision)
                    self.assertInvalid(fixture, decision, diagnostic)

    def test_non_json_and_ambiguous_json_bytes_are_rejected(self) -> None:
        with _temporary_git_project() as fixture:
            decision = _decision(fixture)
            encoded = json.dumps(decision, ensure_ascii=False)
            cases = (
                (
                    "duplicate-key",
                    encoded.replace(
                        '"schema_version": 1,',
                        '"schema_version": 1, "schema_version": 1,',
                        1,
                    ).encode("utf-8"),
                    "duplicate object key",
                ),
                (
                    "nan",
                    encoded.replace(
                        '"schema_version": 1', '"schema_version": NaN', 1
                    ).encode("utf-8"),
                    "non-json number",
                ),
                ("invalid-utf8", b'{"value":"\xff"}', "invalid utf-8 json"),
                ("top-level-array", b"[]", "top level must be an object"),
            )
            for name, raw, diagnostic in cases:
                with self.subTest(name=name):
                    fixture.decision.write_bytes(raw)
                    result = _run(fixture, decision=fixture.decision)
                    self.assertEqual(
                        result.returncode, 1, result.stdout + result.stderr
                    )
                    self.assertIn(diagnostic, _combined(result))

    def test_decision_path_must_be_absolute_external_and_non_symlink(self) -> None:
        with _temporary_git_project() as fixture:
            decision = _decision(fixture)
            _write(fixture.decision, decision)

            relative = _run(
                fixture, decision="../external-hardware-review-decision.json"
            )
            self.assertEqual(relative.returncode, 1, relative.stdout + relative.stderr)
            self.assertIn("path must be absolute", _combined(relative))

            inside = fixture.project / "decision.json"
            _write(inside, decision)
            inside_result = _run(fixture, decision=inside)
            self.assertEqual(
                inside_result.returncode, 1, inside_result.stdout + inside_result.stderr
            )
            self.assertIn("outside the source checkout", _combined(inside_result))

            link = fixture.root / "decision-link.json"
            link.symlink_to(fixture.decision)
            linked = _run(fixture, decision=link)
            self.assertEqual(linked.returncode, 1, linked.stdout + linked.stderr)
            self.assertIn("non-symlink external file", _combined(linked))

    def test_case_insensitive_checkout_alias_is_not_an_external_path(self) -> None:
        with _temporary_git_project() as fixture:
            alias_project = fixture.project.with_name(fixture.project.name.swapcase())
            try:
                aliases_checkout = alias_project.samefile(fixture.project)
            except OSError:
                aliases_checkout = False
            if not aliases_checkout:
                self.skipTest("filesystem has no case-insensitive checkout alias")
            decision_path = fixture.project / "decision.json"
            _write(decision_path, _decision(fixture))
            result = _run(
                fixture,
                decision=alias_project / decision_path.name,
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("outside the source checkout", _combined(result))

    def test_reviewed_revision_must_exist_be_a_commit_and_hold_fixed_blobs(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            missing = _decision(fixture, reviewed_revision="f" * 40)
            self.assertInvalid(fixture, missing, "existing full commit")

            tree_revision = _git(fixture.project, "rev-parse", "HEAD^{tree}")
            noncommit = _decision(fixture, reviewed_revision=tree_revision)
            self.assertInvalid(fixture, noncommit, "existing full commit")

        with _temporary_git_project() as fixture:
            path = fixture.project / REVIEW_REQUEST_PATH
            original = path.read_bytes()
            path.unlink()
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "missing request blob")
            missing_path_revision = _git(fixture.project, "rev-parse", "HEAD")
            path.write_bytes(original)
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "restore request blob")
            decision = _decision(fixture, reviewed_revision=missing_path_revision)
            self.assertInvalid(fixture, decision, "missing required closure paths")

        with _temporary_git_project() as fixture:
            path = fixture.project / REVIEW_REQUEST_PATH
            original = path.read_bytes()
            path.unlink()
            path.symlink_to("../manifest.json")
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "symlink request blob")
            symlink_revision = _git(fixture.project, "rev-parse", "HEAD")
            path.unlink()
            path.write_bytes(original)
            _git(fixture.project, "add", "-A")
            _git(fixture.project, "commit", "--quiet", "-m", "restore regular request")
            decision = _decision(fixture, reviewed_revision=symlink_revision)
            self.assertInvalid(fixture, decision, "must be a 100644 blob")

    def test_nonancestor_commit_with_the_exact_fixed_closure_is_accepted(self) -> None:
        with _temporary_git_project() as fixture:
            main_branch = _git(fixture.project, "branch", "--show-current")
            _git(fixture.project, "switch", "--quiet", "-c", "review-side")
            side_note = fixture.project / "side-note.txt"
            side_note.write_text("review side branch\n", encoding="utf-8")
            _git(fixture.project, "add", side_note.name)
            _git(fixture.project, "commit", "--quiet", "-m", "side review note")
            side_revision = _git(fixture.project, "rev-parse", "HEAD")
            _git(fixture.project, "switch", "--quiet", main_branch)
            main_note = fixture.project / "main-note.txt"
            main_note.write_text("main branch\n", encoding="utf-8")
            _git(fixture.project, "add", main_note.name)
            _git(fixture.project, "commit", "--quiet", "-m", "main note")
            ancestry = subprocess.run(
                (
                    "git",
                    "-C",
                    str(fixture.project),
                    "merge-base",
                    "--is-ancestor",
                    side_revision,
                    "HEAD",
                ),
                check=False,
            )
            self.assertEqual(ancestry.returncode, 1)
            decision = _decision(fixture, reviewed_revision=side_revision)
            result = _write_and_run(fixture, decision)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(json.loads(result.stdout)["hardware_review_complete"], True)

    def test_git_replace_cannot_substitute_a_different_reviewed_tree(self) -> None:
        with _temporary_git_project() as fixture:
            reviewed_revision = fixture.reviewed_revision
            manifest = fixture.design / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            _git(fixture.project, "add", str(manifest.relative_to(fixture.project)))
            _git(fixture.project, "commit", "--quiet", "-m", "drift fixed closure")
            replacement_revision = _git(fixture.project, "rev-parse", "HEAD")
            _git(
                fixture.project,
                "replace",
                reviewed_revision,
                replacement_revision,
            )
            replaced_bytes = _git_bytes(
                fixture.project,
                "show",
                f"{reviewed_revision}:hardware_informed_design/manifest.json",
            )
            original_bytes = _git_bytes(
                fixture.project,
                "--no-replace-objects",
                "show",
                f"{reviewed_revision}:hardware_informed_design/manifest.json",
            )
            self.assertEqual(replaced_bytes, manifest.read_bytes())
            self.assertNotEqual(original_bytes, manifest.read_bytes())
            decision = _decision(fixture, reviewed_revision=reviewed_revision)
            self.assertInvalid(
                fixture,
                decision,
                "closure differs byte-for-byte from the live worktree",
            )

    def test_git_environment_cannot_redirect_binding_to_a_foreign_repository(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            foreign = fixture.root / "foreign"
            shutil.copytree(
                fixture.project,
                foreign,
                ignore=shutil.ignore_patterns(".git"),
            )
            _git(foreign, "init", "--quiet")
            _git(foreign, "config", "user.name", "Foreign Contract Test")
            _git(foreign, "config", "user.email", "foreign@example.invalid")
            _git(foreign, "add", ".")
            _git(foreign, "commit", "--quiet", "-m", "foreign same-byte closure")
            foreign_revision = _git(foreign, "rev-parse", "HEAD")
            self.assertNotEqual(foreign_revision, fixture.reviewed_revision)
            decision = _decision(fixture, reviewed_revision=foreign_revision)
            _write(fixture.decision, decision)
            result = _run(
                fixture,
                decision=fixture.decision,
                environment={"GIT_DIR": str(foreign / ".git")},
            )

            local_decision = _decision(fixture)
            _write(fixture.decision, local_decision)
            worktree_result = _run(
                fixture,
                decision=fixture.decision,
                environment={"GIT_WORK_TREE": str(foreign)},
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("existing full commit", _combined(result))
        self.assertEqual(worktree_result.returncode, 0, worktree_result.stderr)
        self.assertIs(
            json.loads(worktree_result.stdout)["hardware_review_complete"], True
        )

    def test_git_subprocesses_override_caller_lazy_fetch_setting(self) -> None:
        with _temporary_git_project() as fixture:
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            wrapper_directory = fixture.root / "git-wrapper"
            wrapper_directory.mkdir()
            wrapper = wrapper_directory / "git"
            wrapper.write_text(
                "#!/bin/sh\n"
                'if [ "${GIT_NO_LAZY_FETCH-}" != "1" ]; then\n'
                '  echo "GIT_NO_LAZY_FETCH was not forced to 1" >&2\n'
                "  exit 97\n"
                "fi\n"
                f'exec {shlex.quote(str(real_git))} "$@"\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            decision = _decision(fixture)
            _write(fixture.decision, decision)
            result = _run(
                fixture,
                decision=fixture.decision,
                environment={
                    "GIT_NO_LAZY_FETCH": "0",
                    "PATH": (
                        str(wrapper_directory) + os.pathsep + os.environ.get("PATH", "")
                    ),
                },
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIs(json.loads(result.stdout)["hardware_review_complete"], True)

    def test_missing_promisor_blob_is_not_lazy_fetched_when_git_supports_guard(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            remote = fixture.root / "promisor.git"
            clone = subprocess.run(
                (
                    "git",
                    "clone",
                    "--quiet",
                    "--bare",
                    str(fixture.project),
                    str(remote),
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(clone.returncode, 0, clone.stderr)
            _git(remote, "config", "uploadpack.allowFilter", "true")
            _git(fixture.project, "remote", "add", "origin", str(remote))
            _git(fixture.project, "config", "remote.origin.promisor", "true")
            _git(
                fixture.project,
                "config",
                "remote.origin.partialclonefilter",
                "blob:none",
            )
            blob_id = _git(
                fixture.project,
                "rev-parse",
                f"{fixture.reviewed_revision}:hardware_informed_design/manifest.json",
            )
            loose_blob = (
                fixture.project / ".git" / "objects" / blob_id[:2] / blob_id[2:]
            )
            if not loose_blob.is_file():
                self.skipTest("reviewed manifest blob is not a loose Git object")
            loose_blob.unlink()

            probe_fetch = fixture.root / "probe-fetch"
            probe_guard = fixture.root / "probe-guard"
            shutil.copytree(fixture.project, probe_fetch)
            shutil.copytree(fixture.project, probe_guard)

            def cat_blob(
                project: Path, lazy_fetch: str
            ) -> subprocess.CompletedProcess[bytes]:
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.upper().startswith("GIT_")
                }
                environment["GIT_NO_LAZY_FETCH"] = lazy_fetch
                return subprocess.run(
                    (
                        "git",
                        "--no-replace-objects",
                        "-C",
                        str(project),
                        "cat-file",
                        "blob",
                        blob_id,
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    check=False,
                )

            if cat_blob(probe_fetch, "0").returncode != 0:
                self.skipTest("local Git cannot exercise promisor lazy fetch")
            if cat_blob(probe_guard, "1").returncode == 0:
                self.skipTest("local Git does not implement GIT_NO_LAZY_FETCH")

            before_objects = _object_store_snapshot(fixture.project)
            decision = _decision(fixture)
            _write(fixture.decision, decision)
            result = _run(
                fixture,
                decision=fixture.decision,
                environment={"GIT_NO_LAZY_FETCH": "0"},
            )
            after_objects = _object_store_snapshot(fixture.project)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("could not read reviewed tree blob", _combined(result))
        self.assertEqual(after_objects, before_objects)

    def test_repository_root_must_not_fall_back_to_a_parent_git_worktree(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            shutil.rmtree(fixture.project / ".git")
            _git(fixture.root, "init", "--quiet")
            _git(fixture.root, "config", "user.name", "Parent Contract Test")
            _git(fixture.root, "config", "user.email", "parent@example.invalid")
            _git(fixture.root, "add", "project")
            _git(fixture.root, "commit", "--quiet", "-m", "parent repository")
            parent_revision = _git(fixture.root, "rev-parse", "HEAD")
            decision = _decision(fixture, reviewed_revision=parent_revision)
            result = _write_and_run(fixture, decision)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("repository-root", _combined(result))
        self.assertIn("git worktree discovery", _combined(result))

    def test_fixed_closure_drift_is_rejected_but_unrelated_drift_is_accepted(
        self,
    ) -> None:
        with _temporary_git_project() as fixture:
            manifest = fixture.design / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            _git(fixture.project, "add", str(manifest.relative_to(fixture.project)))
            _git(fixture.project, "commit", "--quiet", "-m", "format manifest")
            decision = _decision(fixture)
            self.assertInvalid(
                fixture,
                decision,
                "closure differs byte-for-byte from the live worktree",
            )

        with _temporary_git_project() as fixture:
            attributes = fixture.project / ".gitattributes"
            attributes.write_text(
                "hardware_informed_design/manifest.json text eol=lf\n",
                encoding="utf-8",
            )
            _git(fixture.project, "add", ".gitattributes")
            _git(
                fixture.project,
                "commit",
                "--quiet",
                "-m",
                "add line-ending normalization",
            )
            manifest = fixture.design / "manifest.json"
            manifest.write_bytes(manifest.read_bytes().replace(b"\n", b"\r\n"))
            decision = _decision(fixture)
            self.assertInvalid(
                fixture,
                decision,
                "closure differs byte-for-byte from the live worktree",
            )

        with _temporary_git_project() as fixture:
            readme = fixture.design / "README.md"
            readme.write_text(
                readme.read_text(encoding="utf-8") + "\nUnrelated review note.\n",
                encoding="utf-8",
            )
            _git(fixture.project, "add", str(readme.relative_to(fixture.project)))
            _git(fixture.project, "commit", "--quiet", "-m", "unrelated readme note")
            decision = _decision(fixture)
            result = _write_and_run(fixture, decision)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIs(json.loads(result.stdout)["hardware_review_complete"], True)

    def test_cli_formats_gate_exit_and_read_only_intake_are_explicit(self) -> None:
        with _temporary_git_project() as fixture:
            decision = _decision(fixture)
            _write(fixture.decision, decision)
            before = {
                str(path.relative_to(fixture.project)): path.read_bytes()
                for path in fixture.project.rglob("*")
                if path.is_file() and ".git" not in path.parts
            }
            decision_before = fixture.decision.read_bytes()
            expected_markers = {
                "text": "hardware review decision present: yes",
                "markdown": "| Hardware review decision present | Yes |",
                "json": '"hardware_review_decision_present": true',
            }
            for output_format, marker in expected_markers.items():
                with self.subTest(output_format=output_format):
                    result = _run(
                        fixture,
                        decision=fixture.decision,
                        output_format=output_format,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(marker, result.stdout)
            after = {
                str(path.relative_to(fixture.project)): path.read_bytes()
                for path in fixture.project.rglob("*")
                if path.is_file() and ".git" not in path.parts
            }

            self.assertEqual(after, before)
            self.assertEqual(fixture.decision.read_bytes(), decision_before)
            self.assertFalse((fixture.design / "decisions").exists())
            request = _read(fixture.project / REVIEW_REQUEST_PATH)
            self.assertEqual(
                request["decision_artifact"],
                {"status": "absent", "path": None, "decision_id": None},
            )
            self.assertTrue(
                all(value is False for value in request["authorizations"].values())
            )

    def test_cli_syntax_errors_exit_two(self) -> None:
        result = subprocess.run(
            (sys.executable, str(TOOL), "--not-a-validator-option"),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("unrecognized arguments", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
