"""Contracts for review-pending hardware-informed design artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "operator_library"
EXTRACTION = ROOT / "abstraction_extraction"
DESIGN = ROOT / "hardware_informed_design"
TARGET = ROOT / "compiler" / "targets" / "apple_gpu_family9.json"
TOOL = ROOT / "tools" / "validate_hardware_informed_design.py"

REVIEW_REQUEST_RELATIVE_PATH = (
    "review_requests/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json"
)
REVIEW_ITEM_IDS = frozenset(
    {
        "vendor-fact-applicability",
        "width-policy-and-fail-closed-selection",
        "partial-cardinality-and-final-reduction",
        "barrier-participation-and-publication",
        "scratch-initialization-lifetime-and-reuse",
        "variant-owner-split",
        "resource-accounting-and-runtime-gates",
        "numeric-scope-and-oracle-boundary",
        "observed-scope-and-authority-boundary",
    }
)
RESOURCE_PHASE_AMBIGUITY_ID = "resource-accounting-phase-order"
RESOURCE_PHASE_OPTION_IDS_ORDERED = (
    "preimplementation-standalone-metal-prototypes",
    "amend-gate-to-port-acceptance-after-bounded-implementation",
)
RESOURCE_PHASE_OPTION_IDS = frozenset(RESOURCE_PHASE_OPTION_IDS_ORDERED)
REVIEWED_ARTIFACT_PATHS = (
    "hardware_informed_design/schema.json",
    "hardware_informed_design/manifest.json",
    "hardware_informed_design/evidence/apple-metal-hierarchical-reduction-v1.json",
    "hardware_informed_design/proposals/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json",
    "hardware_informed_design/review_requests/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json",
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _review_request(design: Path = DESIGN) -> tuple[Path, dict[str, object]]:
    manifest = _read(design / "manifest.json")
    paths = manifest["review_requests"]
    assert paths == [REVIEW_REQUEST_RELATIVE_PATH]
    path = design / REVIEW_REQUEST_RELATIVE_PATH
    return path, _read(path)


def _public_schema_decision() -> dict[str, object]:
    """Return schema-test bytes, not an authoritative review decision fixture."""

    _, request = _review_request()
    return {
        "$schema": (
            "https://open-cake-ir.local/hardware-informed-design/"
            "schema-v1.json#/$defs/hardware_review_decision"
        ),
        "schema_version": 1,
        "decision_id": "schema-contract-only-hardware-review-decision",
        "kind": "external_human_hardware_review_decision",
        "authority": "external_human_hardware_reviewer_only",
        "binding": {
            "design_stage_id": "apple-metal-hardware-informed-design-v1",
            "request_id": request["review_request_id"],
            "path": (f"hardware_informed_design/{REVIEW_REQUEST_RELATIVE_PATH}"),
            "reviewed_git_revision": "0" * 40,
            "reviewed_artifact_paths": list(REVIEWED_ARTIFACT_PATHS),
        },
        "reviewer_attestation": {
            "reviewer_identity": "Schema contract test reviewer",
            "authorship_attestation": "external-human-outside-automation",
            "decision_basis": "Exercise the closed public decision schema only.",
        },
        "global_disposition": "approved",
        "item_verdicts": [
            {
                "review_item_id": item["review_item_id"],
                "verdict": "approved",
                "localized_reason": (
                    f"Schema contract reason for {item['review_item_id']}."
                ),
            }
            for item in request["review_items"]
        ],
        "ambiguity_resolution": {
            "ambiguity_id": RESOURCE_PHASE_AMBIGUITY_ID,
            "selected_option_id": RESOURCE_PHASE_OPTION_IDS_ORDERED[0],
            "localized_reason": (
                "The schema test selects the approval-compatible option."
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


def _definition_validator(
    schema: dict[str, object], definition: str
) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        format_checker=FormatChecker(),
    )


def _candidate_closure_digest(extraction: Path, design: Path) -> str:
    closure = _read(design / "manifest.json")["input_candidate_closure"]
    assert isinstance(closure, dict)

    def extraction_path(repository_relative: object) -> Path:
        relative = Path(str(repository_relative))
        return extraction / relative.relative_to("abstraction_extraction")

    value = {
        "candidate": _read(extraction_path(closure["candidate_path"])),
        "observations": {
            str(relative): _read(extraction_path(relative))
            for relative in sorted(closure["observation_paths"])
        },
    }
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _refresh_candidate_closure(extraction: Path, design: Path) -> None:
    path = design / "manifest.json"
    manifest = _read(path)
    manifest["input_candidate_closure"]["candidate_closure_sha256"] = (
        _candidate_closure_digest(extraction, design)
    )
    _write(path, manifest)


def _run(
    library: Path = LIBRARY,
    extraction: Path = EXTRACTION,
    design: Path = DESIGN,
    target: Path = TARGET,
    output_format: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(TOOL),
        "--library-root",
        str(library),
        "--extraction-root",
        str(extraction),
        "--design-root",
        str(design),
        "--target-path",
        str(target),
    ]
    if output_format is not None:
        command.extend(("--format", output_format))
    return subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


Mutation = Callable[[Path, Path, Path, Path], None]


class HardwareInformedDesignSchemaTests(unittest.TestCase):
    def test_every_object_schema_is_closed(self) -> None:
        schema = _read(DESIGN / "schema.json")
        missing: list[str] = []

        def visit(value: object, path: str) -> None:
            if isinstance(value, dict):
                if (
                    value.get("type") == "object"
                    and value.get("additionalProperties") is not False
                ):
                    missing.append(path)
                for key, item in value.items():
                    visit(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(schema, "$")
        self.assertEqual(missing, [])

    def test_all_five_document_kinds_conform_to_the_public_schema(self) -> None:
        schema = _read(DESIGN / "schema.json")
        Draft202012Validator.check_schema(schema)
        manifest = _read(DESIGN / "manifest.json")
        documents = {
            "manifest": [manifest],
            "evidence": [_read(DESIGN / path) for path in manifest["evidence"]],
            "proposal": [_read(DESIGN / path) for path in manifest["proposals"]],
            "review_request": [
                _read(DESIGN / path) for path in manifest["review_requests"]
            ],
            "hardware_review_decision": [_public_schema_decision()],
        }
        root_validator = Draft202012Validator(schema, format_checker=FormatChecker())
        for definition, values in documents.items():
            self.assertTrue(values)
            for value in values:
                for mode, validator in (
                    ("definition", _definition_validator(schema, definition)),
                    ("root", root_validator),
                ):
                    with self.subTest(definition=definition, mode=mode):
                        errors = sorted(
                            validator.iter_errors(value),
                            key=lambda error: tuple(
                                str(part) for part in error.absolute_path
                            ),
                        )
                        self.assertEqual(
                            errors,
                            [],
                            "\n".join(
                                f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                                for error in errors
                            ),
                        )


class HardwareInformedDesignValidatorTests(unittest.TestCase):
    def _assert_mutation_rejected(
        self,
        mutate: Mutation,
        expected_diagnostics: str | tuple[str, ...],
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            design = fixture / "hardware_informed_design"
            target = fixture / "apple_gpu_family9.json"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            shutil.copytree(DESIGN, design)
            shutil.copy2(TARGET, target)
            mutate(library, extraction, design, target)
            result = _run(library, extraction, design, target)

        combined = (result.stderr + result.stdout).lower()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        for diagnostic in (
            (expected_diagnostics,)
            if isinstance(expected_diagnostics, str)
            else expected_diagnostics
        ):
            self.assertIn(diagnostic.lower(), combined)

    def test_cli_validates_metadata_without_setting_review_readiness(self) -> None:
        result = _run(output_format="json")

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        manifest = _read(DESIGN / "manifest.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["design_stage_id"], manifest["design_stage_id"])
        self.assertEqual(summary["counts"], manifest["counts"])
        self.assertEqual(summary["review_request_count"], 1)
        self.assertIs(summary["hardware_review_decision_present"], False)
        self.assertIs(summary["resource_phase_ambiguity_exposed"], True)
        self.assertIs(summary["ready_for_principle_review"], False)
        self.assertIs(summary["hardware_semantics_verified"], False)
        self.assertIs(summary["performance_claim_authorized"], False)
        self.assertIs(summary["compiler_change_authorized"], False)
        self.assertIs(summary["remote_source_bytes_verified"], False)
        self.assertIs(summary["engineering_observation_record_consistent"], True)
        self.assertIs(summary["retained_local_probe_reported_outcome"], True)
        self.assertEqual(summary["retained_local_probe_case_count"], 6)
        self.assertEqual(summary["retained_local_probe_epoch_count"], 12)
        self.assertEqual(
            summary["retained_local_probe_scope"],
            "observed_apple_m4_fp32_rms_probe_pipeline_only",
        )
        self.assertIs(summary["hardware_review_complete"], False)
        self.assertIs(summary["evaluation_evidence_present"], False)
        self.assertEqual(
            summary["verification_scope"],
            "offline_metadata_and_derivation_validation",
        )

    def test_cli_supports_text_markdown_and_json_reports(self) -> None:
        expected = {
            None: "hardware-informed design metadata consistent:",
            "text": "hardware-informed design metadata consistent:",
            "markdown": "# Hardware-informed design metadata consistency",
            "json": '"ready_for_principle_review": false',
        }
        for output_format, marker in expected.items():
            with self.subTest(output_format=output_format or "default"):
                result = _run(output_format=output_format)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(marker, result.stdout)

    def test_review_request_manifest_membership_and_count_are_exact(self) -> None:
        manifest = _read(DESIGN / "manifest.json")
        self.assertEqual(manifest["review_requests"], [REVIEW_REQUEST_RELATIVE_PATH])
        self.assertEqual(manifest["counts"]["review_request_count"], 1)

        mutations = (
            ("review_requests", []),
            ("review_request_count", 0),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: object = replacement,
                ) -> None:
                    del library, extraction, target
                    path = design / "manifest.json"
                    manifest = _read(path)
                    if field == "review_requests":
                        manifest[field] = replacement
                    else:
                        manifest["counts"][field] = replacement
                    _write(path, manifest)

                self._assert_mutation_rejected(mutate, field)

    def test_review_request_binds_the_actual_stage_three_document_closures(
        self,
    ) -> None:
        manifest = _read(DESIGN / "manifest.json")
        evidence_path = DESIGN / str(manifest["evidence"][0])
        proposal_path = DESIGN / str(manifest["proposals"][0])
        evidence = _read(evidence_path)
        proposal = _read(proposal_path)
        _, request = _review_request()

        self.assertEqual(
            request["stage_binding"],
            {
                "manifest": {
                    "design_stage_id": manifest["design_stage_id"],
                    "path": "hardware_informed_design/manifest.json",
                },
                "evidence": {
                    "evidence_id": evidence["evidence_id"],
                    "path": f"hardware_informed_design/{manifest['evidence'][0]}",
                },
                "proposal": {
                    "proposal_id": proposal["proposal_id"],
                    "path": f"hardware_informed_design/{manifest['proposals'][0]}",
                },
            },
        )

        closure = request["closure"]
        expected_closures = {
            "fact_ids": {value["fact_id"] for value in evidence["facts"]},
            "decision_ids": {value["decision_id"] for value in proposal["decisions"]},
            "hypothesis_ids": {
                value["hypothesis_id"] for value in proposal["hypotheses"]
            },
            "observed_scope_finding_ids": {
                value["finding_id"] for value in proposal["observed_scope_findings"]
            },
            "falsifier_ids": {
                value["falsifier_id"] for value in proposal["falsifiers"]
            },
        }
        for field, expected in expected_closures.items():
            with self.subTest(field=field):
                self.assertEqual(set(closure[field]), expected)
        self.assertEqual(
            closure["counts"],
            {
                "fact_count": len(expected_closures["fact_ids"]),
                "decision_count": len(expected_closures["decision_ids"]),
                "hypothesis_count": len(expected_closures["hypothesis_ids"]),
                "observed_scope_finding_count": len(
                    expected_closures["observed_scope_finding_ids"]
                ),
                "falsifier_count": len(expected_closures["falsifier_ids"]),
            },
        )

    def test_review_request_stage_and_atomic_reference_drift_is_rejected(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            request["stage_binding"]["proposal"]["proposal_id"] = (
                "different-hardware-proposal"
            )
            request["closure"]["fact_ids"].pop()
            request["closure"]["counts"]["fact_count"] -= 1
            request["review_items"][0]["fact_ids"].append("imaginary-hardware-fact")
            _write(path, request)

        self._assert_mutation_rejected(
            mutate,
            (
                "stage_binding.proposal.proposal_id",
                "fact_ids",
                "imaginary-hardware-fact",
            ),
        )

    def test_review_items_have_exact_ids_and_cover_every_atomic_reference(
        self,
    ) -> None:
        _, request = _review_request()
        items = request["review_items"]
        item_ids = {value["review_item_id"] for value in items}
        self.assertEqual(item_ids, REVIEW_ITEM_IDS)
        contract = request["external_human_decision_contract"]
        self.assertEqual(set(contract["required_review_item_ids"]), REVIEW_ITEM_IDS)

        reference_fields = (
            "fact_ids",
            "decision_ids",
            "hypothesis_ids",
            "observed_scope_finding_ids",
            "falsifier_ids",
        )
        for field in reference_fields:
            with self.subTest(field=field):
                closure = set(request["closure"][field])
                used = {reference for item in items for reference in item[field]}
                self.assertEqual(used, closure)
                self.assertTrue(
                    all(set(item[field]).issubset(closure) for item in items)
                )
                self.assertTrue(all(item[field] for item in items))
        self.assertTrue(all(value["required_human_judgment"] for value in items))

    def test_review_item_id_and_coverage_weakening_is_rejected(self) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
            (
                "review_items",
                lambda request: request["review_items"].pop(),
            ),
            (
                "required_review_item_ids",
                lambda request: request["external_human_decision_contract"][
                    "required_review_item_ids"
                ].pop(),
            ),
            (
                "observed_scope_finding_ids",
                lambda request: [
                    item["observed_scope_finding_ids"].remove(
                        "observed-runtime-resource-gates"
                    )
                    for item in request["review_items"]
                    if "observed-runtime-resource-gates"
                    in item["observed_scope_finding_ids"]
                ],
            ),
        )
        for expected, edit in mutations:
            with self.subTest(expected=expected):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    edit: Callable[[dict[str, object]], None] = edit,
                ) -> None:
                    del library, extraction, target
                    path, request = _review_request(design)
                    edit(request)
                    _write(path, request)

                self._assert_mutation_rejected(mutate, expected)

    def test_review_item_reference_mapping_cannot_be_remapped_with_known_ids(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            item = next(
                value
                for value in request["review_items"]
                if value["review_item_id"] == "vendor-fact-applicability"
            )
            item["fact_ids"].append("observed-m4-hierarchical-runtime-resource-gates")
            _write(path, request)

        self._assert_mutation_rejected(
            mutate, ("vendor-fact-applicability", "fact_ids")
        )

    def test_resource_phase_order_ambiguity_requires_one_external_human_choice(
        self,
    ) -> None:
        _, request = _review_request()
        ambiguity = request["phase_order_ambiguity"]
        self.assertEqual(ambiguity["ambiguity_id"], RESOURCE_PHASE_AMBIGUITY_ID)
        self.assertEqual(ambiguity["status"], "unresolved_requires_human_choice")
        self.assertIs(ambiguity["selection_required"], True)
        self.assertIs(ambiguity["automation_may_select"], False)
        self.assertEqual(
            {value["option_id"] for value in ambiguity["resolution_options"]},
            RESOURCE_PHASE_OPTION_IDS,
        )
        contract = request["external_human_decision_contract"][
            "ambiguity_resolution_contract"
        ]
        self.assertEqual(contract["required_ambiguity_id"], RESOURCE_PHASE_AMBIGUITY_ID)
        self.assertEqual(set(contract["allowed_option_ids"]), RESOURCE_PHASE_OPTION_IDS)
        self.assertEqual(
            contract["approval_compatible_option_id"],
            RESOURCE_PHASE_OPTION_IDS_ORDERED[0],
        )
        self.assertEqual(
            contract["successor_required_option_id"],
            RESOURCE_PHASE_OPTION_IDS_ORDERED[1],
        )
        self.assertEqual(
            contract["successor_required_option_allowed_global_dispositions"],
            ["changes_requested", "rejected"],
        )
        self.assertEqual(
            contract["successor_required_option_required_non_approved_review_item_id"],
            "resource-accounting-and-runtime-gates",
        )

    def test_resource_phase_order_options_cannot_be_removed_or_auto_resolved(
        self,
    ) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
            (
                "status",
                lambda request: request["phase_order_ambiguity"].__setitem__(
                    "status", "automatically_resolved"
                ),
            ),
            (
                "resolution_options",
                lambda request: request["phase_order_ambiguity"][
                    "resolution_options"
                ].pop(),
            ),
            (
                "selection_required",
                lambda request: request["phase_order_ambiguity"].__setitem__(
                    "selection_required", False
                ),
            ),
            (
                "automation_may_select",
                lambda request: request["phase_order_ambiguity"].__setitem__(
                    "automation_may_select", True
                ),
            ),
            (
                "allowed_option_ids",
                lambda request: request["external_human_decision_contract"][
                    "ambiguity_resolution_contract"
                ]["allowed_option_ids"].pop(),
            ),
            (
                "approval_compatible_option_id",
                lambda request: request["external_human_decision_contract"][
                    "ambiguity_resolution_contract"
                ].__setitem__(
                    "approval_compatible_option_id",
                    RESOURCE_PHASE_OPTION_IDS_ORDERED[1],
                ),
            ),
            (
                "successor_required_option_allowed_global_dispositions",
                lambda request: request["external_human_decision_contract"][
                    "ambiguity_resolution_contract"
                ]["successor_required_option_allowed_global_dispositions"].append(
                    "approved"
                ),
            ),
            (
                "successor_required_option_required_non_approved_review_item_id",
                lambda request: request["external_human_decision_contract"][
                    "ambiguity_resolution_contract"
                ].__setitem__(
                    "successor_required_option_required_non_approved_review_item_id",
                    "vendor-fact-applicability",
                ),
            ),
        )
        for expected, edit in mutations:
            with self.subTest(expected=expected):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    edit: Callable[[dict[str, object]], None] = edit,
                ) -> None:
                    del library, extraction, target
                    path, request = _review_request(design)
                    edit(request)
                    _write(path, request)

                self._assert_mutation_rejected(mutate, expected)

    def test_review_request_human_resolution_prose_is_semantically_pinned(
        self,
    ) -> None:
        mutations: list[tuple[str, Callable[[dict[str, object]], None]]] = [
            (
                "phase_order_ambiguity.contradiction",
                lambda request: request["phase_order_ambiguity"].__setitem__(
                    "contradiction", "Automation has resolved the phase order."
                ),
            ),
            (
                "stage_transition_rule",
                lambda request: request["external_human_decision_contract"].__setitem__(
                    "stage_transition_rule",
                    "Any global approval advances Stage 3 despite item verdicts.",
                ),
            ),
        ]
        for option_index, option_id in enumerate(RESOURCE_PHASE_OPTION_IDS_ORDERED):
            for field in ("action", "consequence"):
                mutations.append(
                    (
                        f"{option_id}.{field}",
                        lambda request, option_index=option_index, field=field: request[
                            "phase_order_ambiguity"
                        ]["resolution_options"][option_index].__setitem__(
                            field, f"Automatically rewritten {field}."
                        ),
                    )
                )

        for expected, edit in mutations:
            with self.subTest(expected=expected):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    edit: Callable[[dict[str, object]], None] = edit,
                ) -> None:
                    del library, extraction, target
                    path, request = _review_request(design)
                    edit(request)
                    _write(path, request)

                expected_parts = (
                    ("resolution_options", expected.rsplit(".", 1)[-1])
                    if expected.startswith(tuple(RESOURCE_PHASE_OPTION_IDS))
                    else tuple(expected.split("."))
                )
                self._assert_mutation_rejected(mutate, expected_parts)

    def test_review_request_contains_no_decision_reviewer_or_readiness(self) -> None:
        _, request = _review_request()
        self.assertEqual(
            request["decision_artifact"],
            {"status": "absent", "path": None, "decision_id": None},
        )
        self.assertEqual(request["authority"], "external_human_hardware_reviewer_only")
        self.assertIs(
            request["external_human_decision_contract"][
                "automation_may_write_decision"
            ],
            False,
        )
        dispositions = {"approved", "changes_requested", "rejected"}
        self.assertEqual(
            set(request["external_human_decision_contract"]["global_dispositions"]),
            dispositions,
        )
        self.assertEqual(
            set(
                request["external_human_decision_contract"][
                    "per_item_verdict_contract"
                ]["verdict_values"]
            ),
            dispositions,
        )
        self.assertTrue(
            all(value is False for value in request["authorizations"].values())
        )
        for forbidden in (
            "reviewer",
            "decision",
            "approval",
            "ready",
            "ready_for_principle_review",
            "hardware_review_complete",
        ):
            self.assertNotIn(forbidden, request)

    def test_review_request_cannot_inject_a_decision_reviewer_or_readiness(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            request["reviewer"] = "automation"
            request["decision"] = "approved"
            request["approval"] = True
            request["ready"] = True
            request["ready_for_principle_review"] = True
            request["hardware_review_complete"] = True
            _write(path, request)

        self._assert_mutation_rejected(
            mutate,
            (
                "reviewer",
                "decision",
                "approval",
                "ready",
                "ready_for_principle_review",
                "hardware_review_complete",
            ),
        )

    def test_review_request_state_and_authority_cannot_be_automated(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            request["state"] = "approved"
            request["authority"] = "offline_validator"
            contract = request["external_human_decision_contract"]
            contract["decision_authority"] = "offline_validator"
            contract["per_item_verdict_contract"]["per_item_verdict_required"] = False
            contract["per_item_verdict_contract"]["localized_reason_required"] = False
            contract["reviewed_git_revision_required"] = False
            _write(path, request)

        self._assert_mutation_rejected(
            mutate,
            (
                "state",
                "authority",
                "decision_authority",
                "per_item_verdict_required",
                "localized_reason_required",
                "reviewed_git_revision_required",
            ),
        )

    def test_per_item_aggregate_gate_requires_every_item_to_be_approved(
        self,
    ) -> None:
        aggregate_fields = (
            "stage_clear_requires_all_item_verdicts_approved",
            "non_approved_item_blocks_stage_transition",
        )
        _, request = _review_request()
        contract = request["external_human_decision_contract"][
            "per_item_verdict_contract"
        ]
        self.assertEqual(
            contract["global_disposition_aggregation"],
            "maximum_review_item_verdict_severity",
        )
        self.assertEqual(
            contract["verdict_severity_order"],
            ["approved", "changes_requested", "rejected"],
        )
        for field in aggregate_fields:
            self.assertIs(contract[field], True)

        for field, replacement in (
            ("global_disposition_aggregation", "first_item_wins"),
            ("verdict_severity_order", ["rejected", "changes_requested", "approved"]),
        ):
            with self.subTest(field=field):

                def mutate_aggregation(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: object = replacement,
                ) -> None:
                    del library, extraction, target
                    path, request = _review_request(design)
                    request["external_human_decision_contract"][
                        "per_item_verdict_contract"
                    ][field] = replacement
                    _write(path, request)

                self._assert_mutation_rejected(mutate_aggregation, field)

        mutations: tuple[tuple[str, str], ...] = tuple(
            (field, action)
            for field in aggregate_fields
            for action in ("false", "delete")
        )
        for field, action in mutations:
            with self.subTest(field=field, action=action):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    action: str = action,
                ) -> None:
                    del library, extraction, target
                    path, request = _review_request(design)
                    contract = request["external_human_decision_contract"][
                        "per_item_verdict_contract"
                    ]
                    if action == "delete":
                        del contract[field]
                    else:
                        contract[field] = False
                    _write(path, request)

                self._assert_mutation_rejected(mutate, field)

    def test_review_request_cannot_grant_any_authority(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            for field in request["authorizations"]:
                request["authorizations"][field] = True
            request["external_human_decision_contract"][
                "automation_may_write_decision"
            ] = True
            _write(path, request)

        _, request = _review_request()
        self._assert_mutation_rejected(
            mutate,
            tuple(request["authorizations"]) + ("automation_may_write_decision",),
        )

    def test_passing_m4_probe_cannot_substitute_for_a_human_decision(self) -> None:
        manifest = _read(DESIGN / "manifest.json")
        evidence = _read(DESIGN / str(manifest["evidence"][0]))
        aggregate = evidence["hierarchical_reduction_probe_observation"][
            "normalized_output"
        ]["aggregate"]
        _, request = _review_request()
        result = _run(output_format="json")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)

        self.assertIs(aggregate["all_cases_passed"], True)
        self.assertEqual(request["decision_artifact"]["status"], "absent")
        self.assertIs(summary["hardware_review_decision_present"], False)
        self.assertIs(summary["hardware_review_complete"], False)
        self.assertIs(summary["ready_for_principle_review"], False)

        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, request = _review_request(design)
            request["decision_artifact"] = {
                "status": "approved_from_local_probe",
                "path": None,
                "decision_id": (
                    "local-apple-m4-hierarchical-reduction-probe-2026-08-26"
                ),
            }
            request["authorizations"]["human_hardware_review_cleared"] = True
            request["authorizations"]["principle_driven_iteration_authorized"] = True
            request["authorizations"]["implementation_authorized"] = True
            _write(path, request)

        self._assert_mutation_rejected(
            mutate,
            (
                "decision_artifact",
                "human_hardware_review_cleared",
                "principle_driven_iteration_authorized",
                "implementation_authorized",
            ),
        )

    def test_cli_rejects_a_weakened_public_review_request_schema(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path = design / "schema.json"
            schema = _read(path)
            schema["$defs"]["review_request"] = {}
            _write(path, schema)

        self._assert_mutation_rejected(mutate, "schema.json.$defs.review_request")

    def test_cli_rejects_broken_public_review_request_safety_references(
        self,
    ) -> None:
        mutations: tuple[tuple[str, dict[str, object]], ...] = (
            ("stage_binding", {"type": "object"}),
            ("closure", {"type": "object"}),
            ("review_items", {"type": "array"}),
            ("phase_order_ambiguity", {"type": "object"}),
            ("decision_artifact", {"type": "object"}),
            ("external_human_decision_contract", {"type": "object"}),
            ("authorizations", {"type": "object"}),
            ("non_claims", {"type": "array"}),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: dict[str, object] = replacement,
                ) -> None:
                    del library, extraction, target
                    path = design / "schema.json"
                    schema = _read(path)
                    schema["$defs"]["review_request"]["properties"][field] = replacement
                    _write(path, schema)

                self._assert_mutation_rejected(
                    mutate,
                    f"schema.json.$defs.review_request.properties.{field}",
                )

    def test_cli_rejects_weakened_external_human_decision_schema_gates(
        self,
    ) -> None:
        mutations: tuple[tuple[str, dict[str, object]], ...] = (
            (
                "global_dispositions",
                {"type": "array", "items": {"type": "string"}},
            ),
            ("required_review_item_ids", {"type": "array"}),
            ("per_item_verdict_contract", {"type": "object"}),
            ("reviewed_git_revision_required", {"type": "boolean"}),
            ("ambiguity_resolution_contract", {"type": "object"}),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: dict[str, object] = replacement,
                ) -> None:
                    del library, extraction, target
                    path = design / "schema.json"
                    schema = _read(path)
                    schema["$defs"]["external_human_decision_contract"]["properties"][
                        field
                    ] = replacement
                    _write(path, schema)

                self._assert_mutation_rejected(
                    mutate,
                    (
                        "schema.json.$defs.external_human_decision_contract."
                        f"properties.{field}"
                    ),
                )

    def test_cli_rejects_weakened_public_review_authority_schema_fields(
        self,
    ) -> None:
        authorization_fields = (
            "human_hardware_review_cleared",
            "principle_driven_iteration_authorized",
            "compiler_change_authorized",
            "implementation_authorized",
            "evaluation_authorized",
            "performance_claim_authorized",
            "scientific_claim_authorized",
            "promotion_authorized",
        )
        mutations: list[tuple[str, str, dict[str, object]]] = [
            (
                "review_authorizations",
                field,
                {"type": "boolean"},
            )
            for field in authorization_fields
        ]
        mutations.extend(
            (
                "absent_decision_artifact",
                field,
                (
                    {"type": "string"}
                    if field == "status"
                    else {"type": ["null", "string"]}
                ),
            )
            for field in ("status", "path", "decision_id")
        )
        mutations.extend(
            (
                definition,
                field,
                {"type": "boolean"},
            )
            for definition, field in (
                ("phase_order_ambiguity", "automation_may_select"),
                (
                    "external_human_decision_contract",
                    "automation_may_write_decision",
                ),
                (
                    "per_item_verdict_contract",
                    "stage_clear_requires_all_item_verdicts_approved",
                ),
                (
                    "per_item_verdict_contract",
                    "non_approved_item_blocks_stage_transition",
                ),
                (
                    "hierarchical_probe_aggregate",
                    "all_command_buffers_completed",
                ),
                ("hierarchical_probe_aggregate", "all_cases_passed"),
            )
        )

        for definition, field, replacement in mutations:
            with self.subTest(definition=definition, field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    definition: str = definition,
                    field: str = field,
                    replacement: dict[str, object] = replacement,
                ) -> None:
                    del library, extraction, target
                    path = design / "schema.json"
                    schema = _read(path)
                    schema["$defs"][definition]["properties"][field] = replacement
                    _write(path, schema)

                self._assert_mutation_rejected(
                    mutate,
                    f"schema.json.$defs.{definition}.properties.{field}",
                )

    def test_cli_rejects_weakened_public_phase_resolution_schema_fields(
        self,
    ) -> None:
        mutations: list[tuple[str, str, dict[str, object]]] = [
            ("phase_order_ambiguity", "contradiction", {"type": "string"}),
            ("phase_order_ambiguity", "resolution_options", {"type": "array"}),
            (
                "external_human_decision_contract",
                "stage_transition_rule",
                {"type": "string"},
            ),
        ]
        for definition in (
            "preimplementation_standalone_metal_prototypes_option",
            "amend_gate_after_bounded_implementation_option",
        ):
            for field in ("option_id", "action", "consequence"):
                mutations.append((definition, field, {"type": "string"}))

        for definition, field, replacement in mutations:
            with self.subTest(definition=definition, field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    definition: str = definition,
                    field: str = field,
                    replacement: dict[str, object] = replacement,
                ) -> None:
                    del library, extraction, target
                    path = design / "schema.json"
                    schema = _read(path)
                    schema["$defs"][definition]["properties"][field] = replacement
                    _write(path, schema)

                self._assert_mutation_rejected(
                    mutate,
                    f"schema.json.$defs.{definition}.properties.{field}",
                )

    def test_cli_rejects_weakened_public_hardware_review_decision_schema(
        self,
    ) -> None:
        def remove_root_entry(schema: dict[str, object]) -> None:
            schema["oneOf"].pop()

        def replace_definition(schema: dict[str, object]) -> None:
            schema["$defs"]["hardware_review_decision"] = {}

        mutations: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
            ("schema.json.oneOf", remove_root_entry),
            (
                "schema.json.$defs.hardware_review_decision",
                replace_definition,
            ),
            (
                "hardware_review_decision.properties.binding",
                lambda schema: schema["$defs"]["hardware_review_decision"][
                    "properties"
                ].__setitem__("binding", {"type": "object"}),
            ),
            (
                "hardware_review_decision.properties.authority",
                lambda schema: schema["$defs"]["hardware_review_decision"][
                    "properties"
                ].__setitem__("authority", {"type": "string"}),
            ),
            (
                "review_decision_request_binding.properties.reviewed_artifact_paths",
                lambda schema: schema["$defs"]["review_decision_request_binding"][
                    "properties"
                ].__setitem__("reviewed_artifact_paths", {"type": "array"}),
            ),
            (
                "review_item_verdict.properties.review_item_id",
                lambda schema: schema["$defs"]["review_item_verdict"][
                    "properties"
                ].__setitem__("review_item_id", {"$ref": "#/$defs/id"}),
            ),
            (
                "review_scope_acknowledgements.properties.implementation_authorized",
                lambda schema: schema["$defs"]["review_scope_acknowledgements"][
                    "properties"
                ].__setitem__("implementation_authorized", {"type": "boolean"}),
            ),
            (
                "schema.json.$defs.git_commit",
                lambda schema: schema["$defs"]["git_commit"].__setitem__(
                    "pattern", ".+"
                ),
            ),
            (
                "schema.json.$defs.human_review_text",
                lambda schema: schema["$defs"]["human_review_text"].__setitem__(
                    "pattern", ".*"
                ),
            ),
            (
                "per_item_verdict_contract.properties.global_disposition_aggregation",
                lambda schema: schema["$defs"]["per_item_verdict_contract"][
                    "properties"
                ].__setitem__("global_disposition_aggregation", {"type": "string"}),
            ),
            (
                "ambiguity_resolution_contract.properties.approval_compatible_option_id",
                lambda schema: schema["$defs"]["ambiguity_resolution_contract"][
                    "properties"
                ].__setitem__("approval_compatible_option_id", {"type": "string"}),
            ),
        )
        for expected, edit in mutations:
            with self.subTest(expected=expected):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    edit: Callable[[dict[str, object]], None] = edit,
                ) -> None:
                    del library, extraction, target
                    path = design / "schema.json"
                    schema = _read(path)
                    edit(schema)
                    _write(path, schema)

                self._assert_mutation_rejected(mutate, expected)

    def test_cli_rejects_a_weakened_public_falsifier_schema(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path = design / "schema.json"
            schema = _read(path)
            schema["$defs"]["falsifier"] = {}
            _write(path, schema)

        self._assert_mutation_rejected(mutate, "schema.json.$defs.falsifier")

    def test_abstraction_extraction_is_validated_before_design_artifacts(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, target
            manifest = _read(extraction / "manifest.json")
            candidate_path = extraction / str(manifest["candidates"][0])
            candidate = _read(candidate_path)
            candidate["state"] = "prematurely_reviewed"
            _write(candidate_path, candidate)
            _write(design / "schema.json", {})

        self._assert_mutation_rejected(mutate, "extracted_unreviewed")

    def test_live_candidate_closure_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, design, target
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["summary"] = f"{candidate['summary']} Drift."
            _write(path, candidate)

        self._assert_mutation_rejected(mutate, ("candidate closure", "sha256"))

    def test_candidate_identity_is_bound_beyond_the_closure_digest(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, target
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["candidate_id"] = "different-reviewed-candidate"
            _write(path, candidate)
            _refresh_candidate_closure(extraction, design)

        self._assert_mutation_rejected(mutate, "candidate_id")

    def test_candidate_binding_requires_the_extracted_state_and_stage_three_gate(
        self,
    ) -> None:
        fields = {
            "state": ("prematurely_reviewed", "extracted_unreviewed"),
            "next_gate": ("principle_driven_iteration", "hardware_informed_design"),
        }
        for field, (replacement, expected) in fields.items():
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: str = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    binding_field = {
                        "state": "expected_state",
                        "next_gate": "expected_next_gate",
                    }[field]
                    proposal["candidate_binding"][binding_field] = replacement
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, expected)

    def test_compiler_target_is_explicitly_non_evidentiary_context(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            evidence["non_evidentiary_context"]["role"] = "hardware_evidence"
            _write(path, evidence)

        self._assert_mutation_rejected(mutate, "non_evidentiary_context_only")

    def test_repository_snapshot_parent_revisions_and_paths_are_bound(self) -> None:
        mutations = (
            (
                "local_probe_observation",
                "git_parent_revision",
                "0" * 40,
            ),
            (
                "local_probe_observation",
                "probe_path",
                "tools/different_probe.py",
            ),
            (
                "non_evidentiary_context",
                "git_parent_revision",
                "0" * 40,
            ),
            (
                "non_evidentiary_context",
                "target_path",
                "compiler/targets/different_target.json",
            ),
            (
                "non_evidentiary_context",
                "lowering_path",
                "src/open_cake_ir/compiler/different_lowering.py",
            ),
        )
        for section, field, replacement in mutations:
            with self.subTest(section=section, field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    section: str = section,
                    field: str = field,
                    replacement: str = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["evidence"][0])
                    evidence = _read(path)
                    evidence[section]["repository_snapshot"][field] = replacement
                    _write(path, evidence)

                self._assert_mutation_rejected(mutate, field)

    def test_device_threadgroup_dimension_source_and_fact_are_exactly_bound(
        self,
    ) -> None:
        def mutate_source(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            source = next(
                value
                for value in evidence["sources"]
                if value["source_id"] == "apple-api-maxthreadsperthreadgroup"
            )
            source["url"] = "https://example.invalid/maxthreadsperthreadgroup"
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate_source,
            "sources[5].url",
        )

        def mutate_fact(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            fact = next(
                value
                for value in evidence["facts"]
                if value["fact_id"]
                == "metal-device-threadgroup-dimension-limits-must-be-queried"
            )
            fact["statement"] = "A weakened per-dimension limit statement."
            fact["source_ids"] = ["apple-api-maxthreadsperthreadgroup"]
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate_fact,
            ("statement", "source_ids"),
        )

    def test_hierarchical_probe_source_and_external_artifact_metadata_are_bound(
        self,
    ) -> None:
        def mutate_source_binding(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            snapshot = evidence["hierarchical_reduction_probe_observation"][
                "repository_snapshot"
            ]
            snapshot["repository_revision"] = "0" * 40
            snapshot["worktree_clean_at_observation"] = False
            snapshot["source_paths"][-1] = (
                "tools/metal_hierarchical_reduction_probe/different_runner.swift"
            )
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate_source_binding,
            (
                "repository_revision",
                "worktree_clean_at_observation",
                "source_paths",
            ),
        )

        def mutate_artifact_binding(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            artifact = evidence["hierarchical_reduction_probe_observation"][
                "retained_artifact"
            ]
            artifact["relative_path"] = (
                "different-revision/metal-hierarchical-reduction.json"
            )
            artifact["reference_status"] = "verified_external_artifact"
            artifact["offline_validator_access"] = "available"
            _write(path, evidence)

        # The validator receives no artifact root. It validates an explicitly
        # unverified reference and never claims access to or byte identity for it.
        self._assert_mutation_rejected(
            mutate_artifact_binding,
            (
                "retained_artifact.relative_path",
                "retained_artifact.reference_status",
                "retained_artifact.offline_validator_access",
            ),
        )

    def test_hierarchical_probe_no_claim_fields_cannot_grant_authority(self) -> None:
        false_observation_fields = (
            "raw_runner_output_retained",
            "stable_device_identifiers_retained",
            "external_workload_oracle_used",
            "performance_measured",
            "evaluation_evidence",
            "scientific_claim_authorized",
            "compiler_change_authorized",
            "human_hardware_review_cleared",
            "promotion_authorized",
        )

        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            observation = evidence["hierarchical_reduction_probe_observation"]
            for field in false_observation_fields:
                observation[field] = True
            coverage = observation["normalized_output"]["coverage"]
            coverage["layer_all_simdgroups_tested"] = True
            coverage["bfloat_input_conversion_tested"] = True
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate,
            false_observation_fields
            + ("layer_all_simdgroups_tested", "bfloat_input_conversion_tested"),
        )

    def test_hierarchical_probe_case_matrix_and_results_are_exactly_bound(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            output = evidence["hierarchical_reduction_probe_observation"][
                "normalized_output"
            ]
            output["coverage"]["simdgroup_counts"][-1] = 31
            output["coverage"]["threads_per_threadgroup"][-1] = 992
            output["case_results"][0]["expected_owner_sums"][0] = 13.5
            output["case_results"][1]["observed_owner_sums"][1] = -50.0
            output["case_results"][2]["broadcast_thread_counts"][0] = 127
            output["case_results"][3]["simdgroup_count"] = 7
            output["aggregate"]["broadcast_consumer_check_count"] = 4031
            output["aggregate"]["all_cases_passed"] = False
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate,
            (
                "coverage.simdgroup_counts",
                "coverage.threads_per_threadgroup",
                "case_results[0].expected_owner_sums",
                "case_results[1].observed_owner_sums",
                "case_results[2].broadcast_thread_counts",
                "case_results[3].simdgroup_count",
                "aggregate.broadcast_consumer_check_count",
                "aggregate.all_cases_passed",
            ),
        )

    def test_hierarchical_probe_runtime_resource_observations_are_exactly_bound(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            output = evidence["hierarchical_reduction_probe_observation"][
                "normalized_output"
            ]
            output["device"]["max_threadgroup_memory_bytes"] = 16384
            output["pipeline"]["thread_execution_width"] = 64
            output["pipeline"]["max_total_threads_per_threadgroup"] = 512
            output["pipeline"]["static_threadgroup_memory_bytes"] = 16
            output["pipeline"]["dynamic_threadgroup_memory_bytes"] = 128
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate,
            (
                "device.max_threadgroup_memory_bytes",
                "pipeline.thread_execution_width",
                "pipeline.max_total_threads_per_threadgroup",
                "pipeline.static_threadgroup_memory_bytes",
                "pipeline.dynamic_threadgroup_memory_bytes",
            ),
        )

    def test_hierarchical_probe_falsifier_outcomes_have_exact_closure_and_mapping(
        self,
    ) -> None:
        manifest = _read(DESIGN / "manifest.json")
        evidence = _read(DESIGN / str(manifest["evidence"][0]))
        proposal = _read(DESIGN / str(manifest["proposals"][0]))
        outcomes = evidence["hierarchical_reduction_probe_observation"][
            "falsifier_outcomes"
        ]
        self.assertEqual(
            {value["falsifier_id"] for value in outcomes},
            {value["falsifier_id"] for value in proposal["falsifiers"]},
        )

        def remove_outcome(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            evidence["hierarchical_reduction_probe_observation"][
                "falsifier_outcomes"
            ].pop()
            _write(path, evidence)

        self._assert_mutation_rejected(remove_outcome, "falsifier_outcomes")

        def remap_outcome(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            outcomes = evidence["hierarchical_reduction_probe_observation"][
                "falsifier_outcomes"
            ]
            barrier = next(
                value
                for value in outcomes
                if value["falsifier_id"] == "barrier-not-uniform"
            )
            barrier["outcome"] = "outside_probe_scope"
            _write(path, evidence)

        self._assert_mutation_rejected(remap_outcome, ("falsifier_outcomes", "outcome"))

    def test_local_outcomes_track_the_actual_proposal_falsifier_closure(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            proposal["falsifiers"][0]["falsifier_id"] = "replacement-falsifier"
            _write(path, proposal)

        self._assert_mutation_rejected(
            mutate,
            ("required falsifier id closure", "cross_document_falsifier_closure"),
        )

    def test_new_hierarchical_probe_facts_cannot_widen_or_change_observation(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            facts = {value["fact_id"]: value for value in evidence["facts"]}
            facts["observed-m4-hierarchical-width32-case-matrix"]["scope"] = (
                "gpu_family"
            )
            facts["observed-m4-hierarchical-exact-owner-and-broadcast"][
                "source_ids"
            ] = ["local-apple-m4-metal-operator-probe-2026-08-26"]
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate,
            (
                "observed-m4-hierarchical-width32-case-matrix",
                "observed-m4-hierarchical-exact-owner-and-broadcast",
                "source_ids",
            ),
        )

    def test_observed_scope_findings_cannot_widen_their_exact_mappings(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            findings = {
                value["finding_id"]: value
                for value in proposal["observed_scope_findings"]
            }
            findings["observed-width32-case-matrix"]["status"] = "reported_globally"
            findings["observed-exact-owner-and-broadcast"]["fact_ids"].append(
                "observed-m4-hierarchical-runtime-resource-gates"
            )
            findings["observed-two-epoch-scratch-reuse"][
                "does_not_resolve_hypothesis_ids"
            ].append("specialization-is-performance-competitive")
            findings["observed-runtime-resource-gates"]["scope"] = (
                "observed_apple_m4_all_workloads"
            )
            _write(path, proposal)

        self._assert_mutation_rejected(
            mutate,
            (
                "observed_scope_findings",
                "status",
                "fact_ids",
                "does_not_resolve_hypothesis_ids",
                "scope",
            ),
        )

    def test_observed_scope_findings_leave_hypotheses_unresolved_and_unbound(
        self,
    ) -> None:
        manifest = _read(DESIGN / "manifest.json")
        proposal = _read(DESIGN / str(manifest["proposals"][0]))
        self.assertTrue(proposal["hypotheses"])
        self.assertTrue(
            all(
                hypothesis["status"] == "unresolved"
                and hypothesis["result_binding"] is None
                for hypothesis in proposal["hypotheses"]
            )
        )

        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            proposal["hypotheses"][0]["status"] = "reported_in_observed_scope"
            proposal["hypotheses"][1]["result_binding"] = (
                "local-apple-m4-hierarchical-reduction-probe-2026-08-26"
            )
            _write(path, proposal)

        self._assert_mutation_rejected(mutate, ("status", "result_binding"))

    def test_hardware_evidence_does_not_claim_the_proposal_scope(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            evidence["claim_scope"] = "hardware_mapping_only"
            _write(path, evidence)

        self._assert_mutation_rejected(mutate, "hardware_evidence_only")

    def test_live_target_resource_context_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, design
            value = _read(target)
            value["resource_limits"]["maximum_threads_per_workgroup"] = 512
            _write(target, value)

        self._assert_mutation_rejected(mutate, ("target", "512"))

    def test_pipeline_observation_cannot_be_promoted_to_a_family_width_fact(
        self,
    ) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["evidence"][0])
            evidence = _read(path)
            fact = next(
                fact
                for fact in evidence["facts"]
                if fact["fact_id"] == "observed-m4-pipeline-width-32"
            )
            fact["scope"] = "gpu_family"
            _write(path, evidence)

        self._assert_mutation_rejected(
            mutate, ("observed-m4-pipeline-width-32", "scope")
        )

    def test_proposal_and_decisions_cannot_reference_unknown_atomic_facts(self) -> None:
        def add_to_proposal_binding(proposal: dict[str, object]) -> None:
            proposal["evidence_binding"]["fact_ids"].append("imaginary-family-fact")

        def add_to_decision(proposal: dict[str, object]) -> None:
            proposal["decisions"][0]["fact_ids"].append("imaginary-family-fact")

        for location, edit in (
            ("evidence_binding", add_to_proposal_binding),
            ("decision", add_to_decision),
        ):
            with self.subTest(location=location):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    edit: Callable[[dict[str, object]], None] = edit,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    edit(proposal)
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, "imaginary-family-fact")

    def test_hardware_hypothesis_requires_at_least_one_falsifier(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            del proposal["hypotheses"][0]["falsifier"]
            _write(path, proposal)

        self._assert_mutation_rejected(mutate, "falsifier")

    def test_resource_phase_hypothesis_falsifier_contract_is_semantically_pinned(
        self,
    ) -> None:
        fields = ("method", "observable", "reject_when", "retained_result_contract")
        for field in fields:
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    hypothesis = next(
                        value
                        for value in proposal["hypotheses"]
                        if value["hypothesis_id"]
                        == "resource-accounting-fits-selected-pipelines"
                    )
                    hypothesis["falsifier"][field] = (
                        "Never reject overflow; always retain the configuration."
                        if field == "reject_when"
                        else f"Automation weakened the {field} boundary."
                    )
                    _write(path, proposal)

                self._assert_mutation_rejected(
                    mutate,
                    ("hypotheses[2]", f"falsifier.{field}"),
                )

    def test_resource_derivations_are_fixed_to_1024_over_32_and_128_bytes(self) -> None:
        mutations = {
            "maximum_simdgroups_at_family_ceiling": (31, "32"),
            "scratch_bytes_per_accumulator": (124, "128"),
        }
        for field, (replacement, expected) in mutations.items():
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: int = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    proposal["design"]["resource_derivation"][field] = replacement
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, expected)

    def test_variant_resource_derivations_are_fail_closed(self) -> None:
        mutations = (
            ("rms_norm", "raw_dynamic_bytes", 128, "132"),
            ("rms_norm", "aligned_dynamic_bytes", 132, "144"),
            ("layer_norm", "accumulator_count", 1, "2"),
            ("layer_norm", "raw_dynamic_bytes", 128, "256"),
            ("layer_norm", "aligned_dynamic_bytes", 240, "256"),
        )
        for variant, field, replacement, expected in mutations:
            with self.subTest(variant=variant, field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    variant: str = variant,
                    field: str = field,
                    replacement: int = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    proposal["design"]["resource_derivation"][
                        "variant_resource_requirements"
                    ][variant][field] = replacement
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, expected)

    def test_publication_barrier_and_width_mismatch_policy_are_fail_closed(
        self,
    ) -> None:
        mutations = (
            (
                ("reduction_sequence", "publication_barrier"),
                "simdgroup_barrier(mem_threadgroup)",
                "threadgroup_barrier(mem_flags::mem_threadgroup)",
            ),
            (
                ("width_policy", "mismatch_action"),
                "dispatch_anyway",
                "fail_closed_before_dispatch",
            ),
            (
                ("width_policy", "threadgroup_shape_contract"),
                "MTLSize(width=N,height=M,depth=1)",
                "MTLSize(width=N,height=1,depth=1)",
            ),
            (
                ("width_policy", "shape_mismatch_action"),
                "dispatch_anyway",
                "fail_closed_before_dispatch",
            ),
            (
                ("resource_derivation", "runtime_dimension_limit_query"),
                "pipeline.maxTotalThreadsPerThreadgroup",
                "MTLDevice.maxThreadsPerThreadgroup",
            ),
            (
                ("resource_derivation", "runtime_dimension_limit_gate"),
                "threadsPerThreadgroup.width<=device.maxThreadsPerThreadgroup.width",
                "runtime_dimension_limit_gate",
            ),
            (
                ("resource_derivation", "runtime_thread_limit_gate"),
                "threadsPerThreadgroup.width<=pipeline.maxTotalThreadsPerThreadgroup",
                "runtime_thread_limit_gate",
            ),
            (
                ("scratch_lifecycle", "initialization_barrier"),
                "threadgroup_barrier(mem_threadgroup)",
                "threadgroup_barrier(mem_flags::mem_threadgroup)",
            ),
        )
        for (section, field), replacement, expected in mutations:
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    section: str = section,
                    field: str = field,
                    replacement: str = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    proposal["design"][section][field] = replacement
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, expected)

    def test_required_top_level_falsifiers_have_exact_closure(self) -> None:
        required_ids = {
            "ambiguous-final-reducer-owner",
            "barrier-not-uniform",
            "bfloat-simd-sum-input",
            "bitwise-order-required",
            "current-lowering-capability-missing",
            "partial-final-simdgroup-outside-v1",
            "partial-set-exceeds-final-simdgroup",
            "pipeline-thread-limit-exceeded",
            "runtime-width-not-32",
            "threadgroup-shape-or-dimension-limit-unsupported",
            "scratch-read-before-publication",
            "scratch-reuse-race",
            "threadgroup-memory-budget-exceeded",
        }
        manifest = _read(DESIGN / "manifest.json")
        proposal = _read(DESIGN / str(manifest["proposals"][0]))
        self.assertEqual(
            {value["falsifier_id"] for value in proposal["falsifiers"]},
            required_ids,
        )

        for missing_id in sorted(required_ids):
            with self.subTest(missing_id=missing_id):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    missing_id: str = missing_id,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    proposal["falsifiers"] = [
                        value
                        for value in proposal["falsifiers"]
                        if value["falsifier_id"] != missing_id
                    ]
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, missing_id)

    def test_threadgroup_shape_decision_and_falsifier_cannot_be_weakened(
        self,
    ) -> None:
        def mutate_decision(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            decision = next(
                value
                for value in proposal["decisions"]
                if value["decision_id"] == "refine-explicit-resource-accounting"
            )
            decision["fact_ids"].remove(
                "metal-device-threadgroup-dimension-limits-must-be-queried"
            )
            _write(path, proposal)

        self._assert_mutation_rejected(mutate_decision, "fact_ids")

        def mutate_width_decision(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            decision = next(
                value
                for value in proposal["decisions"]
                if value["decision_id"] == "refine-width32-fail-closed-specialization"
            )
            decision["fact_ids"].remove(
                "metal-device-threadgroup-dimension-limits-must-be-queried"
            )
            _write(path, proposal)

        self._assert_mutation_rejected(mutate_width_decision, "fact_ids")

        def mutate_falsifier(
            library: Path, extraction: Path, design: Path, target: Path
        ) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            proposal = _read(path)
            falsifier = next(
                value
                for value in proposal["falsifiers"]
                if value["falsifier_id"]
                == "threadgroup-shape-or-dimension-limit-unsupported"
            )
            falsifier["condition"] = "Only width is checked."
            _write(path, proposal)

        self._assert_mutation_rejected(mutate_falsifier, "condition")

    def test_review_pending_validation_cannot_claim_readiness(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path = design / "manifest.json"
            manifest = _read(path)
            manifest["gate_separation"]["readiness"]["ready"] = True
            _write(path, manifest)

        self._assert_mutation_rejected(mutate, "ready")

    def test_claim_scope_and_nonclaims_cannot_expand_to_later_stage_results(
        self,
    ) -> None:
        mutations = {
            "claim_scope": "performance_result",
            "non_claims": [],
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):

                def mutate(
                    library: Path,
                    extraction: Path,
                    design: Path,
                    target: Path,
                    *,
                    field: str = field,
                    replacement: object = replacement,
                ) -> None:
                    del library, extraction, target
                    manifest = _read(design / "manifest.json")
                    path = design / str(manifest["proposals"][0])
                    proposal = _read(path)
                    proposal[field] = replacement
                    _write(path, proposal)

                self._assert_mutation_rejected(mutate, field)

    def test_extra_design_artifact_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            (design / "unreviewed-kernel.metal").write_text(
                "kernel void not_authorized() {}\n", encoding="utf-8"
            )

        self._assert_mutation_rejected(mutate, "unlisted")

    def test_extra_review_request_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            _, request = _review_request(design)
            _write(design / "review_requests" / "unlisted-request.json", request)

        self._assert_mutation_rejected(mutate, "unlisted")

    def test_listed_proposal_symlink_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
            path.unlink()
            path.symlink_to("../manifest.json")

        self._assert_mutation_rejected(mutate, "regular non-symlink file")

    def test_listed_review_request_symlink_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            path, _ = _review_request(design)
            path.unlink()
            path.symlink_to("../manifest.json")

        self._assert_mutation_rejected(mutate, "regular non-symlink file")

    def test_design_artifacts_are_not_clean_start_compiler_sources(self) -> None:
        source_set = _read(ROOT / "compiler" / "source_set.json")
        self.assertFalse(
            any(
                path == "hardware_informed_design"
                or path.startswith("hardware_informed_design/")
                for path in source_set["paths"]
            )
        )


if __name__ == "__main__":
    unittest.main()
