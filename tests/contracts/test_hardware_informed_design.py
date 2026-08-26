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


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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

    def test_all_three_document_kinds_conform_to_the_public_schema(self) -> None:
        schema = _read(DESIGN / "schema.json")
        Draft202012Validator.check_schema(schema)
        manifest = _read(DESIGN / "manifest.json")
        documents = {
            "manifest": [manifest],
            "evidence": [_read(DESIGN / path) for path in manifest["evidence"]],
            "proposal": [_read(DESIGN / path) for path in manifest["proposals"]],
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
        self.assertIs(summary["ready_for_principle_review"], False)
        self.assertIs(summary["hardware_semantics_verified"], False)
        self.assertIs(summary["performance_claim_authorized"], False)
        self.assertIs(summary["compiler_change_authorized"], False)
        self.assertIs(summary["remote_source_bytes_verified"], False)
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
                "threadgroup_barrier(mem_threadgroup)",
            ),
            (
                ("width_policy", "mismatch_action"),
                "dispatch_anyway",
                "fail_closed_before_dispatch",
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

    def test_listed_proposal_symlink_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path, design: Path, target: Path) -> None:
            del library, extraction, target
            manifest = _read(design / "manifest.json")
            path = design / str(manifest["proposals"][0])
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
