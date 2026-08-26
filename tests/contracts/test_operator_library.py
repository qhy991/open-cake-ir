"""Contracts for the source-fact-only operator library.

The library is deliberately outside the Compiler revision.  These tests exercise its
public boundary: the checked-in JSON Schema and the offline validator command.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "operator_library"
TOOL = ROOT / "tools" / "validate_operator_library.py"


def _read(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _catalog_path(library: Path, relative: object) -> Path:
    """Manifest paths are canonical paths relative to the library root."""

    return library / str(relative)


def _definition_validator(
    schema: dict[str, object], definition: str
) -> Draft202012Validator:
    """Keep local ``#/$defs`` references rooted at the complete schema."""

    return Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        format_checker=FormatChecker(),
    )


def _assert_schema_valid(
    testcase: unittest.TestCase,
    validator: Draft202012Validator,
    document: dict[str, object],
) -> None:
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    testcase.assertEqual(
        errors,
        [],
        "\n".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        ),
    )


def _run_validator(
    library: Path | None = None, output_format: str | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOL)]
    if library is not None:
        command.extend(("--library-root", str(library)))
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


class OperatorLibrarySchemaTests(unittest.TestCase):
    def test_all_three_document_kinds_conform_to_the_authoring_schema(self) -> None:
        schema = _read(LIBRARY / "schema.json")
        Draft202012Validator.check_schema(schema)
        manifest = _read(LIBRARY / "manifest.json")

        by_definition = {
            "manifest": [manifest],
            "source_snapshot": [
                _read(_catalog_path(LIBRARY, path)) for path in manifest["sources"]
            ],
            "operator_occurrence": [
                _read(_catalog_path(LIBRARY, path)) for path in manifest["operators"]
            ],
        }
        self.assertTrue(by_definition["source_snapshot"])
        self.assertTrue(by_definition["operator_occurrence"])
        for definition, documents in by_definition.items():
            validator = _definition_validator(schema, definition)
            for document in documents:
                with self.subTest(
                    definition=definition,
                    identity=document.get(
                        "implementation_id", document.get("source_id", "manifest")
                    ),
                ):
                    _assert_schema_valid(self, validator, document)

    def test_operator_catalog_is_not_part_of_the_compiler_source_set(self) -> None:
        source_set = _read(ROOT / "compiler" / "source_set.json")
        paths = source_set["paths"]
        self.assertIsInstance(paths, list)
        self.assertFalse(
            any(
                isinstance(path, str)
                and (path == "operator_library" or path.startswith("operator_library/"))
                for path in paths
            )
        )

    def test_mlx_online_summary_occurrences_bind_algorithm_headers(self) -> None:
        manifest = _read(LIBRARY / "manifest.json")
        occurrences = {
            occurrence["implementation_id"]: occurrence
            for relative in manifest["operators"]
            for occurrence in [_read(_catalog_path(LIBRARY, relative))]
        }
        expected = {
            "mlx.metal-scaled-dot-product-attention": {
                (
                    "mlx/backend/metal/kernels/scaled_dot_product_attention.metal",
                    None,
                ),
                ("mlx/backend/metal/kernels/sdpa_vector.h", "sdpa_vector"),
            },
            "mlx.metal-softmax": {
                ("mlx/backend/metal/kernels/softmax.h", "softmax_looped"),
                ("mlx/backend/metal/kernels/softmax.metal", None),
            },
        }

        for implementation_id, locators in expected.items():
            with self.subTest(implementation_id=implementation_id):
                self.assertEqual(
                    {
                        (locator["value"], locator["symbol"])
                        for locator in occurrences[implementation_id]["locators"]
                        if locator["kind"] == "repository_path"
                    },
                    locators,
                )

    def test_mlx_runtime_index_occurrences_bind_algorithm_symbols(self) -> None:
        manifest = _read(LIBRARY / "manifest.json")
        occurrences = {
            occurrence["implementation_id"]: occurrence
            for relative in manifest["operators"]
            for occurrence in [_read(_catalog_path(LIBRARY, relative))]
        }
        expected = {
            "mlx.metal-gather": (
                "mlx/backend/metal/kernels/indexing/gather.h",
                "gather_impl",
            ),
            "mlx.metal-scatter": (
                "mlx/backend/metal/kernels/indexing/scatter.h",
                "scatter_impl",
            ),
        }

        for implementation_id, locator_pair in expected.items():
            with self.subTest(implementation_id=implementation_id):
                self.assertEqual(
                    {
                        (locator["value"], locator["symbol"])
                        for locator in occurrences[implementation_id]["locators"]
                        if locator["kind"] == "repository_path"
                    },
                    {locator_pair},
                )

    def test_mlx_matrix_occurrences_bind_reviewed_algorithm_symbols(self) -> None:
        manifest = _read(LIBRARY / "manifest.json")
        occurrences = {
            occurrence["implementation_id"]: occurrence
            for relative in manifest["operators"]
            for occurrence in [_read(_catalog_path(LIBRARY, relative))]
        }
        algorithms = {
            "mlx.metal-convolution": (
                "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h",
                "implicit_gemm_conv_2d",
            ),
            "mlx.metal-quantized-matmul": (
                "mlx/backend/metal/kernels/quantized.h",
                "qmm_n_impl",
            ),
            "mlx.metal-steel-gemm": (
                "mlx/backend/metal/kernels/steel/gemm/gemm.h",
                "mlx::steel::GEMMKernel::gemm_loop",
            ),
        }

        for implementation_id, locator_pair in algorithms.items():
            with self.subTest(implementation_id=implementation_id):
                locators = occurrences[implementation_id]["locators"]
                self.assertIn(
                    locator_pair,
                    {
                        (locator["value"], locator["symbol"])
                        for locator in locators
                        if locator["kind"] == "repository_path"
                    },
                )

        wrappers = {
            "mlx.metal-convolution": (
                "mlx/backend/metal/kernels/conv.metal",
                None,
            ),
            "mlx.metal-quantized-matmul": (
                "mlx/backend/metal/kernels/quantized.metal",
                None,
            ),
        }
        for implementation_id, locator_pair in wrappers.items():
            with self.subTest(wrapper=implementation_id):
                self.assertIn(
                    locator_pair,
                    {
                        (locator["value"], locator["symbol"])
                        for locator in occurrences[implementation_id]["locators"]
                        if locator["kind"] == "repository_path"
                    },
                )


class OperatorLibraryValidatorTests(unittest.TestCase):
    def test_cli_validates_and_reports_each_required_count(self) -> None:
        text_result = _run_validator()
        self.assertEqual(text_result.returncode, 0, text_result.stderr)
        for label in (
            "sources",
            "operators",
            "mathematical operators",
            "family counts",
            "review counts",
            "scope counts",
            "mechanism counts",
        ):
            self.assertIn(label, text_result.stdout.lower())

        json_result = _run_validator(output_format="json")
        self.assertEqual(json_result.returncode, 0, json_result.stderr)
        report = json.loads(json_result.stdout)
        self.assertTrue(report["valid"])
        for key in (
            "source_count",
            "operator_count",
            "mathematical_operator_count",
            "family_counts",
            "review_counts",
            "scope_counts",
            "mechanism_counts",
        ):
            self.assertIn(key, report)
        manifest = _read(LIBRARY / "manifest.json")
        occurrences = [
            _read(_catalog_path(LIBRARY, path)) for path in manifest["operators"]
        ]
        self.assertEqual(report["source_count"], len(manifest["sources"]))
        self.assertEqual(report["operator_count"], len(occurrences))
        self.assertEqual(
            report["mathematical_operator_count"],
            len({occurrence["operator_id"] for occurrence in occurrences}),
        )
        self.assertEqual(
            report["family_counts"],
            {
                family: Counter(occurrence["family"] for occurrence in occurrences)[
                    family
                ]
                for family in manifest["family_vocabulary"]
            },
        )
        schema = _read(LIBRARY / "schema.json")
        operator_properties = schema["$defs"]["operator_occurrence"]["properties"]
        for field, report_key in (
            ("review_status", "review_counts"),
            ("scope", "scope_counts"),
        ):
            self.assertEqual(
                report[report_key],
                {
                    value: Counter(occurrence[field] for occurrence in occurrences)[
                        value
                    ]
                    for value in operator_properties[field]["enum"]
                },
            )
        self.assertEqual(
            report["mechanism_counts"],
            {
                mechanism: Counter(
                    item
                    for occurrence in occurrences
                    for item in occurrence["mechanisms"]
                )[mechanism]
                for mechanism in manifest["mechanism_vocabulary"]
            },
        )

        markdown_result = _run_validator(output_format="markdown")
        self.assertEqual(markdown_result.returncode, 0, markdown_result.stderr)
        self.assertIn("| Metric | Count |", markdown_result.stdout)
        self.assertIn("## Mechanism counts", markdown_result.stdout)

    def _assert_mutation_rejected(
        self,
        mutate: Callable[[Path, dict[str, object]], None],
        expected_diagnostic: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "operator_library"
            shutil.copytree(LIBRARY, library)
            manifest = _read(library / "manifest.json")
            mutate(library, manifest)

            result = _run_validator(library)

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected_diagnostic, result.stderr.lower())

    def _append_secondary_reviewed_repository_locator(
        self,
        library: Path,
        manifest: dict[str, object],
        field: str,
        value: object,
    ) -> None:
        operators = manifest["operators"]
        assert isinstance(operators, list)
        for relative in operators:
            path = _catalog_path(library, relative)
            occurrence = _read(path)
            if occurrence["review_status"] not in {
                "source_verified",
                "semantics_reviewed",
            }:
                continue
            locators = occurrence["locators"]
            assert isinstance(locators, list)
            repository_locator = next(
                (
                    locator
                    for locator in locators
                    if isinstance(locator, dict)
                    and locator.get("kind") == "repository_path"
                ),
                None,
            )
            if repository_locator is None:
                continue
            broken = copy.deepcopy(repository_locator)
            broken["value"] = f"{broken['value']}.secondary"
            broken[field] = value
            locators.append(broken)
            locators.sort(key=lambda locator: (locator["kind"], locator["value"]))
            path.write_text(json.dumps(occurrence, indent=2) + "\n", encoding="utf-8")
            return
        self.fail("fixture has no reviewed repository-path occurrence")

    def test_path_traversal_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            sources = copy.deepcopy(manifest["sources"])
            assert isinstance(sources, list)
            sources[0] = "../outside.json"
            manifest["sources"] = sources
            (library / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

        self._assert_mutation_rejected(mutate, "safe relative path")

    def test_unlisted_non_json_artifact_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            del manifest
            (library / "sources" / "z-vendored-kernel.ptx").write_text(
                "// copied upstream source\n", encoding="utf-8"
            )
            (library / "sources" / "a-vendored-kernel.cu").write_text(
                "// copied upstream source\n", encoding="utf-8"
            )

        self._assert_mutation_rejected(
            mutate,
            "unlisted directory entries: sources/a-vendored-kernel.cu, "
            "sources/z-vendored-kernel.ptx",
        )

    def test_unlisted_library_root_artifact_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            del manifest
            (library / "vendored.metallib").write_bytes(b"copied binary")

        self._assert_mutation_rejected(
            mutate,
            "unlisted operator_library entries: vendored.metallib",
        )

    def test_v1_assessments_directory_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            del manifest
            (library / "assessments").mkdir()

        self._assert_mutation_rejected(
            mutate,
            "unlisted operator_library entries: assessments",
        )

    def test_unlisted_subdirectory_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            del manifest
            (library / "operators" / "vendored-tree").mkdir()

        self._assert_mutation_rejected(
            mutate,
            "unlisted directory entries: operators/vendored-tree",
        )

    def test_unlisted_symlink_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            del manifest
            (library / "sources" / "linked-source.json").symlink_to(
                library / "manifest.json"
            )

        self._assert_mutation_rejected(
            mutate,
            "unlisted directory entries: sources/linked-source.json",
        )

    def test_unknown_mechanism_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            operators = manifest["operators"]
            assert isinstance(operators, list)
            path = _catalog_path(library, operators[0])
            occurrence = _read(path)
            mechanisms = copy.deepcopy(occurrence["mechanisms"])
            assert isinstance(mechanisms, list)
            mechanisms.append("unknown-mechanism")
            occurrence["mechanisms"] = sorted(mechanisms)
            path.write_text(json.dumps(occurrence, indent=2) + "\n", encoding="utf-8")

        self._assert_mutation_rejected(mutate, "mechanism_vocabulary")

    def test_source_verified_repository_path_requires_a_git_blob(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            operators = manifest["operators"]
            assert isinstance(operators, list)
            for relative in operators:
                path = _catalog_path(library, relative)
                occurrence = _read(path)
                if occurrence["review_status"] not in {
                    "source_verified",
                    "semantics_reviewed",
                }:
                    continue
                locators = occurrence["locators"]
                assert isinstance(locators, list)
                if any(
                    isinstance(locator, dict)
                    and locator.get("kind") == "repository_path"
                    for locator in locators
                ):
                    for locator in locators:
                        assert isinstance(locator, dict)
                        if locator["kind"] == "repository_path":
                            locator["git_object"] = None
                    path.write_text(
                        json.dumps(occurrence, indent=2) + "\n", encoding="utf-8"
                    )
                    return
            self.fail("fixture has no reviewed repository-path occurrence")

        self._assert_mutation_rejected(mutate, "40-hex git_object")

    def test_every_reviewed_repository_path_requires_a_git_blob(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            self._append_secondary_reviewed_repository_locator(
                library, manifest, "git_object", None
            )

        self._assert_mutation_rejected(
            mutate,
            "every reviewed repository_path requires a 40-hex git_object",
        )

    def test_every_reviewed_repository_path_requires_the_source_revision(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            self._append_secondary_reviewed_repository_locator(
                library, manifest, "revision", None
            )

        self._assert_mutation_rejected(
            mutate,
            "every reviewed repository_path revision must equal the referenced "
            "source revision",
        )

    def test_clean_start_access_is_rejected(self) -> None:
        def mutate(library: Path, manifest: dict[str, object]) -> None:
            sources = manifest["sources"]
            assert isinstance(sources, list)
            path = _catalog_path(library, sources[0])
            source = _read(path)
            access = source["authoring_access"]
            assert isinstance(access, dict)
            access["clean_start"] = True
            path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

        self._assert_mutation_rejected(mutate, "clean_start")


if __name__ == "__main__":
    unittest.main()
