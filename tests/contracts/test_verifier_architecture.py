"""Public contracts needed when Compiler delegates its common checks once."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from open_cake_ir.compiler import diagnostics, verifier
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _document() -> dict:
    return json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())


def _findings(document: dict, target: Target = TARGET):
    return verifier.verify(Schedule.from_dict(document), target)


class VerifierArchitectureTests(unittest.TestCase):
    def test_public_diagnostics_reexport_one_canonical_model(self) -> None:
        from open_cake_ir import compiler

        for name in ("Finding", "FindingCategory", "FindingSeverity"):
            with self.subTest(name=name):
                canonical = getattr(diagnostics, name)
                self.assertIs(getattr(verifier, name), canonical)
                self.assertIs(getattr(compiler, name), canonical)
                self.assertEqual(canonical.__module__, diagnostics.__name__)

    def test_explicit_and_derived_grid_share_the_target_limit(self) -> None:
        target = replace(
            TARGET, resource_limits=replace(TARGET.resource_limits, maximum_grid=(8, 1, 1))
        )
        for derived in (False, True):
            for extent in (8, 9):
                with self.subTest(derived=derived, extent=extent):
                    document = _document()
                    if derived:
                        document["buffers"][0]["shape"][0] = extent
                    else:
                        document.pop("program_map")
                        document["grid"] = [extent, 1, 1]
                    self.assertEqual(verifier.resolve_grid(Schedule.from_dict(document)), (extent, 1, 1))
                    findings = [f for f in _findings(document, target) if f.code == "TARGET_GRID_LIMIT"]
                    self.assertEqual(len(findings), int(extent > 8))
                    if findings:
                        self.assertEqual(findings[0].path, "grid[0]")
                        self.assertIs(findings[0].category, diagnostics.FindingCategory.HARDWARE_CONFORMANCE)
                        self.assertTrue(findings[0].blocks_acceptance)

    def test_invalid_program_axes_stay_localized_without_projection_errors(self) -> None:
        for update, code in (
            ({"axis": 3}, "PROGRAM_AXIS_NUMBER_RANGE"),
            ({"buffer": "missing"}, "PROGRAM_AXIS_BUFFER_UNKNOWN"),
            ({"dimension": 2}, "PROGRAM_AXIS_DIMENSION_RANGE"),
        ):
            with self.subTest(update=update):
                document = _document()
                document["program_map"]["axes"][0].update(update)
                self.assertEqual(verifier.resolve_grid(Schedule.from_dict(document)), (1, 1, 1))
                matches = [f for f in _findings(document) if f.code == code]
                self.assertEqual(len(matches), 1)
                self.assertTrue(matches[0].blocks_acceptance)
                self.assertTrue(matches[0].path.startswith("program_map.axes[0]"))

    def test_duplicate_axis_does_not_replace_first_launch_extent(self) -> None:
        document = _document()
        duplicate = copy.deepcopy(document["program_map"]["axes"][0])
        duplicate.update(name="duplicate", dimension=1)
        document["program_map"]["axes"].append(duplicate)
        target = replace(
            TARGET, resource_limits=replace(TARGET.resource_limits, maximum_grid=(8, 1, 1))
        )
        codes = {f.code for f in _findings(document, target)}
        self.assertEqual(verifier.resolve_grid(Schedule.from_dict(document)), (8, 1, 1))
        self.assertIn("PROGRAM_AXIS_DUPLICATE_NUMBER", codes)
        self.assertNotIn("TARGET_GRID_LIMIT", codes)

    def test_ordering_edge_without_data_flow_must_precede_its_consumer(self) -> None:
        document = _document()
        document["operations"][0]["depends_on"] = ["load_b"]
        findings = _findings(document)
        codes = {f.code for f in findings}
        self.assertNotIn("OP_READ_BEFORE_WRITE", codes)
        self.assertNotIn("OP_DEPENDENCY_CYCLE", codes)
        order = [f for f in findings if f.code == "OPERATION_DEPENDENCY_ORDER"]
        self.assertEqual(len(order), 1)
        self.assertEqual(order[0].path, "operations[0].depends_on")
        self.assertIs(order[0].category, diagnostics.FindingCategory.PROGRAM_SAFETY)
        self.assertTrue(order[0].blocks_acceptance)
        document["operations"][0], document["operations"][1] = (
            document["operations"][1], document["operations"][0]
        )
        self.assertFalse(any(f.blocks_acceptance for f in _findings(document)))

    def test_unknown_and_self_dependencies_keep_their_specific_diagnostics(self) -> None:
        for dependency, code in (("missing", "OP_DEPENDENCY_UNKNOWN"), ("load_a", "OP_SELF_DEPENDENCY")):
            with self.subTest(dependency=dependency):
                document = _document()
                document["operations"][0]["depends_on"] = [dependency]
                findings = _findings(document)
                self.assertEqual(sum(f.code == code for f in findings), 1)
                self.assertNotIn("OPERATION_DEPENDENCY_ORDER", {f.code for f in findings})

    def test_one_allocation_extent_rule_covers_each_declared_reference(self) -> None:
        for space in ("global", "register", "shared", "tensor"):
            for owner_space in ("shared", "tensor"):
                for size in (511, 512):
                    with self.subTest(space=space, owner_space=owner_space, size=size):
                        document = _document()
                        document["allocations"] = [{"name": "tile", "space": owner_space, "size_bytes": size}]
                        if owner_space == "tensor":
                            document["allocations"][0]["allocating_role"] = "compute"
                        # a_tile occupies 128 fp32 elements, independent of the
                        # separate rules for its declared storage space.
                        document["buffers"][4].update(space=space, allocation="tile")
                        findings = _findings(document)
                        overflow = [f for f in findings if f.code == "BUFFER_ALLOCATION_OVERFLOW"]
                        self.assertEqual(len(overflow), int(size < 512))
                        if overflow:
                            self.assertEqual(overflow[0].path, "buffers[4].byte_offset")
                            self.assertTrue(overflow[0].blocks_acceptance)
                        if space in ("shared", "tensor") and space != owner_space:
                            self.assertIn("BUFFER_SPACE_MISMATCH", {f.code for f in findings})

    def test_stages_and_byte_offset_contribute_to_the_single_extent(self) -> None:
        document = _document()
        document["allocations"] = [{"name": "tile", "space": "shared", "size_bytes": 1024}]
        document["buffers"][4].update(space="shared", allocation="tile", stages=2, byte_offset=4)
        overflow = [f for f in _findings(document) if f.code == "BUFFER_ALLOCATION_OVERFLOW"]
        self.assertEqual(len(overflow), 1)
        self.assertIn("1028", overflow[0].message)

    def test_unknown_allocation_reference_is_checked_in_every_space(self) -> None:
        for space in ("global", "register", "shared", "tensor"):
            with self.subTest(space=space):
                document = _document()
                document["buffers"][4].update(space=space, allocation="missing")
                matches = [f for f in _findings(document) if f.code == "BUFFER_ALLOCATION_UNKNOWN"]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].path, "buffers[4].allocation")

    def test_duplicate_names_are_findings_for_each_named_schedule_group(self) -> None:
        document = json.loads((ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json").read_text())
        for group in ("roles", "allocations", "buffers", "pipelines", "barriers", "operations"):
            with self.subTest(group=group):
                mutated = copy.deepcopy(document)
                mutated[group].append(copy.deepcopy(mutated[group][0]))
                matches = [f for f in _findings(mutated) if f.code == "NAME_DUPLICATE" and f.path == group]
                self.assertEqual(len(matches), 1)
                self.assertIs(matches[0].category, diagnostics.FindingCategory.SCHEDULE_SEMANTICS)
                self.assertTrue(matches[0].blocks_acceptance)


if __name__ == "__main__":
    unittest.main()
