"""Python map options use the same Schedule, admission and lowering as JSON."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import FrontendError, parse
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError


ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(target="gfx938", backend="triton",
               residency={"ctas_per_multiprocessor": 2},
               program_map={"persistent": True, "traversal": ("column", "row")})
def copy_tiles(lm, x: cake.Tensor((257, 130), "fp32"),
               y: cake.Tensor((257, 130), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    column = lm.program(x, axis=1, dimension=1, tile=64)
    with compute:
        values = lm.load(x[row, column], reuse="streamed")
        lm.store(y[row, column], values)
'''


def json_document():
    """Independent JSON spelling, including the same masked column tail."""
    return {
        "schema_version": 2, "schedule_id": "copy_tiles", "target": "gfx938",
        "lowering": {"backend": "triton", "entry_point": "copy_tiles"},
        "residency": {"ctas_per_multiprocessor": 2},
        "roles": [{"name": "compute", "execution_groups": [0, 1, 2, 3]}],
        "allocations": [], "pipelines": [], "barriers": [], "tile_loops": [],
        "buffers": [
            {"name": "x", "space": "global", "dtype": "fp32", "shape": [257, 130], "mode": "input"},
            {"name": "y", "space": "global", "dtype": "fp32", "shape": [257, 130], "mode": "output"},
            {"name": "values", "space": "register", "dtype": "fp32", "shape": [64], "mode": "scratch"},
        ],
        "program_map": {
            "axes": [
                {"name": "row", "axis": 0, "buffer": "x", "dimension": 0, "tile": 1},
                {"name": "column", "axis": 1, "buffer": "x", "dimension": 1, "tile": 64},
            ],
            "persistent": True, "traversal": ["column", "row"],
        },
        "operations": [
            {"id": "values", "kind": "load", "role": "compute", "reads": ["x"],
             "writes": ["values"], "parameters": {"movement": "global", "reuse": "streamed"}},
            {"id": "store_y", "kind": "store", "role": "compute", "reads": ["values"],
             "writes": ["y"], "parameters": {"coalesced": True}, "depends_on": ["values"]},
        ],
        "access_maps": [
            {"operation": operation, "buffer": buffer, "boundary": "mask_tiled_axes",
             "indices": [{"source": "program", "name": "row"},
                         {"source": "program_tile", "name": "column"}]}
            for operation, buffer in (("values", "x"), ("store_y", "y"))
        ],
        "outputs": ["y"], "metadata": {},
    }


class ProgramMapFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_python_and_json_preserve_persistent_walk_and_emission(self):
        for target in ("gfx938", "sm_100a"):
            for traversal in (("column", "row"), ("row", "column"), None):
                with self.subTest(target=target, traversal=traversal):
                    source = SOURCE.replace('"gfx938"', repr(target))
                    expected = json_document()
                    expected["target"] = target
                    if traversal is None:
                        source = source.replace(', "traversal": ("column", "row")', '')
                        del expected["program_map"]["traversal"]
                    else:
                        source = source.replace('("column", "row")', repr(traversal))
                        expected["program_map"]["traversal"] = list(traversal)
                    authored = parse(source)
                    self.assertEqual(authored.document, expected)
                    original = self.compiler.assess(expected)
                    observed = self.compiler.assess(authored.document)
                    self.assertEqual(observed, original)
                    self.assertTrue(observed.accepted, observed.findings)
                    self.assertTrue(observed.lowering_eligible, observed.findings)
                    lowered = self.compiler.lower(observed)
                    self.assertEqual(lowered, self.compiler.lower(original))
                    ast.parse(lowered.source)
                    self.assertIn("for _work in tl.range(tl.program_id(0), TOTAL_TILES, NUM_CTAS)", lowered.source)
                    self.assertIn("TOTAL_TILES=771", lowered.source)
                    order = traversal or ("row", "column")
                    self.assertEqual(tuple(axis.name for axis in Schedule.from_dict(authored.document).program_map.walk_order()), order)

    def test_default_map_options_preserve_the_existing_axis_only_document(self):
        old = SOURCE.replace(',\n               program_map={"persistent": True, "traversal": ("column", "row")}', '')
        expected = json_document()
        expected["program_map"] = {"axes": expected["program_map"]["axes"]}
        self.assertEqual(parse(old).document, expected)
        empty = old.replace('backend="triton",', 'backend="triton", program_map={},')
        self.assertEqual(parse(empty).document, expected)
        explicit = old.replace('backend="triton",', 'backend="triton", program_map={"persistent": False},')
        before = self.compiler.assess(expected)
        after = self.compiler.assess(parse(explicit).document)
        self.assertTrue(after.lowering_eligible, after.findings)
        self.assertEqual(self.compiler.lower(before).source, self.compiler.lower(after).source)

    def test_residency_and_occupancy_refusals_match_json(self):
        for residency in (None, {"registers_per_thread": 32}):
            with self.subTest(residency=residency):
                source = SOURCE.replace('residency={"ctas_per_multiprocessor": 2},',
                                        '' if residency is None else f'residency={residency!r},')
                expected = json_document()
                if residency is None:
                    del expected["residency"]
                else:
                    expected["residency"] = residency
                self.assert_refusal(source, expected, "PERSISTENT_WITHOUT_RESIDENCY")
        expected = json_document()
        expected["target"] = "sm_103a"
        self.assert_refusal(SOURCE.replace('"gfx938"', '"sm_103a"'), expected,
                            "TRITON_PERSISTENT_TARGET_FACTS_MISSING")

    def test_invalid_traversal_is_a_localized_canonical_finding(self):
        for traversal in (("row",), ("row", "row"), ("row", "unknown")):
            with self.subTest(traversal=traversal):
                source = SOURCE.replace('("column", "row")', repr(traversal))
                expected = json_document()
                expected["program_map"]["traversal"] = list(traversal)
                self.assert_refusal(source, expected, "TRAVERSAL_NOT_A_PERMUTATION")
                authored = parse(source, filename="candidate.py")
                location = authored.location_for("program_map.traversal")
                self.assertEqual(location.filename, "candidate.py")
                self.assertEqual(source.splitlines()[location.line - 1][location.column - 1:location.end_column - 1], repr(traversal))

    def test_malformed_options_use_the_existing_structural_parser(self):
        for options, path in (
            ({"persistent": 1}, "schedule.program_map.persistent"),
            ({"persistent": True, "traversal": "row"}, "schedule.program_map.traversal"),
            ({"persistent": True, "traversal": []}, "schedule.program_map.traversal"),
            ({"persistent": False, "traversal": ["column", "row"]}, "schedule.program_map.traversal"),
            ({"persistent": True, "num_ctas": 1}, "schedule.program_map"),
        ):
            with self.subTest(options=options):
                source = SOURCE.replace('{"persistent": True, "traversal": ("column", "row")}', repr(options))
                expected = json_document()
                expected["program_map"] = dict(axes=expected["program_map"]["axes"], **options)
                with self.assertRaises(ScheduleParseError) as json_error:
                    Schedule.from_dict(expected)
                with self.assertRaises(FrontendError) as python_error:
                    parse(source)
                self.assertEqual(python_error.exception.code, "SCHEDULE_STRUCTURE")
                self.assertEqual(python_error.exception.canonical_path, path)
                self.assertIn(str(json_error.exception).removeprefix("schedule."), str(python_error.exception))

    def test_axes_are_owned_only_by_program_declarations(self):
        for options in ('{"axes": []}', '{"axes": "row"}', 'None', 'True', '[]'):
            with self.subTest(options=options):
                source = SOURCE.replace('{"persistent": True, "traversal": ("column", "row")}', options)
                with self.assertRaisesRegex(FrontendError, 'lm.program declarations own|dictionary of map options'):
                    parse(source)
        for field in ('persistent=True', 'traversal=("row", "column")'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(FrontendError, 'program_map.axes\\[0\\] unknown fields'):
                    parse(SOURCE.replace('axis=0, dimension=0, tile=1', f'axis=0, dimension=0, tile=1, {field}'))

    def test_map_options_cannot_replace_axes_or_coexist_with_grid(self):
        no_axes = '\n'.join(line for line in SOURCE.splitlines() if 'lm.program(' not in line)
        no_axes = no_axes.replace('[row, column]', '[:, :]')
        with self.assertRaisesRegex(FrontendError, 'program_map.axes must not be empty'):
            parse(no_axes)
        with self.assertRaisesRegex(FrontendError, 'cannot coexist with grid'):
            parse(SOURCE.replace('backend="triton",', 'backend="triton", grid=(1, 1, 1),'))
        with self.assertRaisesRegex(FrontendError, 'exactly one of grid or program_map'):
            parse(no_axes.replace('backend="triton",', 'backend="triton", grid=(1, 1, 1),'))

    def test_duplicate_map_options_never_silently_override(self):
        for source in (
            SOURCE.replace('backend="triton",', 'backend="triton", program_map={},'),
            SOURCE.replace('"persistent": True', '"persistent": False, "persistent": True'),
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(FrontendError, 'duplicate keywords|unique strings'):
                    parse(source)

    def assert_refusal(self, source, expected, code):
        authored = parse(source)
        self.assertEqual(authored.document, expected)
        observed = self.compiler.assess(authored.document)
        self.assertEqual(observed, self.compiler.assess(expected))
        self.assertFalse(observed.lowering_eligible)
        self.assertIn(code, [finding.code for finding in observed.findings])


if __name__ == "__main__":
    unittest.main()
