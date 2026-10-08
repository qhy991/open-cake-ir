"""Singleton logical axes do not distinguish loop stores; CPU semantics only."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]
OWNERSHIP = "TRITON_LOOP_STORE_OWNERSHIP"


def copy_document(*, columns=17, program_tile=32, target="gfx938", rows=None):
    shape = f"({columns},)" if rows is None else f"({rows}, {columns})"
    dimension = 0 if rows is None else 1
    row_axis = "" if rows is None else "    row = lm.program(x, axis=0, dimension=0, tile=1)\n"
    indices = "i" if rows is None else "row, i"
    return frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="singleton-loop-copy", target="{target}", backend="triton", entry_point="copy")
def candidate(lm, x: cake.Tensor({shape}, "fp32"), out: cake.Tensor({shape}, "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
{row_axis}    whole = lm.program(x, axis={dimension}, dimension={dimension}, tile={program_tile})
    with compute:
        for i in lm.range(x, dimension={dimension}, tile=4, name="walk", num_stages=4):
            value = lm.load(x[{indices}], id="load_x")
            lm.store(out[{indices}], value, coalesced=True, id="store_out")
''').document


def nested_document():
    return frontend.parse('''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="singleton-nested-copy", target="gfx938", backend="triton", entry_point="copy")
def candidate(lm, x: cake.Tensor((8, 8), "fp32"), out: cake.Tensor((8, 8), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    whole = lm.program(x, axis=0, dimension=0, tile=8)
    with compute:
        for row in lm.range(x, dimension=0, tile=4, name="rows"):
            for col in lm.range(x, dimension=1, tile=4, name="columns"):
                value = lm.load(x[row, col], id="load_x")
                lm.store(out[row, col], value, id="store_out")
''').document


class SingletonLoopStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def lower(self, document, target=None):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        self.compiler.lower(assessment)
        return emit(Schedule.from_dict(document),
                    target or self.compiler._revision.targets[document["target"]])

    def ownership_refused(self, document, target=None):
        # Exercise this backend's own refusal even when an invalid access also
        # violates a common shape/index rule. A borrowed block is insufficient.
        findings = preflight(Schedule.from_dict(document),
                             target or self.compiler._revision.targets[document["target"]])
        matching = [finding for finding in findings if finding.code == OWNERSHIP]
        self.assertTrue(matching, findings)
        self.assertTrue(all(finding.blocks_lowering and finding.path for finding in matching))

    def assert_copy(self, emission, size):
        values = [float(index - 7) for index in range(size)]
        memories = {"x": values[:], "out": [None] * size}
        observed = _execute(emission, memories)
        self.assertEqual(memories["out"], values)
        self.assertEqual(memories["x"], values)
        self.assertEqual(len(observed.stores), size)
        self.assertEqual(set(observed.stores.values()), {1})

    def test_singleton_axis_copy_preserves_tails_and_exactly_one_store_on_peer_targets(self):
        for target in ("sm_100a", "gfx938", "gfx1151", "xcore1002"):
            for columns in (16, 17):
                with self.subTest(target=target, columns=columns):
                    emission = self.lower(copy_document(columns=columns, target=target))
                    self.assertEqual(emission.toolchain["grid"], [1, 1, 1])
                    self.assert_copy(emission, columns)

    def test_varying_axis_remains_required_in_single_and_multi_axis_maps(self):
        d = copy_document(program_tile=16)
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertFalse(assessment.lowering_eligible)
        self.ownership_refused(d)
        d = copy_document(rows=3)
        self.assert_copy(self.lower(d), 3 * 17)
        store = next(access for access in d["access_maps"] if access["operation"] == "store_out")
        store["indices"][0] = dict(source="dimension", dimension=0, extent=1)
        self.ownership_refused(d)

    def test_one_physical_persistent_cta_does_not_make_a_varying_logical_axis_singleton(self):
        d = copy_document(rows=3)
        d["program_map"]["persistent"] = True
        d["residency"] = {"ctas_per_multiprocessor": 1}
        original = self.compiler._revision.targets["gfx938"]
        # Synthetic one-multiprocessor CPU fixture, not a gfx938 hardware fact.
        target = replace(original, occupancy=replace(original.occupancy, multiprocessor_count=1))
        emission = self.lower(d, target)
        self.assertEqual(emission.toolchain["grid"], [1, 1, 1])
        self.assertEqual(emission.constants["TOTAL_TILES"], 3)
        self.assertEqual(emission.constants["NUM_CTAS"], 1)
        self.assert_copy(emission, 3 * 17)
        store = next(access for access in d["access_maps"] if access["operation"] == "store_out")
        store["indices"][0] = dict(source="dimension", dimension=0, extent=1)
        self.ownership_refused(d, target)

    def test_missing_and_duplicate_nested_loop_coordinates_stay_refused(self):
        original = nested_document()
        self.assert_copy(self.lower(original), 64)
        for first, second in ((dict(source="dimension", dimension=0, extent=4),
                               dict(source="loop_tile", name="col")),
                              (dict(source="loop_tile", name="row"),
                               dict(source="loop_tile", name="row"))):
            with self.subTest(first=first, second=second):
                d = deepcopy(original)
                store = next(access for access in d["access_maps"] if access["operation"] == "store_out")
                store["indices"] = [first, second]
                self.ownership_refused(d)

    def test_state_destinations_and_indirect_store_indices_stay_refused(self):
        d = copy_document()
        next(buffer for buffer in d["buffers"] if buffer["name"] == "out")["mode"] = "state"
        d["outputs"] = []
        self.ownership_refused(d)
        for source in ("buffer", "scalar_buffer"):
            with self.subTest(source=source):
                d = copy_document(rows=1)
                store = next(access for access in d["access_maps"] if access["operation"] == "store_out")
                # Keep the loop coordinate present so the no-indirect-index
                # condition owns refusal independently of loop coverage.
                store["indices"][0] = dict(source=source, name="row")
                self.ownership_refused(d)

    def test_unresolved_axis_owner_or_dimension_cannot_prove_singleton(self):
        for change in ({"buffer": "missing"}, {"dimension": 2}):
            with self.subTest(change=change):
                d = copy_document()
                d["program_map"]["axes"][0].update(change)
                self.ownership_refused(d)
