"""Reduction placement and logical singleton rank; CPU source semantics only."""
from __future__ import annotations

import copy
import itertools
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from tests.contracts.test_triton_loop_scopes import _execute, _scalar_store

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _chain(producer="resident", *, op="sum", dtype="fp32", singleton=False,
           in_loop=False, carried=False, broadcast=False):
    document = _scalar_store(producer, op=op, singleton=singleton)
    result_dtype = "int32" if dtype == "int32" else "fp32"
    load_only = producer in {"scalar_load", "block_load"}
    for buffer in document["buffers"]:
        buffer["dtype"] = dtype if buffer["name"] in {"x", "tile"} or (
            load_only and buffer["name"] == "rows") else result_dtype
    document["buffers"].append(dict(name="folded", shape=[1], dtype=result_dtype,
                                   mode="scratch", space="register"))
    store = document["operations"].pop()
    previous = document["operations"][-1]["id"]
    document["operations"].append(dict(
        id="fold_again", kind="reduce", role="compute", reads=["rows"], writes=["folded"],
        depends_on=[previous], parameters=dict(op=op, axis=0, scope="cta", across_loop=carried)))
    if in_loop:
        loop = document["tile_loops"][0]
        next(item for item in document["operations"] if item["id"] == "fold")["parameters"]["across_loop"] = False
        loop["body"].append("fold_again")
    store.update(reads=["folded"], depends_on=["fold_again"])
    if broadcast:
        document["buffers"].append(dict(name="expanded", shape=[4], dtype=result_dtype,
                                       mode="scratch", space="register"))
        next(buffer for buffer in document["buffers"] if buffer["name"] == "y")["shape"] = [2, 4]
        document["operations"].append(dict(
            id="expand", kind="broadcast_in_dim", role="compute", reads=["folded"], writes=["expanded"],
            depends_on=["fold_again"], parameters=dict(dimensions=[])))
        document["access_maps"][-1]["indices"].append(dict(source="dimension", dimension=1))
        store.update(reads=["expanded"], depends_on=["expand"])
    document["operations"].append(store)
    return document


class TritonReductionStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowered = self.compiler.lower(assessment)
        for operation in document["operations"]:
            self.assertIn(operation["id"], lowered.source_map)
        target = Target.load(ROOT / "compiler/targets" / f'{document["target"]}.json')
        return emit(Schedule.from_dict(document), target)

    def test_global_reduce_is_a_pointer_storage_refusal(self):
        document = _scalar_store("resident")
        document["operations"] = document["operations"][1:]
        fold = document["operations"][0]
        fold.update(reads=["x"], depends_on=[])
        document["buffers"] = [b for b in document["buffers"] if b["name"] != "tile"]
        next(b for b in document["buffers"] if b["name"] == "x")["shape"] = [8]
        document["program_map"]["axes"][0].update(buffer="y")
        document["access_maps"] = document["access_maps"][1:]
        for target in ("sm_100a", "sm_103a", "gfx938", "gfx1151", "xcore1002"):
            with self.subTest(target=target):
                document["target"] = target
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertFalse(assessment.lowering_eligible)
                findings = [f for f in assessment.findings if f.code == "TRITON_REDUCE_STORAGE"]
                self.assertEqual([f.path for f in findings], ["operations[0].reads[0]"])
                self.assertIn("'x' is global", findings[0].message)
                with self.assertRaisesRegex(EmitError, "Triton reduction requires register values"):
                    emit(Schedule.from_dict(document), Target.load(ROOT / "compiler/targets" / f"{target}.json"))

    def test_each_nonregister_reduce_edge_is_refused_by_its_own_rule(self):
        for edge, space in itertools.product(("reads", "writes"), ("shared", "tensor", "global")):
            with self.subTest(edge=edge, space=space):
                document = _scalar_store("resident")
                fold = document["operations"][1]
                name = fold[edge][0]
                buffer = next(b for b in document["buffers"] if b["name"] == name)
                buffer["space"] = space
                if space in {"shared", "tensor"}:
                    buffer.update(allocation="pool", byte_offset=0)
                    document["allocations"] = [dict(name="pool", space=space, size_bytes=128)]
                findings = preflight(Schedule.from_dict(document), TARGET)
                matching = [f for f in findings if f.code == "TRITON_REDUCE_STORAGE"]
                self.assertEqual([f.path for f in matching], [f"operations[1].{edge}[0]"])
                self.assertTrue(matching[0].blocks_lowering)
                self.assertFalse(matching[0].blocks_acceptance)

    def test_chained_register_reductions_keep_values_for_scalar_and_block_producers(self):
        for producer, singleton, op, dtype in itertools.product(
            ("resident", "scalar_load", "block_load", "carried"), (False, True),
            ("sum", "max"), ("fp32", "fp16", "bf16", "int32")
        ):
            if producer in {"scalar_load", "block_load"} and singleton:
                continue
            with self.subTest(producer=producer, singleton=singleton, op=op, dtype=dtype):
                document = _chain(producer, op=op, dtype=dtype, singleton=singleton)
                emission = self.lower(document)
                acc_dtype = "tl.int32" if dtype == "int32" else "tl.float32"
                self.assertIn(f"tl.{op}(tl.reshape(rows, (1,)).to({acc_dtype}), axis=0)", emission.source)
                load_only = producer in {"scalar_load", "block_load"}
                values = [-16777219, -16777217] if dtype == "int32" else [-3.0, -7.0]
                data = values if load_only else [values[0]] * 8 + [values[1]] * 8
                memories = dict(x=data, y=[None] * 2)
                _execute(emission, memories)
                expected = values if load_only or op == "max" else [8 * value for value in values]
                self.assertEqual(memories["y"], expected)

    def test_singleton_fold_can_be_carried_after_a_resident_loop_fold(self):
        for op, dtype in itertools.product(("sum", "max"), ("fp32", "int32")):
            with self.subTest(op=op, dtype=dtype):
                document = _chain("carried", op=op, dtype=dtype, in_loop=True, carried=True)
                emission = self.lower(document)
                memories = dict(x=[-3] * 8 + [-7] * 8, y=[None] * 2)
                _execute(emission, memories)
                self.assertEqual(memories["y"], [-24, -56] if op == "sum" else [-3, -7])

    def test_chained_singleton_then_broadcast_preserves_output_domain(self):
        for producer, op in itertools.product(("resident", "scalar_load", "block_load"), ("sum", "max")):
            with self.subTest(producer=producer, op=op):
                document = _chain(producer, op=op, broadcast=True)
                emission = self.lower(document)
                load_only = producer != "resident"
                memories = dict(x=[-3, -7] if load_only else [-3] * 8 + [-7] * 8, y=[None] * 8)
                observed = _execute(emission, memories)
                factor = 1 if load_only or op == "max" else 8
                self.assertEqual(memories["y"], [-3 * factor] * 4 + [-7 * factor] * 4)
                self.assertEqual(set(observed.stores.values()), {1})

    def test_nonsingleton_reduction_still_folds_the_declared_axis(self):
        for op in ("sum", "max"):
            document = _scalar_store("resident", op=op, singleton=True)
            emission = self.lower(document)
            self.assertIn(f"rows = tl.{op}(tile.to(tl.float32), axis=1)", emission.source)
            self.assertNotIn("tl.reshape", emission.source)

    def test_malformed_axis_shape_and_dtype_remain_semantic_refusals(self):
        original = _chain()
        mutations = (
            (lambda d: d["operations"][-2]["parameters"].update(axis=1), "REDUCE_AXIS_OUT_OF_RANGE"),
            (lambda d: next(b for b in d["buffers"] if b["name"] == "folded").update(shape=[2]), "REDUCE_SHAPE_MISMATCH"),
            (lambda d: next(b for b in d["buffers"] if b["name"] == "folded").update(dtype="fp16"), "REDUCE_DTYPE_MISMATCH"),
        )
        for mutate, code in mutations:
            with self.subTest(code=code):
                document = copy.deepcopy(original)
                mutate(document)
                assessment = self.compiler.assess(document)
                self.assertFalse(assessment.accepted)
                self.assertIn(code, [f.code for f in assessment.findings])
