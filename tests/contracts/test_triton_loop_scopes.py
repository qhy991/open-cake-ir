"""Two-deep source semantics; the CPU evaluator makes no GPU/numerics claim."""
from __future__ import annotations

import ast
import copy
import itertools
import json
import operator
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.performance.work import operation_repetitions, work_bound

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def _gemm(*, tail=False, invariant_b=False):
    d = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke.json").read_text())
    m, n, k = (35, 19, 33) if tail else (32, 16, 32)
    for buffer in d["buffers"]:
        buffer["shape"] = {
            "a": [m, k], "b": [n, k], "bias": [n], "c": [m, n],
            "a_tile": [16, 16], "b_tile": [16, 16], "acc": [16, 16],
            "bias_tile": [16], "c_tile": [16, 16],
        }[buffer["name"]]
    next(op for op in d["operations"] if op["id"] == "dot")["parameters"]["tile_shape"] = [16, 16, 16]
    inner = d["tile_loops"][0]
    inner["tile"] = 16
    outer = copy.deepcopy(inner)
    outer.update(name="m_loop", iterator="m", dimension=0,
                 body=["k_loop", "add_bias", "store_c"])
    outer["range_options"]["num_stages"] = 1
    d["tile_loops"].append(outer)  # declaration order is deliberately inner-first
    d["program_map"]["axes"] = [dict(name="n_block", axis=0, buffer="b", dimension=0, tile=16)]
    for access in d["access_maps"]:
        for index in access["indices"]:
            if index.get("name") == "m_block":
                index.update(source="loop_tile", name="m")
    bias = next(op for op in d["operations"] if op["id"] == "load_bias")
    d["operations"].remove(bias)
    d["operations"].insert(0, bias)  # invariant across both loops
    if invariant_b:
        # A loaded B tile is reused for each K tile: the declared math is explicit.
        b = next(op for op in d["operations"] if op["id"] == "load_b")
        d["operations"].remove(b)
        d["operations"].insert(1, b)
        inner["body"].remove("load_b")
        access = next(a for a in d["access_maps"] if a["operation"] == "load_b")
        access["indices"][1] = dict(source="dimension", dimension=1, extent=16)
    return d


def _reduction(op="sum", resident=False):
    def buf(name, shape, mode="scratch"):
        return dict(name=name, space="register" if mode == "scratch" else "global",
                    dtype="fp32", shape=shape, mode=mode)
    def operation(name, kind, reads, writes, parameters, depends_on=()):
        return dict(id=name, kind=kind, role="compute", reads=reads, writes=writes,
                    parameters=parameters, depends_on=list(depends_on))
    d = _gemm()
    d["buffers"] = [buf("x", [2, 5, 8], "input"), buf("y", [2, 5], "output"),
                    buf("tile", [2, 4]), buf("rows", [2])]
    d["outputs"] = ["y"]
    d["program_map"] = dict(axes=[dict(name="batch", axis=0, buffer="x", dimension=0, tile=1)])
    d["operations"] = [
        operation("load_x", "load", ["x"], ["tile"], dict(movement="global", reuse="streamed")),
        operation("fold", "reduce", ["tile"], ["rows"], dict(op=op, axis=1, scope="cta"), ["load_x"]),
        operation("store_y", "store", ["rows"], ["y"], dict(coalesced=True), ["fold"]),
    ]
    options = d["tile_loops"][0]["range_options"]
    d["tile_loops"] = [
        dict(name="rows_loop", iterator="row", buffer="x", dimension=1, tile=2,
             body=["features_loop", "store_y"], range_options=options),
        dict(name="features_loop", iterator="feature", buffer="x", dimension=2, tile=4,
             body=["load_x", "fold"], range_options=options),
    ]
    d["access_maps"] = [
        dict(operation="load_x", buffer="x", boundary="mask_tiled_axes", indices=[
            dict(source="program", name="batch"), dict(source="loop_tile", name="row"),
            dict(source="loop_tile", name="feature")]),
        dict(operation="store_y", buffer="y", boundary="mask_tiled_axes", indices=[
            dict(source="program", name="batch"), dict(source="loop_tile", name="row")]),
    ]
    if resident:
        # Reduction is local to a row tile, with no implicit carry over that output loop.
        d["tile_loops"] = d["tile_loops"][:1]
        d["tile_loops"][0]["body"] = ["load_x", "fold", "store_y"]
        d["buffers"][2]["shape"] = [2, 8]
        d["access_maps"][0]["indices"][2] = dict(source="dimension", dimension=2)
        d["operations"][1]["parameters"]["across_loop"] = False
    return d


def _scalar_store(producer="carried", *, op="sum", singleton=False, vector_store=False):
    """One logical value with scalar or one-element-block native producers."""
    d = _reduction(op)
    d["schedule_id"] = f"scalar-{producer}-{op}-{int(singleton)}-{int(vector_store)}"
    load_only = producer in {"scalar_load", "block_load"}
    features = 4 if producer == "carried" else 8
    x_shape = ([2] if producer == "scalar_load" else [2, 1]) if load_only else (
        [2, 1, 8] if singleton else [2, 8]
    )
    shapes = {"x": x_shape, "y": [2, 1] if vector_store else [2],
              "tile": [1, features] if singleton else [features], "rows": [1]}
    for buffer in d["buffers"]:
        buffer["shape"] = shapes[buffer["name"]]
    indices = [dict(source="program", name="batch")]
    if load_only:
        d["tile_loops"] = []
        d["buffers"] = [buffer for buffer in d["buffers"] if buffer["name"] != "tile"]
        load, _, store = d["operations"]
        load["writes"] = ["rows"]
        store["depends_on"] = ["load_x"]
        d["operations"] = [load, store]
        if producer == "block_load":
            indices.append(dict(source="dimension", dimension=1))
    else:
        dimension = 2 if singleton else 1
        if singleton:
            indices.append(dict(source="dimension", dimension=1))
        if producer == "carried":
            loop = d["tile_loops"][1]
            loop["dimension"] = dimension
            d["tile_loops"] = [loop]
            indices.append(dict(source="loop_tile", name="feature"))
        else:
            d["tile_loops"] = []
            d["operations"][1]["parameters"]["across_loop"] = False
            indices.append(dict(source="dimension", dimension=dimension))
        d["operations"][1]["parameters"]["axis"] = 1 if singleton else 0
    d["access_maps"][0]["indices"] = indices
    d["access_maps"][1]["indices"] = [dict(source="program", name="batch")]
    if vector_store:
        d["access_maps"][1]["indices"].append(dict(source="dimension", dimension=1))
    return d


class _Tile:
    """Only broadcasting/indexing needed to execute the emitted arithmetic on CPU."""

    def __init__(self, shape, values):
        self.shape, self.values = tuple(shape), list(values)

    def at(self, index):
        offset = 0
        for extent, coordinate in zip(self.shape, index):
            offset = offset * extent + coordinate
        return self.values[offset]

    def binary(self, other, fn):
        if not isinstance(other, _Tile):
            other = _Tile((), [other])
        rank = max(len(self.shape), len(other.shape))
        a, b = (1,) * (rank - len(self.shape)) + self.shape, (1,) * (rank - len(other.shape)) + other.shape
        assert all(x == y or x == 1 or y == 1 for x, y in zip(a, b))
        shape = tuple(max(x, y) for x, y in zip(a, b))
        def value(tile, padded, index):
            mapped = tuple(0 if size == 1 else i for size, i in zip(padded, index))
            return tile.at(mapped[rank-len(tile.shape):])
        return _Tile(shape, [fn(value(self, a, i), value(other, b, i))
                            for i in itertools.product(*(range(n) for n in shape))])

    def __add__(self, x): return self.binary(x, operator.add)
    __radd__ = __add__
    def __mul__(self, x): return self.binary(x, operator.mul)
    __rmul__ = __mul__
    def __sub__(self, x): return self.binary(x, operator.sub)
    def __lt__(self, x): return self.binary(x, operator.lt)
    def __eq__(self, x): return self.binary(x, operator.eq)
    def __and__(self, x): return self.binary(x, operator.and_)
    def __or__(self, x): return self.binary(x, operator.or_)
    def to(self, dtype): return self

    def __getitem__(self, indices):
        assert len(self.shape) == 1
        assert all(i is None or i == slice(None) for i in indices)
        return _Tile(tuple(1 if i is None else self.shape[0] for i in indices), self.values)


class _Pointer:
    def __init__(self, memory, offsets=0):
        self.memory, self.offsets = memory, offsets

    def __add__(self, x): return _Pointer(self.memory, self.offsets + x)


class _TL:
    """Execute source control flow and memory effects, without Triton or Torch."""
    float32 = object()
    int32 = object()
    range = staticmethod(lambda *args, **kwargs: range(*args))
    arange = staticmethod(lambda start, stop: _Tile((stop-start,), range(start, stop)))
    full = staticmethod(lambda shape, value, dtype: _Tile(shape, itertools.repeat(value, _size(shape))))
    zeros = staticmethod(lambda shape, dtype: _TL.full(shape, 0, dtype))
    maximum = staticmethod(lambda a, b: a.binary(b, max))

    @staticmethod
    def where(condition, yes, no):
        if not isinstance(condition, _Tile):
            condition = _Tile((), [condition])
        return condition.binary(yes, lambda c, y: (c, y)).binary(
            no, lambda choice, n: choice[1] if choice[0] else n)

    def __init__(self):
        self.program = (0, 0, 0)
        self.loads, self.stores = {}, {}
        self.store_types = []

    def program_id(self, axis): return self.program[axis]

    def load(self, pointer, mask=None, other=None, **kwargs):
        self.loads[id(pointer.memory)] = self.loads.get(id(pointer.memory), 0) + 1
        offsets = pointer.offsets
        if not isinstance(offsets, _Tile):
            offsets = _Tile((), [offsets])
        valid = mask.binary(offsets, lambda enabled, _: enabled).values if mask is not None else [True] * len(offsets.values)
        # Bounds assertions catch missing/wrong masks rather than accepting Python's
        # negative indexing. False lanes must never touch memory.
        values = []
        for offset, enabled in zip(offsets.values, valid):
            if enabled:
                assert 0 <= offset < len(pointer.memory)
            values.append(pointer.memory[offset] if enabled else other)
        return _Tile(offsets.shape, values)

    def store(self, pointer, values, mask=None):
        offsets = pointer.offsets
        if not isinstance(offsets, _Tile):
            offsets = _Tile((), [offsets])
        if not isinstance(values, _Tile):
            values = _Tile((), [values])
        # This is a type contract, not merely a numerical zip of two value lists:
        # a scalar pointer takes a scalar value; block pointers broadcast values/masks
        # to their own shape. It is not a native Triton compilation claim.
        if not offsets.shape and values.shape:
            raise TypeError("scalar pointer cannot store a block value")
        self.store_types.append((offsets.shape, values.shape))
        values = values.binary(offsets, lambda value, _: value)
        if values.shape != offsets.shape:
            raise TypeError("store value cannot broadcast to the pointer shape")
        if mask is not None:
            if not isinstance(mask, _Tile):
                mask = _Tile((), [mask])
            mask = mask.binary(offsets, lambda enabled, _: enabled)
            if mask.shape != offsets.shape:
                raise TypeError("store mask cannot broadcast to the pointer shape")
        valid = mask.values if mask is not None else [True] * len(offsets.values)
        for offset, value, enabled in zip(offsets.values, values.values, valid):
            if enabled:
                assert 0 <= offset < len(pointer.memory)
                key = (id(pointer.memory), offset)
                self.stores[key] = self.stores.get(key, 0) + 1
                pointer.memory[offset] = value

    @staticmethod
    def trans(a):
        m, n = a.shape
        return _Tile((n, m), [a.at((i, j)) for j in range(n) for i in range(m)])

    @staticmethod
    def dot(a, b, **kwargs):
        m, k = a.shape
        kb, n = b.shape
        assert k == kb
        return _Tile((m, n), [sum(a.at((i, t)) * b.at((t, j)) for t in range(k))
                              for i in range(m) for j in range(n)])

    @staticmethod
    def sum(a, axis):
        return _TL.reduce(a, axis, sum)

    @staticmethod
    def max(a, axis):
        return _TL.reduce(a, axis, max)

    @staticmethod
    def min(a, axis):
        return _TL.reduce(a, axis, min)

    @staticmethod
    def argmin(a, axis, tie_break_left=True):
        assert tie_break_left
        return _TL.reduce(a, axis, lambda values: min(
            enumerate(values), key=lambda item: (item[1], item[0]))[0])

    @staticmethod
    def reduce(a, axis, function):
        assert 0 <= axis < len(a.shape)
        shape = a.shape[:axis] + a.shape[axis + 1:]
        return _Tile(shape, [function(a.at(index[:axis] + (j,) + index[axis:])
                                     for j in range(a.shape[axis]))
                            for index in itertools.product(*(range(n) for n in shape))])


def _size(shape):
    result = 1
    for n in shape: result *= n
    return result


def _execute(emission, memories):
    tree = ast.parse(emission.source)
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    kernel.decorator_list = []
    for arg in kernel.args.args: arg.annotation = None
    tl = _TL()
    env = {"tl": tl}
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), "<emitted-kernel>", "exec"), env)
    constants = emission.toolchain["compile_constants"]
    for program in itertools.product(*(range(n) for n in emission.toolchain["grid"])):
        tl.program = program
        env[kernel.name](**{name: _Pointer(memory) for name, memory in memories.items()}, **constants)
    return tl


class TritonLoopScopesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowered = self.compiler.lower(assessment)
        for operation in document["operations"]:
            self.assertIn(operation["id"], lowered.source_map)
        return emit(Schedule.from_dict(document), TARGET)

    def refuse(self, document, code):
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertTrue(any(f.code == code for f in assessment.findings), assessment.findings)
        self.assertTrue(all(f.path for f in assessment.findings if f.code == code))

    def preflight_refuse(self, document, code):
        # Test the owning backend rule even when the same unsupported fixture also
        # violates a common semantic rule. Such a borrowed block is insufficient.
        findings = preflight(Schedule.from_dict(document), TARGET)
        self.assertTrue(any(f.code == code for f in findings), findings)
        self.assertTrue(all(f.path for f in findings if f.code == code))
        self.assertFalse(self.compiler.assess(document).lowering_eligible)

    def test_two_deep_gemm_resets_inner_carry_and_stores_each_outer_tile(self):
        for tail, invariant in ((False, False), (True, False), (False, True)):
            with self.subTest(tail=tail, invariant=invariant):
                d = _gemm(tail=tail, invariant_b=invariant)
                emission = self.lower(d)
                shape = {b["name"]: b["shape"] for b in d["buffers"]}
                m, k = shape["a"]
                n = shape["b"][0]
                a = [(i * 3 + t) % 7 - 3 for i in range(m) for t in range(k)]
                b = [(j + t * 2) % 5 - 2 for j in range(n) for t in range(k)]
                bias = [j - 2 for j in range(n)]
                memories = dict(a=a, b=b, bias=bias, c=[None] * (m*n))
                tl = _execute(emission, memories)
                expected = [sum(a[i*k+t] * b[j*k+(t % 16 if invariant else t)] for t in range(k)) + bias[j]
                            for i in range(m) for j in range(n)]
                self.assertEqual(memories["c"], expected)
                self.assertEqual(set(tl.stores.values()), {1})
                self.assertEqual(tl.loads[id(bias)], (n+15)//16)
                if invariant:
                    self.assertEqual(tl.loads[id(b)], 1)
                self.assertIn("for m in tl.range(0, N_M_LOOP, BLOCK_M_LOOP, num_stages=1", emission.source)
                self.assertIn("for k in tl.range(0, N_K_LOOP, BLOCK_K_LOOP, num_stages=2", emission.source)

    def test_outer_invariant_load_executes_once_per_output_tile(self):
        d = _gemm()
        d["tile_loops"][1]["body"].insert(0, "load_bias")
        emission = self.lower(d)
        memories = dict(a=[1]*1024, b=[2]*512, bias=list(range(16)), c=[None]*512)
        tl = _execute(emission, memories)
        self.assertEqual(memories["c"], [64+j for _ in range(32) for j in range(16)])
        self.assertEqual(tl.loads[id(memories["bias"])], 2)

    def test_nested_reduction_uses_result_shape_and_resets_each_outer_iteration(self):
        for op in ("sum", "max"):
            with self.subTest(op=op):
                emission = self.lower(_reduction(op))
                x = [-1 - (batch * 40 + row * 8 + k) % 13
                     for batch in range(2) for row in range(5) for k in range(8)]
                memories = dict(x=x, y=[None]*10)
                tl = _execute(emission, memories)
                fold = sum if op == "sum" else max
                self.assertEqual(memories["y"], [fold(x[i*8:(i+1)*8]) for i in range(10)])
                self.assertEqual(set(tl.stores.values()), {1})

    def test_scalar_carried_store_keeps_singleton_state_for_both_reduction_ranks(self):
        for singleton, op in itertools.product((False, True), ("sum", "max")):
            with self.subTest(singleton=singleton, op=op):
                emission = self.lower(_scalar_store(op=op, singleton=singleton))
                identity = ('tl.zeros((1,), tl.float32)' if op == "sum" else
                            'tl.full((1,), float("-inf"), tl.float32)')
                self.assertIn("rows = " + identity, emission.source)
                x = [float(i-17) for i in range(16)]
                memories = dict(x=x, y=[None]*2)
                tl = _execute(emission, memories)
                fold = sum if op == "sum" else max
                self.assertEqual(memories["y"], [fold(x[:8]), fold(x[8:])])
                self.assertEqual(tl.store_types, [((1,), (1,))]*2)
                self.assertEqual(set(tl.stores.values()), {1})

    def test_single_value_producers_store_through_scalar_and_vector_addresses(self):
        producers = (("resident", False, ()), ("resident", True, (1,)),
                     ("scalar_load", False, ()), ("block_load", False, (1,)))
        for (producer, singleton, value_shape), vector_store in itertools.product(producers, (False, True)):
            with self.subTest(producer=producer, singleton=singleton, vector_store=vector_store):
                emission = self.lower(_scalar_store(producer, singleton=singleton,
                                                    vector_store=vector_store))
                x = [float(i+1) for i in range(16 if producer == "resident" else 2)]
                memories = dict(x=x, y=[None]*2)
                tl = _execute(emission, memories)
                expected = [sum(x[:8]), sum(x[8:])] if producer == "resident" else x
                self.assertEqual(memories["y"], expected)
                self.assertEqual(tl.store_types, [((1,), value_shape)]*2)
                self.assertEqual(set(tl.stores.values()), {1})

    def test_store_type_model_rejects_block_to_scalar_pointer_and_preserves_masks(self):
        tl, memory = _TL(), [None]
        with self.assertRaisesRegex(TypeError, "scalar pointer"):
            tl.store(_Pointer(memory), _Tile((1,), [7]))
        self.assertEqual(memory, [None])
        tl.store(_Pointer(memory), _Tile((), [3]))
        self.assertEqual(memory, [3])
        pointer = _Pointer(memory, _Tile((1,), [0]))
        tl.store(pointer, _Tile((), [9]), mask=False)
        self.assertEqual(memory, [3])
        tl.store(pointer, _Tile((), [7]), mask=True)
        self.assertEqual(memory, [7])

    def test_resident_reduction_inside_output_loop_does_not_carry(self):
        emission = self.lower(_reduction(resident=True))
        memories = dict(x=list(range(80)), y=[None]*10)
        _execute(emission, memories)
        self.assertEqual(memories["y"], [sum(range(i*8, (i+1)*8)) for i in range(10)])
        self.assertNotIn("rows +=", emission.source)

    def test_analysis_counts_nested_operations_and_outer_stores(self):
        schedule = Schedule.from_dict(_gemm())
        counts = {r.operation: r.whole_grid for r in operation_repetitions(schedule)}
        self.assertEqual(counts, dict(load_bias=1, load_a=4, load_b=4, dot=4, add_bias=2, store_c=2))
        self.assertEqual(work_bound(schedule).mma_flops, 2*32*16*32)
        self.assertEqual([l.name for l in schedule.enclosing_loops(schedule.operation("dot"))], ["m_loop", "k_loop"])

    def test_inner_accumulator_cannot_escape_outer_output_loop(self):
        d = _gemm()
        d["tile_loops"][1]["body"] = ["k_loop"]
        self.refuse(d, "BUFFER_ESCAPES_LOOP")

    def test_parent_invariant_is_visible_to_descendants_but_child_tile_cannot_escape(self):
        d = _gemm()
        d["operations"][4]["reads"] = ["a_tile", "bias_tile"]
        self.refuse(d, "BUFFER_ESCAPES_LOOP")

    def test_loop_coordinate_cannot_be_used_after_its_loop(self):
        d = _gemm()
        d["access_maps"][2]["indices"] = [dict(source="loop_tile", name="k")]
        self.refuse(d, "ACCESS_LOOP_SCOPE")

    def test_out_of_order_child_and_interleaved_root_operation_are_refused(self):
        d = _gemm()
        d["tile_loops"][1]["body"] = ["add_bias", "k_loop", "store_c"]
        self.refuse(d, "LOOP_BODY_SCOPE_ORDER")
        d = _gemm()
        bias = d["operations"].pop(0)
        d["operations"].insert(2, bias)
        self.refuse(d, "LOOP_BODY_SCOPE_ORDER")

    def test_loop_store_requires_distinct_affine_output_coordinates(self):
        d = _gemm()
        d["access_maps"][3]["indices"][0] = dict(source="dimension", dimension=0, extent=16)
        self.refuse(d, "TRITON_LOOP_STORE_OWNERSHIP")

    def test_in_loop_scan_stays_a_localized_refusal(self):
        d = _reduction()
        d["operations"][1].update(kind="scan", parameters=dict(op="sum", axis=1))
        self.preflight_refuse(d, "TRITON_OPERATION_POSITION")

    def test_nested_special_reductions_and_unproven_mma_operands_are_refused(self):
        d = _reduction()
        d["operations"][1].update(kind="reduce_argmin", parameters=dict(tie_break="lowest_index", nan_policy="reject_input", across_loop=True))
        self.preflight_refuse(d, "TRITON_NESTED_OPERATION_UNSUPPORTED")
        d = _gemm()
        d["operations"][1].update(kind="cast", reads=["b_tile"], parameters=dict(to="bf16"))
        self.preflight_refuse(d, "TRITON_NESTED_MMA_OPERAND")

    def test_mma_carry_over_an_ancestor_is_explicitly_refused(self):
        d = _gemm()
        d["access_maps"][0]["indices"].reverse()
        self.preflight_refuse(d, "TRITON_MMA_ANCESTOR_CARRY")

    def test_sibling_deeper_and_dynamic_nests_remain_explicitly_refused(self):
        d = _gemm()
        d["tile_loops"][1]["body"].remove("k_loop")
        self.preflight_refuse(d, "TRITON_LOOP_NEST_UNSUPPORTED")
        d = _gemm()
        third = copy.deepcopy(d["tile_loops"][1])
        third.update(name="third", iterator="third_i", body=["m_loop"])
        d["tile_loops"].append(third)
        self.preflight_refuse(d, "TRITON_TILE_LOOP_COUNT")
        d = _reduction()
        d["tile_loops"][1]["stop"] = dict(program="batch", add=1, floor_div=1)
        self.refuse(d, "TRITON_NESTED_LOOP_STOP")
        for option in ("flatten", "warp_specialize"):
            d = _gemm()
            d["tile_loops"][0]["range_options"][option] = True
            self.refuse(d, "TRITON_NESTED_LOOP_OPTION")
