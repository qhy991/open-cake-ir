"""Qualification specialization probes; CPU/source checks are not GPU evidence."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import native_cuda_contract as c
import native_cuda_evaluate as e
from open_cake_ir.compiler import Compiler


class KMeansPartialSpecializations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The public draft loader is the supported source-development boundary.
        # Production prepare/run/verify continue to require a released Compiler.
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.rows = [row for row in c.cases() if row["family"] == "kmeans"]

    def test_formal_row_lowering_preserves_all_cases_and_both_abis(self):
        lowered = e.lower_rows(self.compiler)
        self.assertEqual([r["id"] for r, _, _ in lowered], [r["id"] for r in c.cases()])
        self.assertEqual(len(lowered), 12)
        self.assertEqual(sum("baseline" in r for r, _, _ in lowered), 11)
        for row, document, source in lowered:
            with self.subTest(row=row["id"]):
                abi = c.owner(row).tensor_abi(row["case_id"])
                c.validate_metadata(row["metadata"], target=row["target"], abi=abi)
                if row["family"] == "two_mma":
                    with self.assertRaises(c.QualificationError):
                        c.baseline_schedule(row)
                    continue
                baseline = c.baseline_schedule(row)
                globals_ = [{k: b[k] for k in ("name", "shape", "dtype", "mode")}
                            for b in baseline["buffers"] if b["space"] == "global"]
                self.assertEqual(globals_, row["metadata"]["arguments"])
                metadata = row["baseline"]["metadata"]
                self.assertEqual(metadata["compiler"], "triton")
                self.assertEqual(metadata["target"], row["target"])
                self.assertEqual(metadata["signature"], {
                    a.name: "*" + ("i32" if a.dtype == "int32" else a.dtype) for a in abi})

    def test_all_kmeans_cases_keep_complementary_coordinates_and_two_results(self):
        for row in self.rows:
            depth = c.owner(row).case(row["case_id"])["shape"]["D"]
            expected = [[k for k in range(depth) if (k // 16) % 2 == parity] for parity in (0, 1)]
            for native, specialize in ((True, c.specialize), (False, c.baseline_schedule)):
                with self.subTest(row=row["id"], native=native):
                    document = specialize(row)
                    operations = document["operations"]
                    mmas = [op for op in operations if op["kind"] == "mma"]
                    self.assertEqual(len(mmas), 2)
                    contraction = [loop for loop in document.get("tile_loops", [])
                                   if loop["buffer"] == "tokens" and loop["dimension"] == 2]
                    self.assertEqual(len(contraction), int(native))
                    offsets = range(0, depth, contraction[0]["tile"]) if native else (0,)
                    coordinates = [[offset + k for offset in offsets
                                    for start, end in op["parameters"]["k_ranges"]
                                    for k in range(start, end)] for op in mmas]
                    self.assertEqual(coordinates, expected)
                    self.assertEqual(sorted(coordinates[0] + coordinates[1]), list(range(depth)))
                    self.assertEqual(mmas[0]["reads"], mmas[1]["reads"])
                    self.assertNotEqual(mmas[0]["writes"], mmas[1]["writes"])
                    results = [op["writes"][0] for op in mmas]
                    buffers = {b["name"]: b for b in document["buffers"]}
                    for result in results:
                        self.assertEqual(buffers[result]["dtype"], "fp32")
                    if native:
                        self.assertNotEqual(mmas[0]["signals"], mmas[1]["signals"])
                        readouts = [next(op for op in operations if op["kind"] == "load"
                                         and op["reads"] == [result]) for result in results]
                        for mma, readout in zip(mmas, readouts):
                            self.assertEqual(readout["waits"], mma["signals"])
                            self.assertEqual(readout["parameters"]["movement"], "tmem")
                            self.assertIn(mma["id"], contraction[0]["body"])
                        results = [op["writes"][0] for op in readouts]
                    combine, = [op for op in operations if op["kind"] == "elementwise"
                                and op["reads"] == results]
                    self.assertEqual(combine["parameters"], {"op": "add"})
                    self.assertTrue(self.compiler.assess(document).lowering_eligible)

    def test_single_centroid_normalization_preserves_both_contributions(self):
        row = next(row for row in self.rows if row["case_id"] == "duplicate_tie")
        for specialize in (c.specialize, c.baseline_schedule):
            document = specialize(row)
            self.assertFalse(any(loop["buffer"] == "centroids" for loop in document["tile_loops"]))
            argmin, = [op for op in document["operations"] if op["kind"] == "reduce_argmin"]
            self.assertFalse(argmin["parameters"]["across_loop"])
            centroid_axis, = [axis for axis in document["program_map"]["axes"]
                              if axis["buffer"] == "centroids"]
            self.assertEqual(centroid_axis["axis"], 2)
            lowered = self.compiler.lower(self.compiler.assess(document))
            self.assertEqual(lowered.toolchain_requirements["grid"], [1, 1, 1])

    def test_emitted_triton_executes_selected_operands_and_materializes_each_partial(self):
        # Execute only the emitted contraction expressions on exact integer-valued
        # CPU arrays. This checks coordinates and explicit result boundaries;
        # downstream Triton IR and GPU rounding require their own actual receipts.
        for row in self.rows:
            with self.subTest(row=row["id"]):
                document = c.baseline_schedule(row)
                mmas = [op for op in document["operations"] if op["kind"] == "mma"]
                results = [op["writes"][0] for op in mmas]
                combine, = [op for op in document["operations"] if op["reads"] == results]
                names = {*results, combine["writes"][0]}
                source = self.compiler.lower(self.compiler.assess(document)).source
                assignments = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                               and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
                buffers = {b["name"]: b for b in document["buffers"]}
                rng = np.random.default_rng(42)
                operands = [rng.integers(-2, 3, size=buffers[name]["shape"]).astype(np.float32)
                            for name in mmas[0]["reads"]]
                events = []
                gathered = []
                def gather(value, indices, axis):
                    result = np.take_along_axis(value, indices, axis)
                    gathered.append(result)
                    return result
                def dot(a, b, *, out_dtype):
                    self.assertIs(out_dtype, np.float32)
                    events.append("dot")
                    return a @ b
                def materialize(assembly, *, constraints, args, dtype, is_pure, pack):
                    self.assertEqual((assembly, constraints, dtype, is_pure, pack),
                                     ("mov.b32 $0, $1;", "=f,f", np.float32, True, 1))
                    self.assertEqual(len(args), 1)
                    events.append("materialize")
                    return args[0].copy()
                class LogicalTL:
                    float32 = np.float32
                    arange = staticmethod(np.arange)
                    where = staticmethod(np.where)
                    broadcast_to = staticmethod(np.broadcast_to)
                    trans = staticmethod(np.transpose)
                tl = LogicalTL()
                tl.gather, tl.dot, tl.inline_asm_elementwise = gather, dot, materialize
                env = dict(zip(mmas[0]["reads"], operands), tl=tl)
                for node in assignments:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "<CPU emitted partials>", "exec"), env)
                self.assertEqual(events, ["dot", "materialize", "dot", "materialize"])
                self.assertEqual(len(gathered), 4)
                for parity in (0, 1):
                    selected = [k for k in range(128) if (k // 16) % 2 == parity]
                    a, b = [operand[:, selected] for operand in operands]
                    np.testing.assert_array_equal(gathered[2 * parity], a)
                    np.testing.assert_array_equal(gathered[2 * parity + 1], b)
                    np.testing.assert_array_equal(env[results[parity]], a @ b.T)
                np.testing.assert_array_equal(env[combine["writes"][0]], operands[0] @ operands[1].T)


if __name__ == "__main__":
    unittest.main()
