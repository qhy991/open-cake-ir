"""Public lowering contracts; CPU source checks are not on-device qualification."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, EmitError
from open_cake_ir.compiler.backends import cutedsl, cutedsl_register
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.target import Target


ROOT = Path(__file__).resolve().parents[2]


def register_schedule(*, shape=(33, 19, 21), tile=(16, 16, 16)) -> dict:
    """Adapt the mathematical corpus fixture to explicit warp hardware commitments."""
    document = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke.json").read_text())
    document["target"] = "sm_103a"
    document["lowering"] = {"backend": "cutlass_cute_dsl", "entry_point": "warp_matmul"}
    document["roles"][0]["warps"] = [0]
    document.pop("residency")
    m, n, k = shape
    bm, bn, bk = tile
    shapes = {"a": [m, k], "b": [n, k], "bias": [n], "c": [m, n],
              "a_tile": [bm, bk], "b_tile": [bn, bk], "acc": [bm, bn],
              "bias_tile": [bn], "c_tile": [bm, bn]}
    for buffer in document["buffers"]:
        buffer["shape"] = shapes[buffer["name"]]
    for axis, size in zip(document["program_map"]["axes"], (bm, bn)):
        axis["tile"] = size
    document["tile_loops"][0]["tile"] = bk
    document["tile_loops"][0]["range_options"]["num_stages"] = 1
    document["operations"][2]["parameters"] = {
        "accumulator": "fp32", "tile_shape": list(tile), "instruction": {
            "contract": cutedsl_register.MMA_CONTRACT, "shape": [16, 8, 16],
            "cta_group": 1, "operand_source": "register", "operand_major": ["k", "k"],
        },
    }
    document["operations"][-1]["parameters"]["coalesced"] = False
    return document


def rename(document: dict, old: str, new: str) -> dict:
    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return new if value == old else value
    return replace(document)


class RegisterCuTeTests(unittest.TestCase):
    def test_current_register_cute_refuses_selected_ranges_at_public_boundaries(self):
        from open_cake_ir.compiler.backends import cutedsl, cutedsl_register
        from open_cake_ir.compiler.backends.common import EmitError
        from open_cake_ir.compiler.target import Target

        value = json.loads((ROOT / 'corpus/schedules/b300-cute-register-primary.json').read_text())
        operation = next(op for op in value['operations'] if op['kind'] == 'mma')
        extent = operation['parameters']['tile_shape'][2]
        operation['parameters']['k_ranges'] = [[0, extent // 2]]
        schedule = Schedule.from_dict(value)
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        for backend in (cutedsl, cutedsl_register):
            with self.subTest(backend=backend.__name__):
                self.assertIn('CUTE_MMA_K_RANGES_UNSUPPORTED', [f.code for f in backend.requirements(schedule)])
                self.assertIn('CUTE_MMA_K_RANGES_UNSUPPORTED', [f.code for f in backend.preflight(schedule, target)])
                with self.assertRaises(EmitError):
                    backend.emit(schedule, target)

    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_103a.json")

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        return self.compiler.lower(assessment)

    def refuses(self, document, code, path=None):
        schedule = Schedule.from_dict(document)
        direct = cutedsl.preflight(schedule, self.target)
        matching = [finding for finding in direct if finding.code == code]
        self.assertTrue(matching, direct)
        if path is not None:
            self.assertIn(path, [finding.path for finding in matching])
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        if assessment.accepted:
            self.assertTrue(set(matching) <= set(assessment.findings))
        with self.assertRaises(EmitError):
            cutedsl.emit(schedule, self.target)

    def test_shapes_tiles_and_independent_tails_have_exact_pointer_grid_contract(self):
        for shape in ((32, 16, 128), (33, 16, 128), (32, 17, 128), (32, 16, 129), (1, 1, 65), (17, 9, 71)):
            for tile in ((16, 8, 16), (16, 16, 32), (16, 32, 64)):
                with self.subTest(shape=shape, tile=tile):
                    document = register_schedule(shape=shape, tile=tile)
                    lowering = self.lower(document)
                    requirements = lowering.toolchain_requirements
                    self.assertEqual(dict(requirements), {
                        "compiler": "cutlass_cute_dsl", "source_language": "python", "target": "sm_103a",
                        "kernel_entry_point": "warp_matmul",
                        "signature": [{"name": name, "dtype": dtype} for name, dtype in (
                            ("a", "bf16"), ("b", "bf16"), ("bias", "fp32"), ("c", "fp32"))],
                        "grid": [(shape[0] + 15) // 16, (shape[1] + tile[1] - 1) // tile[1], 1],
                        "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0,
                    })
                    module = ast.parse(lowering.source)
                    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
                    self.assertEqual(len(functions), 1)
                    kernel = functions[0]
                    self.assertEqual(kernel.name, requirements["kernel_entry_point"])
                    self.assertEqual([arg.arg for arg in kernel.args.args], [row["name"] for row in requirements["signature"]])
                    self.assertEqual([ast.unparse(arg.annotation) for arg in kernel.args.args], ["cute.Pointer"] * 4)
                    self.assertEqual([ast.unparse(node) for node in kernel.decorator_list], ["cute.kernel"])
                    self.assertEqual(set(lowering.source_map), {op["id"] for op in document["operations"]})

    def test_names_and_global_declaration_order_are_not_operator_dispatch(self):
        document = register_schedule()
        for old, new in (("a", "z_operand"), ("b", "a_operand"), ("bias", "weights"),
                         ("c", "answer"), ("dot", "contraction"), ("add_bias", "combine"),
                         ("compute", "worker"), ("m_block", "row_block"), ("k_loop", "contraction_loop")):
            document = rename(document, old, new)
        document["schedule_id"] = "unrelated mathematical schedule"
        # Declaration order need not follow mathematical roles; ABI groups inputs then outputs.
        buffers = document["buffers"]
        document["buffers"] = [buffers[3], buffers[2], buffers[0], buffers[1], *buffers[4:]]
        lowering = self.lower(document)
        self.assertEqual([row["name"] for row in lowering.toolchain_requirements["signature"]],
                         ["weights", "z_operand", "a_operand", "answer"])
        self.assertIn("contraction", lowering.source_map)
        self.assertNotIn("triton", lowering.source)
        self.assertNotIn("torch", lowering.source)

    def test_grid_axis_swap_is_emitted_from_program_map(self):
        document = register_schedule(shape=(65, 19, 21))
        document["program_map"]["axes"][0]["axis"] = 1
        document["program_map"]["axes"][1]["axis"] = 0
        lowering = self.lower(document)
        self.assertEqual(lowering.toolchain_requirements["grid"], [2, 5, 1])
        self.assertIn("_cake_m0 = _cake_by * 16", lowering.source)
        self.assertIn("_cake_n0 = _cake_bx * 16", lowering.source)

    def test_collective_mma_is_outside_lane_predicates_and_operand_fragments_reset_each_trip(self):
        module = ast.parse(self.lower(register_schedule()).source)
        parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
        gemm = next(node for node in ast.walk(module) if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.gemm")
        ancestors = []
        node = gemm
        while node in parents:
            node = parents[node]
            ancestors.append(node)
        self.assertFalse(any(isinstance(node, ast.If) for node in ancestors))
        k_loop = next(node for node in ancestors if isinstance(node, ast.For))
        self.assertEqual(ast.unparse(k_loop.iter), "cutlass.range(2)")
        resets = [ast.unparse(node.value) for node in k_loop.body if isinstance(node, ast.Expr)]
        self.assertIn("_cake_r_a.fill(0.0)", resets)
        self.assertIn("_cake_r_b.fill(0.0)", resets)
        self.assertEqual(sum("cute.gemm" in reset for reset in resets), 1)

    def test_cache_intents_are_actual_copy_atom_options(self):
        for reuse, spelling in ((None, None), ("streamed", "LoadCacheMode.GLOBAL"),
                               ("reused", "CacheEvictionPriority.EVICT_LAST")):
            document = register_schedule()
            for index in (0, 1, 3):
                parameters = document["operations"][index]["parameters"]
                parameters.pop("reuse", None)
                if reuse is not None:
                    parameters["reuse"] = reuse
            source = self.lower(document).source
            self.assertEqual(source.count("cute.nvgpu.CopyG2ROp()"), 3)
            if spelling is None:
                self.assertNotIn("load_cache_mode", source)
                self.assertNotIn("l1c_evict_priority", source)
            else:
                self.assertEqual(source.count(spelling), 3)

    def test_each_unimplemented_loop_control_is_locally_refused(self):
        for field, value in (("num_stages", 2), ("loop_unroll_factor", 2), ("flatten", True),
                             ("warp_specialize", True), ("disable_licm", True)):
            with self.subTest(field=field):
                document = register_schedule()
                document["tile_loops"][0]["range_options"][field] = value
                self.refuses(document, "CUTE_RANGE_OPTION_UNSUPPORTED", f"tile_loops[0].range_options.{field}")
        document = register_schedule()
        document["tile_loops"][0]["range_options"]["disallow_acc_multi_buffer"] = False
        self.lower(document)

    def test_single_trip_loop_preserves_the_canonical_ir_refusal(self):
        for k in (1, 7, 16):
            document = register_schedule(shape=(1, 1, k))
            assessment = self.compiler.assess(document)
            self.assertFalse(assessment.accepted)
            self.assertIn("TILE_LOOP_SINGLE_TRIP", {finding.code for finding in assessment.findings})
            self.refuses(document, "CUTE_REGISTER_LOOP", "tile_loops[0]")

    def test_index_arithmetic_refuses_extents_that_can_overflow(self):
        self.refuses(register_schedule(shape=(2**31 - 9, 1, 17)), "CUTE_REGISTER_INDEX_RANGE", "buffers")

    def test_malformed_operation_graph_is_refused_before_emission(self):
        mutations = (
            (lambda d: d["operations"].pop(3), "CUTE_REGISTER_OPERATIONS"),
            (lambda d: d["operations"][4]["reads"].__setitem__(0, "a_tile"), "CUTE_REGISTER_DATAFLOW"),
            (lambda d: d["operations"][4].__setitem__("depends_on", ["dot"]), "CUTE_REGISTER_DEPENDENCIES"),
            (lambda d: d["tile_loops"][0].__setitem__("body", ["load_a", "dot"]), "CUTE_REGISTER_LOOP"),
            (lambda d: d["operations"].__setitem__(slice(0, 2), d["operations"][1::-1]), "CUTE_REGISTER_ORDER"),
            (lambda d: d["operations"][4]["parameters"].__setitem__("op", "mul"), "CUTE_REGISTER_ELEMENTWISE"),
            (lambda d: d["operations"][4]["parameters"].__setitem__("broadcast_axis", 0), "CUTE_REGISTER_ELEMENTWISE"),
            (lambda d: d["operations"][-1]["parameters"].__setitem__("coalesced", True), "CUTE_REGISTER_STORE_COALESCING"),
        )
        for mutation, code in mutations:
            with self.subTest(code=code):
                document = register_schedule()
                mutation(document)
                self.refuses(document, code)

    def test_instruction_storage_shape_access_and_role_boundaries(self):
        mutations = (
            (lambda d: d["operations"][2]["parameters"]["instruction"].__setitem__("contract", "triton.dot.bf16_fp32"), "CUTE_REGISTER_MMA"),
            (lambda d: d["operations"][2]["parameters"]["instruction"].__setitem__("shape", [16, 8, 8]), "CUTE_REGISTER_MMA"),
            (lambda d: d["operations"][2]["parameters"]["instruction"].__setitem__("operand_source", "shared"), "CUTE_REGISTER_MMA"),
            (lambda d: d["buffers"][1]["shape"].__setitem__(1, 7), "CUTE_REGISTER_SHAPE"),
            (lambda d: d["buffers"][4].__setitem__("stages", 2), "CUTE_REGISTER_BUFFER_OPTIONS"),
            (lambda d: d["buffers"][0].__setitem__("byte_offset", 2), "CUTE_REGISTER_BUFFER_OPTIONS"),
            (lambda d: d["buffers"][2].__setitem__("mode", "state"), "CUTE_REGISTER_BUFFER"),
            (lambda d: d["roles"][0].__setitem__("warps", [0, 1]), "CUTE_REGISTER_ROLE"),
            (lambda d: d.__setitem__("residency", {"registers_per_thread": 128}), "CUTE_REGISTER_RESIDENCY"),
            (lambda d: d["access_maps"].pop(), "CUTE_REGISTER_ACCESS"),
            (lambda d: d["access_maps"][1]["indices"][0].__setitem__("name", "m_block"), "CUTE_REGISTER_ACCESS"),
            (lambda d: d["program_map"].__setitem__("persistent", True), "CUTE_REGISTER_PROGRAM_MAP"),
        )
        for mutation, code in mutations:
            with self.subTest(code=code):
                document = register_schedule()
                mutation(document)
                self.refuses(document, code)

    def test_python_symbols_are_refused_at_assessment_and_local_names_do_not_capture_pointers(self):
        for name in ("cute", "cutlass", "warp", "class", "open_cake_cute_launch", "warp_matmul", "__arg"):
            with self.subTest(name=name):
                self.refuses(rename(register_schedule(), "a", name), "CUTE_REGISTER_IDENTIFIER")
        document = rename(register_schedule(), "a", "_cake_r_a")
        document = rename(document, "b", "_")
        source = self.lower(document).source
        module = ast.parse(source)
        names = {arg.arg for arg in module.body[-1].args.args}
        writes = {node.id for node in ast.walk(module) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.assertFalse(names & writes)

    def test_non_ascii_buffer_symbols_are_refused_before_python_normalizes_the_abi(self):
        for name in ("Ｋ", "ｃｕｔｅ", "K", "é"):
            with self.subTest(name=name):
                self.refuses(rename(register_schedule(), "a", name),
                             "CUTE_REGISTER_IDENTIFIER", "buffers[0].name")
        lowering = self.lower(rename(register_schedule(), "a", "K"))
        kernel = ast.parse(lowering.source).body[-1]
        self.assertEqual([arg.arg for arg in kernel.args.args],
                         [row["name"] for row in lowering.toolchain_requirements["signature"]])

    def test_non_ascii_entry_symbols_are_refused_at_public_and_direct_boundaries(self):
        schedule = Schedule.from_dict(register_schedule())
        for name in ("Ｋ", "ｃｕｔｅ", "K", "é"):
            with self.subTest(name=name):
                document = register_schedule()
                document["lowering"]["entry_point"] = name
                assessment = self.compiler.assess(document)
                self.assertFalse(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                self.assertEqual(assessment.findings[0].code, "SCHEDULE_STRUCTURE")
                self.assertEqual(assessment.findings[0].path, "schedule.lowering.entry_point")
                typed = replace(schedule, lowering=replace(schedule.lowering, entry_point=name))
                findings = cutedsl.preflight(typed, self.target)
                self.assertIn(("CUTE_REGISTER_IDENTIFIER", "lowering.entry_point"),
                              {(finding.code, finding.path) for finding in findings})
                with self.assertRaises(EmitError):
                    cutedsl.emit(typed, self.target)
                with self.assertRaises(EmitError):
                    cutedsl.emit(schedule, self.target, entry_point=name)

    def test_provenance_comments_cannot_inject_python_statements(self):
        document = register_schedule()
        document["schedule_id"] = "test\nraise RuntimeError('injected')"
        module = ast.parse(self.lower(document).source)
        self.assertEqual(len(module.body), 4)
        bad = rename(register_schedule(), "load_a", "load_a\nraise RuntimeError('injected')")
        self.refuses(bad, "CUTE_REGISTER_IDENTIFIER", "operations[0].id")

    def test_legacy_emission_and_b300_refusal_are_preserved(self):
        schedule = Schedule.load(ROOT / "corpus/schedules/flash-kmeans-assignment-full.json")
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        self.assertFalse(cutedsl_register.applies(schedule))
        self.assertEqual(cutedsl.emit(schedule, target), cutedsl._Emitter(schedule, target).emit())
        document = json.loads((ROOT / "corpus/schedules/flash-kmeans-assignment-full.json").read_text())
        document["target"] = "sm_103a"
        self.refuses(document, "CUTE_TARGET_UNSUPPORTED", "target")
        self.assertEqual(cutedsl.SUPPORTED_OPERATION_KINDS,
                         frozenset(cutedsl.BODY_EMITTERS) | cutedsl_register.SUPPORTED_OPERATION_KINDS)
        self.assertIn(OperationKind.ELEMENTWISE, cutedsl.SUPPORTED_OPERATION_KINDS)


if __name__ == "__main__":
    unittest.main()
