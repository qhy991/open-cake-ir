"""Kernel parameters follow body use; host geometry and launch options stay intact."""
import ast
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from open_cake_ir.tasks.workloads import create_task
from tests.contracts.test_triton_loop_scopes import _execute, _reduction, _scalar_store

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


class LiveTritonConstants(unittest.TestCase):
    def assert_interface(self, source, requirements):
        """The actual host call and compile metadata must name the same parameters."""
        tree = ast.parse(source)
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == requirements["kernel_entry_point"])
        self.assertEqual([arg.arg for arg in kernel.args.args],
                         [*requirements["signature"], *requirements["compile_constants"]])
        parameters = {arg.arg for arg in kernel.args.args if arg.annotation is not None}
        reads = {node.id for statement in kernel.body for node in ast.walk(statement)
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        self.assertLessEqual(parameters, reads)
        launch = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id.startswith("_cake_launch_"))
        options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in launch.keywords}
        self.assertEqual(options, {**requirements["compile_constants"], **requirements["compile_options"]})

    def assert_native_interface(self, source, requirements):
        self.assert_interface(source, requirements)
        projected = project_triton_kernel(source.encode(), requirements)
        validate_triton_kernel(projected, requirements)

    def lower(self, document):
        compiler = Compiler.load(ROOT)
        assessed = compiler.assess(document)
        self.assertTrue(assessed.lowering_eligible, assessed.findings)
        lowered = compiler.lower(assessed)
        self.assert_native_interface(lowered.source, lowered.toolchain_requirements)
        for operation in document["operations"]:
            start, _ = lowered.source_map[operation["id"]]
            self.assertIn(operation["id"], lowered.source.splitlines()[start - 1])
        return emit(Schedule.from_dict(document), TARGET)

    def test_grid_only_extents_do_not_become_parameters_or_change_output_coverage(self):
        emission = self.lower(_scalar_store("scalar_load"))
        self.assertEqual(emission.toolchain["compile_constants"], {})
        self.assertEqual(emission.constants["N_BATCH"], 2)
        self.assertEqual(emission.toolchain["grid"], [2, 1, 1])
        memories = dict(x=[7, -3], y=[None, None])
        observed = _execute(emission, memories)
        self.assertEqual(memories["y"], [7, -3])
        self.assertEqual(set(observed.stores.values()), {1})

    def test_loop_bounds_tail_masks_and_launch_options_survive(self):
        document = _reduction()
        emission = self.lower(document)
        self.assertNotIn("N_BATCH", emission.toolchain["compile_constants"])
        self.assertIn("N_ROWS_LOOP", emission.toolchain["compile_constants"])
        self.assertIn("N_FEATURES_LOOP", emission.toolchain["compile_constants"])
        values = list(range(80))
        memories = dict(x=values, y=[None] * 10)
        observed = _execute(emission, memories)
        self.assertEqual(memories["y"], [sum(values[i:i + 8]) for i in range(0, 80, 8)])
        self.assertEqual(set(observed.stores.values()), {1})

        single_loop = self.lower(_scalar_store("carried"))
        self.assertNotIn("NUM_STAGES", single_loop.toolchain["compile_constants"])
        self.assertEqual(single_loop.toolchain["compile_options"]["num_stages"], 2)
        memories = dict(x=list(range(16)), y=[None, None])
        _execute(single_loop, memories)
        self.assertEqual(memories["y"], [28, 92])

    def test_persistent_grid_keeps_stride_constants_and_visits_every_element_once(self):
        document = _scalar_store("scalar_load")
        count = TARGET.occupancy.multiprocessor_count + 3
        for buffer in document["buffers"]:
            if buffer["space"] == "global":
                buffer["shape"] = [count]
        document["program_map"].update(persistent=True, traversal=["batch"])
        document["residency"] = dict(ctas_per_multiprocessor=1, registers_per_thread=32)
        emission = self.lower(document)
        self.assertEqual(emission.toolchain["compile_constants"], {
            "TOTAL_TILES": count, "NUM_CTAS": TARGET.occupancy.multiprocessor_count})
        self.assertEqual(emission.toolchain["compile_options"]["maxnreg"], 32)
        memories = dict(x=list(range(count)), y=[None] * count)
        observed = _execute(emission, memories)
        self.assertEqual(memories["y"], memories["x"])
        self.assertEqual(set(observed.stores.values()), {1})

    def test_registered_multi_output_and_state_sources_keep_native_abi(self):
        for task in ("adadelta", "adamw", "momentum_sgd"):
            with self.subTest(task=task):
                _, source = create_task(task, backend="triton-dcu", rows=128, columns=1024)
                schedule = Schedule.from_dict(frontend.parse(source).document)
                emission = emit(schedule, Target.load(ROOT / "compiler/targets/gfx938.json"))
                self.assert_native_interface(emission.source, emission.toolchain)
                self.assertNotIn("N_ROW", emission.toolchain["compile_constants"])
                self.assertEqual(len(emission.toolchain["signature"]),
                                 sum(buffer.space.value == "global" for buffer in schedule.buffers))
        document = json.loads((ROOT / "corpus/schedules/atomic-reservation-b8-smoke.json").read_text())
        # Compiler state emission exceeds the narrower native-author subset.
        emission = emit(Schedule.from_dict(document), TARGET)
        self.assert_interface(emission.source, emission.toolchain)
        self.assertNotIn("N_TOKEN", emission.toolchain["compile_constants"])

    def test_program_stages_use_the_same_native_compile_contract(self):
        source = (ROOT / "examples/python/epilogue_program.py").read_text()
        program = parse_program(source, program_id="rounded-epilogue-python").program
        lowered = Compiler.load(ROOT).lower_program(program)
        for stage in lowered.lowerings:
            with self.subTest(stage=stage.schedule_id):
                self.assert_native_interface(stage.source, stage.toolchain_requirements)
                self.assertNotIn("N_ROW", stage.toolchain_requirements["compile_constants"])
