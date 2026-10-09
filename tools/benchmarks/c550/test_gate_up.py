"""Software ABI and rounding contracts; CPU fixtures are not device qualification."""
from contextlib import ExitStack
from copy import deepcopy
import ast
import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _Pointer

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('gate_up', Path(__file__).with_name('gate_up.py'))
gate_up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_up)


def execute(source, inputs):
    """Execute emitted arithmetic with the existing bounds-checked memory double.

    The double models the BF16 rounding boundaries. It does not model native dot
    reduction order, native tanh error, or compiler FP32 instruction contraction.
    """
    document = frontend.parse('from open_cake_ir.compiler import frontend as cake\n' + source).document
    emission = emit(Schedule.from_dict(document), Target.load(ROOT / 'compiler/targets/xcore1002.json'))
    tree = ast.parse(emission.source)
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    kernel.decorator_list = []
    for arg in kernel.args.args:
        arg.annotation = None
    memory = deepcopy(inputs)
    for buffer in document['buffers']:
        if buffer['space'] == 'global' and buffer['mode'] == 'output':
            memory[buffer['name']] = [float('nan')] * math.prod(buffer['shape'])
    tl = _TL()
    libdevice = SimpleNamespace(tanh=lambda tile: _Tile(tile.shape,
        [rounded(math.tanh(value), 'fp32') for value in tile.values]))
    env = {'tl': tl, 'libdevice': libdevice}
    def cast(tile, dtype):
        return _Tile(tile.shape, [rounded(value, 'fp32' if dtype is _TL.float32 else dtype)
                                 for value in tile.values])
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile, 'to', cast))
        stack.enter_context(patch.object(_TL, 'bfloat16', 'bf16', create=True))
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), '<CAKE CPU contract>', 'exec'), env)
        for indices in itertools.product(*(range(n) for n in emission.toolchain['grid'])):
            tl.program = indices
            env[kernel.name](**{name: _Pointer(values) for name, values in memory.items()},
                             **emission.toolchain['compile_constants'])
    return {name: memory[name] for name in document['outputs']}, tl


class GateUpContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_original_rank_and_three_distinct_stages_survive_tail_lowering(self):
        for batch, sequence in ((3, 129), (1, 8192), (2, 1321)):
            program = gate_up.program_for(batch, sequence)
            self.assertEqual(program.inputs, ('x', 'gate_proj', 'up_proj'))
            self.assertEqual(program.outputs, ('output',))
            self.assertEqual(program.tensors['x'].shape, (batch, sequence, 3072))
            self.assertEqual(program.tensors['output'].shape, (batch, sequence, 24576))
            self.assertEqual([stage.name for stage in program.stages],
                             ['gate_projection', 'up_projection', 'gelu_product'])
            self.assertEqual(program.tensors['gate_output'].dtype.value, 'bf16')
            self.assertEqual(program.tensors['up_output'].dtype.value, 'bf16')
            lowered = self.compiler.lower_program(program)
            self.assertEqual([item.toolchain_requirements['grid'] for item in lowered.lowerings],
                             [[(sequence + 15) // 16, 768, batch]] * 2 + [[sequence, 96, batch]])
            for stage in program.stages[:2]:
                document = json.loads(stage.schedule_bytes)
                cast = next(op for op in document['operations'] if op['id'] == 'round_projection')
                self.assertEqual(cast['parameters']['to'], 'bf16')
            self.assertIn('activated_bf16 = activated.to(tl.bfloat16)', lowered.lowerings[2].source)
            self.assertIn('activated_fp32 = activated_bf16.to(tl.float32)', lowered.lowerings[2].source)

    def test_projection_output_rounds_the_bf16_midpoint_before_activation(self):
        source = gate_up._projection_source(1, 1, 64, 32,
            name='projection_test', weight='weight', output='projected')
        x = [1., 1.] + [0.] * 62
        weight = ([1., 2**-8] + [0.] * 62) * 32
        result, trace = execute(source, {'x': x, 'weight': weight})
        self.assertEqual(result['projected'], [1.] * 32)
        self.assertNotEqual(1., 1. + 2**-8)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_skipping_activation_round_changes_the_emitted_result(self):
        source = gate_up._activation_source(1, 1, 8)
        inputs = {'gate_output': [-3.] * 8, 'up_output': [0.0322265625] * 8}
        result, trace = execute(source, inputs)
        mutant = source.replace('lm.cast(activated_bf16, to="fp32", id="widen_activation")',
                                'lm.cast(activated, to="fp32", id="widen_activation")')
        unrounded, _ = execute(mutant, inputs)
        self.assertEqual(result['output'], [-0.00011682510375976562] * 8)
        self.assertNotEqual(result, unrounded)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_workload_binding_refuses_changed_order_dtype_or_shape(self):
        program = gate_up.program_for(2, 131)
        abi = [TensorABI(name, program.tensors[name].shape, program.tensors[name].dtype.value, mode)
               for mode, names in [('input', program.inputs), ('output', program.outputs)] for name in names]
        workload = SimpleNamespace(target='xcore1002', tensor_abi=lambda case: abi)
        self.assertEqual(gate_up.source_for_workload(workload, 'primary'), gate_up.source_for(2, 131))
        for mutation in ('order', 'dtype', 'shape'):
            changed = list(abi)
            if mutation == 'order': changed[1], changed[2] = changed[2], changed[1]
            elif mutation == 'dtype': changed[0] = TensorABI('x', abi[0].shape, 'fp32', 'input')
            else: changed[0] = TensorABI('x', (262, 3072), 'bf16', 'input')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate_up.source_for_workload(SimpleNamespace(target='xcore1002', tensor_abi=lambda case: changed), 'primary')
        for batch, sequence in ((0, 1), (1, True), (-1, 3)):
            with self.assertRaises(ValueError): gate_up.program_for(batch, sequence)


if __name__ == '__main__':
    unittest.main()
