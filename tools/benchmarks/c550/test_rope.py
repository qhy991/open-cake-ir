"""RoPE source and arithmetic contracts; no GPU or target qualification."""
import ast
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Target, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import DType, Schedule
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _Pointer

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('rope_starter', Path(__file__).with_name('rope.py'))
rope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rope)


class ScalarBinding(unittest.TestCase):
    def test_factory_scalar_is_required_as_a_checked_specialization(self):
        abi = (TensorABI('position_ids', (2, 131), 'int64', 'input'),
               TensorABI('inv_freq', (64,), 'fp32', 'input'),
               TensorABI('cos_sin', (2, 131, 128, 2), 'bf16', 'output'))
        semantics = {'fixed_scalar_inputs': {'attention_scaling': {'dtype': 'float32', 'value': 1.0}}}
        workload = SimpleNamespace(target='xcore1002', tensor_abi=lambda case: abi,
                                   document={'semantics': semantics})
        self.assertEqual(rope.source_for_workload(workload, 'primary'), rope.source_for(2, 131))
        for scalars in ({}, {'attention_scaling': {'dtype': 'float32', 'value': True}},
                        {'attention_scaling': {'dtype': 'float32', 'value': float('nan')}}):
            semantics['fixed_scalar_inputs'] = scalars
            with self.assertRaises(ValueError): rope.source_for_workload(workload, 'primary')


@unittest.skipUnless('int64' in {dtype.value for dtype in DType}, 'requires the INT64 storage successor')
class RoPEArithmetic(unittest.TestCase):
    def emission(self, batch, sequence, scale=1.0):
        schedule = Schedule.from_dict(frontend.parse(rope.source_for(batch, sequence,
                                                                   attention_scaling=scale)).document)
        real = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', {item.code for item in verify(schedule, real)})
        probe = replace(real, instruction_contracts=real.instruction_contracts | {'maca.sin.f32', 'maca.cos.f32'})
        findings = (*verify(schedule, probe), *preflight(schedule, probe))
        self.assertFalse([item for item in findings if item.blocks_lowering], findings)
        return emit(schedule, probe)

    def test_rank_int64_cast_duplicate_frequency_and_interleaved_output(self):
        positions = [0, 1, 16777217, 2147483657]
        frequencies = [rounded(1. / (index + 1), 'fp32') for index in range(64)]
        scale = 1.25
        emitted = self.emission(1, len(positions), scale)
        self.assertEqual(emitted.toolchain['signature'],
                         {'position_ids': '*i64', 'inv_freq': '*fp32', 'cos_sin': '*bf16'})
        self.assertEqual(emitted.toolchain['grid'], [1, len(positions), 128])
        tree = ast.parse(emitted.source)
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        kernel.decorator_list = []
        for arg in kernel.args.args: arg.annotation = None
        memory = {'position_ids': positions[:], 'inv_freq': frequencies[:],
                  'cos_sin': [float('nan')] * (len(positions) * 128 * 2)}
        tl = _TL()
        def cast(tile, dtype):
            return _Tile(tile.shape, [int(value) if dtype is _TL.int32 else
                         rounded(value, 'fp32' if dtype is _TL.float32 else dtype) for value in tile.values])
        def multiply(tile, other):
            return tile.binary(other, lambda x, y: rounded(x*y, 'fp32')
                               if isinstance(x, float) or isinstance(y, float) else x*y)
        library = SimpleNamespace(**{name: (lambda tile, op=getattr(math, name):
            _Tile(tile.shape, [rounded(op(value), 'fp32') for value in tile.values])) for name in ('sin', 'cos')})
        env = {'tl': tl, 'libdevice': library}
        with ExitStack() as stack:
            for name, method in [('to', cast), ('__mul__', multiply), ('__rmul__', multiply),
                ('__floordiv__', lambda a, b: a.binary(b, lambda x, y: x//y)),
                ('__mod__', lambda a, b: a.binary(b, lambda x, y: x%y)),
                ('__ge__', lambda a, b: a.binary(b, lambda x, y: x>=y))]:
                stack.enter_context(patch.object(_Tile, name, method, create=True))
            stack.enter_context(patch.object(_TL, 'bfloat16', 'bf16', create=True))
            exec(compile(ast.Module(body=[kernel], type_ignores=[]), '<emitted RoPE CPU contract>', 'exec'), env)
            for indices in itertools.product(*(range(n) for n in emitted.toolchain['grid'])):
                tl.program = indices
                env[kernel.name](**{name: _Pointer(values) for name, values in memory.items()},
                                 **emitted.toolchain['compile_constants'])
        expected = []
        for position in positions:
            for index in range(128):
                angle = rounded(rounded(position, 'fp32') * frequencies[index % 64], 'fp32')
                for function in (math.cos, math.sin):
                    expected.append(rounded(rounded(rounded(function(angle), 'fp32') * scale, 'fp32'), 'bf16'))
        self.assertEqual(memory['cos_sin'], expected)
        self.assertEqual(memory['position_ids'], positions)
        self.assertEqual(memory['inv_freq'], frequencies)
        self.assertEqual(set(tl.stores.values()), {1})

    def test_sequence_tails_keep_original_axes(self):
        for batch, sequence in ((1, 613), (16, 919), (64, 541)):
            self.assertEqual(self.emission(batch, sequence).toolchain['grid'], [batch, sequence, 128])


if __name__ == '__main__':
    unittest.main()
