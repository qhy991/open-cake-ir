"""RoPE source and arithmetic contracts; no GPU or target qualification."""
import ast
from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import struct
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Target, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import DType, Schedule
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.tasks.c550_bench.binding import BenchProblem
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from tests.contracts.test_c550_bench_binding import ORACLE_NUMERICS
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _Pointer

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('rope_starter', Path(__file__).with_name('rope.py'))
rope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rope)


def fp32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def rne10(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return fp32((bits + 0xfff + ((bits >> 13) & 1)) & 0xffffe000)


def scaled_half(value):
    return rounded(rounded(rounded(value * 32., 'fp32'), 'fp16') / 32., 'fp32')


class FrequencyDomain(unittest.TestCase):
    def test_normal_domain_preserves_every_retained_significand_and_halfway_parity(self):
        checked = 0
        for exponent in range(-19, 11):
            for retained in range(1024):
                for tail in (0, 4095, 4096, 4097, 8191):
                    value = fp32(((exponent + 127) << 23) | (retained << 13) | tail)
                    if value <= 2047.:
                        self.assertEqual(scaled_half(value), rne10(value), (exponent, retained, tail))
                        checked += 1
        self.assertEqual(checked, 153596)
        self.assertEqual(scaled_half(0.), 0.)

    def test_subnormal_overflow_and_nonexact_positions_are_outside_the_proof(self):
        value = 2.**-20 + 2.**-30
        self.assertEqual(rne10(value), value)
        self.assertEqual(scaled_half(value), 2.**-20)
        self.assertNotEqual(scaled_half(value), rne10(value))
        self.assertEqual(rne10(2047.5), 2048.)
        with self.assertRaises(OverflowError):
            scaled_half(2047.5)
        self.assertEqual(rne10(2049.), 2048.)
        self.assertNotEqual(2049. * scaled_half(.125), rne10(2049.) * rne10(.125))


def original_problem(sequence=131):
    """Small original-authority fixture, not a copy of private Bench material."""
    tensor = lambda dtype: SimpleNamespace(dtype=SimpleNamespace(value=dtype))
    reference = '''def get_inputs(axes_and_scalars, device):
    seq_len = axes_and_scalars['seq_len']
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    inv_freq = torch.full((64,), 0.125, device=device)
    return {'position_ids': positions, 'inv_freq': inv_freq, 'attention_scaling': 1.0}

def run(position_ids, inv_freq, attention_scaling):
    return position_ids, inv_freq, attention_scaling
'''
    definition = SimpleNamespace(reference=reference, custom_inputs_entrypoint='get_inputs',
        inputs={name: tensor(dtype) for name, dtype in [('position_ids', 'int64'),
            ('inv_freq', 'float32'), ('attention_scaling', 'float32')]},
        outputs={'cos_sin': tensor('bfloat16')},
        get_input_shapes=lambda axes: {'position_ids': (2, axes['seq_len']), 'inv_freq': (64,), 'attention_scaling': None},
        get_output_shapes=lambda axes: {'cos_sin': (2, axes['seq_len'], 128, 2)})
    tolerance = SimpleNamespace(model_dump=lambda: {'max_atol': 1e-5, 'max_rtol': .05,
        'required_matched_ratio': .99, 'max_error_cap': None})
    selected = SimpleNamespace(uuid='original-fixture', axes={'batch_size': 2, 'seq_len': sequence}, tolerance=tolerance)
    raw = {'uuid': selected.uuid, 'inputs': {name: {'type': 'custom'} for name in definition.inputs}, 'tolerance': {}}
    return BenchProblem(Path('/fixture/original-bench'), rope.TASK,
        SimpleNamespace(document=lambda name: {'seed': 200}), definition, (selected,), (raw,),
        Path('/fixture/original-bench/rope'))


class ScalarBinding(unittest.TestCase):
    def test_factory_scalar_is_required_as_a_checked_specialization(self):
        original = original_problem()
        document = original.workload_document('original-fixture', oracle_numerics=ORACLE_NUMERICS)
        with patch.object(BenchProblem, 'open', return_value=original) as opening:
            self.assertEqual(rope.source_for_workload(BenchWorkload(document), 'primary'), rope.source_for_rne10(2, 131))
            opening.assert_called_once_with(original.root, rope.TASK)
        for scalars in ({}, {'attention_scaling': {'dtype': 'float32', 'value': True, 'binding': 'literal_input'}},
                        {'attention_scaling': {'dtype': 'float32', 'value': float('nan'), 'binding': 'literal_input'}},
                        {'attention_scaling': {'dtype': 'float32', 'value': 1., 'binding': 'unchecked'}}):
            changed = deepcopy(document); changed['semantics']['fixed_scalar_inputs'] = scalars
            with self.assertRaises(ValueError): rope.source_for_workload(BenchWorkload(changed), 'primary')

    def test_wrong_original_identity_policy_and_position_domain_are_refused(self):
        original = original_problem()
        document = original.workload_document('original-fixture', oracle_numerics=ORACLE_NUMERICS)
        mutations = [
            ('fixed original Bench', lambda d: d['semantics']['benchmark'].update(task='other')),
            ('fixed original Bench', lambda d: d['semantics']['benchmark'].update(commit='other')),
            ('HIGH numerical', lambda d: d['semantics']['oracle_numerics'].update(float32_matmul_precision='highest')),
            ('HIGH numerical', lambda d: d['semantics']['oracle_numerics'].update(allow_tf32=False)),
            ('HIGH numerical', lambda d: d['semantics']['oracle_numerics']['initialization'].update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=None)),
            ('known original input factory', lambda d: d['oracle'].update(custom_inputs_entrypoint='other')),
            ('known original input factory', lambda d: d['semantics']['original_input_specifications']['inv_freq'].update(type='random')),
        ]
        for message, mutate in mutations:
            changed = deepcopy(document); mutate(changed)
            with self.subTest(message=message), patch.object(BenchProblem, 'open') as opening:
                with self.assertRaisesRegex(ValueError, message):
                    rope.source_for_workload(BenchWorkload(changed), 'primary')
                opening.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'primary case'):
            rope.source_for_workload(BenchWorkload(document), 'unseen')
        too_long = original_problem(2050).workload_document('original-fixture', oracle_numerics=ORACLE_NUMERICS)
        with self.assertRaisesRegex(ValueError, 'positions in'):
            rope.source_for_workload(BenchWorkload(too_long), 'primary')
        with self.assertRaisesRegex(ValueError, 'positions in'):
            rope.source_for_rne10(1, 2050)

    def test_original_document_owner_rejects_factory_value_domain_changes(self):
        original = original_problem()
        document = original.workload_document('original-fixture', oracle_numerics=ORACLE_NUMERICS)
        for transform in (
            lambda text: text.replace('positions = torch.arange(seq_len, dtype=torch.int64, device=device)',
                                     'positions = torch.arange(seq_len, dtype=torch.int64, device=device) + 2049'),
            lambda text: text.replace('0.125', '0.0000001'),
        ):
            changed = deepcopy(document)
            changed['oracle']['reference_source'] = transform(document['oracle']['reference_source'])
            with patch.object(BenchProblem, 'open', return_value=original):
                with self.assertRaisesRegex(ValueError, 'original ABI, scalar, oracle or tolerance'):
                    rope.source_for_workload(BenchWorkload(changed), 'primary')


@unittest.skipUnless('int64' in {dtype.value for dtype in DType}, 'requires the INT64 storage successor')
class RoPEArithmetic(unittest.TestCase):
    def emission(self, batch, sequence, scale=1.0, *, rne10=False):
        source = (rope.source_for_rne10(batch, sequence) if rne10 else
                  rope.source_for(batch, sequence, attention_scaling=scale))
        schedule = Schedule.from_dict(frontend.parse(source).document)
        real = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        findings = (*verify(schedule, real), *preflight(schedule, real))
        self.assertFalse([item for item in findings if item.blocks_lowering], findings)
        return emit(schedule, real)

    def execute(self, emitted, positions, frequencies):
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
            stack.enter_context(patch.object(_TL, 'float16', 'fp16', create=True))
            exec(compile(ast.Module(body=[kernel], type_ignores=[]), '<emitted RoPE CPU contract>', 'exec'), env)
            for indices in itertools.product(*(range(n) for n in emitted.toolchain['grid'])):
                tl.program = indices
                env[kernel.name](**{name: _Pointer(values) for name, values in memory.items()},
                                 **emitted.toolchain['compile_constants'])
        self.assertEqual(memory['position_ids'], positions)
        self.assertEqual(memory['inv_freq'], frequencies)
        self.assertEqual(set(tl.stores.values()), {1})
        return memory['cos_sin']

    def test_rank_int64_cast_duplicate_frequency_and_interleaved_output(self):
        positions = [0, 1, 16777217, 2147483657]
        frequencies = [rounded(1. / (index + 1), 'fp32') for index in range(64)]
        scale = 1.25
        emitted = self.emission(1, len(positions), scale)
        self.assertEqual(emitted.toolchain['signature'],
                         {'position_ids': '*i64', 'inv_freq': '*fp32', 'cos_sin': '*bf16'})
        self.assertEqual(emitted.toolchain['grid'], [1, len(positions), 128])
        observed = self.execute(emitted, positions, frequencies)
        expected = []
        for position in positions:
            for index in range(128):
                angle = rounded(rounded(position, 'fp32') * frequencies[index % 64], 'fp32')
                for function in (math.cos, math.sin):
                    expected.append(rounded(rounded(rounded(function(angle), 'fp32') * scale, 'fp32'), 'bf16'))
        self.assertEqual(observed, expected)

    def test_bounded_casts_reproduce_rne10_angles_and_preserve_bf16_output_seams(self):
        positions = [0, 1, 1023, 1024, 2046, 2047]
        # Halfway values distinguish even/odd ties; other values include the
        # original frequency extrema and the conservative proof boundaries.
        frequencies = [fp32(0x3b801000), fp32(0x3b803000), 2.**-19,
                       2047., rounded(2.4551407022954663e-6, 'fp32'), .125, .0031415927, .031234567] * 8
        frequencies = [rounded(f, 'fp32') for f in frequencies]
        emitted = self.emission(1, len(positions), rne10=True)
        self.assertIn('scaled_frequency.to(tl.float16)', emitted.source)
        self.assertIn('frequency_half.to(tl.float32)', emitted.source)
        self.assertNotIn('tl.dot', emitted.source)
        actual = self.execute(emitted, positions, frequencies)
        expected = []
        for position in positions:
            for index in range(128):
                angle = rounded(position * rne10(frequencies[index % 64]), 'fp32')
                for function in (math.cos, math.sin):
                    expected.append(rounded(rounded(function(angle), 'fp32'), 'bf16'))
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, self.execute(self.emission(1, len(positions)), positions, frequencies))

    def test_sequence_tails_keep_original_axes(self):
        for batch, sequence in ((1, 613), (16, 919), (64, 541)):
            self.assertEqual(self.emission(batch, sequence).toolchain['grid'], [batch, sequence, 128])


if __name__ == '__main__':
    unittest.main()
