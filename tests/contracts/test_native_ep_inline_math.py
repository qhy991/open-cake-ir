"""Only an explicit worker rewrite may inline complete Cake expert math."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from open_cake_ir.compiler import Program, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_ep_math import lower_ep_math


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / 'examples/programs/weave-local-expert-ffn-native-b300.json'


def material(tokens=7):
    program = Program.from_dict(json.loads(PROGRAM.read_text()))
    combine = Schedule.from_dict(json.loads((ROOT / f'examples/schedules/native/'
                            f'weave-weighted-combine-t{tokens}-h16.json').read_text()))
    target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
    return program, combine, target


class NativeEpInlineMath(unittest.TestCase):
    def test_explicit_rewrite_maps_every_leaf_operation(self):
        for tokens in (7, 8):
            with self.subTest(tokens=tokens):
                program, combine, target = material(tokens)
                lowered = lower_ep_math(program, combine, target,
                                        entry='cake_ep4', rewrite='ranked_mailbox')
                expected = {f'{stage.name}.{op.op_id}' for stage in program.stages
                            for op in stage.schedule.operations}
                expected.update(f'combine.{op.op_id}' for op in combine.operations)
                self.assertEqual(set(lowered.source_map), expected)
                self.assertEqual(len(expected), 29)
                self.assertEqual(lowered.tokens, tokens)
                self.assertEqual((lowered.hidden, lowered.intermediate,
                                  lowered.local_experts, lowered.activated_shared_bytes),
                                 (16, 32, 2, 128))
                self.assertIn('cake_ep4_expert(', lowered.source)
                self.assertIn('cake_ep4_combine(', lowered.source)
                self.assertIn('__syncthreads();', lowered.source)
                self.assertIn('__float2bfloat16_rn', lowered.source)
                self.assertNotIn('fmaf(', lowered.source)
                self.assertNotIn('tl.', lowered.source)

    def test_math_drift_and_implicit_fusion_refuse(self):
        program, combine, target = material()
        with self.assertRaisesRegex(ValueError, 'ranked inline expert math refused'):
            lower_ep_math(program, combine, target,
                          entry='cake_ep4', rewrite='ordered_program')
        changed = json.loads(PROGRAM.read_text())
        next(op for op in changed['stages'][1]['schedule']['operations']
             if op['id'] == 'exp_gate')['parameters']['op'] = 'tanh'
        with self.assertRaisesRegex(ValueError, 'NATIVE_ACTIVATION_ARITHMETIC'):
            lower_ep_math(Program.from_dict(changed), combine, target,
                          entry='cake_ep4', rewrite='ranked_mailbox')
        changed = json.loads((ROOT / 'examples/schedules/native/'
                              'weave-weighted-combine-t7-h16.json').read_text())
        next(op for op in changed['operations']
             if op['id'] == 'reduce_routes')['parameters']['axis'] = 1
        with self.assertRaisesRegex(ValueError, 'REDUCE_SHAPE_MISMATCH|NATIVE_COMBINE_REDUCE'):
            lower_ep_math(program, Schedule.from_dict(changed), target,
                          entry='cake_ep4', rewrite='ranked_mailbox')

    def test_generated_device_helpers_have_host_cpp_syntax(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('host C++ compiler unavailable')
        program, combine, target = material()
        source = lower_ep_math(program, combine, target,
                               entry='cake_ep4', rewrite='ranked_mailbox').source
        source = source.replace('#include <cuda_bf16.h>', '''#include <cstdint>
#define __device__
#define __forceinline__ inline
using __nv_bfloat16 = float;
inline float __bfloat162float(float value) { return value; }
inline float __float2bfloat16_rn(float value) { return value; }
inline void __syncthreads() {}
''')
        with tempfile.TemporaryDirectory(prefix='cake-ep-inline-syntax-') as directory:
            path = Path(directory) / 'math.cpp'
            path.write_text(source)
            subprocess.run([compiler, '-std=c++17', '-fsyntax-only', str(path)],
                           check=True, capture_output=True, text=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
