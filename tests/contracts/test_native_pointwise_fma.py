"""The native PTX FMA leaf is eligible only for its owned row mapping."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda import emit, preflight
from open_cake_ir.compiler.backends.common import EmitError


ROOT = Path(__file__).resolve().parents[2]


def document(*, persistent: bool = False) -> dict:
    value = json.loads((ROOT / 'corpus/schedules/fma-b8-smoke.json').read_text())
    value['schedule_id'] = 'native-fma-row-b8'
    value['target'] = 'sm_103a'
    value['lowering'] = {'backend': 'native_cuda', 'entry_point': 'cake_native_fma_row'}
    value['roles'][0]['execution_groups'] = [0]
    for buffer in value['buffers']:
        buffer['shape'][-1] = 32
    for op in value['operations']:
        if op['kind'] == 'load':
            op['parameters'].pop('reuse')
    value['program_map']['persistent'] = persistent
    if persistent:
        value['program_map']['cooperative'] = True
        value['residency'] = {'ctas_per_multiprocessor': 1}
    else:
        value.pop('residency')
    return value


class NativePointwiseFma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def codes(self, value):
        return [f.code for f in self.compiler.assess(value).findings]

    def test_explicit_ptx_fma_and_host_abi(self):
        assessment = self.compiler.assess(document())
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowering = self.compiler.lower(assessment)
        self.assertIn('fma.rn.f32 %0, %1, %2, %3;', lowering.source)
        self.assertIn('cake_work * 32 + cake_lane', lowering.source)
        self.assertEqual(lowering.toolchain_requirements['grid'], [8, 1, 1])
        self.assertEqual(lowering.toolchain_requirements['block'], [32, 1, 1])
        self.assertEqual(set(lowering.source_map),
                         {op['id'] for op in document()['operations']})
        self.assertNotIn('tcgen05.mma', lowering.source)

    def test_cooperative_persistent_path_and_exact_target(self):
        value = document(persistent=True)
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        source = self.compiler.lower(assessment).source
        self.assertIn('for (int cake_work=int(blockIdx.x); cake_work<8; cake_work+=8)', source)
        self.assertIn('cudaLaunchCooperativeKernel', source)
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor', source)
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        no_fma = replace(target, instruction_contracts=target.instruction_contracts
                         - {'ptx.fma.rn.f32'})
        schedule = Schedule.from_dict(value)
        self.assertIn('NATIVE_POINTWISE_FMA_CONTRACT',
                      [f.code for f in preflight(schedule, no_fma)])
        with self.assertRaisesRegex(EmitError, 'TARGET_INSTRUCTION_UNSUPPORTED'):
            emit(schedule, no_fma)

    def test_cache_mapping_and_body_counterexamples(self):
        value = document()
        value['operations'][0]['parameters']['reuse'] = 'streamed'
        self.assertIn('NATIVE_POINTWISE_LOAD', self.codes(value))
        value = document()
        value['access_maps'][0]['indices'][1]['dimension'] = 0
        self.assertIn('ACCESS_TILE_MISMATCH', self.codes(value))
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        self.assertIn('NATIVE_POINTWISE_ACCESS', [f.code for f in preflight(
            Schedule.from_dict(value), target)])
        value = document()
        value['operations'].pop()
        self.assertIn('OUTPUT_UNWRITTEN', self.codes(value))
        self.assertIn('NATIVE_POINTWISE_BODY', [f.code for f in preflight(
            Schedule.from_dict(value), target)])
        value = document()
        value['roles'][0]['execution_groups'] = [0, 1]
        self.assertIn('NATIVE_POINTWISE_ROLE', self.codes(value))


if __name__ == '__main__':
    unittest.main()
