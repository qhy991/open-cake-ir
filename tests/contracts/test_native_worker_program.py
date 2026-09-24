"""Native worker lowering keeps leaf mathematics and scheduling effects distinct."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.program import LoweredWorkerProgram
from tests.contracts.test_worker_execution import rank_document


ROOT = Path(__file__).resolve().parents[2]
LEAF = ROOT / 'examples/schedules/native/fma-row-b512.json'


def document() -> dict:
    leaf = json.loads(LEAF.read_text())
    shape = [512, 16]
    tensors = {name: {'shape': shape, 'dtype': 'fp32'}
               for name in ('a', 'b', 'c', 'middle0', 'middle1', 'out')}
    tensors.update({name: {'shape': [1], 'dtype': 'int32'}
                    for name in ('first_ctas', 'chunks', 'steal_budget')})
    stages = [{'name': name, 'schedule': deepcopy(leaf),
               'bindings': {'a': source, 'b': 'b', 'c': 'c', 'y': destination}}
              for name, source, destination in (
                  ('send', 'a', 'middle0'), ('calculate', 'middle0', 'middle1'),
                  ('finish', 'middle1', 'out'))]
    return {
        'schema_version': 2, 'program_id': 'native-fma-worker-b512',
        'target': 'sm_103a', 'tensors': tensors,
        'inputs': ['a', 'b', 'c', 'first_ctas', 'chunks', 'steal_budget'],
        'outputs': ['out'], 'stages': stages,
        'execution': {
            'kind': 'cooperative_workers',
            'lowering': {'backend': 'native_cuda', 'entry_point': 'cake_fma_workers'},
            'controls': {'first_class_ctas': 'first_ctas', 'chunk_count': 'chunks',
                         'steal_budget': 'steal_budget'},
            'workers': [
                {'name': 'first', 'phases': ['send', 'finish']},
                {'name': 'second', 'phases': ['calculate', 'finish']},
            ],
            'queues': [
                {'stage': 'send', 'workers': ['first']},
                {'stage': 'calculate', 'workers': ['first', 'second']},
                {'stage': 'finish', 'workers': ['first', 'second']},
            ],
            'handoffs': [
                {'payload': 'middle0', 'producer': 'send', 'consumer': 'calculate',
                 'order': 'release_acquire', 'scope': 'device'},
                {'payload': 'middle1', 'producer': 'calculate', 'consumer': 'finish',
                 'order': 'release_acquire', 'scope': 'device'},
            ],
            'steal': {'borrower': 'first', 'stage': 'calculate',
                      'after': 'send', 'before': 'finish'},
        },
    }


def rank_native_document() -> dict:
    value = document()
    value['schema_version'] = 3
    value['execution']['workers'][1]['phases'] = ['calculate']
    value['execution']['queues'][2]['workers'] = ['first']
    for handoff in value['execution']['handoffs']:
        handoff['scope'] = 'system'
    value['execution']['placement'] = {
        'world_size': 2, 'state_rank': 0,
        'worker_ranks': {'first': 0, 'second': 1},
        'tensor_ranks': {name: (1 if name == 'middle0' else 0)
                         for name in value['tensors']},
    }
    return value


class NativeWorkerProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_complete_fma_program_emits_one_cooperative_kernel(self):
        lowered = self.compiler.lower_program(Program.from_dict(document()))
        self.assertIsInstance(lowered, LoweredWorkerProgram)
        lowered.validate_binding()
        self.assertEqual(lowered.toolchain_requirements['grid'], [148, 1, 1])
        self.assertEqual(lowered.toolchain_requirements['state_bytes'], 6184)
        self.assertEqual(lowered.toolchain_requirements['state_stolen_tiles_offset_bytes'], 32)
        self.assertEqual(lowered.toolchain_requirements[
            'state_first_combine_compute_done_plus_one_offset_bytes'], 36)
        self.assertEqual(lowered.toolchain_requirements['argument_order'],
                         list(document()['tensors']))
        source = lowered.source
        self.assertEqual(source.count('{'), source.count('}'))
        self.assertEqual(source.count('fma.rn.f32 %0, %1, %2, %3;'), 3)
        self.assertIn('atom.relaxed.gpu.global.add.s32', source)
        self.assertIn('st.release.gpu.global.s32', source)
        self.assertIn('ld.acquire.gpu.global.s32', source)
        self.assertIn('cudaMemsetAsync(h->state, 0, 6184', source)
        self.assertIn('cake_claim(state + 8);', source)
        self.assertIn('atomicCAS(state + 9, 0, cake_relaxed(state + 4) + 1)', source)
        self.assertIn('cudaLaunchCooperativeKernel', source)
        self.assertNotIn('cudaPointerGetAttributes', source)
        self.assertNotIn('st.release.sys.global.s32', source)
        self.assertIn('// CAKE_OP: send.fma', source)
        self.assertIn('// CAKE_OP: calculate.fma', source)
        self.assertIn('// CAKE_OP: finish.fma', source)
        self.assertNotIn('triton', source.lower())

    def test_system_handoff_publishes_a_peer_payload_with_runtime_admission(self):
        value = document()
        value['execution']['handoffs'][0]['scope'] = 'system'
        lowered = self.compiler.lower_program(Program.from_dict(value))
        self.assertTrue(lowered.toolchain_requirements['peer_payload_runtime_check'])
        self.assertIn('cake_publish_system(state + 10 + tile)', lowered.source)
        self.assertIn('cake_acquire_system(state + 10 + tile)', lowered.source)
        self.assertIn('st.release.sys.global.s32', lowered.source)
        self.assertIn('ld.acquire.sys.global.s32', lowered.source)
        self.assertIn('cudaPointerGetAttributes', lowered.source)
        self.assertIn('cudaDevP2PAttrNativeAtomicSupported', lowered.source)
        self.assertIn('cudaDeviceEnablePeerAccess', lowered.source)
        self.assertIn('state_attrs.device != device', lowered.source)
        value = document()
        value['execution']['handoffs'][1]['scope'] = 'system'
        second = self.compiler.lower_program(Program.from_dict(value)).source
        self.assertIn('cake_publish_system(state + 522 + tile)', second)
        self.assertIn('cake_acquire_system(state + 522 + tile)', second)
        self.assertIn('cake_publish(state + 10 + tile)', second)

    def test_two_rank_source_owns_shared_system_queue_and_exact_peer_checks(self):
        lowered = self.compiler.lower_program(Program.from_dict(rank_native_document()))
        requirements = lowered.toolchain_requirements
        self.assertEqual(requirements['world_size'], 2)
        self.assertEqual(requirements['state_rank'], 0)
        self.assertEqual(requirements['tensor_ranks']['middle0'], 1)
        self.assertEqual(requirements['grid_per_rank'], [[148, 1, 1], [148, 1, 1]])
        self.assertTrue(requirements['peer_pair_runtime_check'])
        self.assertEqual(requirements['state_bytes'], 6184)
        self.assertEqual(set(requirements['host_abi']),
                         {'create_rank', 'launch_two', 'destroy_rank'})
        source = lowered.source
        self.assertEqual(source.count('{'), source.count('}'))
        self.assertEqual(source.count('fma.rn.f32 %0, %1, %2, %3;'), 3)
        self.assertIn('atom.relaxed.sys.global.add.s32', source)
        self.assertIn('st.release.sys.global.s32', source)
        self.assertIn('ld.acquire.sys.global.s32', source)
        self.assertIn('cudaPointerGetAttributes', source)
        self.assertIn('cudaDevP2PAttrNativeAtomicSupported', source)
        self.assertIn('cudaStreamSynchronize', source)
        self.assertEqual(source.count('cudaLaunchCooperativeKernel'), 2)
        self.assertIn('rank == 0 && int(blockIdx.x) < c', source)
        self.assertIn('rank == 1 && int(blockIdx.x) >= c', source)
        for stage in ('send', 'calculate', 'finish'):
            self.assertIn(f'// CAKE_OP: {stage}.fma', source)

    def test_refuses_unproved_routes_and_handoffs(self):
        with self.assertRaisesRegex(ValueError, 'exact Target needs'):
            self.compiler.lower_program(Program.from_dict(rank_document()))
        value = rank_native_document()
        value['execution']['placement']['tensor_ranks']['middle0'] = 0
        with self.assertRaisesRegex(ValueError, 'first intermediate on rank 1'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['stages'][1]['schedule']['lowering']['backend'] = 'triton'
        with self.assertRaisesRegex(ValueError, 'stage .*native_cuda leaf route'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['stages'][1]['schedule']['program_map']['cooperative'] = False
        with self.assertRaisesRegex(ValueError, 'cooperative one-CTA-per-SM'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['execution']['lowering']['entry_point'] = 'bad();'
        with self.assertRaisesRegex(ValueError, 'entry_point must be an identifier'):
            self.compiler.lower_program(Program.from_dict(value))


if __name__ == '__main__':
    unittest.main()
