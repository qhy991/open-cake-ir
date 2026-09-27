"""Exact no-FTZ packed FP32 state mapping and scalar fallback controls."""
from __future__ import annotations

from copy import deepcopy
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from tests.contracts.test_native_kda_prepared_beta_bridge import document
from tests.contracts.test_native_two_phase_carried import target


class NativeKdaF32x2StateTest(unittest.TestCase):
    def test_exact_row_owned_pair_emits_two_non_ftz_instructions(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        pair = native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('scale_state'))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.op_id, 'combine_state')
        emission = native_cuda.emit(schedule, target())
        self.assertEqual(emission.source.count('CAKE_NATIVE_FP32X2_STATE_NO_FTZ'), 2)
        self.assertIn('mul.rn.f32x2', emission.source)
        self.assertIn('add.rn.f32x2', emission.source)
        self.assertNotIn('mul.rn.ftz.f32x2', emission.source)
        self.assertNotIn('add.rn.ftz.f32x2', emission.source)
        self.assertEqual(emission.toolchain['grid'], [64, 1, 1])
        self.assertEqual(emission.toolchain['dynamic_shared_bytes'], 57472)

    def test_other_multiply_with_narrow_token_tile_stays_scalar(self):
        schedule = Schedule.from_dict(document())
        self.assertIsNone(native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('apply_beta')))

    def test_wrong_broadcast_does_not_claim_packed_mapping(self):
        value = deepcopy(document())
        next(op for op in value['operations'] if op['id'] == 'scale_state')[
            'parameters']['broadcast_axis'] = 0
        schedule = Schedule.from_dict(value)
        self.assertIsNone(native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('scale_state')))

    def test_non_add_consumer_does_not_claim_packed_mapping(self):
        value = deepcopy(document())
        next(op for op in value['operations'] if op['id'] == 'combine_state')[
            'parameters']['op'] = 'sub'
        schedule = Schedule.from_dict(value)
        self.assertIsNone(native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('scale_state')))

    def test_shared_product_reader_does_not_claim_packed_mapping(self):
        value = deepcopy(document())
        next(op for op in value['operations'] if op['id'] == 'scale_output')[
            'reads'].append('decayed_state')
        schedule = Schedule.from_dict(value)
        self.assertIsNone(native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('scale_state')))

    def test_non_carried_fp32_pair_does_not_claim_packed_mapping(self):
        value = deepcopy(document())
        value['tile_loops'][0].pop('carried_buffers')
        schedule = Schedule.from_dict(value)
        self.assertIsNone(native_cuda._packed_state_pair(
            schedule, target(), schedule.operation('scale_state')))


if __name__ == '__main__':
    unittest.main()
