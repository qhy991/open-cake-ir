"""Ranked mailbox lowering binds visible Cake math to exact B300 effects."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler import (Compiler, LoweredRankedMailbox, Program,
                                   RankedMailboxEffects, Schedule, Target)
from open_cake_ir.compiler.backends.native_cuda_ranked_mailbox import _admit


ROOT = Path(__file__).resolve().parents[2]
EFFECTS = ROOT / 'examples/programs/weave-ranked-mailbox-effects-b300.json'
LOCAL = ROOT / 'examples/programs/weave-local-expert-ffn-native-b300.json'


def material(tokens):
    effects = RankedMailboxEffects.from_dict(json.loads(EFFECTS.read_text()))
    local = Program.from_dict(json.loads(LOCAL.read_text()))
    combine = Schedule.from_dict(json.loads((ROOT / f'examples/schedules/native/'
                                              f'weave-weighted-combine-t{tokens}-h16.json').read_text()))
    return effects, local, combine


class NativeRankedMailbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_t7_and_t8_complete_source_bind_math_effects_and_target(self):
        for tokens, payloads, tasks in ((7, 21, 56), (8, 24, 64)):
            with self.subTest(tokens=tokens):
                effects, local, combine = material(tokens)
                lowered = self.compiler.lower_ranked_mailbox(effects, local, combine)
                self.assertIsInstance(lowered, LoweredRankedMailbox)
                lowered.validate_binding()
                self.assertEqual((lowered.analysis.remote_payload_capacity_per_rank,
                                  lowered.analysis.compute_task_capacity_per_rank),
                                 (payloads, tasks))
                requirements = lowered.toolchain_requirements
                self.assertEqual(requirements['world_size'], 4)
                self.assertEqual(requirements['tokens_per_rank'], tokens)
                self.assertEqual(requirements['payload_capacity'], payloads)
                self.assertEqual(requirements['task_capacity'], tasks)
                self.assertEqual(requirements['grid_per_rank'], [[148, 1, 1]] * 4)
                self.assertTrue(requirements['cooperative_grid'])
                self.assertTrue(requirements['peer_pair_runtime_check'])
                self.assertTrue(requirements['input_domain_runtime_check'])
                self.assertEqual(requirements['activated_shared_bytes'], 128)
                self.assertIn('atom.acq_rel.sys.global.add.s32', lowered.source)
                self.assertIn('atom.release.sys.global.add.s32', lowered.source)
                self.assertIn('cudaDevP2PAttrNativeAtomicSupported', lowered.source)
                self.assertIn('cudaPointerGetAttributes', lowered.source)
                self.assertIn('host_ids[T * K]', lowered.source)
                self.assertIn('host_weights[T * K]', lowered.source)
                self.assertNotIn('fmaf(', lowered.source)
                self.assertNotIn('@ENTRY@', lowered.source)
                math = {f'{stage.name}.{op.op_id}' for stage in local.stages
                        for op in stage.schedule.operations}
                math.update(f'combine.{op.op_id}' for op in combine.operations)
                self.assertEqual(len(math), 29)
                self.assertTrue(math <= set(lowered.source_map))
                self.assertTrue({'effect.payload.reserve', 'effect.task.claim',
                                 'effect.return.acquire', 'effect.peer_pair.admit'}
                                <= set(lowered.source_map))

    def test_typed_refusals_precede_generated_code(self):
        effects, local, combine = material(7)
        wrong = effects.document
        wrong['lowering']['backend'] = 'triton'
        with self.assertRaisesRegex(ValueError, 'dedicated backend lowering'):
            self.compiler.lower_ranked_mailbox(
                RankedMailboxEffects.from_dict(wrong), local, combine)
        changed = json.loads(LOCAL.read_text())
        next(op for op in changed['stages'][1]['schedule']['operations']
             if op['id'] == 'exp_gate')['parameters']['op'] = 'rsqrt'
        with self.assertRaisesRegex(ValueError, 'NATIVE_ACTIVATION_ARITHMETIC'):
            self.compiler.lower_ranked_mailbox(effects, Program.from_dict(changed),
                                               combine)
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        missing = replace(target, instruction_contracts=target.instruction_contracts
                          - {'ptx.atom.release.sys.global.add.s32'})
        fake = SimpleNamespace(_revision=SimpleNamespace(targets={local.target: missing}))
        with self.assertRaisesRegex(ValueError, 'PTX system contracts'):
            _admit(fake, effects, local, combine, effects.analyze(local, combine))


if __name__ == '__main__':
    unittest.main()
