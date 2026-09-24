"""Ranked queues name ownership and capacity without admitting a GPU backend."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program, RankedMailboxEffects, Schedule


ROOT = Path(__file__).resolve().parents[2]
EFFECTS = ROOT / 'examples/programs/weave-ranked-mailbox-effects-b300.json'
LOCAL = ROOT / 'examples/programs/weave-local-expert-ffn-b300.json'


def combine(tokens):
    return Schedule.from_dict(json.loads((ROOT / f'examples/schedules/triton/'
                                          f'weave-weighted-combine-t{tokens}-h16.json').read_text()))


class RankedMailboxEffectsContract(unittest.TestCase):
    def test_exact_queues_derive_skew_and_tail_capacity_from_math(self):
        document = json.loads(EFFECTS.read_text())
        effects = RankedMailboxEffects.from_dict(document)
        self.assertEqual(effects.document, document)
        local = Program.from_dict(json.loads(LOCAL.read_text()))
        compiler = Compiler.load(ROOT)
        for tokens, payloads, tasks, returns in ((7, 21, 56, 14),
                                                  (8, 24, 64, 16)):
            with self.subTest(tokens=tokens):
                analysis = compiler.assess_ranked_mailbox(effects, local,
                                                           combine(tokens))
                self.assertEqual((analysis.world_size, analysis.items_per_rank,
                                  analysis.routes_per_item, analysis.feature_width),
                                 (4, tokens, 2, 16))
                self.assertEqual((analysis.remote_payload_capacity_per_rank,
                                  analysis.compute_task_capacity_per_rank,
                                  analysis.return_slots_per_rank),
                                 (payloads, tasks, returns))
                with self.assertRaisesRegex(ValueError,
                                            'dedicated backend lowering'):
                    compiler.lower_ranked_mailbox(effects, local, combine(tokens))
        self.assertEqual(len(compiler.lower_program(local).lowerings), 3)

    def test_wrong_owner_key_scope_reset_and_math_refuse(self):
        base = json.loads(EFFECTS.read_text())
        changes = (
            (lambda d: d['channels']['payload'].__setitem__('owner', 'source_rank'),
             'keys, owners'),
            (lambda d: d['channels']['task']['key'].__setitem__(2, 'destination_rank'),
             'keys, owners'),
            (lambda d: d['channels']['return'].__setitem__('ready',
                                                             'release_acquire_device'),
             'keys, owners'),
            (lambda d: d.__setitem__('reset', 'optional'), 'reset'),
            (lambda d: d.__setitem__('world_size', 1), 'at least two'),
            (lambda d: d.__setitem__('schema_version', True), 'fields or version'),
        )
        for mutate, reason in changes:
            value = deepcopy(base)
            mutate(value)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                RankedMailboxEffects.from_dict(value)
        local = Program.from_dict(json.loads(LOCAL.read_text()))
        wrong = json.loads((ROOT / 'examples/schedules/triton/'
                            'weave-weighted-combine-t7-h16.json').read_text())
        next(buffer for buffer in wrong['buffers']
             if buffer['name'] == 'output')['shape'][1] = 32
        effects = RankedMailboxEffects.from_dict(base)
        with self.assertRaisesRegex(ValueError, 'combine shape'):
            effects.analyze(local, Schedule.from_dict(wrong))


if __name__ == '__main__':
    unittest.main()
