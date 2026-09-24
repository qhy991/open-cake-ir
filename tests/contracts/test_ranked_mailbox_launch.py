"""Four rank Evaluation binding preserves owners, reset and one launch."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import (LoweredRankedMailbox, Program,
                                   RankedMailboxEffects, Schedule)
from open_cake_ir.compiler.ir import DType
from open_cake_ir.evaluation.ranked_launch import (
    RankedMailboxExecutable, prepare_ranked_mailbox,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Tensor:
    pointer: int
    nbytes: int
    device: int
    shape: tuple[int, ...]
    dtype: object


def fixture():
    effects = RankedMailboxEffects.from_dict(json.loads((ROOT / 'examples/programs/'
        'weave-ranked-mailbox-effects-b300.json').read_text()))
    local = Program.from_dict(json.loads((ROOT / 'examples/programs/'
        'weave-local-expert-ffn-b300.json').read_text()))
    combine = Schedule.from_dict(json.loads((ROOT / 'examples/schedules/triton/'
        'weave-weighted-combine-t7-h16.json').read_text()))
    analysis = effects.analyze(local, combine)
    names = [f'{stage.name}.{op.op_id}' for stage in local.stages
             for op in stage.schedule.operations]
    names.extend(f'combine.{op.op_id}' for op in combine.operations)
    source = '\n'.join(f'// CAKE_OP: {name}' for name in names) + '\n'
    specs = [
        {'name': 'hidden', 'shape': [7, 16], 'dtype': 'bf16'},
        {'name': 'expert_ids', 'shape': [7, 2], 'dtype': 'int32'},
        {'name': 'route_weights', 'shape': [7, 2], 'dtype': 'fp32'},
        {'name': 'w_up_gate', 'shape': [2, 64, 16], 'dtype': 'bf16'},
        {'name': 'w_down', 'shape': [2, 16, 32], 'dtype': 'bf16'},
    ]
    requirements = {
        'target': 'sm_103a', 'entry_point': effects.lowering.entry_point,
        'world_size': 4, 'tokens_per_rank': 7, 'routes_per_token': 2,
        'feature_width': 16, 'payload_capacity': 21,
        'task_capacity': 56, 'return_slots': 14,
        'rank_inputs': specs, 'grid_per_rank': [[148, 1, 1]] * 4,
        'block': [32, 1, 1], 'cooperative_grid': True,
        'peer_pair_runtime_check': True, 'input_domain_runtime_check': True,
        'state_reset': 'zero_all_rank_mailboxes_before_launch',
        'host_abi': {name: name for name in (
            'mailbox_bytes', 'status_offset', 'output_offset',
            'output_bytes', 'prepare_rank', 'launch')},
    }
    lowered = LoweredRankedMailbox(local, combine, effects, analysis, 'fixture',
        source, {name: (i + 1, i + 1) for i, name in enumerate(names)}, requirements)
    inputs = {}
    for rank in range(4):
        inputs[rank] = {}
        for index, row in enumerate(specs):
            shape = tuple(row['shape'])
            dtype = DType(row['dtype'])
            nbytes = dtype.itemsize
            for extent in shape:
                nbytes *= extent
            inputs[rank][row['name']] = Tensor(1000000 + rank * 1000000 + index * 8192,
                                                nbytes, rank, shape, dtype)
    plans = {rank: {'communication_ctas': (12, 36, 72, 120)[rank],
                    'chunks': (2, 3, 7, 1)[rank], 'steal_budget': 14 if rank == 0 else 0}
             for rank in range(4)}
    return lowered, inputs, plans


class RankedMailboxLaunch(unittest.TestCase):
    def callbacks(self, *, status=(0, 0, 0, 0), mailbox_bytes=8192,
                  output_delta=0, contexts=None):
        calls = {'launch': 0, 'reset': 0, 'status': 0, 'bind': 0}
        current = contexts if contexts is not None else ['stream'] * 4

        def check_tensor(tensor, spec):
            if (tensor.shape != spec.shape or tensor.dtype is not spec.dtype
                    or tensor.nbytes != spec.nbytes):
                raise ValueError('fake tensor shape/dtype differs')

        def span(tensor):
            return tensor.device, tensor.pointer, tensor.pointer + tensor.nbytes

        def context(rank):
            return (rank, current[rank])

        def load(lowered):
            lowered.validate_binding()

            def bind(inputs, mailboxes, plans, bound):
                calls['bind'] += 1
                self.assertEqual(set(inputs), set(mailboxes) | set(plans))
                self.assertEqual(len(bound), 4)

                def launch(contexts):
                    self.assertEqual(contexts, bound)
                    calls['launch'] += 1

                return launch

            return RankedMailboxExecutable(mailbox_bytes, 32, 4096, 224, bind)

        def allocate(rank, size):
            return Tensor(1000000 + rank * 1000000 + 100000, size,
                          rank, (size,), None)

        def view(mailbox, offset, spec):
            return Tensor(mailbox.pointer + offset + output_delta, spec.nbytes,
                          mailbox.device, spec.shape, spec.dtype)

        def reset(mailboxes, bound):
            self.assertEqual(len(mailboxes), len(bound))
            calls['reset'] += 1

        def read_status(mailboxes, offset, bound):
            self.assertEqual(offset, 32)
            calls['status'] += 1
            return status

        return calls, current, dict(load_source=load, allocate_mailbox=allocate,
            check_tensor=check_tensor, storage_span=span, execution_context=context,
            output_view=view, reset_mailboxes=reset, read_status=read_status)

    def test_one_reset_and_combined_launch_per_run(self):
        lowered, inputs, plans = fixture()
        calls, _, callbacks = self.callbacks()
        prepared = prepare_ranked_mailbox(lowered, inputs, plans, **callbacks)
        self.assertEqual(calls['bind'], 1)
        self.assertEqual(set(prepared.outputs), {0, 1, 2, 3})
        self.assertEqual(prepared.outputs[0].pointer,
                         prepared.mailboxes[0].pointer + 4096)
        prepared.run()
        prepared.run()
        self.assertEqual((calls['reset'], calls['launch'], calls['status'],
                          prepared.launch_calls), (2, 2, 2, 2))

    def test_owner_alias_controls_layout_and_status_refuse(self):
        lowered, inputs, plans = fixture()
        _, _, callbacks = self.callbacks()
        bad = {rank: dict(row) for rank, row in inputs.items()}
        bad[2]['hidden'] = Tensor(inputs[2]['hidden'].pointer,
                                  inputs[2]['hidden'].nbytes, 1,
                                  inputs[2]['hidden'].shape, DType.BF16)
        with self.assertRaisesRegex(ValueError, 'owner, extent or alias'):
            prepare_ranked_mailbox(lowered, bad, plans, **callbacks)
        bad = {rank: dict(row) for rank, row in inputs.items()}
        bad[0]['expert_ids'] = bad[0]['hidden']
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            prepare_ranked_mailbox(lowered, bad, plans, **callbacks)
        for field, value in (('communication_ctas', 0), ('chunks', 8),
                             ('steal_budget', 57)):
            wrong = {rank: dict(row) for rank, row in plans.items()}
            wrong[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,
                                                                    'exceed declared bounds'):
                prepare_ranked_mailbox(lowered, inputs, wrong, **callbacks)
        _, _, shifted = self.callbacks(output_delta=4)
        with self.assertRaisesRegex(ValueError, 'exact mailbox view'):
            prepare_ranked_mailbox(lowered, inputs, plans, **shifted)
        _, _, short = self.callbacks(mailbox_bytes=200)
        with self.assertRaisesRegex(ValueError, 'compiled mailbox size'):
            prepare_ranked_mailbox(lowered, inputs, plans, **short)
        wrong_requirements = dict(lowered.toolchain_requirements)
        wrong_requirements['grid_per_rank'] = [[148, 1, 1], [148, 1, 1],
                                               [147, 1, 1], [148, 1, 1]]
        with self.assertRaisesRegex(ValueError, 'source, target or mailbox ABI'):
            prepare_ranked_mailbox(replace(lowered,
                toolchain_requirements=wrong_requirements), inputs, plans, **callbacks)
        _, _, failure = self.callbacks(status=(0, 0, 3, 0))
        with self.assertRaisesRegex(ValueError, 'statuses differ'):
            prepare_ranked_mailbox(lowered, inputs, plans, **failure).run()
        calls, contexts, changing = self.callbacks()
        prepared = prepare_ranked_mailbox(lowered, inputs, plans, **changing)
        contexts[1] = 'changed'
        with self.assertRaisesRegex(ValueError, 'contexts changed'):
            prepared.run()
        self.assertEqual(calls['launch'], 0)


if __name__ == '__main__':
    unittest.main()
