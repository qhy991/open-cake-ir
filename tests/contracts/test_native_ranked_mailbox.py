"""Ranked mailbox lowering binds visible Cake math to exact B300 effects."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler import (Compiler, LoweredRankedMailbox, Program,
                                   RankedMailboxEffects, Schedule, Target)
from open_cake_ir.compiler.ir import DType
from open_cake_ir.compiler.backends.native_cuda_ranked_mailbox import _admit, _emit_source
from open_cake_ir.evaluation.ranked_launch import (
    RankedMailboxExecutable, prepare_ranked_mailbox,
)
from open_cake_ir.evaluation.ranked_manifest import RankedMailboxLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.weave_ep.workload import validate_contract


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
                self.assertEqual(requirements['rank_inputs'], [
                    {'name': 'hidden', 'shape': [tokens, 16], 'dtype': 'bf16'},
                    {'name': 'expert_ids', 'shape': [tokens, 2], 'dtype': 'int32'},
                    {'name': 'route_weights', 'shape': [tokens, 2], 'dtype': 'fp32'},
                    {'name': 'w_up_gate', 'shape': [2, 64, 16], 'dtype': 'bf16'},
                    {'name': 'w_down', 'shape': [2, 16, 32], 'dtype': 'bf16'},
                ])
                self.assertTrue(requirements['cooperative_grid'])
                self.assertTrue(requirements['peer_pair_runtime_check'])
                self.assertTrue(requirements['input_domain_runtime_check'])
                self.assertEqual(requirements['activated_shared_bytes'], 128)
                self.assertIn('--gpu-architecture=compute_103a',
                              requirements['nvcc_flags'])
                self.assertIn('--gpu-code=sm_103a', requirements['nvcc_flags'])
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
        observed = replace(target, occupancy=replace(
            target.occupancy, multiprocessor_count=147))
        synthetic = SimpleNamespace(_revision=SimpleNamespace(targets={local.target: observed}))
        admitted_target, inline_math = _admit(
            synthetic, effects, local, combine, effects.analyze(local, combine))
        source, _ = _emit_source(effects.lowering.entry_point,
                                 effects.analyze(local, combine),
                                 admitted_target, inline_math)
        self.assertIn('constexpr int SMS = 147;', source)

    def test_generated_rank_input_abi_binds_one_evaluation_launch(self):
        effects, local, combine = material(8)
        lowered = self.compiler.lower_ranked_mailbox(effects, local, combine)
        rows = lowered.toolchain_requirements['rank_inputs']
        self.assertEqual([row['name'] for row in rows],
                         ['hidden', 'expert_ids', 'route_weights',
                          'w_up_gate', 'w_down'])
        inputs = {}
        for rank in range(4):
            inputs[rank] = {}
            for index, row in enumerate(rows):
                dtype = DType(row['dtype'])
                shape = tuple(row['shape'])
                nbytes = dtype.itemsize
                for extent in shape:
                    nbytes *= extent
                inputs[rank][row['name']] = SimpleNamespace(
                    pointer=1000000 + rank * 1000000 + index * 8192,
                    nbytes=nbytes, device=rank, shape=shape, dtype=dtype)
        plans = {rank: {'communication_ctas': (147, 12, 36, 72)[rank],
                        'chunks': (2, 4, 8, 1)[rank],
                        'steal_budget': 32 if rank == 0 else 0}
                 for rank in range(4)}
        calls = {'reset': 0, 'launch': 0, 'status': 0}

        def span(tensor):
            return tensor.device, tensor.pointer, tensor.pointer + tensor.nbytes

        def check(tensor, spec):
            if (tensor.shape != spec.shape or tensor.dtype is not spec.dtype
                    or tensor.nbytes != spec.nbytes):
                raise ValueError('bound generated ABI differs')

        def load(source):
            self.assertIs(source, lowered)

            def bind(buffers, mailboxes, controls, contexts):
                self.assertEqual(set(buffers), set(mailboxes))
                self.assertEqual(controls[0]['steal_budget'], 32)

                def launch(bound):
                    self.assertEqual(bound, contexts)
                    calls['launch'] += 1

                return launch

            return RankedMailboxExecutable(8192, 32, 4096, 256, bind)

        def allocate(rank, size):
            return SimpleNamespace(pointer=1100000 + rank * 1000000,
                                   nbytes=size, device=rank, shape=(size,), dtype=None)

        def view(mailbox, offset, spec):
            return SimpleNamespace(pointer=mailbox.pointer + offset,
                                   nbytes=spec.nbytes, device=mailbox.device,
                                   shape=spec.shape, dtype=spec.dtype)

        def reset(mailboxes, contexts):
            calls['reset'] += 1

        def statuses(mailboxes, offset, contexts):
            self.assertEqual(offset, 32)
            calls['status'] += 1
            return (0, 0, 0, 0)

        prepared = prepare_ranked_mailbox(
            lowered, inputs, plans, load_source=load, allocate_mailbox=allocate,
            check_tensor=check, storage_span=span,
            execution_context=lambda rank: (rank, None), output_view=view,
            reset_mailboxes=reset, read_status=statuses)
        prepared.run()
        self.assertEqual((calls['reset'], calls['launch'], calls['status'],
                          prepared.launch_calls), (1, 1, 1, 1))

    def test_actual_lowering_binds_frozen_distributed_workload_case(self):
        effects, local, combine = material(7)
        lowered = self.compiler.lower_ranked_mailbox(effects, local, combine)
        contract_path = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'
        workload = WorkloadContract.from_document(
            json.loads(contract_path.read_text()), contract_path,
            validate=validate_contract)
        combine_document = json.loads((ROOT / 'examples/schedules/native/'
                                       'weave-weighted-combine-t7-h16.json').read_text())
        bindings = {'hidden_states': 'hidden', 'expert_ids': 'expert_ids',
                    'route_weights': 'route_weights', 'w_up_gate': 'w_up_gate',
                    'w_down': 'w_down'}
        plans = [{'rank': rank, 'communication_ctas': (147, 12, 36, 72)[rank],
                  'chunks': (2, 3, 7, 1)[rank],
                  'steal_budget': 14 if rank == 0 else 0}
                 for rank in range(4)]
        manifest = RankedMailboxLaunchManifest.from_lowered(
            lowered, combine_document=combine_document, workload=workload,
            case_id='tail_tokens', tensor_bindings=bindings, plans=plans)
        manifest.check_lowered(lowered)
        self.assertEqual(manifest.as_dict()['case_id'], 'tail_tokens')
        self.assertEqual(manifest.as_dict()['plans'][2]['chunks'], 7)
        with self.assertRaisesRegex(ValueError, 'Workload|public output'):
            manifest.check_workload(workload, 'skew_to_rank0')


if __name__ == '__main__':
    unittest.main()
