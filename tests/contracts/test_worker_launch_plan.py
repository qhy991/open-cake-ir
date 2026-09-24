"""Worker Programs allocate hidden state and execute one owned kernel."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import unittest

from open_cake_ir.compiler import Program
from open_cake_ir.compiler.program import LoweredWorkerProgram
from open_cake_ir.evaluation.launch_plan import prepare_worker_program
from tests.contracts.test_worker_execution import document


@dataclass
class Storage:
    address: int
    nbytes: int
    device: str = 'gpu0'
    status: int = 0


def lowered_worker():
    program = Program.from_dict(document())
    return LoweredWorkerProgram(program, 'fixture', 'generated CUDA source', {
        'target': program.target,
        'entry_point': program.execution.lowering.entry_point,
        'state_bytes': 32,
        'state_status_offset_bytes': 28,
        'argument_order': list(program.tensors),
        'state_reset': 'zero_before_each_launch_on_launch_stream',
        'host_abi': {'create': 'make', 'launch': 'run', 'destroy': 'close'},
    })


class WorkerLaunchPlanTests(unittest.TestCase):
    def setUp(self):
        self.lowered = lowered_worker()
        self.inputs = {name: Storage(10000 + index * 100000,
                                     self.lowered.program.tensors[name].nbytes)
                       for index, name in enumerate(self.lowered.program.inputs)}
        self.next_address = 1000000
        self.calls = []
        self.context = ('gpu0', 'stream0')

    def allocate(self, name, spec):
        self.next_address += 100000
        return Storage(self.next_address, spec.nbytes)

    def allocate_state(self, size):
        self.next_address += 100000
        return Storage(self.next_address, size)

    @staticmethod
    def span(tensor):
        return tensor.device, tensor.address, tensor.address + tensor.nbytes

    def prepare(self, **changes):
        arguments = dict(
            allocate=self.allocate, allocate_state=self.allocate_state,
            load_worker=lambda lowered, buffers, state: self.load(lowered, buffers, state),
            check_tensor=lambda tensor, spec: self.assertEqual(tensor.nbytes, spec.nbytes),
            storage_span=self.span,
            execution_context=lambda: self.context,
            read_status=lambda state, offset, context: self.status(state, offset, context),
        )
        arguments.update(changes)
        return prepare_worker_program(self.lowered, self.inputs, **arguments)

    def load(self, lowered, buffers, state):
        self.assertIs(lowered, self.lowered)
        self.assertEqual(tuple(buffers), tuple(lowered.program.tensors))
        self.calls.append('load')
        def launch(context):
            self.assertEqual(context, self.context)
            self.calls.append('launch')
            state.status = 0
        return launch

    def status(self, state, offset, context):
        self.assertEqual(offset, 28)
        self.assertEqual(context, self.context)
        self.calls.append('status')
        return state.status

    def test_one_kernel_and_one_status_check_per_call(self):
        prepared = self.prepare()
        self.assertEqual(self.calls, ['load'])
        self.assertEqual(set(prepared.buffers), set(self.lowered.program.tensors))
        self.assertEqual(set(prepared.outputs), {'out'})
        for _ in range(2):
            self.assertIs(prepared.run()['out'], prepared.buffers['out'])
        self.assertEqual(self.calls, ['load', 'launch', 'status', 'launch', 'status'])
        self.assertEqual(prepared.launch_calls, 2)

    def test_state_and_tensor_storage_cannot_alias_or_cross_devices(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.prepare(allocate_state=lambda size: Storage(self.inputs['a'].address, size))
        with self.assertRaisesRegex(ValueError, 'share one device'):
            self.prepare(allocate_state=lambda size: Storage(2000000, size, 'gpu1'))
        with self.assertRaisesRegex(ValueError, 'extent differs'):
            self.prepare(allocate_state=lambda size: Storage(2000000, size - 4))

    def test_metadata_status_and_context_fail_closed(self):
        original = self.lowered
        for field, value in (
            ('state_bytes', 0), ('state_status_offset_bytes', 32),
            ('argument_order', ['a']), ('state_reset', 'caller_might_reset'),
        ):
            with self.subTest(field=field):
                self.lowered = replace(original,
                    toolchain_requirements={**original.toolchain_requirements, field: value})
                with self.assertRaisesRegex(ValueError, 'state, argument order or host ABI'):
                    self.prepare()
        self.lowered = original
        prepared = self.prepare(read_status=lambda state, offset, context: -1)
        with self.assertRaisesRegex(ValueError, 'returned status -1'):
            prepared.run()
        prepared = self.prepare()
        self.context = ('gpu0', 'other-stream')
        with self.assertRaisesRegex(ValueError, 'device/stream changed'):
            prepared.run()

    def test_only_declared_system_payload_may_use_peer_storage(self):
        program_document = document()
        program_document['execution']['handoffs'][0]['scope'] = 'system'
        program = Program.from_dict(program_document)
        requirements = {**self.lowered.toolchain_requirements,
                        'peer_payload_runtime_check': True}
        self.lowered = replace(self.lowered, program=program,
                               toolchain_requirements=requirements)
        def allocate(name, spec):
            tensor = self.allocate(name, spec)
            if name == 'middle0':
                tensor.device = 'gpu1'
            return tensor
        prepared = self.prepare(allocate=allocate, launch_device='gpu0')
        self.assertEqual(prepared.buffers['middle0'].device, 'gpu1')
        self.assertEqual(prepared.buffers['middle1'].device, 'gpu0')
        self.assertEqual(prepared.run()['out'].device, 'gpu0')
        with self.assertRaisesRegex(ValueError, 'explicit launch device'):
            self.prepare(allocate=allocate)
        self.lowered = replace(self.lowered, toolchain_requirements={
            key: value for key, value in requirements.items()
            if key != 'peer_payload_runtime_check'})
        with self.assertRaisesRegex(ValueError, 'peer-aware host ABI'):
            self.prepare(allocate=allocate, launch_device='gpu0')
        self.lowered = replace(self.lowered, toolchain_requirements=requirements)
        with self.assertRaisesRegex(ValueError, 'needs peer storage'):
            self.prepare(launch_device='gpu0')
        def undeclared_peer(name, spec):
            tensor = allocate(name, spec)
            if name == 'middle1':
                tensor.device = 'gpu1'
            return tensor
        with self.assertRaisesRegex(ValueError, 'local storage must share one device'):
            self.prepare(allocate=undeclared_peer, launch_device='gpu0')


if __name__ == '__main__':
    unittest.main()
