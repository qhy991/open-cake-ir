"""Complete Program event intervals using the real ordered loader and CPU devices.

These contracts do not qualify a physical MetaX target for performance Runs.
"""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.evaluation import metax_event_benchmark as events
from open_cake_ir.evaluation.program import admit_program_execution
from tests.contracts.test_native_program_tensors import build
from tests.contracts.test_ordered_launch_plan import document
from tests.contracts import test_program_evaluation as program_fixtures

ROOT = Path(__file__).resolve().parents[2]


class ProgramEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = document()
        value['target'] = 'xcore1002'
        for stage in value['stages']:
            stage['schedule']['target'] = 'xcore1002'
        cls.program = Program.from_dict(value)
        cls.workload = program_fixtures.workload_for(cls.program)
        cls.candidate = build(cls.program, cls.workload, Compiler.load(ROOT))

    def fixture(self, *, omit=False):
        native, manifest, tensor, calls, children = program_fixtures.ProgramEvaluationTests().loaded(self.candidate)
        native._stream = 0  # The fixture kernel has no real device stream.
        sets = []
        for _ in range(16):
            args = [tensor([1, 2, 3, 4], (4,), 'fp32'),
                    tensor([float('nan')] * 4, (4,), 'fp32')]
            native.prepare_arguments(args)
            sets.append(args)
        order = []
        stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: order.append('sync'))
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, actual):
                self.assert_stream = actual is stream
                if not self.assert_stream: raise ValueError('wrong stream')
                order.append('event')
            def synchronize(self): order.append('end_sync')
            def elapsed_time(self, other): return .25
        torch = SimpleNamespace(version=SimpleNamespace(maca='CPU fixture'), float32=object(),
            empty=lambda *args, **kwargs: SimpleNamespace(fill_=lambda value: order.append('reset')),
            cuda=SimpleNamespace(default_stream=lambda device: stream,
                get_device_properties=lambda device: SimpleNamespace(L2_cache_size=8388608),
                Event=Event, stream=lambda value: nullcontext()))
        def launch(args):
            order.append('program')
            if not omit:
                native.launch(args, tensor_contract=manifest, stream=0)
        wrapper = SimpleNamespace(manifest=manifest, candidate=self.candidate,
                                  loaded=native, launch=launch)
        self.addCleanup(lambda: native.close(synchronize=lambda: None))
        return wrapper, sets, torch, calls, order

    def test_full_program_has_fresh_outputs_and_ordered_reset_event_interval(self):
        loaded, sets, torch, calls, order = self.fixture()
        timer = events.MacaProgramEventBenchmark(loaded.manifest, l2_cache_bytes=8388608)
        with patch.dict(sys.modules, torch=torch):
            values = timer.capture_loaded_cohort(loaded, sets, dry_run_iters=11, repeat_iters=5)
        self.assertEqual(values, [.25] * 5)
        self.assertEqual(calls, ['first', 'second'] * 16)
        self.assertTrue(all(args[-1].data == [3, 4, 5, 6] for args in sets))
        self.assertEqual(order[-25:], ['reset', 'event', 'program', 'event', 'end_sync'] * 5)
        record = {'native_activity': timer.last_activity, 'samples_ms': values}
        events.validate_cohort(record, loaded.manifest, sample_count=5)
        for name, value in [('completed_stage_calls', 31), ('profiler_enabled', True),
                            ('stage_names', ['second', 'first']), ('interval', events.INTERVAL)]:
            bad = deepcopy(record)
            bad['native_activity'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                events.validate_cohort(bad, loaded.manifest, sample_count=5)

    def test_missing_program_dispatch_cannot_become_a_latency(self):
        loaded, sets, torch, _, _ = self.fixture(omit=True)
        timer = events.MacaProgramEventBenchmark(loaded.manifest, l2_cache_bytes=8388608)
        with patch.dict(sys.modules, torch=torch), self.assertRaisesRegex(ValueError, 'stage count'):
            timer.capture_loaded_cohort(loaded, sets, dry_run_iters=11, repeat_iters=5)

    def test_implementation_does_not_imply_device_qualification(self):
        with self.assertRaisesRegex(ValueError, 'not implemented|not qualified'):
            admit_program_execution('xcore1002', timing=True, attribution=True)


if __name__ == '__main__':
    unittest.main()
