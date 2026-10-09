"""CPU observations do not qualify device timing or explain a historical spike."""
from contextlib import nullcontext
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from benchmarks.c550.host_timing_trace import HostTimeline, instrument_host_phases
from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
from open_cake_ir.evaluation.metax_event_benchmark import MacaEventBenchmark


class HostTimingTrace(unittest.TestCase):
    def fixture(self):
        manifest = NS(target='xcore1002', aligned_variant=False, kernel_name='test',
            grid=(1,1,1), block=(256,1,1), dynamic_shared_memory_bytes=0,
            hidden_null_pointer_parameters=0, tensor_abi=(('x',(4,),'bf16','input'),))
        argument = NS(is_contiguous=lambda: True, shape=(4,), dtype='torch.bfloat16',
            device=NS(type='cuda',index=0), data_ptr=lambda: 4096)
        calls = []
        api = NS(mcModuleLaunchKernel=lambda *args: calls.append(args) or 0)
        loaded = LoadedMetaxCandidate(None, manifest, api, None, ctypes.c_void_p(1), None, {})
        stream = NS(cuda_stream=0, synchronize=lambda: None)
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, actual):
                if actual is not stream: raise ValueError('stream changed')
            def synchronize(self): pass
            def elapsed_time(self, other):
                if type(other) is not Event: raise ValueError('event not unwrapped')
                return 1.0  # Synthetic sentinel, never a device measurement.
        torch = NS(version=NS(maca='CPU double'), float32=object(),
            empty=lambda *a,**k:NS(fill_=lambda value: None),cuda=NS(Event=Event,
            default_stream=lambda device:stream, stream=lambda stream:nullcontext(),
            get_device_properties=lambda device:NS(L2_cache_size=8388608)))
        return loaded, argument, torch, calls

    def test_real_timer_keeps_checks_dispatch_count_and_event_order(self):
        loaded, arg, torch, calls = self.fixture()
        original_api, original_event = loaded._api, torch.cuda.Event
        timeline = HostTimeline()
        with patch.dict(sys.modules, torch=torch), instrument_host_phases(loaded,torch,timeline):
            values = MacaEventBenchmark(loaded.manifest,l2_cache_bytes=8388608)(
                lambda:loaded.launch([arg],tensor_contract=loaded.manifest),
                dry_run_iters=11,repeat_iters=5,cold_l2_cache=True,use_cuda_graph=False)
        self.assertEqual(values,[1.0]*5)
        self.assertEqual(len(calls),16)
        phases = [row['phase'] for row in timeline.rows]
        self.assertEqual(phases[:3],['begin_record','end_record','end_synchronize'])
        self.assertEqual(phases[3:25],['prepare_arguments','native_submit']*11)
        self.assertEqual(phases[25:],['begin_record','prepare_arguments','native_submit','end_record','end_synchronize']*5)
        self.assertIs(loaded._api,original_api)
        self.assertIs(torch.cuda.Event,original_event)
        self.assertNotIn('prepare_arguments',vars(loaded))
        self.assertTrue(all(row['wall_end_ns']>=row['wall_start_ns'] and
                            row['cpu_end_ns']>=row['cpu_start_ns'] for row in timeline.rows))

    def test_invalid_abi_still_refuses_before_native_submission(self):
        loaded,arg,torch,calls = self.fixture();arg.shape=(5,)
        timeline=HostTimeline();original_api=loaded._api;original_event=torch.cuda.Event
        with self.assertRaisesRegex(ValueError,'sealed ABI'):
            with instrument_host_phases(loaded,torch,timeline):
                loaded.launch([arg],tensor_contract=loaded.manifest)
        self.assertEqual(calls,[])
        self.assertEqual([r['phase'] for r in timeline.rows],['prepare_arguments'])
        self.assertEqual(timeline.rows[0]['error_type'],'ValueError')
        self.assertIs(loaded._api,original_api);self.assertIs(torch.cuda.Event,original_event)
        self.assertNotIn('prepare_arguments',vars(loaded))

    def test_exception_identity_and_independent_clock_observations(self):
        wall=iter([100,900]);cpu=iter([50,70]);error=RuntimeError('retained')
        timeline=HostTimeline(wall_clock=lambda:next(wall),cpu_clock=lambda:next(cpu))
        def fail():raise error
        with self.assertRaises(RuntimeError) as caught:timeline.call('native_submit',fail)
        self.assertIs(caught.exception,error)
        row=timeline.rows[0]
        self.assertEqual(row['wall_end_ns']-row['wall_start_ns'],800)
        self.assertEqual(row['cpu_end_ns']-row['cpu_start_ns'],20)

    def test_overlap_refuses_and_restores_the_first_observer(self):
        loaded,arg,torch,_=self.fixture();timeline=HostTimeline()
        with instrument_host_phases(loaded,torch,timeline):
            observed=loaded._api
            with self.assertRaisesRegex(RuntimeError,'overlap'):
                with instrument_host_phases(loaded,torch,HostTimeline()):pass
            self.assertIs(loaded._api,observed)
        with instrument_host_phases(loaded,torch,HostTimeline()):pass

    def test_record_budget_refuses_without_silently_dropping_calls(self):
        timeline=HostTimeline();calls=[]
        for _ in range(256):timeline.call('phase',lambda:calls.append(1))
        with self.assertRaisesRegex(RuntimeError,'record limit'):
            timeline.call('phase',lambda:calls.append(1))
        self.assertEqual(len(calls),256)
