"""Real Program owner and native gate under CPU runtime doubles."""
import ctypes as C
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from benchmarks.c550 import default_stream_adapter as adapter
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from tests.contracts import test_metax_program_events as fixtures


class Runtime:
    def __init__(self, *, fail_drain=False, invalid_sample=False):
        self.threads=[];self.counter=0;self.fail_drain=fail_drain
        Create=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_void_p),C.c_uint)
        Record=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)
        One=C.CFUNCTYPE(C.c_int,C.c_void_p)
        Elapsed=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_float),C.c_void_p,C.c_void_p)
        Wait=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_uint)
        Host=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_void_p)
        def create(out,flags):self.counter+=1;out[0]=self.counter;return 0
        def join(handle):
            for t in self.threads:t.join(timeout=3)
            return 99 if any(t.is_alive() for t in self.threads) else 0
        def elapsed(out,begin,end):out[0]=float('nan') if invalid_sample else .125;return 0
        def host(stream,callback,pointer):
            cb=C.CFUNCTYPE(None,C.c_void_p)(callback)
            t=threading.Thread(target=cb,args=(pointer,));t.start();self.threads.append(t);return 0
        def drain(stream):
            status=join(stream)
            return status or (88 if fail_drain and stream else 0)
        callbacks=[Create(create),Record(lambda *a:0),One(join),Elapsed(elapsed),One(lambda h:0),
            Create(create),One(lambda h:0),Wait(lambda *a:0),Host(host),One(drain)]
        self.symbols=NS(**dict(zip(adapter.SYMBOLS,callbacks)))


class CompleteLaunchAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter.prepare_helper()
        fixtures.ProgramEvents.setUpClass()

    def fixture(self, *, fail_drain=False, invalid_sample=False):
        holder=fixtures.ProgramEvents()
        loaded,arguments,torch,calls,_=holder.fixture()
        torch.cuda.current_stream=lambda:NS(cuda_stream=0)
        runtime=Runtime(fail_drain=fail_drain,invalid_sample=invalid_sample)
        admission=MetaxDeviceAdmission('maca-123456789abc','xcore1002','xcore1002',
            'MetaX C550',64,'0000:0f:00','/cpu-runtime-double.so')
        with patch.object(adapter.C,'CDLL',return_value=runtime.symbols):
            timer=adapter.CompleteLaunchCapture(loaded.manifest,admission)
        def cleanup():
            # Only CPU fixtures are released here; no device resource exists.
            adapter._RETAIN_UNTIL_EXIT.clear()
            holder.doCleanups()
        self.addCleanup(cleanup)
        return timer,loaded,arguments,torch,calls,runtime

    def capture(self, timer, loaded, arguments, torch):
        with patch.dict(sys.modules,torch=torch):
            return timer.capture_loaded_cohort(loaded,arguments,dry_run_iters=11,repeat_iters=5)

    def test_complete_program_uses_real_owner_and_all_fresh_outputs(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        self.assertEqual(self.capture(timer,loaded,args,torch),[.125]*5)
        self.assertEqual(calls,['first','second']*16)
        self.assertTrue(all(values[-1].data==[3,4,5,6] for values in args))
        self.assertEqual(timer.last_activity['stage_calls_per_invocation'],[2]*16)
        self.assertFalse(timer.last_activity['performance_qualified'])

    def test_non_default_torch_stream_refuses_before_any_stage(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        torch.cuda.current_stream=lambda:NS(cuda_stream=9)
        with self.assertRaisesRegex(ValueError,'default stream'):self.capture(timer,loaded,args,torch)
        self.assertEqual(calls,[])

    def test_program_bound_stream_cannot_be_bypassed(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        loaded.loaded._stream=9  # Deliberately bind the CPU fixture to another stream.
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        self.assertIn('ordered stream differs',str(caught.exception.__cause__))
        self.assertEqual(calls,[])

    def test_changed_storage_still_refuses_in_program_owner(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        args[0][-1].address+=4
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        self.assertIn('storage changed after preparation',str(caught.exception.__cause__))
        self.assertEqual(calls,[])

    def test_omitted_dispatch_has_no_successful_observation(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        loaded.launch=lambda values:None
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        self.assertIn('stage count',str(caught.exception.__cause__))
        self.assertEqual(caught.exception.observation['completed_callbacks'],0)
        self.assertEqual(calls,[])

    def test_partial_program_fault_preserves_actual_stage_calls(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        original_error=RuntimeError('second stage rejected')
        def boundary(name,position):
            if name=='second' and position=='before':raise original_error
        loaded.launch=lambda values:loaded.loaded.launch(values,tensor_contract=loaded.manifest,
                                                        stream=0,boundary=boundary)
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        self.assertIs(caught.exception.__cause__,original_error)
        self.assertEqual(calls,['first'])
        self.assertEqual(caught.exception.observation['observed_stage_calls'],1)
        self.assertEqual(caught.exception.observation['completed_callbacks'],0)
        self.assertFalse(caught.exception.unsafe_to_release)

    def test_failed_drain_retains_owners_and_forbids_reuse(self):
        timer,loaded,args,torch,calls,_=self.fixture(fail_drain=True)
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        self.assertTrue(caught.exception.unsafe_to_release)
        self.assertFalse(loaded.loaded.closed)
        self.assertIs(adapter._RETAIN_UNTIL_EXIT[-1][1],loaded)
        self.assertIs(adapter._RETAIN_UNTIL_EXIT[-1][2],args)
        with self.assertRaises(ValueError):self.capture(timer,loaded,args,torch)
        self.assertEqual(len(calls),32)

    def test_codegen_family_is_not_physical_device_identity(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        wrong=MetaxDeviceAdmission('maca-123456789abc','xcore1002','xcore1000',
            'MetaX C550',64,'0000:0f:00','/cpu-runtime-double.so')
        with patch.object(adapter.C,'CDLL') as library,self.assertRaisesRegex(ValueError,'exact admitted'):
            adapter.CompleteLaunchCapture(loaded.manifest,wrong)
        library.assert_not_called()

    def test_duplicate_argument_set_refuses_before_callbacks(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        args[1]=args[0]
        with self.assertRaises(ValueError):self.capture(timer,loaded,args,torch)
        self.assertEqual(calls,[])

    def test_native_status_zero_does_not_admit_invalid_event_values(self):
        timer,loaded,args,torch,calls,_=self.fixture(invalid_sample=True)
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,args,torch)
        observation=caught.exception.observation
        self.assertEqual(observation['status'],0)
        self.assertFalse(observation['capture_completed'])
        self.assertFalse(observation['performance_qualified'])
        self.assertFalse(caught.exception.unsafe_to_release)
