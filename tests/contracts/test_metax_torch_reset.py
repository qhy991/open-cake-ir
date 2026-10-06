"""The production reset callback runs before the native event submission begins."""
import ctypes as C
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import metax_native_events as native
from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate


class TorchResetTests(unittest.TestCase):
    def capture(self, *, fail=False):
        reset_calls=[]
        class ResetBuffer:
            def data_ptr(self):return 1234
            def numel(self):return 8388608
            def fill_(self,value):
                if fail:raise RuntimeError('reset failed')
                reset_calls.append(value)
        dummy=C.CFUNCTYPE(C.c_int)(lambda:0)
        api=SimpleNamespace(**{name:dummy for name in native.SYMBOLS})
        kernel=object.__new__(LoadedMetaxCandidate)
        kernel.closed=False;kernel._api=api;kernel._function=C.c_void_p(1);kernel.launch_calls=0
        pointer=C.c_void_p(4321);slots=(C.c_void_p*1)(C.addressof(pointer))
        kernel.prepare_arguments=lambda *args,**kwargs:([pointer],slots)
        manifest=SimpleNamespace(grid=(1,1,1),block=(64,1,1),dynamic_shared_memory_bytes=0)
        loaded=SimpleNamespace(loaded=kernel,manifest=manifest)
        def entry(addresses,function,dimensions,shared,args,warmups,samples,buffer,words,elapsed,calls,phase):
            reset=C.CFUNCTYPE(C.c_int,C.c_size_t,C.c_uint,C.c_size_t,C.c_void_p)(addresses[5])
            for _ in range(6):
                code=reset(buffer,0x3f800000,words,None)
                if code:return code
            calls._obj.value=16;phase._obj.value=13
            for i in range(5):elapsed[i]=.001
            return 0
        torch=SimpleNamespace(cuda=SimpleNamespace(default_stream=lambda _:object(),stream=lambda _:nullcontext()))
        with patch.dict(sys.modules,torch=torch),patch.object(native,'_HELPER',SimpleNamespace(cake_maca_event_cohort=entry)):
            values=native.capture(loaded,[object() for _ in range(16)],ResetBuffer(),warmups=11,samples=5,torch_reset=True)
        return values,reset_calls,kernel.launch_calls

    def test_two_complete_fills_before_each_native_sample(self):
        values,reset_calls,calls=self.capture()
        self.assertEqual(reset_calls,[1.0]*12)
        self.assertEqual(calls,16);self.assertTrue(all(value>0 for value in values))

    def test_callback_exception_is_retained_and_never_ignored_by_ctypes(self):
        with self.assertRaises(RuntimeError) as caught:self.capture(fail=True)
        self.assertEqual(caught.exception.native_observations['reset_errors'],['reset failed'])
