"""Real native component with CPU runtime callbacks, never GPU timing evidence."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DefaultStreamGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('c++')
        if compiler is None: raise unittest.SkipTest('host C++ compiler unavailable')
        cls.directory=tempfile.TemporaryDirectory()
        library=Path(cls.directory.name)/'gate.so'
        subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC','-pthread',
            str(ROOT/'tools/benchmarks/c550/default_stream_gate.cc'),'-o',str(library)],check=True,capture_output=True)
        cls.helper=C.CDLL(str(library)).cake_default_stream_cohort
        cls.reset_type=C.CFUNCTYPE(C.c_int,C.c_void_p)
        cls.launch_type=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint)
        cls.helper.argtypes=[C.POINTER(C.c_void_p),cls.reset_type,cls.launch_type,C.c_void_p,
            C.c_uint,C.c_uint,C.POINTER(C.c_float),C.POINTER(C.c_uint),C.POINTER(C.c_uint),
            C.POINTER(C.c_int),C.POINTER(C.c_uint)]
        cls.helper.restype=C.c_int

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def run_component(self, *, fail=None, delay=False):
        order=[];threads=[];closed=[];counter=[0];resets=[];callbacks=[]
        Create=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_void_p),C.c_uint)
        Record=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)
        One=C.CFUNCTYPE(C.c_int,C.c_void_p)
        Elapsed=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_float),C.c_void_p,C.c_void_p)
        Wait=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_uint)
        Host=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_void_p)
        def create(out,flags):
            counter[0]+=1;out[0]=counter[0];order.append(('create',counter[0],flags));return 0
        def record(event,stream):
            order.append(('record',event,stream))
            return 77 if fail=='end' and event==2 and len(resets)==2 else 0
        def drain(stream):
            order.append(('drain',stream))
            if stream==4:
                for thread in threads:thread.join(timeout=3)
                if any(t.is_alive() for t in threads):return 99
                if fail=='drain':return 88
            return 0
        def sync(event):
            order.append(('sync',event))
            if threads:
                threads[-1].join(timeout=3)
                if threads[-1].is_alive():return 99
            return 0
        def elapsed(out,begin,end):out[0]=.125;order.append(('elapsed',));return 0
        def destroy(handle):closed.append(handle);order.append(('destroy',handle));return 0
        def wait(stream,event,flags):order.append(('wait',stream,event,flags));return 0
        def host(stream,callback,pointer):
            order.append(('host',stream))
            function=C.CFUNCTYPE(None,C.c_void_p)(callback)
            def run():function(pointer);order.append(('released',))
            thread=threading.Thread(target=run);thread.start();threads.append(thread);return 0
        def reset(user):resets.append(1);order.append(('reset',));return 0
        def launch(user,index):
            callbacks.append(index);order.append(('stage',index,'first'))
            if fail=='launch' and index==11:return 32
            if delay and index==11:time.sleep(2.05)
            order.append(('stage',index,'second'));return 0
        api_callbacks=[Create(create),Record(record),One(sync),Elapsed(elapsed),One(destroy),
                       Create(create),One(destroy),Wait(wait),Host(host),One(drain)]
        api=(C.c_void_p*10)(*(C.cast(f,C.c_void_p).value for f in api_callbacks))
        values=(C.c_float*5)();count=C.c_uint();phase=C.c_uint();cleanup=C.c_int();drained=C.c_uint()
        reset_cb=self.reset_type(reset);launch_cb=self.launch_type(launch)
        status=self.helper(api,reset_cb,launch_cb,None,11,5,values,C.byref(count),C.byref(phase),C.byref(cleanup),C.byref(drained))
        self.assertTrue(all(not t.is_alive() for t in threads))
        return dict(status=status,count=count.value,phase=phase.value,cleanup=cleanup.value,
                    drained=drained.value,order=order,closed=closed,calls=callbacks,resets=len(resets),samples=list(values))

    def test_full_callback_and_default_stream_order_precede_gate_release(self):
        r=self.run_component()
        self.assertEqual((r['status'],r['count'],r['cleanup'],r['drained']),(0,16,0,1))
        self.assertEqual(r['calls'],list(range(16)));self.assertEqual(r['resets'],6)
        self.assertEqual(sorted(r['closed']),[1,2,3,4])
        order=r['order']
        for index in range(11,16):
            at=order.index(('stage',index,'first'))
            self.assertEqual(order[at-3:at],[('record',3,4),('wait',None,3,0),('record',1,None)])
            self.assertEqual(order[at:at+3],[('stage',index,'first'),('stage',index,'second'),('record',2,None)])
        self.assertEqual(sum(x==('released',) for x in order),5)

    def test_partial_callback_failure_releases_gate_without_a_sample(self):
        r=self.run_component(fail='launch')
        self.assertEqual((r['status'],r['count'],r['phase']),(32,11,16))
        self.assertEqual(r['samples'],[0.0]*5)
        self.assertNotIn(('stage',11,'second'),r['order'])
        self.assertEqual(sorted(r['closed']),[1,2,3,4]);self.assertEqual(r['drained'],1)

    def test_end_submission_failure_also_drains_and_releases(self):
        r=self.run_component(fail='end')
        self.assertEqual((r['status'],r['count'],r['phase']),(77,12,17))
        self.assertEqual(r['samples'],[0.0]*5);self.assertEqual(sorted(r['closed']),[1,2,3,4])

    def test_drain_failure_cannot_destroy_callback_resources_or_report_success(self):
        r=self.run_component(fail='drain')
        self.assertEqual((r['status'],r['cleanup'],r['drained']),(88,88,0))
        self.assertEqual(r['closed'],[])

    def test_gate_timeout_refuses_the_observation(self):
        r=self.run_component(delay=True)
        self.assertEqual((r['status'],r['phase'],r['count']),(-41001,19,12))
        self.assertEqual(r['samples'],[0.0]*5);self.assertEqual(sorted(r['closed']),[1,2,3,4])
