"""Real process locks: no provider, driver, device context or GPU work."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import local_broker

ROOT=Path(__file__).resolve().parents[2]
CHILD='''import json,os,sys
from open_cake_ir.evaluation import local_broker
try:
    job=local_broker.admit_local_job('maca',device=int(sys.argv[1]),lock_scope=sys.argv[2],queue_seconds=float(sys.argv[3]))
except local_broker.LocalBrokerBusy:
    print(json.dumps({'state':'busy'}),flush=True)
    raise SystemExit(7)
print(json.dumps({'state':'ready','job':local_broker.observe_local_job('maca'),
                 'device':os.environ['MACA_VISIBLE_DEVICES']}),flush=True)
sys.stdin.readline()
'''


class DeviceLeaseProcesses(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve()
        self.base=self.root/f'open-cake-ir-maca-{os.geteuid()}.lock'
        self.processes=[]
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)

    def spawn(self,device,scope='device',queue=0):
        env={key:value for key,value in os.environ.items() if not key.startswith(
            ('METAL_', 'GPUQ_', 'OPEN_CAKE_LOCAL_'))}
        env.update(TMPDIR=str(self.root),PYTHONPATH=str(ROOT/'src'))
        p=subprocess.Popen([sys.executable,'-c',CHILD,str(device),scope,str(queue)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        self.processes.append(p)
        return p

    def event(self,process):
        ready,_,_=select.select([process.stdout],[],[],5)
        self.assertTrue(ready,'broker child did not report within five seconds')
        line=process.stdout.readline()
        if not line:
            self.fail(process.stderr.read())
        return json.loads(line)

    def finish(self,process):
        process.stdin.write('\n');process.stdin.flush()
        _,err=process.communicate(timeout=5)
        self.assertEqual(process.returncode,0,err)

    def test_different_devices_overlap_same_device_is_refused_and_exit_releases(self):
        first=self.spawn(1);second=self.spawn(2)
        self.assertEqual(self.event(first)['state'],'ready')
        self.assertEqual(self.event(second)['state'],'ready')
        self.assertIsNone(first.poll());self.assertIsNone(second.poll())
        blocked=self.spawn(1)
        self.assertEqual(self.event(blocked)['state'],'busy')
        blocked.communicate(timeout=5);self.assertEqual(blocked.returncode,7)
        self.finish(first)
        replacement=self.spawn(1);self.assertEqual(self.event(replacement)['state'],'ready')
        self.finish(replacement);self.finish(second)
        fd=local_broker._acquire(self.base);os.close(fd)

    def test_exec_retains_both_admission_descriptors(self):
        child = """import os,sys
from open_cake_ir.evaluation import local_broker
local_broker.admit_local_job('maca',device=2,lock_scope='device')
os.execvpe(sys.executable,[sys.executable,'-c',
    "from open_cake_ir.evaluation.local_broker import observe_local_job; print(observe_local_job('maca'))"],dict(os.environ))
"""
        env={key:value for key,value in os.environ.items() if not key.startswith(
            ('METAL_', 'GPUQ_', 'OPEN_CAKE_LOCAL_'))}
        env.update(TMPDIR=str(self.root),PYTHONPATH=str(ROOT/'src'))
        result=subprocess.run([sys.executable,'-c',child],env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(result.stdout.strip().startswith('maca-'))
        fd=local_broker._acquire(self.base);os.close(fd)

    def test_old_exclusive_user_jobs_and_new_device_jobs_exclude_each_other(self):
        for first_scope,second_scope in [('user','device'),('device','user')]:
            with self.subTest(first=first_scope):
                first=self.spawn(0,first_scope);self.assertEqual(self.event(first)['state'],'ready')
                other=self.spawn(3,second_scope);self.assertEqual(self.event(other)['state'],'busy')
                other.communicate(timeout=5);self.assertEqual(other.returncode,7)
                self.finish(first)

    def test_waiting_for_busy_device_retains_no_shared_legacy_lock(self):
        device=self.base.with_name(self.base.stem+'-device-1.lock')
        held=local_broker._acquire(device)
        waiter=self.spawn(1,queue=3)
        try:
            time.sleep(.2);self.assertIsNone(waiter.poll())
            barrier=None
            for _ in range(50):
                try:barrier=local_broker._acquire(self.base);break
                except BlockingIOError:time.sleep(.01)
            self.assertIsNotNone(barrier,'queued device request retained the legacy barrier')
            os.close(held);held=None
            self.assertFalse(select.select([waiter.stdout],[],[],.15)[0])
            os.close(barrier);barrier=None
            self.assertEqual(self.event(waiter)['state'],'ready');self.finish(waiter)
        finally:
            if held is not None:os.close(held)
            if 'barrier' in locals() and barrier is not None:os.close(barrier)

    def test_killed_owner_releases_both_kernel_locks(self):
        owner=self.spawn(4);self.assertEqual(self.event(owner)['state'],'ready')
        owner.kill();owner.communicate(timeout=5)
        replacement=self.spawn(4);self.assertEqual(self.event(replacement)['state'],'ready')
        self.finish(replacement)
        fd=local_broker._acquire(self.base);os.close(fd)


class DeviceLeaseAdmission(unittest.TestCase):
    def test_invalid_scope_or_missing_device_does_not_acquire(self):
        for kind,device,scope in [('maca',None,'device'),('metal',0,'device'),('maca',1,'unknown')]:
            with self.subTest(kind=kind,device=device,scope=scope), \
                 patch.dict(os.environ,{},clear=True),patch.object(local_broker,'_acquire') as acquire:
                with self.assertRaises(ValueError):local_broker.admit_local_job(kind,device=device,lock_scope=scope)
                acquire.assert_not_called()

    def test_observation_requires_the_matching_device_and_compatibility_descriptors(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=True), \
             patch.object(local_broker,'_lock_path',return_value=Path(directory)/'legacy.lock'):
            job=local_broker.admit_local_job('maca',device=2,lock_scope='device')
            fd=int(os.environ['METAL_BROKER_LOCK_FD']);barrier=int(os.environ['OPEN_CAKE_LOCAL_LEGACY_FD'])
            try:
                self.assertEqual(local_broker.observe_local_job('maca'),job)
                os.environ.pop('OPEN_CAKE_LOCAL_LEGACY_FD')
                with self.assertRaisesRegex(ValueError,'compatibility lock'):local_broker.observe_local_job('maca')
                os.environ['OPEN_CAKE_LOCAL_LEGACY_FD']=str(barrier)
                os.environ['OPEN_CAKE_LOCAL_DEVICE']='3'
                with self.assertRaisesRegex(ValueError,'mapping'):local_broker.observe_local_job('maca')
            finally:os.close(fd);os.close(barrier)
