"""The existing local lock owns mapping, waiting and pre-authoring admission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import local_broker
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.platforms import PLATFORMS
from tools import launch_task, launch_task_matrix


class LocalDeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.lock = self.root / 'local.lock'

    def test_mapping_is_installed_only_after_the_existing_lock_is_acquired(self):
        with patch.dict(os.environ, {'MACA_VISIBLE_DEVICES': '7', 'HIP_VISIBLE_DEVICES': '4'}, clear=True), \
             patch.object(local_broker, '_lock_path', return_value=self.lock):
            fd = local_broker._acquire(self.lock)
            try:
                with self.assertRaises(local_broker.LocalBrokerBusy):
                    local_broker.admit_local_job('maca', device=0)
                self.assertEqual(os.environ['MACA_VISIBLE_DEVICES'], '7')
                self.assertNotIn('OPEN_CAKE_LOCAL_DEVICE', os.environ)
            finally:
                os.close(fd)
            job = local_broker.admit_local_job('maca', device=0)
            held = int(os.environ['METAL_BROKER_LOCK_FD'])
            try:
                self.assertEqual(os.environ['MACA_VISIBLE_DEVICES'], '0')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
                self.assertNotIn('HIP_VISIBLE_DEVICES', os.environ)
                self.assertEqual(local_broker.observe_local_job('maca'), job)
                os.environ['MACA_VISIBLE_DEVICES'] = '3'
                with self.assertRaisesRegex(ValueError, 'mapping differs'):
                    local_broker.observe_local_job('maca')
            finally:
                os.close(held)

    def test_waiting_holds_no_lease_and_proceeds_after_the_owner_releases(self):
        source = '''from pathlib import Path
import os,sys
from open_cake_ir.evaluation import local_broker
local_broker._lock_path=lambda kind:Path(sys.argv[1])
job=local_broker.admit_local_job('maca',device=0,queue_seconds=3)
print(os.environ['MACA_VISIBLE_DEVICES'],flush=True)
'''
        with patch.dict(os.environ, {}, clear=True):
            fd = local_broker._acquire(self.lock)
            process = subprocess.Popen([sys.executable, '-c', source, str(self.lock)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')})
            try:
                time.sleep(.2)
                self.assertIsNone(process.poll())
                os.close(fd)
                fd = None
                out, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, err)
                self.assertEqual(out.strip(), '0')
            finally:
                if fd is not None:os.close(fd)
                if process.poll() is None:
                    process.terminate();process.wait(timeout=5)
        fd = local_broker._acquire(self.lock)
        os.close(fd)

    def test_invalid_selection_or_timeout_does_not_acquire(self):
        for kind, device, timeout in [('metal', 0, 0), ('maca', -1, 0), ('maca', True, 0),
                                      ('maca', 0, -1), ('maca', 0, float('nan')),
                                      ('maca', 0, float('inf'))]:
            with self.subTest(kind=kind,device=device,timeout=timeout), \
                 patch.object(local_broker, '_acquire') as acquire:
                with self.assertRaises(ValueError):
                    local_broker.admit_local_job(kind,device=device,queue_seconds=timeout)
                acquire.assert_not_called()

    def test_selection_api_has_one_platform_owner(self):
        self.assertEqual(PLATFORMS[CodeObject.MCFATBIN].local_visibility_environment,
                         ('MACA_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))
        self.assertEqual(PLATFORMS[CodeObject.METAL_BINARY_ARCHIVE].local_visibility_environment,())

    def test_probe_runs_native_admission_under_the_actual_lock_without_a_kernel(self):
        from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
        output = self.root/'probe.json'
        def observe(target, *, runtime_library):
            local_broker.observe_local_job('maca')
            with self.assertRaises(BlockingIOError):local_broker._acquire(self.lock)
            self.assertEqual(os.environ['MACA_VISIBLE_DEVICES'],'0')
            return MetaxDeviceAdmission(os.environ['METAL_JOB_ID'],target,target,'fixture',64,'0000:00:00',runtime_library)
        args=['probe','--kind','maca','--probe-target','xcore1002','--runtime-library','/fixture/runtime.so',
              '--local-device','0','--output',str(output)]
        with patch.dict(os.environ,{},clear=True), patch.object(sys,'argv',args), \
             patch.object(local_broker,'_lock_path',return_value=self.lock), \
             patch('open_cake_ir.evaluation.triton_metax.observe_local_metax',side_effect=observe):
            try:self.assertEqual(local_broker.main(),0)
            finally:
                if 'METAL_BROKER_LOCK_FD' in os.environ:os.close(int(os.environ['METAL_BROKER_LOCK_FD']))
        result=json.loads(output.read_text())
        self.assertEqual(result['scope'],'local_device_admission_only')
        self.assertEqual((result['kernel_calls'],result['timing_samples']),(0,0))

    def test_pre_authoring_failure_reports_the_retained_device_fault(self):
        executor=SimpleNamespace(document={'host_environment':{'python':{'invocation_path':sys.executable}}},
                                 admit_host=lambda:{'runtime_library':'/fixture/runtime.so'})
        runtime={'broker':{'command':['worker','--local-kind','maca','--local-device','0',
                                     '--local-queue-seconds','120'],'cwd':str(self.root)}}
        def probe(command,**kwargs):
            self.assertIn('--local-device',command)
            self.assertEqual(command[command.index('--local-device')+1],'0')
            self.assertEqual(kwargs['timeout'],240)
            (self.root/'local-device-admission.json').write_text(json.dumps({'admitted':False,
                'error':'MACA admission requires a MACA PyTorch and one visible device'}))
            return SimpleNamespace(returncode=1,stdout=b'',stderr=b'fixture')
        with patch.object(launch_task.subprocess,'run',side_effect=probe):
            with self.assertRaisesRegex(ValueError,'before provider qualification.*one visible device'):
                launch_task._admit_local_allocator(runtime,executor,'xcore1002',self.root)

    def test_matrix_forwards_same_mapping_to_baseline_and_author_launches(self):
        args=SimpleNamespace(backend='triton-metax',harness='claude-code',model='m',effort='high',
            turns=20,max_candidates=3,searches_per_turn=2,wall_seconds=28800,token_budget=None,
            maximum_cv=None,required_pair_wins=None,dispatches_per_sample=None,gpu_run=None,
            broker_socket=None,rows=None,columns=None,depth=None,provider_executable=None,
            provider_revision=None,incumbent_registry=None,local_device=0,local_queue_seconds=120)
        command=launch_task_matrix._command(args,'silu',self.root,None)
        self.assertEqual(command[command.index('--local-device')+1],'0')
        self.assertEqual(command[command.index('--local-queue-seconds')+1],'120')
