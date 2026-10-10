"""CPU ownership and subprocess termination contracts; no device work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
from open_cake_ir.tasks import evaluate as worker
from tests.contracts import test_paired_execution as paired_fixture
from tests.contracts import test_untimed_validation_cases as untimed_fixture

ROOT=Path(__file__).resolve().parents[2]


class UndrainedWorker(unittest.TestCase):
    def test_single_tensor_and_untimed_callers_preserve_terminal_owner(self):
        for untimed in (False, True):
            for error_type in (UndrainedDeviceWork, RuntimeError):
                with self.subTest(untimed=untimed, error=error_type.__name__):
                    fixture=untimed_fixture.UntimedValidationCases();fixture.setUp();self.addCleanup(fixture.doCleanups)
                    authority=fixture.authority;authority.case_id='primary'
                    if not untimed:
                        authority.request['evaluation_protocol']={}
                    loaded=NS(module_count=1,loaded=NS(launch_calls=1,resources={}),close=Mock())
                    error=error_type('injected capture failure')
                    with patch.object(worker,'LoadedTorchTensorCandidate',return_value=loaded), \
                         patch.object(worker,'materialize_evaluation_inputs',return_value={}), \
                         patch.object(worker,'evaluate_tile_workload',side_effect=error), \
                         patch.object(worker,'evaluate_tile_validation_case',side_effect=error), \
                         self.assertRaises(error_type) as caught:
                        worker._evaluate_tile_candidate(authority,fixture.result,None,fixture.admission,
                            False,route_calls_per_cohort=None)
                    self.assertIs(caught.exception,error)
                    if error_type is UndrainedDeviceWork:
                        loaded.close.assert_not_called()
                        self.assertTrue(any(owner is loaded for owner in error._owners))
                    else:
                        loaded.close.assert_called_once_with()

    def test_close_can_replace_an_ordinary_failure_with_terminal_custody(self):
        fixture=untimed_fixture.UntimedValidationCases();fixture.setUp();self.addCleanup(fixture.doCleanups)
        ordinary=RuntimeError('ordinary preflight failure')
        terminal=UndrainedDeviceWork('close could not drain')
        loaded=NS(module_count=1,loaded=NS(launch_calls=1,resources={}),close=Mock(side_effect=terminal))
        with patch.object(worker,'LoadedTorchTensorCandidate',return_value=loaded), \
             patch.object(worker,'materialize_evaluation_inputs',return_value={}), \
             patch.object(worker,'evaluate_tile_validation_case',side_effect=ordinary), \
             self.assertRaises(UndrainedDeviceWork) as caught:
            worker._evaluate_untimed_validation_cases(fixture.authority,fixture.result,fixture.admission)
        self.assertIs(caught.exception,terminal)
        loaded.close.assert_called_once_with()
        self.assertTrue(any(owner is loaded for owner in terminal._owners))
        self.assertTrue(any(owner is ordinary for owner in terminal._owners))

    def test_flash_candidate_finalizer_preserves_terminal_owner(self):
        for error_type in (UndrainedDeviceWork, RuntimeError):
            with self.subTest(error=error_type.__name__):
                error=error_type('injected native launch failure')
                torch=MagicMock()
                admission=NS(device_name='fixture',compute_capability=(10,0),gpu_uuid='fixture-uuid',
                             broker_job_id='gpuq-123456789abc')
                torch.cuda.device_count.return_value=1
                torch.cuda.get_device_name.return_value=admission.device_name
                torch.cuda.get_device_capability.return_value=admission.compute_capability
                torch.cuda.get_device_properties.return_value=NS(uuid=admission.gpu_uuid)
                workload=NS(case=lambda case:{'shape':{'B':1,'N':1,'K':1,'D':1}})
                authority=NS(executor=NS(admit_host=lambda:None),workload=workload,case_id='primary',
                             baseline=None,manifest=object(),candidate=object(),payloads={'cubin':b'fixture'})
                loaded=NS(launch=Mock(side_effect=error),close=Mock())
                inputs=(MagicMock(),MagicMock())
                with patch.dict(sys.modules,{'torch':torch}), \
                     patch.object(worker,'_execution_platform',return_value=worker.CodeObject.CUBIN), \
                     patch.object(worker,'generate_flash_kmeans_case',return_value=inputs), \
                     patch.object(worker.LoadedCudaCandidate,'load',return_value=loaded), \
                     self.assertRaises(error_type) as caught:
                    worker._evaluate_candidate(authority,{'counters':{}},collect_timing=False,admission=admission)
                self.assertIs(caught.exception,error)
                if error_type is UndrainedDeviceWork:
                    loaded.close.assert_not_called()
                    self.assertTrue(any(owner is loaded for owner in error._owners))
                else:
                    loaded.close.assert_called_once_with(synchronize=torch.cuda.synchronize)

    def test_capture_retains_prepared_sets_on_terminal_signal(self):
        args=[object()];loaded=NS(fresh_argument_sets=lambda n:args,release_argument_sets=Mock())
        error=UndrainedDeviceWork('stream drain failed')
        with self.assertRaises(UndrainedDeviceWork) as caught:
            worker.capture_tile_cohort(loaded,Mock(side_effect=error),samples_per_cohort=1,route_calls_per_cohort=1)
        self.assertIs(caught.exception,error);loaded.release_argument_sets.assert_not_called()
        self.assertIn(args,error._owners)

    def test_ordinary_capture_failure_still_releases(self):
        args=[object()];loaded=NS(fresh_argument_sets=lambda n:args,release_argument_sets=Mock())
        with self.assertRaises(RuntimeError):
            worker.capture_tile_cohort(loaded,Mock(side_effect=RuntimeError('ordinary')),samples_per_cohort=1,route_calls_per_cohort=1)
        loaded.release_argument_sets.assert_called_once_with(args)

    def test_terminal_snapshot_failure_does_not_release(self):
        args=[object()];error=UndrainedDeviceWork('copy drain failed')
        loaded=NS(snapshot=Mock(side_effect=error),release_argument_sets=Mock())
        with patch.object(worker,'capture_tile_cohort',return_value=([1.],args)),self.assertRaises(UndrainedDeviceWork):
            worker._fresh_tile_cohort(loaded,None,None,{}, {},samples_per_cohort=1,route_calls_per_cohort=1)
        loaded.release_argument_sets.assert_not_called()

    def paired_failure(self,error):
        fixture=paired_fixture.PairedExecutionTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        with patch.object(worker,'_fresh_tile_cohort',side_effect=error),self.assertRaises(type(error)) as caught:
            fixture.execute()
        self.assertIs(caught.exception,error)
        self.assertEqual(len(fixture.created),2)
        return fixture

    def test_paired_terminal_failure_retains_both_modules(self):
        fixture=self.paired_failure(UndrainedDeviceWork('drain failed'))
        self.assertEqual(fixture.closed,[])

    def test_paired_ordinary_failure_still_closes_both_modules(self):
        fixture=self.paired_failure(RuntimeError('ordinary failure'))
        self.assertEqual(fixture.closed,fixture.created)

    def test_terminal_close_failure_stops_further_module_cleanup(self):
        fixture=paired_fixture.PairedExecutionTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.fail_close_at=0
        error=UndrainedDeviceWork('drain failed during close')
        fixture.close_error=error
        with patch.object(worker,'_fresh_tile_cohort',side_effect=RuntimeError('ordinary')),self.assertRaises(UndrainedDeviceWork) as caught:
            fixture.execute()
        self.assertIs(caught.exception,error)
        self.assertEqual(fixture.closed,[fixture.created[0]])

    def child(self,*,occupied=False,bad_artifact=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';output=root/'result.json';marker=root/'unwound'
            request.write_text('{}')
            if occupied:output.write_text('retained')
            code=r'''
import atexit,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from open_cake_ir.tasks import evaluate as worker
from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
os.environ.pop('GPUQ_BACKEND',None)
marker=Path(sys.argv[3]);atexit.register(lambda:marker.write_text('atexit ran'))
payload=object() if sys.argv[4]=='bad' else b'{"drained":false}'
error=UndrainedDeviceWork('CPU injected undrained work',artifact_payloads={'native_observation':payload})
def die(authority,result):
 result['receipt']={'would_be_success':True}
 raise error
sys.argv=['evaluate','--request',sys.argv[1],'--output',sys.argv[2]]
try:
 with patch.object(worker,'_load_authority',return_value=NS(request={'purpose':'search'})), \
      patch.object(worker,'_platform',return_value=NS(evaluate=die)):
  worker.main()
finally:marker.write_text('finally ran')
'''
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
            result=subprocess.run([sys.executable,'-c',code,str(request),str(output),str(marker),
                'bad' if bad_artifact else 'valid'],env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,74,result.stderr)
            self.assertIn('CPU injected undrained work',result.stderr)
            self.assertFalse(marker.exists())
            if occupied:self.assertEqual(output.read_text(),'retained')
            elif not bad_artifact:
                record=json.loads(output.read_text())
                self.assertEqual(record['failure_class'],'UndrainedDeviceWork')
                self.assertIsNone(record['receipt'])
                self.assertEqual((root/record['failure_artifacts']['native_observation']).read_bytes(),b'{"drained":false}')

    def test_worker_persists_failure_and_exits_without_cleanup(self):self.child()
    def test_result_write_failure_still_exits_without_cleanup(self):self.child(occupied=True)
    def test_artifact_failure_still_exits_without_cleanup(self):self.child(bad_artifact=True)
