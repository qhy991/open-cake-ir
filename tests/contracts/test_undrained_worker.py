"""CPU ownership and subprocess termination contracts; no device work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
from open_cake_ir.tasks import evaluate as worker
from tests.contracts import test_paired_execution as paired_fixture

ROOT=Path(__file__).resolve().parents[2]


class UndrainedWorker(unittest.TestCase):
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
