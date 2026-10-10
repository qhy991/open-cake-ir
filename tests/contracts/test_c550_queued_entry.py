"""Fatal drain handling must retain prepared tensors and skip stack teardown."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import qualify_c550_queued_program as entry
from benchmarks.c550.default_stream_adapter import CaptureFailure


class QueuedEntry(unittest.TestCase):
    def fixture(self):
        args=[object() for _ in range(16)]
        loaded=NS(fresh_argument_sets=Mock(return_value=args),release_argument_sets=Mock(),
            snapshot=Mock(return_value=({},{})),validation_inputs={})
        return loaded,args

    def test_unsafe_failure_does_not_release_program_intermediates(self):
        loaded,args=self.fixture();error=CaptureFailure({'drained':False},unsafe_to_release=True)
        timer=NS(capture_loaded_cohort=Mock(side_effect=error))
        with self.assertRaises(CaptureFailure) as caught:
            entry.capture_checked(loaded,timer,object(),{})
        self.assertIs(caught.exception,error)
        loaded.release_argument_sets.assert_not_called()
        loaded.snapshot.assert_not_called()

    def test_safe_failure_releases_prepared_sets_once(self):
        loaded,args=self.fixture();error=CaptureFailure({'drained':True})
        timer=NS(capture_loaded_cohort=Mock(side_effect=error))
        with self.assertRaises(CaptureFailure):entry.capture_checked(loaded,timer,object(),{})
        loaded.release_argument_sets.assert_called_once_with(args)

    def test_success_checks_every_output_before_release(self):
        loaded,args=self.fixture();timer=NS(capture_loaded_cohort=Mock(return_value=[1.0]*5),last_activity={})
        with patch.object(entry,'compare_tile_outputs',return_value=(True,{'inputs_unchanged':True})) as compare:
            result=entry.capture_checked(loaded,timer,object(),{})
        self.assertEqual(compare.call_count,16);self.assertTrue(result['passed'])
        loaded.release_argument_sets.assert_called_once_with(args)

    def fatal_child(self, *, occupied=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/'unwound';result=root/'result.json'
            if occupied:result.write_text('retained')
            code="""from pathlib import Path
import qualify_c550_queued_program as entry
import sys
try: entry.finish(Path(sys.argv[1]),{'component_controls_passed':False,'unsafe_to_release':True},unsafe=True)
finally: Path(sys.argv[2]).write_text('unsafe teardown ran')
"""
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'tools')}
            process=subprocess.run([sys.executable,'-c',code,str(root),str(marker)],env=env,
                capture_output=True,text=True,timeout=20)
            self.assertEqual(process.returncode,74,process.stderr)
            self.assertFalse(marker.exists())
            if occupied:self.assertEqual(result.read_text(),'retained')
            else:self.assertTrue(json.loads(result.read_text())['unsafe_to_release'])

    def test_fatal_result_is_written_before_exit_without_unwind(self):self.fatal_child()
    def test_failed_result_write_still_cannot_unwind_into_cleanup(self):self.fatal_child(occupied=True)
