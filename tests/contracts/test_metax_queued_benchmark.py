"""Ten-sample capture and terminal propagation through the existing cohort owner."""
import json
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation import metax_queued_events as adapter
from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
from open_cake_ir.tasks.evaluate import capture_tile_cohort
from tests.contracts import test_c550_default_stream_adapter as fixtures


class QueuedBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CompleteLaunchAdapter.setUpClass()

    def fixture(self, **options):
        fixture=fixtures.CompleteLaunchAdapter()
        result=fixture.fixture(timer_type=adapter.MacaQueuedEventBenchmark, **options)
        self.addCleanup(fixture.doCleanups)
        result[1].release_argument_sets=Mock()
        return result

    def capture(self, timer, loaded, torch):
        with patch.dict(sys.modules,torch=torch):
            return capture_tile_cohort(loaded,timer,samples_per_cohort=10,route_calls_per_cohort=21)

    def test_ten_samples_keep_all_twenty_one_complete_calls_and_outputs(self):
        timer,loaded,_,torch,calls,_=self.fixture()
        samples,args=self.capture(timer,loaded,torch)
        self.assertEqual(samples,[.125]*10)
        self.assertEqual(len(args),21)
        self.assertEqual(calls,['first','second']*21)
        self.assertTrue(all(values[-1].data==[3,4,5,6] for values in args))
        self.assertEqual(timer.last_activity['completed_callbacks'],21)
        self.assertEqual(timer.last_activity['sample_count'],10)
        self.assertEqual(timer.last_activity['observed_stage_calls'],42)
        self.assertFalse(timer.last_activity['performance_qualified'])
        loaded.release_argument_sets.assert_not_called()

    def test_adapter_refuses_five_samples_before_any_dispatch(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'exactly ten'):
            timer.capture_loaded_cohort(loaded,args,dry_run_iters=11,repeat_iters=5)
        self.assertEqual(calls,[])

    def test_unsafe_native_drain_reaches_terminal_without_cohort_release(self):
        timer,loaded,_,torch,_,_=self.fixture(fail_drain=True)
        with self.assertRaises(UndrainedDeviceWork) as caught:self.capture(timer,loaded,torch)
        loaded.release_argument_sets.assert_not_called()
        self.assertTrue(any(owner is loaded for owner in caught.exception._owners))
        observation=json.loads(caught.exception.artifact_payloads['native_observation'])
        self.assertFalse(observation['drained'])
        self.assertTrue(observation['requires_process_exit'])

    def test_callback_terminal_is_not_downgraded_by_later_native_drain(self):
        timer,loaded,_,torch,_,_=self.fixture()
        original=UndrainedDeviceWork('callback owner cannot be released')
        loaded.launch=Mock(side_effect=original)
        with self.assertRaises(UndrainedDeviceWork) as caught:self.capture(timer,loaded,torch)
        self.assertIs(caught.exception,original)
        self.assertTrue(timer.last_activity['drained'])
        self.assertTrue(timer.last_activity['requires_process_exit'])
        loaded.release_argument_sets.assert_not_called()

    def test_ordinary_drained_callback_failure_keeps_normal_release(self):
        timer,loaded,_,torch,_,_=self.fixture()
        original=RuntimeError('ordinary callback failure')
        loaded.launch=Mock(side_effect=original)
        with self.assertRaises(adapter.CaptureFailure) as caught:self.capture(timer,loaded,torch)
        self.assertFalse(caught.exception.unsafe_to_release)
        self.assertIs(caught.exception.__cause__,original)
        loaded.release_argument_sets.assert_called_once()

    def test_failed_diagnostic_formatting_cannot_mask_unsafe_capture(self):
        class BadMessage(RuntimeError):
            def __str__(self):raise ValueError('broken diagnostic')
        timer,loaded,_,torch,_,_=self.fixture(fail_drain=True)
        loaded.launch=Mock(side_effect=BadMessage())
        with patch.object(adapter,'canonical_json_bytes',side_effect=ValueError('bad serialization')), \
             self.assertRaises(UndrainedDeviceWork) as caught:
            self.capture(timer,loaded,torch)
        loaded.release_argument_sets.assert_not_called()
        self.assertEqual(timer.last_activity['callback_errors'][0]['message'],'<exception message unavailable>')
        observation=json.loads(caught.exception.artifact_payloads['native_observation'])
        self.assertTrue(observation['diagnostic_serialization_failed'])
        self.assertTrue(observation['requires_process_exit'])

    def test_native_call_interruption_without_drain_retains_owners(self):
        timer,loaded,_,torch,_,_=self.fixture()
        helper=NS(cake_default_stream_cohort=Mock(side_effect=RuntimeError('FFI interrupted')))
        with patch.object(adapter,'_HELPER',helper), self.assertRaises(UndrainedDeviceWork):
            self.capture(timer,loaded,torch)
        loaded.release_argument_sets.assert_not_called()
        self.assertFalse(timer.last_activity['drained'])
        self.assertEqual(timer.last_activity['phase'],'native_call_interrupted')

    def test_new_host_compilation_is_refused_inside_an_allocation(self):
        with patch.object(adapter,'_HELPER',None), \
             patch.dict(os.environ,{'METAL_BROKER_LOCK_FD':'99'}), \
             patch.object(adapter.subprocess,'run') as compile_call, \
             self.assertRaisesRegex(ValueError,'precede device allocation'):
            adapter.prepare_helper()
        compile_call.assert_not_called()


if __name__=='__main__':unittest.main()
