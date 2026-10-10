"""Ten-sample capture and terminal propagation through the existing cohort owner."""
import json
from copy import deepcopy
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation import metax_queued_events as adapter
from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
from open_cake_ir.tasks.evaluate import capture_tile_cohort
from tests.contracts import test_c550_default_stream_adapter as fixtures
from tests.contracts import test_metax_program_events as program_fixtures


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
            return capture_tile_cohort(loaded,timer,samples_per_cohort=5,route_calls_per_cohort=16)

    def test_two_cohorts_keep_ten_samples_and_thirty_two_complete_calls(self):
        timer,loaded,_,torch,calls,_=self.fixture()
        samples,args=self.capture(timer,loaded,torch)
        second,more=self.capture(timer,loaded,torch)
        self.assertEqual(samples+second,[.125]*10)
        self.assertEqual(len(args)+len(more),32)
        self.assertEqual(len({id(values) for values in args+more}),32)
        self.assertEqual(calls,['first','second']*32)
        self.assertTrue(all(values[-1].data==[3,4,5,6] for values in args+more))
        self.assertEqual(timer.last_activity['capture']['completed_callbacks'],16)
        self.assertEqual(timer.last_activity['capture']['sample_count'],5)
        self.assertEqual(timer.last_activity['capture']['observed_stage_calls'],32)
        self.assertFalse(timer.last_activity['capture']['performance_qualified'])
        loaded.release_argument_sets.assert_not_called()

    def test_new_protocol_uses_existing_mean10_and_does_not_relabel_old_events(self):
        from open_cake_ir.tasks.normalization.study import evaluation_policy
        from open_cake_ir.evaluation.paired import paired_protocol, PAIRED_MACA_QUEUED_EVENT_KIND
        policy=evaluation_policy(program_fixtures.ProgramEvents.workload,metax_queued_mean10=True)
        protocol=paired_protocol(policy)
        self.assertEqual(policy['paired_timing']['kind'],PAIRED_MACA_QUEUED_EVENT_KIND)
        self.assertEqual(protocol.samples_per_cohort*len(protocol.pair_order),10)
        self.assertEqual(protocol.route_calls_per_cohort,16)
        self.assertEqual(protocol.statistic,'mean')
        self.assertIsNone(protocol.maximum_cv)
        timer,loaded,_,torch,_,_=self.fixture()
        samples,args=self.capture(timer,loaded,torch)
        record=dict(native_activity=deepcopy(timer.last_activity),samples_ms=samples)
        adapter.validate_cohort(record,loaded.manifest,sample_count=5)
        for field,value in [('kind','maca_program_event_samples_v1'),('interval','including_submission_gaps'),
                            ('profiler_enabled',True),('stage_names',['wrong'])]:
            changed=deepcopy(record);changed['native_activity'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                adapter.validate_cohort(changed,loaded.manifest,sample_count=5)
        for field,value in [('drained',False),('requires_process_exit',True),('completed_callbacks',15),
                            ('status',1),('observed_stage_calls',31),('callback_errors',[{}])]:
            changed=deepcopy(record);changed['native_activity']['capture'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                adapter.validate_cohort(changed,loaded.manifest,sample_count=5)
        changed=deepcopy(record);changed['samples_ms'][0]+=1
        with self.assertRaises(ValueError):adapter.validate_cohort(changed,loaded.manifest,sample_count=5)

    def test_worker_prepares_the_helper_before_local_allocation(self):
        from open_cake_ir.tasks import evaluate as worker
        from open_cake_ir.tasks.normalization.study import evaluation_policy
        holder=program_fixtures.ProgramEvents
        timer,loaded,_,torch,_,_=self.fixture()
        authority=NS(allocation_mode='local_serialized',candidate=holder.candidate,baseline=holder.candidate,
            manifest=loaded.manifest,timed_assay_available=True,
            request={'purpose':'search','evaluation_protocol':evaluation_policy(holder.workload,metax_queued_mean10=True)},
            workload=NS(requires_target_preparation=True))
        with patch.object(adapter,'prepare_helper') as prepare:
            self.assertIs(worker._prepare_local_tensor_work(authority,'maca'),authority)
        prepare.assert_called_once_with()
        from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
        admission=MetaxDeviceAdmission('maca-123456789abc','xcore1002','xcore1002',
            'MetaX C550',64,'0000:0f:00','/cpu-runtime-double.so')
        authority.executor=NS(admit_host=lambda:{'runtime_library':admission.runtime_library})
        with patch.object(worker,'_prepare_target_tensor_work',return_value=authority),              patch.object(adapter,'MacaQueuedEventBenchmark') as factory,              patch.object(worker,'_evaluate_paired_tile',side_effect=lambda a,r,b,d:b('candidate',a.manifest)):
            worker._evaluate_metax_candidate(authority,{},collect_timing=True,admission=admission)
        factory.assert_called_once_with(loaded.manifest,admission)

    def test_real_paired_worker_replays_the_new_receipt_and_ten_samples_per_arm(self):
        from open_cake_ir.tasks.normalization.study import evaluation_policy
        holder=program_fixtures.ProgramEvents()
        self.addCleanup(holder.doCleanups)
        original_fixture=holder.fixture
        def fixture_with_current_stream():
            value=original_fixture()
            value[2].cuda.current_stream=lambda:value[2].cuda.default_stream(0)
            return value
        runtime=fixtures.Runtime()
        with patch.object(holder,'fixture',side_effect=fixture_with_current_stream),              patch.object(program_fixtures,'evaluation_policy',side_effect=lambda workload,**kw:evaluation_policy(workload,metax_queued_mean10=True)),              patch.object(adapter.C,'CDLL',return_value=runtime.symbols):
            receipt,result=holder.execute_worker()
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.timing['kind'],'fixed_baseline_paired_maca_queued_event_v1')
        self.assertEqual(receipt.timing['pooled_sample_counts'],{'candidate':10,'baseline':10})
        self.assertEqual(receipt.timing['pooled_mean_ms'],.125)
        self.assertEqual(result['counters']['kernel_calls'],136)

    def test_adapter_refuses_twenty_sample_pair_before_any_dispatch(self):
        timer,loaded,args,torch,calls,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'five samples'):
            timer.capture_loaded_cohort(loaded,args,dry_run_iters=11,repeat_iters=10)
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
