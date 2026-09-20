"""MCPTI timing must account for the complete observed device sequence."""
from copy import deepcopy
from pathlib import Path
import ctypes
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.metax_activity import McptiActivity, _ApiActivity, _Kernel8Prefix
from open_cake_ir.evaluation.metax_benchmark import dispatch_samples, kernel_records


def kernel(name, correlation, start, *, grid=(8, 1, 1), block=(64, 1, 1)):
    return dict(kind=10, name=name, start_ns=start, end_ns=start+2048, device=0, context=0,
                stream=0, correlation=correlation, grid=list(grid), block=list(block),
                registers_per_thread=16, static_shared_bytes=0, dynamic_shared_bytes=0,
                local_bytes_per_thread=0)


def capture(*kernels):
    return dict(source="mcpti_activity", api_version=18, dropped_records=0, pending_buffers=0,
                records=[*kernels, *[dict(kind=5, cbid=60, start_ns=k['start_ns']-500,
                    end_ns=k['start_ns']-100, correlation=k["correlation"], return_value=0) for k in kernels]])


class McptiMeasurements(unittest.TestCase):
    def setUp(self):
        self.reset = kernel("fill", 10, 1000, grid=(32768,1,1), block=(256,1,1))
        self.raw = capture(self.reset, kernel("cake", 11, 4000),
                           {**self.reset, "correlation": 12, "start_ns": 7000, "end_ns": 9048},
                           kernel("cake", 13, 10000))

    def samples(self, raw):
        return dispatch_samples(raw, kernel_name="cake", grid=(8,1,1), block=(64,1,1),
                                repeats=2, reset_record=self.reset)

    def test_complete_cold_sequence_uses_device_interval_only(self):
        self.assertEqual(self.samples(self.raw), [0.002048, 0.002048])
        # A host API's duration is not included or used to rank the device sample.
        self.raw["records"][-1].update(start_ns=1, end_ns=1000000000)
        self.assertEqual(self.samples(self.raw), [0.002048, 0.002048])

    def test_missing_extra_and_non_kernel_work_never_disappear(self):
        variants = []
        missing = deepcopy(self.raw); missing["records"].pop(1); variants.append(missing)
        extra = deepcopy(self.raw); extra["records"].append(kernel("extra", 99, 14000)); variants.append(extra)
        copy = deepcopy(self.raw); copy["records"].append({"kind":1}); variants.append(copy)
        no_api = deepcopy(self.raw); no_api["records"].pop(); variants.append(no_api)
        dropped = deepcopy(self.raw); dropped["dropped_records"] = 1; variants.append(dropped)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.samples(raw)

    def test_wrong_identity_zero_timestamps_and_repeated_ids_are_refused(self):
        for mutation in ({"name":"cake_extra"}, {"grid":[9,1,1]}, {"block":[128,1,1]},
                         {"start_ns":0}, {"end_ns":4000}, {"correlation":10},
                         {"device":1}, {"stream":1}):
            raw=deepcopy(self.raw);raw["records"][1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.samples(raw)

    def test_reset_identity_and_order_are_observed_not_subtracted(self):
        for mutation in ({"name":"some_other_kernel"}, {"stream":2}, {"end_ns":5000}):
            raw=deepcopy(self.raw);raw["records"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.samples(raw)

    def test_warm_capture_is_still_exact_name_and_complete_count(self):
        raw=capture(kernel("cake",1,1000),kernel("cake",2,5000))
        self.assertEqual(dispatch_samples(raw,kernel_name="cake",grid=(8,1,1),block=(64,1,1),
                                         repeats=2,reset_record=None),[0.002048]*2)
        with self.assertRaises(ValueError):
            dispatch_samples(raw,kernel_name="cak",grid=(8,1,1),block=(64,1,1),repeats=2,reset_record=None)

    def test_device_drop_and_failed_launch_api_invalidate_attribution_too(self):
        for change in ({"api_version":19},{"dropped_records":True},{"dropped_records":2},
                       {"pending_buffers":1}):
            with self.subTest(change=change),self.assertRaises(ValueError):kernel_records({**self.raw,**change})
        raw=deepcopy(self.raw);raw["records"][-1]["return_value"]=2
        with self.assertRaisesRegex(ValueError,"failed"):kernel_records(raw)

    def test_unmatched_launch_api_and_borrowed_correlation_are_refused(self):
        for cbid in (19, 333, 57):
            raw=deepcopy(self.raw);raw['records'][-1]['cbid']=cbid
            with self.subTest(cbid=cbid),self.assertRaises(ValueError):self.samples(raw)
        raw=deepcopy(self.raw);raw['records'].append(dict(kind=5,cbid=60,start_ns=20000,
            end_ns=20100,correlation=99,return_value=0))
        with self.assertRaisesRegex(ValueError,'one to one'):self.samples(raw)
        raw=deepcopy(self.raw);raw['records'][-1]['start_ns']=0
        with self.assertRaises(ValueError):self.samples(raw)

    def test_forced_end_drain_refuses_buffers_the_sdk_did_not_return(self):
        import threading
        collector=object.__new__(McptiActivity)
        collector._buffers={1:object()};collector._errors=[];collector._rows=[];collector._dropped=0
        collector._enabled=[];collector._active=True;collector._owner_thread=threading.get_ident();collector._session=threading.Lock()
        collector._session.acquire();calls=[]
        collector._call=lambda name,flag:calls.append((name,flag))
        with self.assertRaisesRegex(ValueError,'retained activity buffers'):collector.finish()
        self.assertEqual(calls,[('mcptiActivityFlushAll',1),('mcptiActivityFlushAll',1)])
        self.assertFalse(collector._active)

    def test_python_prefix_agrees_with_the_qualified_sdk_layout(self):
        # These are the installed SDK ABI offsets, not C550 hardware constants.
        observed={name:getattr(_Kernel8Prefix,name).offset for name in
                  ("registers","start","end","device","stream","grid","block","static_shared","local_per_thread","correlation","grid_id","name")}
        self.assertEqual(observed,dict(registers=6,start=16,end=24,device=40,stream=48,
            grid=52,block=64,static_shared=76,local_per_thread=84,correlation=92,grid_id=96,name=104))
        self.assertEqual(ctypes.sizeof(_ApiActivity),40)

    def test_callback_errors_are_retained_instead_of_becoming_silent_native_exceptions(self):
        collector=object.__new__(McptiActivity)
        collector._buffers={};collector._errors=[];collector._rows=[];collector._dropped=0
        collector._completed(None,0,123,8,99)
        self.assertIn("unowned or invalid buffer",collector._errors[0])
        with patch("open_cake_ir.evaluation.metax_activity.C.create_string_buffer",side_effect=MemoryError("capacity")):
            pointer=ctypes.c_void_p();size=ctypes.c_size_t();count=ctypes.c_size_t()
            collector._requested(ctypes.pointer(pointer),ctypes.pointer(size),ctypes.pointer(count))
        self.assertEqual((pointer.value,size.value,count.value),(None,0,0))
        self.assertIn("capacity",collector._errors[-1])


class CollectorOwnership(unittest.TestCase):
    def test_a_second_constructor_cannot_replace_live_native_callbacks(self):
        import open_cake_ir.evaluation.metax_activity as activity
        with patch.object(activity, "_COLLECTOR", object()):
            with self.assertRaisesRegex(RuntimeError, "process owner"):
                activity.McptiActivity("/does/not/need/to/exist")

    def test_another_thread_cannot_finish_the_active_session(self):
        import threading
        collector=object.__new__(McptiActivity)
        collector._active=True;collector._owner_thread=threading.get_ident()+1
        with self.assertRaisesRegex(RuntimeError,"owning thread"):collector.finish()
        self.assertTrue(collector._active)

    def test_failed_launch_and_failed_drain_are_both_visible(self):
        from types import SimpleNamespace
        from open_cake_ir.evaluation.metax_benchmark import McptiDispatchBenchmark
        from open_cake_ir.evaluation.loaders import LifecycleError
        def finish():raise RuntimeError("drain failed")
        def launch():raise ValueError("launch failed")
        benchmark=object.__new__(McptiDispatchBenchmark)
        benchmark._collector=SimpleNamespace(begin=lambda:None,finish=finish)
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        with patch.dict("sys.modules",{"torch":torch}),self.assertRaises(LifecycleError) as result:
            benchmark._collect(launch)
        self.assertIn("launch failed",str(result.exception))
        self.assertIn("drain failed",str(result.exception))


class RejectedCaptureEvidence(unittest.TestCase):
    def test_a_rejected_cohort_retains_its_actual_activity_not_the_previous_success(self):
        from types import SimpleNamespace
        from open_cake_ir.evaluation.metax_benchmark import McptiDispatchBenchmark
        benchmark=object.__new__(McptiDispatchBenchmark)
        benchmark.manifest=SimpleNamespace(kernel_name='cake',grid=(8,1,1),block=(64,1,1))
        benchmark.l2_cache_bytes=8388608
        benchmark.last_activity={'old':True};benchmark.non_target_dispatches=0;benchmark.resolution_us=0.256
        raw=capture(kernel('unexpected',1,1000))
        benchmark._collect=lambda function:raw
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        with patch.dict('sys.modules',{'torch':torch}),self.assertRaises(ValueError):
            benchmark(lambda:None,dry_run_iters=1,repeat_iters=1,cold_l2_cache=False,use_cuda_graph=False)
        self.assertEqual(benchmark.last_activity['activity'],raw)
        self.assertIsNone(benchmark.non_target_dispatches)
        self.assertIsNone(benchmark.resolution_us)


class MacaProfileRepresentation(unittest.TestCase):
    def setUp(self):
        from dataclasses import asdict
        from open_cake_ir.evaluation.core import TensorLaunchManifest
        from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.workloads import create_task
        from open_cake_ir.evaluation.metax_observations import NOT_COLLECTED
        doc,_=create_task('rmsnorm',backend='triton-metax',rows=8,columns=128)
        self.manifest=TensorLaunchManifest.for_workload(WorkloadContract(doc),'primary',
            target='xcore1002',kernel_name='cake',grid=[8,1,1],block=[64,1,1],
            dynamic_shared_memory_bytes=0,hidden_null_pointer_parameters=0)
        admission=MetaxDeviceAdmission('maca-123456789abc','xcore1002','xcore1002',
            'MetaX C550',64,'0000:0f:00','/opt/maca-3.5.3/lib/libmcruntime.so')
        self.raw={'activity':capture(kernel('cake',1,1000)),'manifest':self.manifest.as_dict(),
                  'device_admission':asdict(admission),'not_collected':list(NOT_COLLECTED)}
        self.correctness={'correctness_launches':2,'instrumented':{'passed':True,'metrics':{
            'output_mismatches':0,'inputs_unchanged':True,'max_abs_error':0.0}}}

    def test_one_actual_dispatch_projects_resources_and_states_missing_counters(self):
        from open_cake_ir.evaluation.metax_observations import maca_profile_summary
        result=maca_profile_summary(self.raw)
        self.assertEqual(result['device_time_us'],2.048)
        self.assertEqual(result['registers_per_thread'],16)
        self.assertEqual(result['pci_bus_id'],'0000:0f:00')
        self.assertEqual(result['timing_use'],'attribution_only')
        self.assertIn('bandwidth',result['not_collected'])

    def test_manifest_geometry_and_native_device_identity_are_checked(self):
        from open_cake_ir.evaluation.metax_observations import maca_profile_summary
        for field,value in (('target','xcore1000'),('device_arch','xcore1000'),
                            ('device_name','NVIDIA B200'),('warp_size',32),('pci_bus_id','')):
            raw=deepcopy(self.raw);raw['device_admission'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):maca_profile_summary(raw)
        raw=deepcopy(self.raw);raw['activity']['records'][0]['grid']=[9,1,1]
        with self.assertRaises(ValueError):maca_profile_summary(raw)

    def test_loaded_function_resources_are_not_replaced_with_a_different_profile(self):
        from open_cake_ir.evaluation.metax_observations import MACA_PROFILE,maca_profile_summary
        profile={'job_id':'maca-123456789abc','gpu_uuid':None,'candidate_sha256':'a'*64,
                 'raw':self.raw,'summary':maca_profile_summary(self.raw)}
        launch={'job_id':profile['job_id'],'gpu_uuid':None,
                'candidate_sha256':'a'*64,'manifest_sha256':self.manifest.canonical_sha256,
                'device_admission':self.raw['device_admission'],
                'correctness_launches':2,
                'resources':{'registers_per_thread':16,'local_bytes':0,'dynamic_shared_bytes':0}}
        MACA_PROFILE.validate_launch(profile,launch,self.correctness)
        launch['resources']['registers_per_thread']=32
        with self.assertRaisesRegex(ValueError,'resources'):MACA_PROFILE.validate_launch(profile,launch,self.correctness)

    def test_recomputed_profile_cannot_change_the_loaded_manifest_or_device(self):
        import json
        from open_cake_ir.evaluation.metax_observations import MACA_PROFILE, MCPTI_PROFILE_KIND
        profile={'kind':MCPTI_PROFILE_KIND,'candidate_sha256':'a'*64,'case_id':'primary',
            'kernel_name':'cake','job_id':'maca-123456789abc','gpu_uuid':None,
            'allocation_mode':'local_serialized','external_gpu_activity':'not_excluded',
            'separate_instrumented_launch':True,
            'evaluation_protocol':{'case_id':'primary','attribution_evaluation':'correctness_then_profile'}}
        launch={'job_id':profile['job_id'],'gpu_uuid':None,'candidate_sha256':'a'*64,
                'manifest_sha256':self.manifest.canonical_sha256,
                'correctness_launches':2,
                'device_admission':deepcopy(self.raw['device_admission']),
                'resources':{'registers_per_thread':16,'local_bytes':0,'dynamic_shared_bytes':0}}
        mutations = (
            ('manifest', 'workload_sha256', 'b'*64),
            ('manifest', 'hidden_null_pointer_parameters', 2),
            ('manifest', 'dynamic_shared_memory_bytes', 32768),
            ('device_admission', 'pci_bus_id', '0000:10:00'),
            ('device_admission', 'runtime_library', '/other/libmcruntime.so'),
        )
        raws = []
        for section, key, value in mutations:
            raw=deepcopy(self.raw);raw[section][key]=value
            raws.append((key,raw))
        raw=deepcopy(self.raw);raw['manifest']['tensor_abi'][0]['shape'][0]=16
        raws.append(('tensor shape',raw))
        for label,raw in raws:
            with self.subTest(field=label),self.assertRaises(ValueError):
                document={**profile,'raw':raw,'summary':MACA_PROFILE.summary(raw)}
                loaded=MACA_PROFILE.load(json.dumps(document).encode(),
                    expected_candidate_sha256='a'*64,expected_case_id='primary')
                MACA_PROFILE.validate_launch(loaded,launch,self.correctness)

    def test_registered_profile_reaches_receipt_feedback_and_requires_instrumented_correctness(self):
        from hashlib import sha256
        from open_cake_ir.evaluation.core import EvaluationReceipt
        from open_cake_ir.evaluation.metax_observations import MACA_PROFILE
        from open_cake_ir.serialization import canonical_json_bytes as encoded
        policy={'case_id':'primary','attribution_evaluation':'correctness_then_profile'}
        profile={'kind':MACA_PROFILE.kind,'candidate_sha256':'a'*64,'case_id':'primary',
            'kernel_name':'cake','job_id':'maca-123456789abc','gpu_uuid':None,
            'allocation_mode':'local_serialized','external_gpu_activity':'not_excluded',
            'separate_instrumented_launch':True,'evaluation_protocol':policy,
            'raw':self.raw,'summary':MACA_PROFILE.summary(self.raw)}
        launch={'job_id':profile['job_id'],'gpu_uuid':None,'candidate_sha256':'a'*64,
            'manifest_sha256':self.manifest.canonical_sha256,'device_admission':self.raw['device_admission'],
            'correctness_launches':2,'resources':{'registers_per_thread':16,'local_bytes':0,'dynamic_shared_bytes':0}}
        metrics={'output_mismatches':0,'inputs_unchanged':True,'max_abs_error':0.0}
        correctness={**self.correctness,'passed':True,'metrics':metrics}
        def receipt():
            payloads={'profile':encoded(profile),'launch_receipt':encoded(launch),
                      'correctness_output':encoded(correctness)}
            return EvaluationReceipt('a'*64,self.manifest.workload_sha256,sha256(encoded(policy)).hexdigest(),
                'attribution','primary',True,metrics,1,0,sha256(payloads['launch_receipt']).hexdigest(),
                None,artifact_payloads=payloads)
        self.assertEqual(receipt().attribution_feedback['kind'],'maca_dispatch_attribution')
        correctness.pop('instrumented')
        with self.assertRaisesRegex(ValueError,'instrumented output'):receipt()
