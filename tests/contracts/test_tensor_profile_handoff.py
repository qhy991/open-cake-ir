"""A tensor profile format owns its raw validation and reaches the receipt consumer."""
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.attribution import TensorProfileFormat, load_instrumented_profile
from open_cake_ir.evaluation.core import EvaluationReceipt
from open_cake_ir.evaluation.hip_observations import HIP_PROFILE
from open_cake_ir.serialization import canonical_json_bytes


class TensorProfileHandoffTests(unittest.TestCase):
    def setUp(self):
        self.kind='synthetic_tensor_activity_v1'
        self.policy={'case_id':'primary','attribution_evaluation':'correctness_then_profile'}
        self.policy_id=sha256(canonical_json_bytes(self.policy)).hexdigest()
        self.launch={'job_id':'synthetic-123456789abc','gpu_uuid':None,'candidate_sha256':'a'*64,'observed_device':'fixture'}
        self.metrics={'output_mismatches':0,'max_abs_error':0.0,'inputs_unchanged':True}
        self.profile=dict(kind=self.kind,candidate_sha256='a'*64,case_id='primary',kernel_name='kernel',
            job_id=self.launch['job_id'],allocation_mode='local_serialized',external_gpu_activity='not_excluded',
            separate_instrumented_launch=True,evaluation_protocol=self.policy,raw={'value':7},summary={'value':7})
        def summary(raw):
            if raw!={'value':7}:raise ValueError('synthetic raw differs')
            return dict(raw)
        def load(payload,**kwargs):
            return load_instrumented_profile(payload,kind=self.kind,job_prefix='synthetic',
                label='synthetic',summary=summary,raw_name='record',**kwargs)
        def validate_launch(profile,launch,correctness):
            if launch.get('observed_device')!='fixture' or profile['job_id']!=launch['job_id']:
                raise ValueError('synthetic launch differs')
            self.assertEqual(correctness['metrics'],self.metrics)
        self.source=TensorProfileFormat(self.kind,summary,load,
            lambda profile:{'kind':'synthetic_feedback','value':profile['summary']['value']},validate_launch)

    def receipt(self):
        launch=canonical_json_bytes(self.launch)
        artifacts={'profile':canonical_json_bytes(self.profile),'launch_receipt':launch,
                   'correctness_output':canonical_json_bytes({'passed':True,'metrics':self.metrics})}
        return EvaluationReceipt(candidate_sha256='a'*64,workload_sha256='b'*64,
            evaluation_protocol_sha256=self.policy_id,purpose='attribution',case_id='primary',
            correctness_passed=True,correctness=self.metrics,kernel_calls=1,fallback_calls=0,
            launch_receipt_sha256=sha256(launch).hexdigest(),timing=None,artifact_payloads=artifacts)

    def test_new_typed_registration_reaches_validation_and_next_turn_feedback(self):
        with patch('open_cake_ir.evaluation.core.TENSOR_PROFILES',{self.kind:self.source}):
            receipt=self.receipt()
            self.assertEqual(receipt.attribution_feedback,{'kind':'synthetic_feedback','value':7})
            self.profile['raw']={'value':8}
            with self.assertRaisesRegex(ValueError,'synthetic raw'):self.receipt()

    def test_platform_launch_binding_cannot_be_skipped_by_the_shared_reader(self):
        with patch('open_cake_ir.evaluation.core.TENSOR_PROFILES',{self.kind:self.source}):
            self.launch['observed_device']='another device'
            with self.assertRaisesRegex(ValueError,'synthetic launch'):self.receipt()

    def test_unknown_kinds_are_not_interpreted_as_nvidia_or_hip(self):
        with self.assertRaisesRegex(ValueError,'no attribution source'):self.receipt()

    def test_nonobject_profiles_are_shape_refusals_before_source_dispatch(self):
        for value in (None, [], 7):
            self.profile = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'profile must be an object'):
                self.receipt()

    def test_collector_and_format_must_arrive_together_before_device_work(self):
        from open_cake_ir.tasks.evaluate import _evaluate_tile_candidate
        for kwargs in ({'profile_source':lambda *args:{}},{'profile_format':self.source}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'bound together'):
                _evaluate_tile_candidate(None,{},None,None,False,route_calls_per_cohort=None,**kwargs)

    def test_existing_hip_launch_identity_rule_still_owns_its_refusal(self):
        with self.assertRaisesRegex(ValueError,'HIP attribution launch'):
            HIP_PROFILE.validate_launch({'job_id':'hip-123456789abc','gpu_uuid':'unavailable'},
                                        {'job_id':'hip-123456789abd','gpu_uuid':'unavailable'}, {})
