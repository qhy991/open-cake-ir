"""The test driver admits a live broker lease, not a replayed receipt or invented ID."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fib_development', ROOT/'tools/test_flashinfer_b300.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DevelopmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {'schema': 'gpuq.admission-receipt.v1', 'job_id': 'gpuq-fixture',
                        'mode': 'exclusive', 'gpu_count': 1, 'gpu_ids': [2], 'cwd': str(ROOT),
                        'argv_count': len(sys.orig_argv),
                        'argv_sha256': sha256(json.dumps(sys.orig_argv,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                        'resolved_executable': str(Path(sys.executable).resolve()),
                        'broker_instance_id': 'fixture-instance'}
        self.snapshot = {'probe_error': None, 'instance_id': 'fixture-instance',
                         'running': [{'job_id':'gpuq-fixture','state':'running','mode':'exclusive','gpu_ids':[2]}]}

    def admit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json';path.write_text(json.dumps(self.receipt))
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'2'}), \
                 patch.object(driver,'_broker_request', side_effect=[
                     {'ok':True,'receipt':dict(self.receipt)}, {'snapshot':self.snapshot}]), \
                 patch.object(driver,'_observe_cuda_device', return_value='admitted') as observe:
                result = driver._admit_receipt(Path('/socket'),path)
                observe.assert_called_once()
                return result

    def test_live_bound_receipt_reaches_exact_device_observation(self):
        self.assertEqual(self.admit(),'admitted')

    def test_finished_or_missing_job_cannot_reuse_its_receipt(self):
        self.snapshot['running'] = []
        with self.assertRaisesRegex(ValueError,'not live'):self.admit()

    def test_other_command_cannot_reuse_a_live_lease(self):
        self.receipt['argv_count'] += 1
        with self.assertRaisesRegex(ValueError,'command differs'):self.admit()

    def test_shared_allocation_is_refused(self):
        self.receipt['mode'] = 'shared'
        with self.assertRaisesRegex(ValueError,'allocation differs'):self.admit()

    def test_replaced_broker_is_refused(self):
        self.snapshot['instance_id'] = 'another-instance'
        with self.assertRaisesRegex(ValueError,'not live'):self.admit()


class AuthoredPlanAdmissionTests(unittest.TestCase):
    def test_task_plan_admits_but_wrong_target_or_public_outputs_do_not(self):
        from open_cake_ir.tasks.solx_fib import attention
        from open_cake_ir.evaluation.workload import WorkloadContract
        workload = WorkloadContract(attention.workload_document(
            'fib_gqa_paged_decode_h32_kv4_d128_ps1', variant='boundary'))
        plan = attention.launch_plan(workload)
        driver.admit_plan_workload(plan, workload)
        for altered in (replace(plan, target='sm_100a'), replace(plan, outputs=('output',))):
            with self.subTest(plan=altered.program_id), self.assertRaisesRegex(ValueError, 'target or public'):
                driver.admit_plan_workload(altered, workload)
        tensors = dict(plan.tensors)
        tensors['q'] = replace(tensors['q'], shape=(1,))
        with self.assertRaisesRegex(ValueError, 'ABI differs'):
            driver.admit_plan_workload(replace(plan, tensors=tensors), workload)

    def test_probe_inputs_are_explicit_and_bounded(self):
        from tools.test_flashinfer_infra import probe_arguments, selected_tasks
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = '024_rmsnorm_h2048'
            (root/'tasks.json').write_text(json.dumps({'tasks':[task]}))
            self.assertEqual(selected_tasks(root/'tasks.json'), [task])
            self.assertEqual(probe_arguments(root,[task]), [])
            (root/'probe.json').write_text(json.dumps({'kind':'source','rows':7}))
            self.assertIn('--candidate-source', probe_arguments(root,[task]))
            for config in ({'kind':'source','rows':0}, {'kind':'source','rows':True},
                           {'kind':'plan','variant':'other'}, {'kind':'source','path':'../a.py'}):
                (root/'probe.json').write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    probe_arguments(root,[task])
