"""The test driver admits a live broker lease, not a replayed receipt or invented ID."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
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
