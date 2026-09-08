"""Local job admission does not borrow CUDA exclusive-card recovery semantics."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.attempts import BrokerAttempt, is_resubmittable_admission_failure, valid_job_mode
from open_cake_ir.evaluation.local_broker import _acquire, observe_local_metal_job
from open_cake_ir.lab.runtime import _BROKER_JOB_OBSERVATION


class LocalMetalBrokerTests(unittest.TestCase):
    def test_namespaces_preserve_real_allocation_domain(self):
        self.assertTrue(valid_job_mode("metal-123456789abc", "local_serialized"))
        self.assertTrue(valid_job_mode("gpuq-123456789abc", "exclusive"))
        for job, mode in (("metal-123456789abc", "exclusive"), ("gpuq-123456789abc", "local_serialized"), ("metal-000000000000", "local_serialized")):
            self.assertFalse(valid_job_mode(job, mode))
        self.assertEqual(_BROKER_JOB_OBSERVATION.findall(b"[metal-run] accepted job metal-123456789abc\n"), [b"metal-123456789abc"])
        self.assertEqual(_BROKER_JOB_OBSERVATION.findall(b"[gpu-run] accepted job metal-123456789abc\n"), [])

    def test_metal_never_borrows_the_cuda_zero_work_retry(self):
        common = dict(candidate_sha256="a" * 64, manifest_sha256="b" * 64, policy_sha256="c" * 64,
            evaluator_arguments_sha256="d" * 64, admitted=False, error="gpu_admission_differs",
            compiler_invocations=0, module_loads=0, preflight_calls=0, kernel_calls=0, timing_samples=0, fallback_calls=0, receipt=None)
        self.assertFalse(is_resubmittable_admission_failure(BrokerAttempt(job_id="metal-123456789abc", mode="local_serialized", **common)))
        self.assertTrue(is_resubmittable_admission_failure(BrokerAttempt(job_id="gpuq-123456789abc", mode="exclusive", **common)))

    def test_live_lock_descriptor_is_required_and_a_second_job_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metal.lock"
            fd = _acquire(path)
            try:
                with self.assertRaises(BlockingIOError):
                    _acquire(path)
                with patch("open_cake_ir.evaluation.local_broker._lock_path", return_value=path), \
                     patch.dict(os.environ, METAL_JOB_ID="metal-123456789abc", METAL_BROKER_LOCK_FD=str(fd)):
                    self.assertEqual(observe_local_metal_job(), "metal-123456789abc")
                with patch.dict(os.environ, METAL_JOB_ID="metal-123456789abc", METAL_BROKER_LOCK_FD="0"):
                    with self.assertRaises(ValueError):
                        observe_local_metal_job()
            finally:
                os.close(fd)
            next_fd = _acquire(path)
            os.close(next_fd)


if __name__ == "__main__":
    unittest.main()
