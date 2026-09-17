"""The device/allocation identity each declared paired assay implies.

This branch used to be `if Metal ... elif <everything else>`, so an AMDGCN pair went down
CUDA's and was refused for not being a cubin. The replacement checks each declared kind by
name -- and shipped untested, on the path every AMD campaign's evidence must pass.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from open_cake_ir.evaluation.paired import (  # noqa: E402
    PAIRED_HIP_KIND, PAIRED_KIND, PAIRED_METAL_KIND, admit_device_identity,
)


def _participants(target):
    return {"candidate": {"target": target}, "baseline": {"target": target}}


class AmdgcnIdentityTests(unittest.TestCase):
    TARGET = "gfx1151"

    def _records(self, **overrides):
        raw = {"kind": PAIRED_HIP_KIND, "job_id": "hip-0123456789ab",
               "gpu_uuid": "not_reported_by_this_runtime"}
        raw.update(overrides.pop("raw", {}))
        launch = {"gpu_uuid": raw["gpu_uuid"]}
        launch.update(overrides.pop("launch", {}))
        return raw, launch

    def test_an_amdgcn_pair_is_admitted_on_its_own_terms(self):
        raw, launch = self._records()
        admit_device_identity(raw, launch, _participants(self.TARGET))

    def test_the_runtime_that_reports_no_uuid_is_admitted_saying_so(self):
        """`observe_local_hip` records the absence in words rather than inventing an id.
        The check is that both records agree on what was said."""

        raw, launch = self._records()
        self.assertEqual(raw["gpu_uuid"], "not_reported_by_this_runtime")
        admit_device_identity(raw, launch, _participants(self.TARGET))
        raw["gpu_uuid"] = ""
        with self.assertRaisesRegex(ValueError, "AMDGCN"):
            admit_device_identity(raw, {"gpu_uuid": ""}, _participants(self.TARGET))

    def test_the_two_records_must_agree_on_the_device_id(self):
        raw, launch = self._records()
        launch["gpu_uuid"] = "something-else"
        with self.assertRaisesRegex(ValueError, "AMDGCN"):
            admit_device_identity(raw, launch, _participants(self.TARGET))

    def test_a_cluster_job_is_not_an_amdgcn_allocation(self):
        raw, launch = self._records(raw={"job_id": "gpuq-0123456789ab"})
        with self.assertRaisesRegex(ValueError, "AMDGCN"):
            admit_device_identity(raw, launch, _participants(self.TARGET))

    def test_the_worker_placeholder_job_is_refused(self):
        raw, launch = self._records(raw={"job_id": "hip-000000000000"})
        with self.assertRaisesRegex(ValueError, "AMDGCN"):
            admit_device_identity(raw, launch, _participants(self.TARGET))

    def test_a_target_whose_executable_is_not_an_hsaco_is_refused(self):
        raw, launch = self._records()
        with self.assertRaisesRegex(ValueError, "AMDGCN"):
            admit_device_identity(raw, launch, _participants("sm_103a"))


class CudaIdentityKeepsItsOwnTermsTests(unittest.TestCase):
    def test_a_cuda_pair_is_admitted_and_an_amdgcn_job_under_it_is_not(self):
        participants = _participants("sm_103a")
        raw = {"kind": PAIRED_KIND, "job_id": "gpuq-0123456789ab", "gpu_uuid": "GPU-abc"}
        admit_device_identity(raw, {"gpu_uuid": "GPU-abc"}, participants)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            admit_device_identity({**raw, "job_id": "hip-0123456789ab"},
                                  {"gpu_uuid": "GPU-abc"}, participants)

    def test_an_amdgcn_target_is_not_admitted_under_the_cuda_assay(self):
        """The refusal that used to be everything else's: now it names CUDA, and only a
        record declaring the CUDA assay reaches it."""

        raw = {"kind": PAIRED_KIND, "job_id": "gpuq-0123456789ab", "gpu_uuid": "GPU-abc"}
        with self.assertRaisesRegex(ValueError, "CUDA"):
            admit_device_identity(raw, {"gpu_uuid": "GPU-abc"}, _participants("gfx1151"))


class MetalIdentityKeepsItsOwnTermsTests(unittest.TestCase):
    """Metal's branch was already covered indirectly; named here so the file speaks for
    every declared kind, which the commit that split them claimed and did not do."""

    def test_a_metal_record_is_judged_by_metal_s_terms(self):
        """Reaching the branch, not `validate_host`'s refusal.

        Written first without a `host`, this passed on
        `Metal observation host identity differs` raised one call earlier -- the branch it
        names never ran. A block on loan, which is the thing this file exists to stop.
        """

        from open_cake_ir.evaluation import metal_observations

        host = {"device_registry_id": "apple-m2", "target": "apple_gpu_family8"}
        raw = {"kind": PAIRED_METAL_KIND, "job_id": "hip-0123456789ab", "host": host,
               "gpu_uuid": "u", "allocation_mode": "local_serialized",
               "external_gpu_activity": "not_excluded",
               "device_registry_id": host["device_registry_id"]}
        launch = {"gpu_uuid": "u", "allocation_mode": "local_serialized",
                  "external_gpu_activity": "not_excluded", "host": host}
        with mock.patch.object(metal_observations, "validate_host", return_value=host):
            # A metal- job is what this assay allocates; the hip- one above is not.
            with self.assertRaisesRegex(ValueError, "Metal"):
                admit_device_identity(raw, launch, _participants("apple_gpu_family8"))
            admit_device_identity({**raw, "job_id": "metal-0123456789ab"}, launch,
                                  _participants("apple_gpu_family8"))


class UnknownKindTests(unittest.TestCase):
    def test_a_kind_with_no_branch_is_refused_saying_that(self):
        """Not the same statement as the evidence being wrong."""

        raw = {"kind": "fixed_baseline_paired_something_v1", "job_id": "x-0123456789ab",
               "gpu_uuid": "u"}
        with self.assertRaisesRegex(ValueError, "no declared device/host identity check"):
            admit_device_identity(raw, {"gpu_uuid": "u"}, _participants("gfx1151"))


if __name__ == "__main__":
    unittest.main()
