"""The execution path follows the Target's declared object, and no vendor is the else.

The worker selected Metal by the type of its launch manifest and reached CUDA for
everything else, so a candidate built for a target CUDA never names -- an AMDGCN one --
arrived at `observe_exclusive_cuda` and was refused by nvidia-smi rather than by the
class that owns it. These hold the selection to the declared code object, and hold the
manifest and that declaration to each other.

No device is touched here and none of this is evidence that any platform can execute.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.tasks import evaluate as worker


def _authority(target: str, *, metal_manifest: bool = False):
    manifest = object.__new__(MetalTensorLaunchManifest) if metal_manifest else SimpleNamespace()
    return SimpleNamespace(candidate=SimpleNamespace(target=target), manifest=manifest)


class ExecutionPlatformSelection(unittest.TestCase):
    def test_each_declared_object_selects_its_own_platform(self) -> None:
        for target, expected in (("sm_100a", "cubin"), ("sm_103a", "cubin"),
                                 ("gfx938", "hsaco"), ("gfx1151", "hsaco")):
            with self.subTest(target=target):
                self.assertEqual(worker._execution_platform(_authority(target)), expected)
        for target in ("apple_gpu_family7", "apple_gpu_family8", "apple_gpu_family9"):
            with self.subTest(target=target):
                self.assertEqual(
                    worker._execution_platform(_authority(target, metal_manifest=True)),
                    "metal_binary_archive")

    def test_a_manifest_and_a_declaration_that_disagree_are_refused(self) -> None:
        for target, metal_manifest in (("sm_100a", True), ("apple_gpu_family7", False),
                                       ("gfx938", True)):
            with self.subTest(target=target, metal_manifest=metal_manifest):
                with self.assertRaisesRegex(ValueError, "declared execution platform differ"):
                    worker._execution_platform(_authority(target, metal_manifest=metal_manifest))

    def test_an_unimplemented_platform_is_named_rather_than_sent_to_cuda(self) -> None:
        """The measured failure this replaces: an hsaco candidate reached nvidia-smi."""
        result: dict[str, object] = {}
        with patch.object(worker, "observe_exclusive_cuda") as admission, \
             patch.object(worker, "_evaluate_metal_candidate") as metal:
            with self.assertRaisesRegex(ValueError, "no execution platform implements 'hsaco'"):
                worker._evaluate_candidate(_authority("gfx938"), result, collect_timing=True)
            admission.assert_not_called()
            metal.assert_not_called()

    def test_a_metal_candidate_still_reaches_the_metal_path(self) -> None:
        result: dict[str, object] = {}
        authority = _authority("apple_gpu_family7", metal_manifest=True)
        with patch.object(worker, "observe_exclusive_cuda") as admission, \
             patch.object(worker, "_evaluate_metal_candidate") as metal:
            worker._evaluate_candidate(authority, result, collect_timing=True)
            metal.assert_called_once_with(authority, result)
            admission.assert_not_called()


if __name__ == "__main__":
    unittest.main()
