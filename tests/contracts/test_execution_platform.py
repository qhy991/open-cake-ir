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


class EveryDeclaredObjectIsARow(unittest.TestCase):
    """A platform is a row, so an eighth Target cannot land without one."""

    def _declared_objects(self) -> set[str]:
        import json
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        return {json.loads(path.read_text(encoding="utf-8"))["code_object"]
                for path in (root / "compiler/targets").glob("*.json")}

    def test_the_registry_covers_every_object_a_target_declares(self) -> None:
        declared = self._declared_objects()
        self.assertTrue(declared)
        self.assertEqual(declared - set(worker._PLATFORMS), set(),
                         "a declared code object with no execution platform row")

    def test_an_object_no_platform_implements_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "no execution platform implements 'hsaco'"):
            worker._platform(_authority("gfx938"))

    def test_attribution_is_not_taken_on_another_platforms_behalf(self) -> None:
        """The measured failure: a profile request reached CUDA's device admission.

        `hsaco` has no attribution source, and the row says so rather than leaving the
        request to fall through to whichever branch follows.
        """
        self.assertIsNone(worker._PLATFORMS["hsaco"].attribution)
        self.assertIsNone(worker._PLATFORMS["hsaco"].evaluate)
        self.assertFalse(worker._PLATFORMS["hsaco"].profiled_child)

    def test_each_implemented_platform_states_where_its_profile_comes_from(self) -> None:
        self.assertEqual(worker._PLATFORMS["cubin"].attribution, "separate")
        self.assertEqual(
            worker._PLATFORMS["metal_binary_archive"].attribution, "inside_evaluate")
        self.assertTrue(worker._PLATFORMS["cubin"].profiled_child)
        self.assertFalse(worker._PLATFORMS["metal_binary_archive"].profiled_child)


if __name__ == "__main__":
    unittest.main()
