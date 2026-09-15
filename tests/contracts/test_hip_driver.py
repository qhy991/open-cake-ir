"""What the AMDGCN launch path refuses before it reaches a device.

`LoadedHipModuleCandidate` needs a ROCm runtime and one visible DCU, so the launch itself
is exercised on hardware and retained as evidence. Host-testable here is every authority
check it owns -- the ones that decide whether these exact bytes may be retained as a
loaded module at all -- plus the dispatch that sends an AMDGCN candidate to this driver
instead of the CUDA one.
"""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.hip_driver import (
    HipLifecycleError,
    LoadedHipModuleCandidate,
    load_hip_runtime,
)


class RuntimeResolutionTest(unittest.TestCase):
    """No soname is named; the runtime the admitted torch loaded is the runtime."""

    def test_a_process_without_the_hip_runtime_is_refused_by_symbol(self) -> None:
        empty = SimpleNamespace()
        with self.assertRaises(RuntimeError) as raised:
            load_hip_runtime(empty)
        message = str(raised.exception)
        for symbol in ("hipModuleLoadData", "hipModuleLaunchKernel", "hipDeviceSynchronize"):
            self.assertIn(symbol, message)
        # The refusal names the missing symbols, never a vendor's library file: a Hygon
        # DTK host loads libgalaxyhip and a ROCm host libamdhip64, and enumerating those
        # here would be one more per-vendor list in shared code.
        for soname in ("libgalaxyhip", "libamdhip64", ".so"):
            self.assertNotIn(soname, message)

    def test_a_process_holding_every_symbol_is_admitted(self) -> None:
        loaded = SimpleNamespace(**{name: (lambda *a: 0) for name in (
            "hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel",
            "hipModuleUnload", "hipDeviceSynchronize")})
        self.assertIs(load_hip_runtime(loaded), loaded)


class LoadAuthorityTest(unittest.TestCase):
    """Every immutable relation is checked before a module is retained."""

    HSACO = b"\x7fELF" + b"\x00" * 60

    def manifest(self, **overrides):
        fields = {"target": "gfx938", "kernel_name": "cake_rmsnorm",
                  "canonical_sha256": "c" * 64, "grid": (128, 1, 1), "block": (64, 1, 1),
                  "dynamic_shared_memory_bytes": 0, "hidden_null_pointer_parameters": 1,
                  "tensor_abi": ()}
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def candidate(self, **overrides):
        fields = {"target": "gfx938", "entry_point": "cake_rmsnorm",
                  "launch_spec_sha256": "c" * 64,
                  "artifact_roles": {"hsaco": sha256(self.HSACO).hexdigest(),
                                     "amdgcn": "d" * 64},
                  "artifact_payloads": {"amdgcn": b".amdhsa_kernel k\n"}}
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def load(self, candidate, manifest, device_arch="gfx938:sramecc+:xnack-"):
        api = SimpleNamespace(**{name: (lambda *a: 0) for name in (
            "hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel",
            "hipModuleUnload", "hipDeviceSynchronize")})
        return LoadedHipModuleCandidate.load(candidate, self.HSACO, manifest,
                                             device_arch, api=api)

    def test_a_payload_that_is_not_an_elf_object_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "launch authority differs"):
            LoadedHipModuleCandidate.load(self.candidate(), b"not-an-elf", self.manifest(),
                                          "gfx938")

    def test_every_broken_identity_relation_is_refused(self) -> None:
        cases = {
            "target disagrees with the manifest":
                (self.candidate(target="gfx1151"), self.manifest()),
            "entry point disagrees with the kernel name":
                (self.candidate(entry_point="other"), self.manifest()),
            "launch spec digest disagrees":
                (self.candidate(launch_spec_sha256="e" * 64), self.manifest()),
            "sealed hsaco digest disagrees with these bytes":
                (self.candidate(artifact_roles={"hsaco": "f" * 64, "amdgcn": "d" * 64}),
                 self.manifest()),
        }
        for label, (candidate, manifest) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "launch authority differs"):
                    self.load(candidate, manifest)

    def test_a_device_that_is_not_the_built_target_is_refused_by_name(self) -> None:
        for observed in ("gfx1151", "gfx942:xnack-", "gfx9380"):
            with self.subTest(observed=observed):
                with self.assertRaises(ValueError) as raised:
                    self.load(self.candidate(), self.manifest(), observed)
                self.assertIn(observed, str(raised.exception))
                self.assertIn("gfx938", str(raised.exception))

    def test_the_declared_feature_suffix_of_the_built_target_is_admitted(self) -> None:
        with patch("open_cake_ir.evaluation.triton_hip.amdgcn_resource_record",
                   return_value={"registers_per_thread": 34}):
            loaded = self.load(self.candidate(), self.manifest())
        self.assertEqual(loaded.resources, {"registers_per_thread": 34})
        self.assertEqual(loaded.launch_calls, 0)

    def test_a_candidate_without_its_assembly_cannot_report_resources(self) -> None:
        """Resources come from the sealed assembly, not from a driver query.

        `hipFuncGetAttribute`'s enumeration differs between HIP versions and nothing here
        has measured it; the `.amdgpu_metadata` the candidate already seals is the owner.
        """
        with self.assertRaisesRegex(ValueError, "seals its assembly"):
            self.load(self.candidate(artifact_payloads={}), self.manifest())


class TeardownTest(unittest.TestCase):
    """A failure during load never leaves a module loaded, and never hides teardown."""

    def test_a_failed_function_lookup_reports_both_failures(self) -> None:
        def unload_fails(*_arguments):
            return 7

        api = SimpleNamespace(
            hipModuleLoadData=lambda *a: 0,
            hipModuleGetFunction=lambda *a: 3,
            hipModuleLaunchKernel=lambda *a: 0,
            hipModuleUnload=unload_fails,
            hipDeviceSynchronize=lambda *a: 0,
        )
        candidate = SimpleNamespace(
            target="gfx938", entry_point="k", launch_spec_sha256="c" * 64,
            artifact_roles={"hsaco": sha256(b"\x7fELF").hexdigest()},
            artifact_payloads={"amdgcn": b".amdhsa_kernel k\n"})
        manifest = SimpleNamespace(target="gfx938", kernel_name="k",
                                   canonical_sha256="c" * 64)
        with patch("open_cake_ir.evaluation.triton_hip.amdgcn_resource_record",
                   return_value={}):
            with self.assertRaises(HipLifecycleError) as raised:
                LoadedHipModuleCandidate.load(candidate, b"\x7fELF", manifest,
                                              "gfx938", api=api)
        self.assertIn("hipModuleGetFunction", str(raised.exception.primary))
        self.assertIn("hipModuleUnload", str(raised.exception.teardown))


class DispatchTest(unittest.TestCase):
    """The tensor-tile path chooses its driver by the executable the target builds."""

    def test_each_declared_target_selects_its_own_driver(self) -> None:
        from open_cake_ir.evaluation.artifacts import executable_role
        self.assertEqual(executable_role("gfx938"), "hsaco")
        self.assertEqual(executable_role("gfx1151"), "hsaco")
        self.assertEqual(executable_role("sm_103a"), "cubin")
        self.assertEqual(executable_role("apple_gpu_family8"), "metal_binary_archive")


class AdmissionRequirementsTest(unittest.TestCase):
    """The device contract an evaluation admits against comes from the target's route."""

    def test_each_amdgcn_target_states_its_own_isa_and_lane_width(self) -> None:
        from open_cake_ir.evaluation.triton_hip import hip_admission_requirements
        self.assertEqual(hip_admission_requirements("gfx938"), {
            "target": "gfx938", "binary_role": "hsaco", "assembly_role": "amdgcn",
            "triton_target": {"backend": "hip", "arch": "gfx938", "warp_size": 64}})
        # Same code object, different vendor, different wavefront. Reading the route is
        # what keeps these two apart without a second table to maintain.
        self.assertEqual(
            hip_admission_requirements("gfx1151")["triton_target"]["warp_size"], 32)

    def test_a_target_that_does_not_lower_through_hip_is_refused_by_name(self) -> None:
        from open_cake_ir.evaluation.triton_hip import hip_admission_requirements
        for target in ("sm_100a", "sm_103a"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError) as raised:
                    hip_admission_requirements(target)
                self.assertIn(target, str(raised.exception))

    def test_it_is_the_contract_admit_exact_hip_checks(self) -> None:
        """Not a second copy of the build's requirements -- the same fields it validates."""
        from open_cake_ir.evaluation.triton_hip import (
            admit_exact_hip, hip_admission_requirements)
        requirements = hip_admission_requirements("gfx938")
        # Reaches the runtime check, which means every field-shape check above it passed;
        # this host has no ROCm PyTorch, so that is where it stops.
        with self.assertRaises((RuntimeError, ModuleNotFoundError)):
            admit_exact_hip(requirements)
        for missing in ("binary_role", "assembly_role", "target"):
            with self.subTest(missing=missing):
                broken = {k: v for k, v in requirements.items() if k != missing}
                with self.assertRaisesRegex(ValueError, "HIP lowering Target"):
                    admit_exact_hip(broken)
        # The Target block is refused by its own name, before the fields that read it.
        with self.assertRaisesRegex(ValueError, "triton_target"):
            admit_exact_hip({k: v for k, v in requirements.items()
                             if k != "triton_target"})


class WorkerDispatchTest(unittest.TestCase):
    """What the worker does with a request for a target that is not CUDA's."""

    def test_attribution_refuses_a_target_nsight_compute_cannot_profile(self) -> None:
        """This branch was reached by not being Metal, which sent a DCU to ncu."""
        import inspect
        from open_cake_ir.tasks import evaluate
        source = inspect.getsource(evaluate.main)
        self.assertIn('executable_role(authority.candidate.target) != "cubin"', source)
        self.assertIn("Nsight Compute", source)

    def test_timing_is_the_studys_statement_not_the_workers_guess(self) -> None:
        """`collect_timing=True` was unconditional, so every search evaluation timed.

        A Study for a target with no named timer carries a measurement-coverage
        limitation instead of a paired assay, and the worker reads that rather than
        deciding from the target -- which would put the timing question back on the
        code-object axis it does not belong to.
        """
        import inspect
        from open_cake_ir.tasks import evaluate
        self.assertIn("collect_timing=authority.timed_assay_available",
                      inspect.getsource(evaluate.main))
        self.assertIn("timed_assay_available: bool = True",
                      inspect.getsource(evaluate._Authority))

    def test_a_timed_hip_evaluation_is_refused_rather_than_silently_untimed(self) -> None:
        from open_cake_ir.tasks import evaluate
        authority = SimpleNamespace(candidate=SimpleNamespace(target="gfx938"))
        with self.assertRaisesRegex(ValueError, "no timing source"):
            evaluate._evaluate_hip_candidate(authority, {}, collect_timing=True)


if __name__ == "__main__":
    unittest.main()
