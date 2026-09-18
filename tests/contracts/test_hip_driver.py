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

    def test_it_looks_past_the_global_symbol_table(self) -> None:
        """`CDLL(None)` was the whole of this, and the DCU refused it.

        torch loads its extensions with RTLD_LOCAL, so on the device the HIP entry points
        were mapped into the process and not globally visible, and every required symbol
        came back unresolved. The process's own memory map names the file; dlopen of an
        already-mapped library is a reference-count bump, not a second load.
        """
        import inspect
        from open_cake_ir.evaluation import hip_driver
        source = inspect.getsource(hip_driver.load_hip_runtime)
        self.assertIn("_mapped_shared_objects", source)
        self.assertIn("RTLD_LOCAL", source)
        # And the map reader names no vendor's library.
        map_source = inspect.getsource(hip_driver._mapped_shared_objects)
        self.assertIn("/proc/self/maps", map_source)
        self.assertIn(".so", map_source)
        for soname in ("libgalaxyhip.so.5", "libamdhip64.so.5"):
            self.assertNotIn(soname, source.replace("`libgalaxyhip.so.5`", "")
                             .replace("`libamdhip64.so.5`", ""))

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
        for soname in ("libgalaxyhip", "libamdhip64"):
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

    def test_attribution_is_not_taken_on_the_hsaco_platform_s_behalf(self) -> None:
        """Nsight Compute is CUDA's profiler and this platform still never reaches it.

        The branch that once reached it was "not Metal", which would have handed a DCU
        candidate to ncu. The answer was first to declare no source at all, and is now to
        declare this platform's own: roctracer activity, taken inside evaluate. Both
        satisfy the rule; only the second gives an authoring Turn something to act on.
        """
        from open_cake_ir.tasks import evaluate
        self.assertEqual(evaluate._PLATFORMS["hsaco"].attribution, "inside_evaluate")
        self.assertFalse(evaluate._PLATFORMS["hsaco"].profiled_child)
        self.assertEqual(evaluate._PLATFORMS["cubin"].attribution, "separate")

    def test_the_hsaco_profile_declares_its_own_kind_and_states_what_it_omits(self) -> None:
        """The projection is bounded, and names the three metrics ncu has and it does not.

        An omitted field reads as "does not constrain" and means "was not looked at", so
        occupancy, bandwidth and instruction counters are named in the record rather than
        left to be inferred from silence.
        """
        from open_cake_ir.evaluation.hip_observations import (
            HIP_PROFILE_KIND, NOT_COLLECTED, hip_attribution_feedback, hip_profile_summary)
        raw = {"source": "roctracer_device_activity", "instrumented_dispatches": 1,
               "device_time_us": 3.071, "non_target_dispatches": 0,
               "not_collected": list(NOT_COLLECTED)}
        summary = hip_profile_summary(raw)
        self.assertEqual(summary["coverage"], "per_dispatch_device_time")
        self.assertEqual(summary["timing_use"], "attribution_only")
        self.assertEqual(list(summary["not_collected"]),
                         ["occupancy", "bandwidth", "instruction_counters"])
        feedback = hip_attribution_feedback(
            {"kernel_name": "_cake_rmsnorm_fp32_kernel", "summary": summary})
        self.assertEqual(feedback["kind"], "hip_dispatch_attribution")
        self.assertEqual(feedback["device_time_us"], 3.071)
        self.assertNotEqual(HIP_PROFILE_KIND, "ncu_kernel_attribution")

    def test_an_activity_record_this_source_did_not_produce_is_refused(self) -> None:
        from open_cake_ir.evaluation.hip_observations import NOT_COLLECTED, hip_profile_summary
        good = {"source": "roctracer_device_activity", "instrumented_dispatches": 1,
                "device_time_us": 3.071, "non_target_dispatches": 0,
                "not_collected": list(NOT_COLLECTED)}
        for field, value in (("source", "ncu"), ("instrumented_dispatches", 2),
                             ("device_time_us", 0.0), ("device_time_us", 3),
                             ("non_target_dispatches", -1), ("not_collected", [])):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    hip_profile_summary({**good, field: value})

    def test_timing_is_the_studys_statement_not_the_platform_table_s(self) -> None:
        """Every row read `collect_timing=True`, so every search evaluation timed.

        A Study for a target with no named timer carries a measurement-coverage
        limitation instead of a paired assay, and the row reads that. Deciding it from
        the target would put the timing question back on the code-object axis, where two
        targets can share an object and differ in whether anything has measured them.
        """
        from open_cake_ir.tasks import evaluate
        for name in ("cubin", "hsaco"):
            with self.subTest(platform=name):
                authority = SimpleNamespace(
                    candidate=SimpleNamespace(target="gfx938" if name == "hsaco" else "sm_103a"),
                    manifest=SimpleNamespace(), timed_assay_available=False)
                seen = {}
                target = "_evaluate_hip_candidate" if name == "hsaco" else "_evaluate_candidate"
                with patch.object(evaluate, target,
                                  lambda a, r, **kw: seen.update(kw)):
                    evaluate._PLATFORMS[name].evaluate(authority, {})
                self.assertFalse(seen["collect_timing"])
        self.assertIn("timed_assay_available: bool = True",
                      __import__("inspect").getsource(evaluate._Authority))

    def test_a_timed_tile_evaluation_refuses_without_a_source(self) -> None:
        """The source is an argument now, so the absence is caught where it is used.

        `_evaluate_hip_candidate` used to refuse timing outright, because gfx938 had no
        named timer. It has one; what stays is that a timed run cannot proceed without a
        source, rather than quietly producing an untimed receipt.
        """
        from open_cake_ir.evaluation.paired import ROUTE_CALLS_PER_COHORT
        from open_cake_ir.tasks import evaluate
        with self.assertRaisesRegex(ValueError, "requires its timing source"):
            evaluate._evaluate_tile_candidate(
                SimpleNamespace(), {}, None, SimpleNamespace(), True,
                route_calls_per_cohort=ROUTE_CALLS_PER_COHORT["hip_dispatch"])


class CloseContractTest(unittest.TestCase):
    """Teardown drains the device first, and matches the lifecycle that calls it."""

    def loaded(self, api):
        from unittest.mock import patch as _patch
        candidate = SimpleNamespace(
            target="gfx938", entry_point="k", launch_spec_sha256="c" * 64,
            artifact_roles={"hsaco": sha256(b"\x7fELF").hexdigest()},
            artifact_payloads={"amdgcn": b".amdhsa_kernel k\n"})
        manifest = SimpleNamespace(target="gfx938", kernel_name="k",
                                   canonical_sha256="c" * 64)
        with _patch("open_cake_ir.evaluation.triton_hip.amdgcn_resource_record",
                    return_value={}):
            return LoadedHipModuleCandidate.load(candidate, b"\x7fELF", manifest,
                                                 "gfx938", api=api)

    def working_api(self, **overrides):
        api = SimpleNamespace(**{name: (lambda *a: 0) for name in (
            "hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel",
            "hipModuleUnload", "hipDeviceSynchronize")})
        for name, value in overrides.items():
            setattr(api, name, value)
        return api

    def test_it_takes_the_same_arguments_the_shared_lifecycle_passes(self) -> None:
        """The tensor-tile lifecycle closes both drivers through one call.

        Measured on the DCU after the kernel had already launched: close() took no
        keywords and the run failed at teardown with 'unexpected keyword argument
        synchronize', after module_loads=1, preflight_calls=1 and kernel_calls=1.
        """
        import inspect
        from open_cake_ir.evaluation.cuda_driver import CudaModules
        self.assertEqual(str(inspect.signature(LoadedHipModuleCandidate.close)),
                         str(inspect.signature(CudaModules.close)))

    def test_the_device_is_drained_before_anything_is_unloaded(self) -> None:
        order = []
        api = self.working_api(hipModuleUnload=lambda *a: order.append("unload") or 0)
        loaded = self.loaded(api)
        loaded.close(synchronize=lambda: order.append("synchronize"))
        self.assertEqual(order, ["synchronize", "unload"])
        self.assertTrue(loaded.closed)

    def test_a_drain_failure_does_not_skip_the_unload_and_is_not_hidden(self) -> None:
        unloaded = []
        api = self.working_api(hipModuleUnload=lambda *a: unloaded.append(1) or 0)
        loaded = self.loaded(api)

        def drain():
            raise RuntimeError("device drain failed")

        with self.assertRaisesRegex(RuntimeError, "device drain failed"):
            loaded.close(synchronize=drain)
        self.assertEqual(unloaded, [1])

    def test_a_primary_failure_is_kept_beside_the_teardown_one(self) -> None:
        api = self.working_api(hipModuleUnload=lambda *a: 7)
        loaded = self.loaded(api)
        primary = ValueError("the launch failed")
        with self.assertRaises(HipLifecycleError) as raised:
            loaded.close(primary=primary)
        self.assertIs(raised.exception.primary, primary)
        self.assertIn("hipModuleUnload", str(raised.exception.teardown))


if __name__ == "__main__":
    unittest.main()
