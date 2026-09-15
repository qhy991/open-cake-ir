"""Admission and sealing of one AMDGCN candidate, without a device.

`load_hip_candidate` needs a ROCm PyTorch and one visible HIP GPU, so the whole of it is
exercised on hardware and retained as evidence rather than here. What is host-testable is
every refusal it owns before a device is touched, and the arch comparison that decides
whether a device is the one the Target declares -- which is where this route refused the
only DCU it has.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from open_cake_ir.evaluation.triton_hip import (
    LoadedHipCandidate,
    _compile_through_the_module,
    device_arch_matches,
    load_hip_candidate,
)


class DeviceArchTest(unittest.TestCase):
    """A device names its ISA with its features; a Target declares the bare ISA."""

    def test_the_measured_dcu_name_is_the_declared_target(self) -> None:
        # Measured on BW1101: torch and hipGetDeviceProperties both report this string
        # for a Target whose document declares `gfx938`. Comparing them as equal strings
        # refused the device the Target describes.
        self.assertTrue(device_arch_matches("gfx938:sramecc+:xnack-", "gfx938"))

    def test_every_well_formed_feature_suffix_is_admitted(self) -> None:
        for observed in ("gfx938", "gfx938:xnack-", "gfx938:sramecc+",
                         "gfx938:sramecc+:xnack-"):
            with self.subTest(observed=observed):
                self.assertTrue(device_arch_matches(observed, "gfx938"))

    def test_a_longer_isa_name_is_not_the_declared_one(self) -> None:
        for observed in ("gfx9380", "gfx9380:xnack-", "gfx1151", "gfx942:xnack-"):
            with self.subTest(observed=observed):
                self.assertFalse(device_arch_matches(observed, "gfx938"))

    def test_a_malformed_suffix_is_refused(self) -> None:
        # A feature carries a sign; a bare colon or a trailing one is not a feature list.
        for observed in ("gfx938:", "gfx938:xnack", "gfx938::xnack-", "gfx938 xnack-"):
            with self.subTest(observed=observed):
                self.assertFalse(device_arch_matches(observed, "gfx938"))

    def test_a_missing_or_non_string_report_is_refused(self) -> None:
        for observed in (None, 938, b"gfx938", ""):
            with self.subTest(observed=observed):
                self.assertFalse(device_arch_matches(observed, "gfx938"))


class CompileContractTest(unittest.TestCase):
    """The compile contract is refused before Triton is asked for anything."""

    REQUIREMENTS = {
        "target": "gfx938",
        "kernel_entry_point": "_cake_kernel",
        "signature": {"x": "*fp32"},
        "compile_constants": {"BLOCK": 256},
        "compile_options": {"num_warps": 4},
        "triton_target": {"backend": "hip", "arch": "gfx938", "warp_size": 64},
    }

    def test_a_module_without_the_named_kernel_is_refused(self) -> None:
        module = SimpleNamespace()
        with self.assertRaises(ValueError) as raised:
            _compile_through_the_module(module, self.REQUIREMENTS, None)
        self.assertIn("_cake_kernel", str(raised.exception))

    def test_malformed_compile_contracts_are_refused(self) -> None:
        module = SimpleNamespace(_cake_kernel=object())
        for field in ("kernel_entry_point", "signature", "compile_constants",
                      "compile_options"):
            for value in (None, "", 4):
                with self.subTest(field=field, value=value):
                    requirements = {**self.REQUIREMENTS, field: value}
                    with self.assertRaises(ValueError):
                        _compile_through_the_module(module, requirements, None)


class SealedCandidateTest(unittest.TestCase):
    """What the loader returns, and what it refuses to hand back."""

    def candidate(self, **overrides) -> LoadedHipCandidate:
        fields = {
            "target": "gfx938",
            "entry_point": "cake_rmsnorm",
            "source_sha256": "a" * 64,
            "artifacts": {"hsaco": {"sha256": "b" * 64, "size_bytes": 8}},
            "resources": {"registers_per_thread": 134},
            "device_arch": "gfx938:sramecc+:xnack-",
            "warp_size": 64,
            "entry": lambda *a, **k: None,
            "_payloads": {"hsaco": b"\x7fELFfixture", "amdgcn": b".amdhsa_kernel k\n"},
        }
        fields.update(overrides)
        return LoadedHipCandidate(**fields)

    def test_an_admitted_role_returns_its_exact_bytes(self) -> None:
        loaded = self.candidate()
        self.assertEqual(loaded.payload("hsaco"), b"\x7fELFfixture")
        self.assertEqual(loaded.payload("amdgcn"), b".amdhsa_kernel k\n")

    def test_an_unknown_role_is_refused_by_name(self) -> None:
        loaded = self.candidate()
        for role in ("cubin", "ptx", "metallib", ""):
            with self.subTest(role=role):
                with self.assertRaises(ValueError) as raised:
                    loaded.payload(role)
                self.assertIn(repr(role), str(raised.exception))

    def test_closing_without_a_directory_is_not_an_error(self) -> None:
        self.candidate().close()

    def test_it_carries_the_device_it_was_admitted_against(self) -> None:
        """The record says which device answered, not which one was asked for."""
        loaded = self.candidate()
        self.assertEqual(loaded.device_arch, "gfx938:sramecc+:xnack-")
        self.assertTrue(device_arch_matches(loaded.device_arch, loaded.target))
        self.assertEqual(loaded.warp_size, 64)


class NoDeviceNoCandidateTest(unittest.TestCase):
    """Without a ROCm runtime the loader refuses; it never partially seals one."""

    def test_loading_refuses_before_reaching_a_device(self) -> None:
        with self.assertRaises((ValueError, RuntimeError, ModuleNotFoundError)):
            load_hip_candidate(
                SimpleNamespace(source="", source_sha256="a" * 64, entry_point="e"),
                {"target": "gfx938", "binary_role": "hsaco", "assembly_role": "amdgcn",
                 "triton_target": {"backend": "hip", "arch": "gfx938", "warp_size": 64}},
            )


if __name__ == "__main__":
    unittest.main()
