"""The arch comparison that decides whether a device is the one the Target declares.

Loading an AMDGCN candidate needs a ROCm PyTorch and one visible HIP GPU, so it is
exercised on hardware and retained as evidence rather than here. What is host-testable
is the comparison that admits a device -- which is where this route refused the only
DCU it has.
"""

from __future__ import annotations

import unittest

from open_cake_ir.evaluation.triton_hip import device_arch_matches


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


if __name__ == "__main__":
    unittest.main()
