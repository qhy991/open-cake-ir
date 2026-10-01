"""The captured MetaX compiler distribution must describe the installed runtime."""

import importlib.metadata
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.lab.metax_host import installed_triton_distribution, validate_host


ROOT = Path(__file__).resolve().parents[2]


class MetaxHostPackagesTest(unittest.TestCase):
    def test_flagtree_and_triton_are_separate_valid_host_variants(self):
        host = json.loads((ROOT / "runtime/hosts/xcore1002.json").read_text())["host_environment"]
        validate_host(host)
        alternate = copy.deepcopy(host)
        active = next(name for name in ("flagtree", "triton") if name in alternate["packages"])
        del alternate["packages"][active]
        alternate["packages"]["triton" if active == "flagtree" else "flagtree"] = "installed"
        validate_host(alternate)
        host["packages"]["flagtree" if active == "triton" else "triton"] = "installed"
        with self.assertRaisesRegex(ValueError, "MACA host package set differs"):
            validate_host(host)

    def test_capture_refuses_ambiguous_or_missing_compiler_distribution(self):
        def versions(available):
            def version(name):
                if name not in available:
                    raise importlib.metadata.PackageNotFoundError(name)
                return "installed"
            return version

        for available, expected in (({"flagtree"}, "flagtree"), ({"triton"}, "triton")):
            with self.subTest(available=available), patch(
                "open_cake_ir.lab.metax_host.importlib.metadata.version", side_effect=versions(available)
            ):
                self.assertEqual(installed_triton_distribution(), expected)
        for available in (set(), {"flagtree", "triton"}):
            with self.subTest(available=available), patch(
                "open_cake_ir.lab.metax_host.importlib.metadata.version", side_effect=versions(available)
            ):
                with self.assertRaisesRegex(ValueError, "exactly one installed"):
                    installed_triton_distribution()


if __name__ == "__main__":
    unittest.main()
