from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.probe_fma_fp32 import probe  # noqa: E402


class IrNumericProbeTests(unittest.TestCase):
    def test_fma_probe_distinguishes_single_rounding_from_mul_then_add(self) -> None:
        result = probe()

        self.assertTrue(result["distinguished"])
        self.assertEqual(result["fused"]["bits"], "0x41473989")
        self.assertEqual(result["separate_mul_add"]["bits"], "0x4147398a")

        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/probe_fma_fp32.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["schema"], result["schema"])


if __name__ == "__main__":
    unittest.main()
