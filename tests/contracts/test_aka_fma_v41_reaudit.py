from pathlib import Path
import unittest

from tools.verify_aka_fma_v41_reaudit import verify

ROOT=Path(__file__).resolve().parents[2]

class AkaFmaV41ReauditTests(unittest.TestCase):
    def test_published_projection_passes_its_verifier(self):
        result=verify(ROOT/"docs/data/aka-fma-v41-reaudit-20260906")
        self.assertEqual(result["status"],"accepted")
        self.assertEqual(result["cases"],12)
        self.assertEqual(result["semantic_survivors"],4)
        self.assertEqual(result["runtime_scalar_blocked"],5)
        self.assertEqual(result["gpu_test"],"not_run")
        self.assertFalse(result["performance_measured"])

if __name__=="__main__":unittest.main()
