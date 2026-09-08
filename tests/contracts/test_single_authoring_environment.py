"""Same Lab allocation policy for paired science and single-author artifact work."""
import unittest

from open_cake_ir.lab.pairing import comparison_arm, matched_run_arms


class SingleAuthoringEnvironmentTests(unittest.TestCase):
    def test_noncomparative_artifact_optimization_has_one_open_cake_run(self):
        self.assertIsNone(comparison_arm({"open_cake": {}}))
        self.assertEqual(matched_run_arms({"open_cake": {}}, "artifact_optimization_only"), ["open_cake"])

    def test_scientific_and_system_scopes_cannot_omit_comparison(self):
        for scope in ("scientific_matched_search", "system_qualification_only"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "artifact_optimization_only"):
                matched_run_arms({"open_cake": {}}, scope)

    def test_existing_paired_allocations_and_refusals_remain(self):
        for comparison in ("direct_cuda", "native_triton"):
            arms = {"open_cake": {}, comparison: {}}
            self.assertEqual(comparison_arm(arms), comparison)
            self.assertEqual(matched_run_arms(arms, "artifact_optimization_only"), sorted(arms))
            self.assertEqual(matched_run_arms(arms, "scientific_matched_search"), sorted(list(arms) * 3))
        for arms in ({}, {"native_triton": {}}, {"open_cake": {}, "metal": {}}):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                comparison_arm(arms)


if __name__ == "__main__":
    unittest.main()
