from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import WorkloadContract  # noqa: E402


class WorkloadContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dsa_path = ROOT / "contracts/workloads/dsa-attention-sparse-mla-decode-v1.json"
        self.kda_path = ROOT / "contracts/workloads/kimi-k3-kda-fused-decode-v1.json"

    def test_new_workloads_load(self) -> None:
        self.assertEqual(
            WorkloadContract.load(self.dsa_path).workload_id,
            "deepseek-v3.2-dsa-sparse-mla-decode-v1",
        )
        self.assertEqual(
            WorkloadContract.load(self.kda_path).workload_id,
            "kimi-k3-kda-fused-decode-v1",
        )

    def test_dsa_task_matrix_scale_and_timing(self) -> None:
        workload = WorkloadContract.load(self.dsa_path)
        captured = (
            ("0c23b10c7b7645719517828c12eaa1d2", 1),
            ("9d4a5f21268e484ea05a2f2af91d9fa7", 2),
            ("b7668cfd194c4b95ab600feb205ebac6", 2),
            ("0a63b87bb2e54e9db1ca3c4c53a1d521", 2),
            ("05f6de657db543ae9e4c46796522843a", 2),
            ("fc85411e250c41879b6d6a1edb80f0a7", 2),
            ("e6b849f2900446148c01d81152efae23", 2),
            ("9f3f891bfbe24776adfcd5a579d093dd", 2),
            ("f77df5ce21634f7ca0586af7e3c13434", 2),
            ("385742b2717e4f02b918c7349dde23d8", 8),
            ("4c46a94ba2364dc7ab476286dee8dce3", 8),
            ("3838996164a94d728710f913477feba8", 7),
            ("02d6ae9c64ab42ff93f05c23c53bcb7d", 8),
            ("ddfa9e340b264f76abe7418692faa876", 6),
            ("78b2e11c30004cceb84355722a7c6b0a", 8),
            ("68d6817dcfd1433aa8d2ddeefd54b6ea", 6),
            ("564007ac354e4662a62cc4d6352dc494", 8),
            ("ae4219a95f044f45bd10e17ab63c6e8f", 7),
            ("232ed014bafc4835b9881bb308c659b0", 8),
            ("7a389715dcc3479e9b0512309f9d1d56", 8),
            ("5096e459ce3f4cdf82773ce1a0c73c8a", 8),
            ("d57eb9e19f0642e8af9bd76ad0823303", 6),
            ("2207f0fdc96347c59d106e4976cfad57", 7),
        )
        observed_captured = tuple(
            (case_id, workload.case(case_id)["shape"]["num_tokens"])
            for case_id in workload.case_ids[:23]
        )
        self.assertEqual(observed_captured, captured)
        self.assertTrue(
            all(
                workload.case(case_id)["shape"]["num_pages"] == 8462
                and workload.case(case_id)["mode"] == "captured_sparse_indices_small"
                for case_id, _ in captured
            )
        )
        generated = tuple(
            (case_id, workload.case(case_id)["shape"], workload.case(case_id)["mode"])
            for case_id in workload.case_ids[23:]
        )
        self.assertEqual(
            generated,
            (
                ("gen-large-num_pages8192-num_tokens8", {"num_tokens": 8, "num_pages": 8192}, "generated_dense_indices_small"),
                ("gen-large-num_pages32768-num_tokens8", {"num_tokens": 8, "num_pages": 32768}, "generated_dense_indices_large"),
                ("gen-large-num_pages32768-num_tokens16", {"num_tokens": 16, "num_pages": 32768}, "generated_dense_indices_large"),
                ("gen-large-num_pages32768-num_tokens32", {"num_tokens": 32, "num_pages": 32768}, "generated_dense_indices_large"),
                ("gen-large-num_pages32768-num_tokens64", {"num_tokens": 64, "num_pages": 32768}, "generated_dense_indices_large"),
                ("gen-large-num_pages32768-num_tokens128", {"num_tokens": 128, "num_pages": 32768}, "generated_dense_indices_large"),
                ("gen-large-num_pages32768-num_tokens256", {"num_tokens": 256, "num_pages": 32768}, "generated_dense_indices_large"),
            ),
        )
        document = workload.document
        self.assertEqual(document["tensors"]["sm_scale"]["value"], 0.1352337788608801)
        self.assertEqual(
            document["semantics"]["sm_scale_authority"]["binding"],
            "executed_value_is_authoritative",
        )
        measurement = document["validation"]["measurement"]
        self.assertEqual(
            (measurement["required_cupti_major"], measurement["warmup_iterations"], measurement["samples_per_trial"], measurement["trials"]),
            (13, 3, 50, 3),
        )
        self.assertEqual(
            measurement["trial_order"],
            ["baseline_before", "candidate", "baseline_after"],
        )

    def test_kda_task_matrix_abi_state_timing_and_trajectory(self) -> None:
        workload = WorkloadContract.load(self.kda_path)
        self.assertEqual(
            tuple(
                (
                    case_id,
                    workload.case(case_id)["shape"]["num_heads"],
                    workload.case(case_id)["shape"]["batch_size"],
                    workload.case(case_id)["shape"]["active_rows"],
                )
                for case_id in workload.case_ids
            ),
            (
                ("h12-b1-a1", 12, 1, 1),
                ("h12-b4-a3", 12, 4, 3),
                ("h12-b8-a8", 12, 8, 8),
                ("h12-b16-a13", 12, 16, 13),
                ("h12-b32-a32", 12, 32, 32),
                ("h12-b64-a51", 12, 64, 51),
                ("h12-b128-a128", 12, 128, 128),
                ("h6-b32-a25", 6, 32, 25),
                ("h3-b64-a51", 3, 64, 51),
            ),
        )
        document = workload.document
        self.assertEqual(
            document["semantics"]["candidate_abi"],
            [
                "mixed_qkv", "a", "b", "conv_states", "w_q_t", "w_k_t", "w_v_t",
                "conv_bias", "A_log", "dt_bias", "onorm_g", "onorm_weight",
                "ssm_states", "cache_indices", "scale", "onorm_eps", "lower_bound",
            ],
        )
        self.assertEqual(document["semantics"]["state_slots"], "batch_size+4")
        self.assertEqual(
            (
                document["tensors"]["scale"]["value"],
                document["tensors"]["onorm_eps"]["value"],
                document["tensors"]["lower_bound"]["value"],
            ),
            (0.08838834764831843, 1e-6, -5.0),
        )
        self.assertEqual(
            document["tensors"]["ssm_states"]["strides"],
            ["H*128*128+256", 16384, 128, 1],
        )
        tiers = document["validation"]["cell_tiers"]
        self.assertEqual({tuple(ids) for ids in tiers.values()}, {workload.case_ids[2:7], workload.case_ids[:2], workload.case_ids[7:]})
        measurement = document["validation"]["measurement"]
        self.assertEqual(measurement["pair_order"], {"baseline_then_candidate": 13, "candidate_then_baseline": 13})
        self.assertEqual(
            (measurement["warmup_iterations_per_arm"], measurement["samples_per_cohort"], measurement["pairing_protocol_authority"]),
            (5, 50, "workload_contract"),
        )
        trajectory = document["validation"]["trajectory"]
        self.assertEqual(
            (trajectory["coverage"], trajectory["steps_per_cell"], trajectory["promotion_required"]),
            ("all_nine_matrix_cells", 256, True),
        )

    def test_structural_cross_field_inconsistencies_fail_closed(self) -> None:
        cases = (
            (self.dsa_path, lambda d: d["semantics"]["case_matrix"].update(total_rows=31), "DSA case matrix"),
            (self.kda_path, lambda d: d["semantics"].update(state_slots="batch_size"), "KDA state-slot"),
            (self.kda_path, lambda d: d["validation"]["cell_tiers"]["primary_tp8_h12"].append("h12-b1-a1"), "KDA cell tiers"),
            (self.kda_path, lambda d: d["semantics"]["candidate_abi"].pop(), "KDA candidate ABI"),
        )
        for source, mutate, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                document = json.loads(source.read_text(encoding="utf-8"))
                mutate(document)
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    WorkloadContract.load(path)


if __name__ == "__main__":
    unittest.main()
