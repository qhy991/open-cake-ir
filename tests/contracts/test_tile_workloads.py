from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from open_cake_ir.tasks.workloads import load_workload
sys.path.insert(0, str(ROOT / "src"))

from examples.paired_triton.prepare import baseline_schedule, prepare_baseline
from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks.tiles.workload import materialize_case, reference_outputs
from open_cake_ir.evaluation.workload import WorkloadContract


class TileWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workloads = {
            name: load_workload(ROOT / f"contracts/workloads/{name}-v1.json")
            for name in ("rmsnorm-fp32", "gemm-bias-bf16-fp32", "indexed-gather-bf16")
        }
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def load_document(self, document: dict) -> WorkloadContract:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_workload(path)

    def test_primary_shapes_dtypes_and_order_are_preserved(self) -> None:
        expected = {
            "rmsnorm-fp32": (("x", (8, 512, 128), "fp32", "input"), ("gamma", (128,), "fp32", "input"), ("y", (8, 512, 128), "fp32", "output")),
            "gemm-bias-bf16-fp32": (("a", (512, 256), "bf16", "input"), ("b", (256, 256), "bf16", "input"), ("bias", (256,), "fp32", "input"), ("c", (512, 256), "fp32", "output")),
            "indexed-gather-bf16": (("expert_rows", (4, 8, 16), "bf16", "input"), ("expert_ids", (8, 8), "int32", "input"), ("row_ids", (8, 8), "int32", "input"), ("gathered_rows", (8, 8, 16), "bf16", "output")),
        }
        for name, workload in self.workloads.items():
            with self.subTest(name=name):
                self.assertEqual(tuple((arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi("primary")), expected[name])

    def test_abi_dimensions_and_tensor_names_are_data(self) -> None:
        document = self.workloads["rmsnorm-fp32"].document
        rename = {"x": "samples", "gamma": "scale", "y": "result"}
        document["tensors"] = {rename[name]: tensor for name, tensor in reversed(list(document["tensors"].items()))}
        document["semantics"]["candidate_abi"] = {"inputs": ["samples", "scale"], "outputs": ["result"]}
        for tensor in document["tensors"].values():
            tensor["shape"] = ["width" if component == "D" else component for component in tensor["shape"]]
        for case in document["cases"]:
            case["shape"]["width"] = case["shape"].pop("D")
        workload = self.load_document(document)
        self.assertEqual([arg.name for arg in workload.tensor_abi("tiny")], ["samples", "scale", "result"])
        inputs = materialize_case(workload, "tiny")
        self.assertEqual(list(inputs), ["samples", "scale"])
        self.assertEqual(list(reference_outputs(workload, "tiny", inputs)), ["result"])

    def test_rmsnorm_mathematics_and_epsilon(self) -> None:
        workload = self.workloads["rmsnorm-fp32"]
        self.assertEqual(workload.document["semantics"]["epsilon"], 1e-6)
        values = {"x": [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "gamma": [1.0, 2.0, -1.0, 0.0]}
        result = reference_outputs(workload, "tiny", values)["y"]
        self.assertAlmostEqual(result[0], 3 / math.sqrt(6.25 + 1e-6), places=6)
        self.assertAlmostEqual(result[1], 8 / math.sqrt(6.25 + 1e-6), places=6)
        self.assertEqual(result[2:], [0.0] * 6)
        # A hypothetical successor scalar must flow from the contract to both paths.
        document = workload.document
        document["semantics"]["epsilon"] = 4.0
        alternate = self.load_document(document)
        result = reference_outputs(alternate, "tiny", values)["y"]
        self.assertAlmostEqual(result[0], 3 / math.sqrt(10.25), places=6)
        schedule = baseline_schedule(alternate, "tiny")
        shift = next(operation for operation in schedule["operations"] if operation["id"] == "shift")
        self.assertEqual(shift["parameters"]["scalar"], 4.0)

    def test_rmsnorm_near_zero_keeps_epsilon_active(self) -> None:
        workload = self.workloads["rmsnorm-fp32"]
        values = materialize_case(workload, "near_zero")
        result = reference_outputs(workload, "near_zero", values)["y"]
        self.assertTrue(all(9.9e-6 < abs(value) < 1.01e-5 for value in result))
        self.assertEqual(values, materialize_case(workload, "near_zero"))

    def test_gemm_rectangular_rhs_orientation_bias_and_k_tail(self) -> None:
        workload = self.workloads["gemm-bias-bf16-fp32"]
        k = workload.tensor_abi("tiny")[0].shape[1]
        a = [0.0] * (2 * k)
        b = [0.0] * (3 * k)
        a[:4], a[k:k + 4] = [1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 2.0, 0.0]
        b[0], b[k + 1] = 1.0, 1.0
        b[2 * k:2 * k + 4] = [1.0, 2.0, 1.0, 0.0]
        a[64], a[k + 64], b[64] = 2.0, -2.0, 3.0
        inputs = {"a": a, "b": b, "bias": [0.25, -0.5, 1.0]}
        before = json.dumps(inputs)
        self.assertEqual(reference_outputs(workload, "tiny", inputs)["c"], [7.25, 1.5, 9.0, -6.75, -0.5, 2.0])
        self.assertEqual(json.dumps(inputs), before)

    def test_gemm_cancellation_and_zeros(self) -> None:
        workload = self.workloads["gemm-bias-bf16-fp32"]
        self.assertEqual(reference_outputs(workload, "cancellation", materialize_case(workload, "cancellation"))["c"], [0.25] * 6)
        self.assertEqual(reference_outputs(workload, "zeros", materialize_case(workload, "zeros"))["c"], [0.0] * 2)

    def test_gather_is_zipped_with_exact_signed_zero_and_no_negative_wrap(self) -> None:
        workload = self.workloads["indexed-gather-bf16"]
        source = [index / 2 for index in range(24)]
        source[0] = -0.0
        inputs = {"expert_rows": source, "expert_ids": [0, 1, -1, 2], "row_ids": [0, 2, 0, 0]}
        result = reference_outputs(workload, "tiny", inputs)["gathered_rows"]
        self.assertEqual(result, source[:4] + source[20:24] + [0.0] * 8)
        self.assertEqual(struct.pack("<f", result[0]), b"\x00\x00\x00\x80")
        self.assertTrue(all(struct.pack("<f", value) == b"\x00" * 4 for value in result[8:]))

    def test_gather_index_boundary_distribution_and_repetitions(self) -> None:
        workload = self.workloads["indexed-gather-bf16"]
        inputs = materialize_case(workload, "index_boundaries")
        self.assertIn(-(1 << 31), inputs["expert_ids"])
        self.assertIn((1 << 31) - 1, inputs["row_ids"])
        result = reference_outputs(workload, "index_boundaries", inputs)["gathered_rows"]
        self.assertEqual(result[8:40], [0.0] * 32)
        repeated = reference_outputs(workload, "repeated_indices", materialize_case(workload, "repeated_indices"))["gathered_rows"]
        self.assertEqual(repeated, repeated[:4] * 8)

    def test_materialization_rounds_bf16_ties_to_even(self) -> None:
        workload = self.workloads["gemm-bias-bf16-fp32"]
        for random_value, expected in ((0.7509765625, 0.5), (0.7529296875, 0.5078125)):
            with self.subTest(random_value=random_value), patch("open_cake_ir.tasks.tiles.workload.random.Random") as rng:
                rng.return_value.random.return_value = random_value
                inputs = materialize_case(workload, "tiny")
                self.assertEqual(inputs["a"][0], expected)
                reference_outputs(workload, "tiny", inputs)

    def test_all_case_materialization_and_small_case_oracle_sizes(self) -> None:
        for workload in self.workloads.values():
            for case_id in workload.case_ids:
                with self.subTest(workload=workload.workload_id, case=case_id):
                    abi = workload.tensor_abi(case_id)
                    inputs = materialize_case(workload, case_id)
                    self.assertEqual(list(inputs), [arg.name for arg in abi if arg.mode == "input"])
                    self.assertEqual({name: len(values) for name, values in inputs.items()}, {arg.name: math.prod(arg.shape) for arg in abi if arg.mode == "input"})
                    if case_id != "primary":
                        result = reference_outputs(workload, case_id, inputs)
                        self.assertEqual({name: len(values) for name, values in result.items()}, {arg.name: math.prod(arg.shape) for arg in abi if arg.mode == "output"})
                        self.assertEqual(inputs, materialize_case(workload, case_id))

    def test_oracle_rejects_invalid_or_incomplete_inputs(self) -> None:
        rms = self.workloads["rmsnorm-fp32"]
        for bad in (float("nan"), float("inf"), 17.0, 0.1, True, 10 ** 1000):
            with self.subTest(value=repr(bad)[:30]):
                inputs = materialize_case(rms, "tiny")
                inputs["x"][0] = bad
                with self.assertRaises(ValueError):
                    reference_outputs(rms, "tiny", inputs)
        for mutation in (lambda values: values.pop("gamma"), lambda values: values.update(extra=[]), lambda values: values["x"].pop()):
            inputs = materialize_case(rms, "tiny")
            mutation(inputs)
            with self.assertRaises(ValueError):
                reference_outputs(rms, "tiny", inputs)
        gather = self.workloads["indexed-gather-bf16"]
        for bad in (True, 1.0, 1 << 31, -(1 << 31) - 1):
            with self.subTest(index=bad):
                inputs = materialize_case(gather, "tiny")
                inputs["expert_ids"][0] = bad
                with self.assertRaises(ValueError):
                    reference_outputs(gather, "tiny", inputs)

    def test_malformed_contracts_fail_at_admission(self) -> None:
        mutations = (
            lambda doc: doc["semantics"]["candidate_abi"]["inputs"].append("x"),
            lambda doc: doc["semantics"]["candidate_abi"].update(inputs=[]),
            lambda doc: doc["tensors"]["x"].update(shape=["unknown"]),
            lambda doc: doc["tensors"]["x"].update(dtype={}),
            lambda doc: doc["tensors"]["x"].update(layout="strided"),
            lambda doc: doc["semantics"].update(epsilon=0),
            lambda doc: doc["semantics"].update(epsilon=1e-100),
            lambda doc: doc["semantics"].update(epsilon=1e100),
            lambda doc: doc["validation"].update(atol=-1),
            lambda doc: doc["cases"][0].update(seed=None),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                document = self.workloads["rmsnorm-fp32"].document
                mutation(document)
                with self.assertRaises(ValueError):
                    self.load_document(document)
        document = self.workloads["indexed-gather-bf16"].document
        document["semantics"]["indexing"]["invalid"] = "wrap"
        with self.assertRaises(ValueError):
            self.load_document(document)
        document = self.workloads["gemm-bias-bf16-fp32"].document
        document["semantics"]["rhs_storage"] = "K_N"
        with self.assertRaises(ValueError):
            self.load_document(document)

    def test_oversized_tolerances_are_public_load_refusals(self) -> None:
        for workload in self.workloads.values():
            for field in ("atol", "rtol"):
                with self.subTest(workload=workload.workload_id, field=field):
                    document = workload.document
                    document["validation"][field] = 10 ** 1000
                    with self.assertRaisesRegex(ValueError, f"validation.{field}"):
                        self.load_document(document)

    def test_nonobject_provenance_is_refused_at_load_and_baseline(self) -> None:
        for entry in (None, "source", 7):
            with self.subTest(entry=entry):
                document = self.workloads["rmsnorm-fp32"].document
                document["provenance"] = [entry]
                with self.assertRaisesRegex(ValueError, r"provenance\[0\].*object"):
                    self.load_document(document)
                # The public constructor can also reach preparation without load().
                with self.assertRaisesRegex(ValueError, r"provenance\[0\].*object"):
                    baseline_schedule(WorkloadContract(document), "primary")

    def test_source_provenance_required_fields_are_validated(self) -> None:
        for field in ("path", "source_commit", "scope"):
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    document = self.workloads["rmsnorm-fp32"].document
                    if missing:
                        document["provenance"][0].pop(field)
                    else:
                        document["provenance"][0][field] = 7
                    with self.assertRaisesRegex(ValueError, rf"provenance\[0\].{field}"):
                        self.load_document(document)
                    with self.assertRaisesRegex(ValueError, rf"provenance\[0\].{field}"):
                        baseline_schedule(WorkloadContract(document), "primary")

    def test_materialization_honors_each_declared_input_bound(self) -> None:
        document = self.workloads["rmsnorm-fp32"].document
        document["tensors"]["x"]["max_abs"] = 0.125
        document["tensors"]["gamma"]["max_abs"] = 0.25
        workload = self.load_document(document)
        for case_id in ("tiny", "near_zero", "alternating"):
            inputs = materialize_case(workload, case_id)
            self.assertLessEqual(max(map(abs, inputs["x"])), 0.125)
            self.assertLessEqual(max(map(abs, inputs["gamma"])), 0.25)
            reference_outputs(workload, case_id, inputs)
    def test_historical_workload_admission_does_not_infer_new_abi(self) -> None:
        workload = load_workload(ROOT / "contracts/workloads/dsa-attention-sparse-mla-decode-v1.json")
        self.assertTrue(workload.case_ids)
        with self.assertRaises(ValueError):
            workload.tensor_abi(workload.case_ids[0])

    def test_primary_baselines_preserve_original_schedule_semantics(self) -> None:
        for workload in self.workloads.values():
            with self.subTest(workload=workload.workload_id):
                source_path = workload.document["provenance"][0]["path"]
                original = json.loads((ROOT / source_path).read_text())
                baseline = baseline_schedule(workload, "primary")
                self.assertEqual(baseline["metadata"]["workload_contract_sha256"], workload.canonical_sha256)
                for projection in (original, baseline):
                    projection.pop("metadata")
                    projection.pop("schedule_id")
                self.assertEqual(baseline, original)

    def test_all_declared_case_baselines_lower_without_executing_source(self) -> None:
        for workload in self.workloads.values():
            for case_id in workload.case_ids:
                with self.subTest(workload=workload.workload_id, case=case_id):
                    assessment = self.compiler.assess(baseline_schedule(workload, case_id))
                    self.assertTrue(assessment.accepted, assessment.findings)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    lowering = self.compiler.lower(assessment)
                    compile(lowering.source, "<tile-baseline>", "exec")

    def test_export_is_exact_compiler_projection_and_source_only(self) -> None:
        for workload in self.workloads.values():
            with self.subTest(workload=workload.workload_id), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "baseline"
                record = prepare_baseline(workload, "primary", output)
                assessment = self.compiler.assess(json.loads((output / record["schedule_file"]).read_text()))
                lowering = self.compiler.lower(assessment)
                self.assertEqual((output / record["native_source_file"]).read_text(), lowering.source)
                self.assertEqual(record["toolchain_requirements"], dict(lowering.toolchain_requirements))
                self.assertEqual(record["status"], "source_only")
                self.assertEqual(record["target_compilation"], "not_run")
                self.assertEqual(record["gpu_correctness"], "R2_pending")
                before = (output / "preparation.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    prepare_baseline(workload, "primary", output)
                self.assertEqual((output / "preparation.json").read_bytes(), before)

    def test_preparation_refuses_checkout_outputs_and_unmatched_primary(self) -> None:
        workload = self.workloads["rmsnorm-fp32"]
        with self.assertRaisesRegex(ValueError, "outside every checkout"):
            prepare_baseline(workload, "primary", ROOT / "examples/paired_triton/uncreated")
        document = workload.document
        document["cases"][0]["shape"]["N"] = 513
        with self.assertRaisesRegex(ValueError, "primary ABI"):
            baseline_schedule(self.load_document(document), "primary")


if __name__ == "__main__":
    unittest.main()
