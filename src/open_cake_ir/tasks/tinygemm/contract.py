from __future__ import annotations
from typing import Mapping,cast
from open_cake_ir.evaluation.workload import _name,_object

def _validate_tinygemm_contract(document: Mapping[str, object]) -> None:
    workload_id = document.get("workload_id")
    revision = document.get("revision")
    if (workload_id, revision) not in {
        ("tinygemm2-stage4-independent-v1", "1"),
        ("tinygemm2-stage4-independent-v2", "2"),
    }:
        raise ValueError("TinyGEMM2 workload identity or revision differs")
    expected_tensors = {
        "input": {
            "shape": ["batch", "input_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
        "weight": {
            "shape": ["output_features", "input_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
        "bias": {
            "shape": ["output_features"],
            "dtype": "bf16",
            "layout": "contiguous",
        },
        "output": {
            "shape": ["batch", "output_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "bf16_round(input_fp32 @ weight_fp32^T + bias_fp32)",
        "accumulator_dtype": "fp32",
        "tie_break": "not_applicable",
        "nonfinite": "equal_nan_false",
    }
    expected_oracle = {
        "kind": "fp32_linear_then_bf16_round",
        "source_sha256": "9cea3117f9202051a5ff9c63221aa81d212aed7fe38b597c7aa540e96c5cb060",
        "atol": 0.01,
        "rtol": 0.01,
    }
    expected_validation = {
        "canonical": "bitwise_equal_to_pinned_stage4_parent_and_tolerance_equal_to_fp32_oracle",
        "parent_revision": "339a8f4cb2644fb85de79f3eed3624a1e91183d1",
        "paper_acceptance_rule": "not_applicable_known_kernel_lane",
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("TinyGEMM2 workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("TinyGEMM2 workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("TinyGEMM2 workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("TinyGEMM2 workload validation differs")
    cases = cast(list[Mapping[str, object]], document["cases"])
    if len(cases) != 1 or (
        cases[0].get("case_id"),
        cases[0].get("shape"),
        cases[0].get("seed"),
        cases[0].get("mode"),
    ) != (
        "stage4_n8_m1024_k1024",
        {"batch": 8, "output_features": 1024, "input_features": 1024},
        0,
        "random_normal",
    ):
        raise ValueError("TinyGEMM2 workload cases differ")
    materialized = cases[0].get("materialized")
    if revision == "1" and materialized is not None:
        raise ValueError("TinyGEMM2 v1 unexpectedly owns materialization")
    if revision == "2" and set(
        _object(materialized, "TinyGEMM2 v2 materialized authority")
    ) != {
        "input",
        "weight",
        "bias",
        "fp32_linear_bf16_oracle",
        "parent_output",
    }:
        raise ValueError("TinyGEMM2 v2 materialized authority differs")
