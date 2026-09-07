from __future__ import annotations
from typing import Mapping,cast
from open_cake_ir.evaluation.workload import _name,_object

def _validate_flash_contract(document: Mapping[str, object]) -> None:
    workload_id = document.get("workload_id")
    revision = document.get("revision")
    if (workload_id, revision) not in {
        ("flash-kmeans-assign-independent-v1", "1"),
        ("flash-kmeans-assign-independent-v2", "2"),
    }:
        raise ValueError("Flash-KMeans workload identity or revision differs")
    expected_tensors = {
        "tokens": {
            "shape": ["B", "N", "D"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "centroids": {
            "shape": ["B", "K", "D"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "assignments": {
            "shape": ["B", "N"],
            "dtype": "int32",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "argmin_k sum_d (tokens[b,n,d] - centroids[b,k,d])^2",
        "accumulator_dtype": "fp32",
        "tie_break": "lowest_index_for_equal_fp32_score",
        "nonfinite": "reject_before_oracle",
    }
    expected_oracle = {
        "kind": "fp32_full_k_chunked",
        "source_sha256": "a74090ca7e5575f5fd6a027600d3330b5fa098078944c089e47c3077af3114d2",
        "chunk_n": 8192,
        "tf32": False,
    }
    expected_validation = {
        "canonical": "exact_int32_ids_with_lowest_index_ties",
        "tie_diagnostic_atol": 0.01,
        "tie_diagnostic_rtol": 0.01,
        "paper_acceptance_rule": "unknown",
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("Flash-KMeans workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("Flash-KMeans workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("Flash-KMeans workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("Flash-KMeans workload validation differs")
    expected_cases = (
        ("tail_nk", {"B": 1, "N": 257, "K": 257, "D": 128}, 1001, "random_standard_normal"),
        ("batched_tail", {"B": 2, "N": 129, "K": 257, "D": 128}, 1002, "random_standard_normal"),
        ("b32_smoke", {"B": 32, "N": 512, "K": 1024, "D": 128}, 1003, "random_standard_normal"),
        ("duplicate_tie", {"B": 1, "N": 4, "K": 4, "D": 128}, None, "constructed_duplicate_centroids"),
        ("public_b1", {"B": 1, "N": 65536, "K": 1024, "D": 128}, 0, "random_standard_normal"),
        ("headline_b32", {"B": 32, "N": 65536, "K": 1024, "D": 128}, 0, "random_standard_normal"),
    )
    raw_cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in raw_cases
    )
    if observed != expected_cases:
        raise ValueError("Flash-KMeans workload cases differ")
    materialized_by_case = {
        cast(str, case["case_id"]): case.get("materialized") for case in raw_cases
    }
    if materialized_by_case["headline_b32"] is None:
        raise ValueError("Flash-KMeans headline materialized authority is missing")
    if revision == "1" and materialized_by_case["b32_smoke"] is not None:
        raise ValueError("Flash-KMeans v1 unexpectedly owns b32 materialization")
    if revision == "2" and materialized_by_case["b32_smoke"] is None:
        raise ValueError("Flash-KMeans v2 b32 materialized authority is missing")


from open_cake_ir.evaluation.workload import WorkloadContract
from .authoring import flash_tensor_abi

class FlashWorkloadContract(WorkloadContract):
    """Runtime view of the frozen Flash contract and its fixed launch ABI."""

    @property
    def target(self) -> str:
        return "sm_100a"

    def tensor_abi(self, case_id: str):
        return flash_tensor_abi(self, case_id)
