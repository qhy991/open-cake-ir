from __future__ import annotations
from typing import Mapping,cast
from open_cake_ir.evaluation.workload import _name,_object

def _validate_dsa_contract(document: Mapping[str, object]) -> None:
    """Validate structural and cross-field invariants for sparse MLA decode."""

    if (document.get("workload_id"), document.get("revision")) != (
        "deepseek-v3.2-dsa-sparse-mla-decode-v1",
        "1",
    ):
        raise ValueError("DSA workload identity or revision differs")
    tensors = _object(document.get("tensors"), "DSA tensors")
    semantics = _object(document.get("semantics"), "DSA semantics")
    geometry = _object(semantics.get("geometry"), "DSA geometry")
    matrix = _object(semantics.get("case_matrix"), "DSA case matrix")
    oracle = _object(document.get("oracle"), "DSA oracle")
    baseline = _object(oracle.get("timing_baseline"), "DSA timing baseline")
    validation = _object(document.get("validation"), "DSA validation")
    measurement = _object(validation.get("measurement"), "DSA measurement")
    cases = cast(list[Mapping[str, object]], document["cases"])
    def tensor_shape(name: str) -> list[object]:
        raw = _object(tensors.get(name), f"DSA tensor {name}").get("shape")
        if not isinstance(raw, list):
            raise ValueError(f"DSA tensor {name} shape is invalid")
        return raw

    heads = geometry.get("num_query_heads")
    ckv_dim = geometry.get("compressed_kv_dim")
    rope_dim = geometry.get("rope_dim")
    page_size = geometry.get("page_size")
    top_k = geometry.get("top_k")
    expected_shapes = {
        "q_nope": ["num_tokens", heads, ckv_dim],
        "q_pe": ["num_tokens", heads, rope_dim],
        "ckv_cache": ["num_pages", page_size, ckv_dim],
        "kpe_cache": ["num_pages", page_size, rope_dim],
        "sparse_indices": ["num_tokens", top_k],
        "output": ["num_tokens", heads, ckv_dim],
    }
    if any(tensor_shape(name) != shape for name, shape in expected_shapes.items()):
        raise ValueError("DSA tensor geometry is inconsistent")
    scale = _object(tensors.get("sm_scale"), "DSA sm_scale").get("value")
    scale_authority = _object(
        semantics.get("sm_scale_authority"), "DSA scale authority"
    )
    if scale != scale_authority.get("executed_value"):
        raise ValueError("DSA executed scale is inconsistent")
    if baseline.get("sparse_mla_top_k") != top_k:
        raise ValueError("DSA baseline top-k is inconsistent")
    modes = [case.get("mode") for case in cases]
    if any(not isinstance(mode, str) for mode in modes):
        raise ValueError("DSA case mode is invalid")
    observed_matrix = {
        "total_rows": len(cases),
        "captured_rows": sum(mode.startswith("captured_") for mode in modes),
        "generated_rows": sum(mode.startswith("generated_") for mode in modes),
        "small_rows": sum(mode.endswith("_small") for mode in modes),
        "large_rows": sum(mode.endswith("_large") for mode in modes),
    }
    if set(matrix) != set(observed_matrix) or any(
        matrix.get(name) != count for name, count in observed_matrix.items()
    ):
        raise ValueError("DSA case matrix is inconsistent")
    asset_sources = [
        _object(entry, "DSA provenance entry")
        for entry in cast(list[object], document["provenance"])
        if _object(entry, "DSA provenance entry").get("kind")
        == "task_captured_asset_source"
    ]
    if len(asset_sources) != 1 or asset_sources[0].get("asset_count") != observed_matrix[
        "captured_rows"
    ]:
        raise ValueError("DSA captured asset provenance is inconsistent")
    for name in ("required_cupti_major", "warmup_iterations", "samples_per_trial", "trials"):
        value = measurement.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"DSA measurement {name} is invalid")
