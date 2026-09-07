"""Static task selection; each task owns its exact semantic validator."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping
from open_cake_ir.evaluation.workload import WorkloadContract
from .flash_kmeans.contract import _validate_flash_contract,FlashWorkloadContract
from .tinygemm.contract import _validate_tinygemm_contract
from .qsa.workload import _validate_qsa_contract
from .dsa.contract import _validate_dsa_contract
from .kda.contract import _validate_kda_fused_decode_contract, _validate_kda_decode_megaop_b200_contract
from .tiles.workload import validate_tile_contract
from .amd.contract import _validate_swiglu_contract, _validate_llama_rmsnorm_contract, _validate_llama_q4_mmvq_contract

_TASKS = {
    "swiglu_fp32": (_validate_swiglu_contract, WorkloadContract),
    "rmsnorm_mul_fp32": (_validate_llama_rmsnorm_contract, WorkloadContract),
    "llama_q4_0_q8_1_mmvq_f32": (_validate_llama_q4_mmvq_contract, WorkloadContract),
    "flash_kmeans_assign": (_validate_flash_contract, FlashWorkloadContract),
    "tinygemm2_bf16_linear": (_validate_tinygemm_contract, WorkloadContract),
    "qsa_prefill": (_validate_qsa_contract, WorkloadContract),
    "dsa_attention": (_validate_dsa_contract, WorkloadContract),
    "kda_fused_decode": (_validate_kda_fused_decode_contract, WorkloadContract),
    "kimi_k3_kda_decode_megaop_b200": (_validate_kda_decode_megaop_b200_contract, WorkloadContract),
    "rmsnorm_fp32": (validate_tile_contract, WorkloadContract),
    "gemm_bias_bf16_fp32": (validate_tile_contract, WorkloadContract),
    "indexed_gather_bf16": (validate_tile_contract, WorkloadContract),
}

def load_workload(path) -> WorkloadContract:
    source = Path(path).resolve(strict=True)
    document = json.loads(source.read_text(encoding="utf-8"))
    operator = document.get("operator") if isinstance(document, dict) else None
    task = _TASKS.get(operator) if isinstance(operator, str) else None
    if task is None:
        raise ValueError("workload operator is unsupported")
    validator, contract_type = task
    return contract_type.from_document(document, source, validate=validator)
