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
from .tiles import workload as tile_math
from .tiles.workload import validate_tile_contract
from .normalization import workload as normalization_math
from .normalization.authoring import starter_source
from .gemm import workload as gemm_math
from .gemm.authoring import starter_source as gemm_starter_source

_TASKS = {
    "flash_kmeans_assign": (_validate_flash_contract, FlashWorkloadContract),
    "tinygemm2_bf16_linear": (_validate_tinygemm_contract, WorkloadContract),
    "qsa_prefill": (_validate_qsa_contract, WorkloadContract),
    "dsa_attention": (_validate_dsa_contract, WorkloadContract),
    "kda_fused_decode": (_validate_kda_fused_decode_contract, WorkloadContract),
    "kimi_k3_kda_decode_megaop_b200": (_validate_kda_decode_megaop_b200_contract, WorkloadContract),
    "rmsnorm_fp32": (validate_tile_contract, WorkloadContract),
    "gemm_bias_bf16_fp32": (validate_tile_contract, WorkloadContract),
    "indexed_gather_bf16": (validate_tile_contract, WorkloadContract),
    "layernorm_fp32": (normalization_math.validate_normalization_contract, WorkloadContract),
    "residual_rmsnorm_fp32": (normalization_math.validate_normalization_contract, WorkloadContract),
    "gemm_bias_fp32": (gemm_math.validate_gemm_contract, WorkloadContract),
}

def _registered_task(document: Mapping[str, object]):
    """Resolve one document's registered validator and contract type.

    The original tile RMSNorm keeps its own validator; revision 3 is the task-owned
    normalization contract. Every caller routes through this one dispatch.
    """
    operator = document.get("operator") if isinstance(document, Mapping) else None
    task = _TASKS.get(operator) if isinstance(operator, str) else None
    if task is None:
        raise ValueError("workload operator is unsupported")
    validator, contract_type = task
    if operator == "rmsnorm_fp32" and document.get("revision") == "3":
        validator = normalization_math.validate_normalization_contract
    return validator, contract_type


def validate_workload_document(document: Mapping[str, object]) -> None:
    """Route one document to the validator its own operator and revision register."""
    _registered_task(document)[0](document)


def load_workload(path) -> WorkloadContract:
    source = Path(path).resolve(strict=True)
    document = json.loads(source.read_text(encoding="utf-8"))
    validator, contract_type = _registered_task(document)
    return contract_type.from_document(document, source, validate=validator)


def _tensor_math(workload: WorkloadContract):
    """Task-owned routing for the common tensor Evaluation input/oracle interface."""
    operator = workload.document["operator"]
    if (operator in {"layernorm_fp32", "residual_rmsnorm_fp32"}
            or operator == "rmsnorm_fp32" and workload.document["revision"] == "3"):
        return normalization_math
    if operator == "gemm_bias_fp32":
        return gemm_math
    if operator in {"rmsnorm_fp32", "gemm_bias_bf16_fp32", "indexed_gather_bf16"}:
        validate_tile_contract(workload.document)
        return tile_math
    raise ValueError("workload has no registered flat tensor oracle")


def materialize_case(workload: WorkloadContract, case_id: str):
    return _tensor_math(workload).materialize_case(workload, case_id)


def reference_outputs(workload: WorkloadContract, case_id: str, inputs):
    return _tensor_math(workload).reference_outputs(workload, case_id, inputs)


def create_task(task_name: str, *, backend: str = "metal-m1-pro", rows: int = 128,
                columns: int = 1024, depth: int | None = None,
                case_id: str = "primary") -> tuple[dict, str]:
    """Return a real fixed-shape Workload and readable starter for the common Lab.

    The launcher persists these outside the checkout, freezes execution bindings and
    delegates all iteration to Ralph. Every declared input case is required, regardless
    of which case is selected for authoring; all five have the same tensor ABI. GEMM
    owns a third extent because its output column count is unrolled by the Schedule.
    """
    if task_name in gemm_math.TASKS:
        if depth is None:
            raise ValueError("GEMM requires its declared K extent")
        document = gemm_math.workload_document(task_name, backend=backend, rows=rows,
                                               depth=depth, columns=columns)
        return document, gemm_starter_source(WorkloadContract(document), case_id)
    if depth is not None:
        raise ValueError("only GEMM declares a K extent")
    document = normalization_math.workload_document(task_name, backend=backend, rows=rows, columns=columns)
    workload = WorkloadContract(document)
    return document, starter_source(workload, case_id)
