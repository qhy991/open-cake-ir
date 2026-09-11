"""Standalone CSA-indexer and MoE-gate tasks derived from DeepSeek-V4-Pro.

The official inference source owns the model-wide implementation.  These are deliberately
narrow, fixed-shape task contracts: they establish independently checkable routing behavior
without claiming full CSA sparse attention, expert dispatch, TP communication, or serving.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import TensorABI, WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from open_cake_ir.tasks.tiles.workload import _round


MODEL_REPOSITORY = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"
MODEL_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
MODEL_PATH = "inference/model.py"
MODEL_CONFIG_PATH = "config.json"
DEEPSELECT_REPOSITORY = "https://github.com/deepseek-ai/DeepSelect"

TASKS = {
    "csa_indexer_topk": ("deepseek_v4_csa_indexer_topk_fp32", "1"),
    "moe_gate": ("deepseek_v4_moe_gate_fp32", "1"),
}
CASES = {
    "csa_indexer_topk": {
        "primary": ("uniform_distinct", 9501), "skewed": ("skewed_distinct", 9502),
        "boundary": ("boundary_distinct", 9503), "tail": ("tail_distinct", 9504),
    },
    "moe_gate": {
        "primary": ("uniform_distinct", 9601), "bias_changes_selection": ("biased_distinct", 9602),
        "negative": ("negative_distinct", 9603), "tail": ("tail_distinct", 9604),
    },
}
CSA_CANDIDATE_COUNT = 2048
CSA_TOPK = 1024
MOE_EXPERTS = 384
MOE_TOPK = 6
MOE_ROUTE_SCALE = 2.5


def workload_document(task_name: str, *, backend: str = "triton-b200", rows: int = 4) -> dict:
    """Return one fixed B200 contract for an official V4-Pro routing-stage slice."""
    if task_name not in TASKS or backend != "triton-b200" or type(rows) is not int or rows <= 0:
        raise ValueError("unsupported DeepSeek-V4 workload or target")
    operator, revision = TASKS[task_name]
    if task_name == "csa_indexer_topk":
        tensors = {
            "index_scores": {"shape": ["Q", "S"], "max_abs": 16.0},
            "indices": {"shape": ["Q", "K"]},
        }
        shape = {"Q": rows, "S": CSA_CANDIDATE_COUNT, "K": CSA_TOPK}
        inputs, outputs = ["index_scores"], ["indices"]
        definition = "indices[q,:] = the 1024 descending score positions from index_scores[q,:]"
        arithmetic = {"scores": "fp32_routing_scores", "selection": "topk_descending_unique_scores", "indices": "int32"}
        scope = "steady_state_indexer_only; candidate_count=2048 and topk=1024; excludes short-context variable_topk"
    else:
        tensors = {
            "router_logits": {"shape": ["T", "E"], "max_abs": 16.0},
            "selection_bias": {"shape": ["E"], "max_abs": 4.0},
            "weights": {"shape": ["T", "K"]},
            "expert_ids": {"shape": ["T", "K"]},
        }
        shape = {"T": rows, "E": MOE_EXPERTS, "K": MOE_TOPK}
        inputs, outputs = ["router_logits", "selection_bias"], ["weights", "expert_ids"]
        definition = "score=sqrt(softplus(router_logits)); expert_ids=topk(score+selection_bias,6); weights=2.5*score[expert_ids]/sum(score[expert_ids])"
        arithmetic = {"score_function": "sqrtsoftplus", "selection_bias": "selection_only", "topk": 6, "route_scale": 2.5, "indices": "int32"}
        scope = "score_routed_layers_only; excludes first_three_hash_routed_layers, local_expert_dispatch, all_reduce, shared_expert, and expert_MLP"
    for descriptor in tensors.values():
        descriptor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    for name in outputs:
        if name in {"indices", "expert_ids"}:
            tensors[name]["dtype"] = "int32"
            tensors[name].pop("finite_only")
    return {
        "schema_version": 1, "workload_id": f"{operator.replace('_', '-')}-{backend}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "official_model_source", "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
             "path": MODEL_PATH, "scope": "CSA Indexer and MoE Gate formulas, state ownership, and exclusions"},
            {"kind": "official_model_config", "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
             "path": MODEL_CONFIG_PATH, "scope": "V4-Pro geometry: 61 layers, index_topk=1024, 384 routed experts, topk=6"},
            {"kind": "official_csa_topk_kernel_source", "repository": DEEPSELECT_REPOSITORY,
             "scope": "DeepSeek Sparse Attention TopK supported dtype and output ownership contract"},
        ],
        "cases": [{"case_id": name, "shape": shape, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES[task_name].items()],
        "tensors": tensors,
        "semantics": {
            "model": "DeepSeek-V4-Pro", "model_revision": MODEL_REVISION, "target": BACKENDS[backend]["target"],
            "definition": definition, "candidate_abi": {"inputs": inputs, "outputs": outputs},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic, "scope": scope,
            "tie_policy": "input scores after selection bias must be unique; ties are outside this standalone deterministic contract",
        },
        "oracle": {"kind": f"real_{task_name}_standard_library_reference",
                   "callable": "open_cake_ir.tasks.deepseek_v4.workload.reference_outputs",
                   "implementation": "independent_standard_library_math_no_model_or_generated_source",
                   "output_rounding": "round_to_nearest_ties_to_even"},
        "validation": {"primary_case": "primary", "all_cases_required": True, "equal_nan": False,
                       "comparison": "bitwise_int32_plus_elementwise_atol_rtol", "atol": 2e-6, "rtol": 2e-5,
                       "qualification": "CPU_oracle_only; Open-Cake_B200_compile_correctness_sanitizer_timing_profile_pending",
                       "performance_domain": "future_B200_protocol_must_use_four_workloads_spanning_at_least_1000x_work"},
    }


def _task_name(document: Mapping[str, object]) -> str:
    return next((name for name, identity in TASKS.items()
                 if document.get("operator") == identity[0] and document.get("revision") == identity[1]), "")


def validate_deepseek_v4_contract(document: Mapping[str, object]) -> None:
    workload = WorkloadContract(document)
    task_name = _task_name(document)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if not task_name or backend != "triton-b200" or workload.case_ids != tuple(CASES[task_name]):
        raise ValueError("DeepSeek-V4 task identity, target, or case protocol differs")
    shape = workload.case("primary")["shape"]
    rows = shape["Q"] if task_name == "csa_indexer_topk" else shape["T"]
    expected = workload_document(task_name, backend=backend, rows=rows)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("DeepSeek-V4 frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _checked_inputs(workload: WorkloadContract, args: tuple[TensorABI, ...], inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    if not isinstance(inputs, Mapping) or set(inputs) != {arg.name for arg in args}:
        raise ValueError("oracle inputs must match the complete input ABI")
    result: dict[str, list[float]] = {}
    tensors = workload.document["tensors"]
    for arg in args:
        values = inputs[arg.name]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != math.prod(arg.shape):
            raise ValueError(f"oracle input {arg.name} must be a complete flat row-major vector")
        bound = tensors[arg.name]["max_abs"]
        if any(type(value) not in {int, float} or not math.isfinite(value) or abs(value) > bound or _round(value, "fp32") != value for value in values):
            raise ValueError(f"oracle input {arg.name} must contain finite bounded FP32 values")
        result[arg.name] = list(values)
    return result


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    validate_deepseek_v4_contract(workload.document)
    task_name = _task_name(workload.document)
    case = workload.case(case_id)
    result: dict[str, list[float]] = {}
    for position, arg in enumerate(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"):
        rng = random.Random(case["seed"] + position * 10000)
        values = []
        for index in range(math.prod(arg.shape)):
            if case["mode"] == "negative_distinct":
                value = -1.0 - index / (math.prod(arg.shape) + 1)
            elif case["mode"] == "skewed_distinct":
                value = 8.0 - index / (math.prod(arg.shape) + 1)
            elif case["mode"] == "boundary_distinct":
                value = 16.0 - 32.0 * index / max(math.prod(arg.shape) - 1, 1)
            elif case["mode"] == "biased_distinct" and arg.name == "selection_bias":
                value = 4.0 - index / (math.prod(arg.shape) + 1)
            elif case["mode"] == "tail_distinct":
                value = (index % 31) / 32.0 + rng.uniform(-1e-5, 1e-5)
            else:
                value = rng.uniform(-2.0, 2.0) + index / (math.prod(arg.shape) + 1) * 1e-3
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    return result


def _rank_unique(values: Sequence[float], width: int, count: int) -> list[int]:
    selected: list[int] = []
    for row in range(len(values) // width):
        segment = values[row * width:(row + 1) * width]
        if len(set(segment)) != width:
            raise ValueError("standalone V4 routing contracts refuse ambiguous score ties")
        selected.extend(sorted(range(width), key=lambda index: segment[index], reverse=True)[:count])
    return selected


def reference_outputs(workload: WorkloadContract, case_id: str, inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float | int]]:
    validate_deepseek_v4_contract(workload.document)
    task_name = _task_name(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = _checked_inputs(workload, args, inputs)
    shape = workload.case(case_id)["shape"]
    if task_name == "csa_indexer_topk":
        return {"indices": _rank_unique(checked["index_scores"], shape["S"], shape["K"])}
    scores = [_round(math.sqrt(math.log1p(math.exp(value))), "fp32") for value in checked["router_logits"]]
    selection = [score + checked["selection_bias"][expert]
                 for row in range(shape["T"]) for expert, score in enumerate(scores[row * shape["E"]:(row + 1) * shape["E"]])]
    expert_ids = _rank_unique(selection, shape["E"], shape["K"])
    weights = []
    for row in range(shape["T"]):
        ids = expert_ids[row * shape["K"]:(row + 1) * shape["K"]]
        chosen = [scores[row * shape["E"] + expert] for expert in ids]
        total = math.fsum(chosen)
        weights.extend(_round(MOE_ROUTE_SCALE * value / total, "fp32") for value in chosen)
    return {"weights": weights, "expert_ids": expert_ids}
