"""Frozen Flash ABI and Schedule specialization; never Lab-wide defaults."""
from __future__ import annotations

import json
from open_cake_ir.evaluation.workload import TensorABI

LOWERING_ROUTE = {"backend": "triton", "entry_point": "cake_flash_kmeans_assign"}

def flash_tensor_abi(workload, case_id):
    if workload.document.get("operator") != "flash_kmeans_assign":
        raise ValueError("historical Open Cake Workload ABI is unsupported")
    shape = workload.case(case_id)["shape"]
    return (
        TensorABI("tokens", (shape["B"], shape["N"], shape["D"]), "bf16", "input"),
        TensorABI("centroids", (shape["B"], shape["K"], shape["D"]), "bf16", "input"),
        TensorABI("centroid_sq", (shape["B"], shape["K"]), "fp32", "input"),
        TensorABI("assignments", (shape["B"], shape["N"]), "int32", "output"),
    )

def prepare_flash_schedule(schedule, workload, case_id):
    document = json.loads(json.dumps(schedule))
    buffers = {item["name"]:item for item in document["buffers"]}
    for tensor in flash_tensor_abi(workload, case_id):
        buffers[tensor.name ]["shape"] = list(tensor.shape)
    document["schedule_id"] = "open-cake-ir-matched-authoring-skeleton-v1"
    document["metadata"]["workload_contract_sha256"] = workload.canonical_sha256
    return document

def validate_flash_authoring(workload, arms):
    if workload.document.get("operator") != "flash_kmeans_assign":
        raise ValueError("direct-CUDA task adapter requires Flash-KMeans")
    if arms["open_cake"].get("lowering_route") != LOWERING_ROUTE:
        raise ValueError("Flash-KMeans lowering route differs")
