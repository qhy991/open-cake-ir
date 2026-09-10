"""BF16 residual add + RMSNorm with two fresh outputs and an explicit rounding seam.

Derived from OMOE's add_reference/reference semantics. The model's in-place binding
is deliberately outside this tensor task; both returned values are checked here.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping

from open_cake_ir.evaluation.workload import WorkloadContract
from .devices import BACKENDS, backend_for_target
from .tiles.workload import _checked_inputs, _round

TASK = "add_rmsnorm_bf16"
EPSILON = 1e-6
CASES = {"primary": ("uniform", 7601), "zeros": ("zeros", 7602),
         "rounding_ties": ("rounding_ties", 7603),
         "cancellation": ("cancellation", 7604),
         "mixed_magnitude": ("mixed_magnitude", 7605)}
BACKENDS_SUPPORTED = ("triton-b200", "triton-b300")


def workload_document(*, rows: int = 128, columns: int = 2560,
                      backend: str = "triton-b200") -> dict:
    if (backend not in BACKENDS_SUPPORTED or type(rows) is not int or type(columns) is not int
            or rows <= 0 or not 1 <= columns <= 16384 or rows * columns * 2 > 2**31 - 1):
        raise ValueError("add-RMSNorm requires a B200/B300 Triton target and bounded BF16 R/C shape")
    tensors = {name: {"shape": ["C"] if name == "weight" else ["R", "C"],
                      "dtype": "bf16", "layout": "contiguous_row_major", "finite_only": True,
                      **({"max_abs": 1.5 if name == "weight" else 256.0}
                         if name in {"delta", "residual", "weight"} else {})}
               for name in ("delta", "residual", "weight", "out", "residual_out")}
    return {
        "schema_version": 1, "state": "frozen", "revision": "1", "operator": TASK,
        "workload_id": f"add-rmsnorm-bf16-{backend}-r{rows}-c{columns}-v1",
        "provenance": [{"kind": "derived_operator_semantics", "path": "omoe/ops/rmsnorm.py",
                        "repository": "https://github.com/qhy991/omoe",
                        "commit": "19356783c9c58f49a603472c3ecfd9db92bb608f", "symbols": ["add_reference", "reference"],
                        "scope": "fresh_dual_output_tensor_task_not_inplace_model_integration"}],
        "cases": [{"case_id": name, "mode": mode, "seed": seed,
                   "shape": {"R": rows, "C": columns}} for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "target": BACKENDS[backend]["target"],
            "definition": "z=RN_bf16(delta+residual); residual_out=z; out=RN_bf16(fp32(z)*rsqrt(mean(fp32(z)^2)+epsilon)*fp32(weight))",
            "candidate_abi": {"inputs": ["delta", "residual", "weight"], "outputs": ["out", "residual_out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "epsilon": EPSILON,
            "arithmetic": {"residual_add": "fp32_add_then_bf16_round_to_nearest_ties_even",
                           "normalization_input": "the_same_rounded_residual_as_residual_out",
                           "reduction": "fp32_backend_order_within_tolerance",
                           "output": "bf16_round_to_nearest_ties_even"},
            "materialization": "task_owned_seeded_bf16_inputs_with_ties_cancellation_and_mixed_magnitudes_weight0_one_every_17th_zero",
        },
        "oracle": {"kind": "bf16_residual_then_independent_fsum_rmsnorm",
                   "callable": "open_cake_ir.tasks.add_rmsnorm.reference_outputs",
                   "implementation": "standard_library_math_without_candidate_or_compiler"},
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "per_output",
            "outputs": {
                "out": {"comparison": "elementwise_atol_rtol", "atol": 2**-16, "rtol": 1/64},
                "residual_out": {"comparison": "bitwise_bf16", "atol": 0.0, "rtol": 0.0}},
            "tolerance_rationale": "Normalized BF16 output allows two relative BF16 rounding units plus a 2^-16 near-zero floor; residual bytes are exact. This is a prospective tensor contract, not measured qualification or a model top-1 rule.",
            "qualification": "all_distributions_at_fixed_shape_no_framework_or_cross_target_claim",
        },
    }


def validate_contract(document: Mapping) -> None:
    workload = WorkloadContract(document)
    shape = workload.case("primary")["shape"]
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if set(shape) != {"R", "C"}:
        raise ValueError("add-RMSNorm requires exact R/C dimensions")
    expected = workload_document(rows=shape["R"], columns=shape["C"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("add-RMSNorm semantic, ABI, distribution or comparison contract differs")
    for case in workload.case_ids:
        workload.tensor_abi(case)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    validate_contract(workload.document)
    case = workload.case(case_id)
    count = case["shape"]["R"] * case["shape"]["C"]
    rng = random.Random(case["seed"])
    delta, residual = [], []
    for i in range(count):
        if case_id == "zeros":
            d, r = (-0.0, -0.0) if i % 2 else (0.0, 0.0)
        elif case_id == "rounding_ties":
            d, r = 2**-8, 1.0 + (i % 2) * 2**-7
        elif case_id == "cancellation":
            r = _round(rng.uniform(-2, 2), "bf16")
            d = -r if i % 2 else _round(-r + 2**-8, "bf16")
        elif case_id == "mixed_magnitude":
            d, r = ((-1)**i * 2.0**rng.randint(-12, 8),
                    (-1)**(i+1) * 2.0**rng.randint(-12, 8))
        else:
            d, r = rng.uniform(-2, 2), rng.uniform(-2, 2)
        delta.append(_round(d, "bf16")); residual.append(_round(r, "bf16"))
    weights = [_round(1.0 if i == 0 else (0.0 if (i + 1) % 17 == 0 else rng.uniform(-1.5, 1.5)), "bf16")
               for i in range(case["shape"]["C"])]
    return {"delta": delta, "residual": residual, "weight": weights}


def reference_outputs(workload: WorkloadContract, case_id: str, inputs: Mapping) -> dict:
    validate_contract(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    delta, residual, weight = _checked_inputs(workload, args, inputs)
    z = [_round(d + r, "bf16") for d, r in zip(delta, residual, strict=True)]
    width = args[0].shape[-1]
    out = []
    for start in range(0, len(z), width):
        row = z[start:start+width]
        inverse = 1.0 / math.sqrt(math.fsum(v*v for v in row) / width + EPSILON)
        out.extend(_round(v * inverse * w, "bf16") for v, w in zip(row, weight, strict=True))
    return {"out": out, "residual_out": z}


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    validate_contract(workload.document)
    args = workload.tensor_abi(case_id)
    width = args[0].shape[-1]
    # Keep the reduction vector-valued even at width one; masked lanes are zero.
    tile = max(2, 1 << (width - 1).bit_length())
    declarations = [f'{a.name}: cake.Tensor({a.shape!r}, "{a.dtype}"'
                    + (', mode="output")' if a.mode == "output" else ')') for a in args]
    body = [
        'd = lm.load(delta[row, col], id="load_delta")',
        'r = lm.load(residual[row, col], id="load_residual")',
        'w = lm.load(weight[col], id="load_weight")',
        'df = lm.cast(d, to="fp32", id="delta_fp32")',
        'rf = lm.cast(r, to="fp32", id="residual_fp32")',
        'wf = lm.cast(w, to="fp32", id="weight_fp32")',
        'summed = df + rf',
        'rounded = lm.cast(summed, to="bf16", id="round_residual")',
        'lm.store(residual_out[row, col], rounded, coalesced=False, id="store_residual")',
        'z = lm.cast(rounded, to="fp32", id="rounded_fp32")',
        'squares = lm.square(z, id="square")',
        'total = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")',
        f'inverse = lm.rsqrt(total / {float(width)!r} + {EPSILON!r}, id="inverse")',
        'normalized = z * inverse', 'weighted = normalized * wf',
        'result = lm.cast(weighted, to="bf16", id="round_output")',
        'lm.store(out[row, col], result, coalesced=False, id="store_out")',
    ]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            '               backend="triton", entry_point="cake_add_rmsnorm",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0, 1, 2, 3])\n'
            '    row = lm.program(delta, axis=0, dimension=0, tile=1)\n'
            f'    col = lm.program(delta, axis=1, dimension=1, tile={tile})\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
