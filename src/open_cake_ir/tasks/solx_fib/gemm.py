"""Exact FlashInfer FP16 A[M,K] @ B[N,K].T tasks and independent CPU oracles.

The starter is a correctness-first scalar-output reduction. It claims no tensor-core
utilization or upstream performance, and does not inspect an upstream implementation.
"""
from __future__ import annotations

from array import array
import json
import math
import random
from collections.abc import Mapping

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, admit_dtype, admit_operations, admit_width, backend_for_target
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
from .workload import row_spans

SPECS = {'fib_gemm_n128_k2048': {'upstream': '004_gemm_n128_k2048',
                         'N': 128,
                         'K': 2048,
                         'batches': [1,
                                     2,
                                     4,
                                     5,
                                     6,
                                     8,
                                     16,
                                     17,
                                     25,
                                     32,
                                     34,
                                     63,
                                     64,
                                     93,
                                     128,
                                     172,
                                     289,
                                     492,
                                     952,
                                     8828,
                                     11006,
                                     12251,
                                     12853,
                                     14915,
                                     16294]},
 'fib_gemm_n256_k7168': {'upstream': '005_gemm_n256_k7168',
                         'N': 256,
                         'K': 7168,
                         'batches': [1,
                                     4,
                                     14,
                                     15,
                                     16,
                                     32,
                                     53,
                                     54,
                                     55,
                                     56,
                                     57,
                                     58,
                                     63,
                                     80,
                                     901,
                                     11948,
                                     14104]},
 'fib_gemm_n2048_k4096': {'upstream': '006_gemm_n2048_k4096',
                          'N': 2048,
                          'K': 4096,
                          'batches': [1,
                                      2,
                                      4,
                                      5,
                                      6,
                                      8,
                                      15,
                                      16,
                                      17,
                                      25,
                                      32,
                                      34,
                                      63,
                                      64,
                                      93,
                                      128,
                                      172,
                                      289,
                                      492,
                                      952,
                                      969,
                                      8828,
                                      11006,
                                      11938,
                                      12251,
                                      12853,
                                      14915,
                                      15813,
                                      16294]},
 'fib_gemm_n4096_k4096': {'upstream': '007_gemm_n4096_k4096',
                          'N': 4096,
                          'K': 4096,
                          'batches': [1,
                                      2,
                                      4,
                                      7,
                                      8,
                                      15,
                                      16,
                                      24,
                                      32,
                                      35,
                                      40,
                                      48,
                                      56,
                                      64,
                                      70,
                                      72,
                                      80,
                                      88,
                                      96,
                                      104,
                                      112,
                                      120,
                                      128,
                                      136,
                                      144,
                                      152,
                                      160,
                                      168,
                                      176,
                                      184,
                                      192,
                                      200,
                                      208,
                                      216,
                                      224,
                                      232,
                                      240,
                                      248,
                                      256,
                                      972,
                                      2053,
                                      2379,
                                      8192]},
 'fib_gemm_n4096_k14336': {'upstream': '008_gemm_n4096_k14336',
                           'N': 4096,
                           'K': 14336,
                           'batches': [1,
                                       2,
                                       4,
                                       7,
                                       8,
                                       15,
                                       16,
                                       24,
                                       32,
                                       35,
                                       40,
                                       48,
                                       56,
                                       64,
                                       70,
                                       72,
                                       80,
                                       88,
                                       96,
                                       104,
                                       112,
                                       120,
                                       128,
                                       136,
                                       144,
                                       152,
                                       160,
                                       168,
                                       176,
                                       184,
                                       192,
                                       200,
                                       208,
                                       216,
                                       224,
                                       232,
                                       240,
                                       248,
                                       256,
                                       972,
                                       2053,
                                       2379,
                                       8192]},
 'fib_gemm_n5120_k2048': {'upstream': '009_gemm_n5120_k2048',
                          'N': 5120,
                          'K': 2048,
                          'batches': [1,
                                      2,
                                      4,
                                      5,
                                      6,
                                      8,
                                      16,
                                      17,
                                      25,
                                      32,
                                      34,
                                      63,
                                      64,
                                      93,
                                      128,
                                      172,
                                      289,
                                      492,
                                      952,
                                      8828,
                                      11006,
                                      12251,
                                      12853,
                                      14915,
                                      16294]},
 'fib_gemm_n6144_k4096': {'upstream': '010_gemm_n6144_k4096',
                          'N': 6144,
                          'K': 4096,
                          'batches': [1,
                                      2,
                                      4,
                                      7,
                                      8,
                                      15,
                                      16,
                                      24,
                                      32,
                                      35,
                                      40,
                                      48,
                                      56,
                                      64,
                                      70,
                                      72,
                                      80,
                                      88,
                                      96,
                                      104,
                                      112,
                                      120,
                                      128,
                                      136,
                                      144,
                                      152,
                                      160,
                                      168,
                                      176,
                                      184,
                                      192,
                                      200,
                                      208,
                                      216,
                                      224,
                                      232,
                                      240,
                                      248,
                                      256,
                                      972,
                                      2053,
                                      2379,
                                      8192]},
 'fib_gemm_n28672_k4096': {'upstream': '011_gemm_n28672_k4096',
                           'N': 28672,
                           'K': 4096,
                           'batches': [1,
                                       2,
                                       4,
                                       7,
                                       8,
                                       15,
                                       16,
                                       24,
                                       32,
                                       35,
                                       40,
                                       48,
                                       56,
                                       64,
                                       70,
                                       72,
                                       80,
                                       88,
                                       96,
                                       104,
                                       112,
                                       120,
                                       128,
                                       136,
                                       144,
                                       152,
                                       160,
                                       168,
                                       176,
                                       184,
                                       192,
                                       200,
                                       208,
                                       216,
                                       224,
                                       232,
                                       240,
                                       248,
                                       256,
                                       972,
                                       2053,
                                       2379,
                                       8192]}}
TASKS = {name: ("solx_" + name + "_fp16", "1") for name in SPECS}
CASES = ("primary", "zeros", "near_zero", "alternating", "mixed_magnitude")


def workload_document(task_name: str, *, rows: int = 1, columns: int | None = None,
                      depth: int | None = None, backend: str = "triton-b300") -> dict:
    if task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported FlashInfer GEMM task/backend")
    spec = SPECS[task_name]
    n, k = spec["N"], spec["K"]
    if columns not in (None, n) or depth not in (None, k):
        raise ValueError("FlashInfer GEMM N/K are upstream constants")
    if type(rows) is not int or rows not in spec["batches"]:
        raise ValueError("FlashInfer GEMM M must be a declared upstream batch")
    if max(rows*k, n*k, rows*n) * 2 > 2**31 - 1:
        raise ValueError("FlashInfer GEMM shape exceeds the tensor ABI")
    admit_dtype(backend, "fp16")
    admit_operations(backend, ("load", "cast", "elementwise", "reduce", "store"))
    for start, stop in row_spans(k):
        admit_width(backend, stop-start)
    operator, revision = TASKS[task_name]
    tensors = {"a": {"shape": ["M", "K"], "max_abs": 2.0},
               "b": {"shape": ["N", "K"], "max_abs": 2.0},
               "out": {"shape": ["M", "N"]}}
    for tensor in tensors.values():
        tensor.update(dtype="fp16", layout="contiguous_row_major", finite_only=True)
    return {
        "schema_version": 1, "workload_id": f"{operator.replace('_','-')}-{backend}-m{rows}-v1",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "solx_pack_task", "upstream": "flashinfer-bench",
             "path": "flashinfer-bench-tasks/tasks/" + spec["upstream"],
             "scope": "semantic_definition_only_no_candidate_or_timing_import"},
            {"kind": "upstream_definition", "path": "flashinfer_trace/definitions/gemm/" + task_name[4:] + ".json",
             "scope": "FP16_A_MK_times_transposed_B_NK"},
            {"kind": "restricted_artifact", "path": "flashinfer-bench-tasks/tasks/" + spec["upstream"] + "/baseline/",
             "scope": "complete_target_implementation"},
        ],
        "cases": [{"case_id": name, "shape": {"M": rows, "K": k, "N": n},
                   "seed": 2701+i, "mode": name} for i,name in enumerate(CASES)],
        "tensors": tensors,
        "semantics": {
            "definition": "out[m,n] = round_fp16(sum_k(a[m,k] * b[n,k]))",
            "target": BACKENDS[backend]["target"],
            "candidate_abi": {"inputs": ["a", "b"], "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": {"operands": "fp16", "accumulator": "fp32", "output": "fp16"},
            "materialization": "seed_plus_10000_times_input_index; uniform[-2,2]; zeros; uniform[-1e-3,1e-3]; alternating +/-0.5; signed powers two [-8,1]; round_fp16",
            "upstream_batches": spec["batches"],
            "exclusions": ["one_declared_M_only", "no_upstream_random_generator_equivalence", "no_device_or_performance_qualification"],
        },
        "oracle": {"kind": "real_gemm_fsum_then_fp16", "callable": "open_cake_ir.tasks.solx_fib.gemm.reference_outputs",
                   "implementation": "independent_standard_library_math_no_compiler_or_candidate", "output_rounding": "round_to_nearest_ties_to_even"},
        "validation": {"primary_case": "primary", "all_cases_required": True, "equal_nan": False,
                       "comparison": "elementwise_atol_rtol", "atol": 1e-2, "rtol": 1e-2,
                       "qualification": "all_elements_all_five_cases_at_one_M; device_validation_pending",
                       "tolerance_rationale": "Uses the pack numerical tolerance with every element required, not matched_ratio 0.99; FP32 accumulation order may differ from fsum before FP16 rounding."},
    }


def validate_contract(document: Mapping[str, object]) -> None:
    workload = WorkloadContract(document)
    task = next((name for name, (operator, rev) in TASKS.items()
                 if document.get("operator") == operator and document.get("revision") == rev), None)
    backend = backend_for_target(workload.target)
    if task is None or backend is None or workload.case_ids != CASES:
        raise ValueError("FlashInfer GEMM operator, target or cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"M", "N", "K"}:
        raise ValueError("FlashInfer GEMM requires M/N/K")
    expected = workload_document(task, rows=shape["M"], columns=shape["N"], depth=shape["K"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("FlashInfer GEMM frozen contract differs")
    for case in CASES:
        workload.tensor_abi(case)


def materialize_case(workload: WorkloadContract, case_id: str):
    validate_contract(workload.document)
    case = workload.case(case_id)
    result = {}
    for position, arg in enumerate(a for a in workload.tensor_abi(case_id) if a.mode == "input"):
        rng = random.Random(case["seed"] + 10000*position)
        values = array("f")
        for i in range(math.prod(arg.shape)):
            if case_id == "zeros": value = 0.0
            elif case_id == "near_zero": value = rng.uniform(-1e-3, 1e-3)
            elif case_id == "alternating": value = (1 if (i + position) % 2 else -1) * 0.5
            elif case_id == "mixed_magnitude": value = (1 if i % 2 else -1) * 2.0**rng.randint(-8,1)
            else: value = rng.uniform(-2,2)
            values.append(_round(value, "fp16"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str, inputs):
    validate_contract(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    a,b = _checked_inputs(workload,args,inputs)
    m,k = args[0].shape
    n = args[1].shape[0]
    return {"out": [_round(math.fsum(a[row*k+t]*b[col*k+t] for t in range(k)), "fp16")
                    for row in range(m) for col in range(n)]}


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    validate_contract(workload.document)
    args = workload.tensor_abi(case_id)
    k = args[0].shape[1]
    body = []
    for i,(start,stop) in enumerate(row_spans(k)):
        body.extend([
            f'a_{i} = lm.load(a[row, {start}:{stop}], id="load_a_{i}")',
            f'b_{i} = lm.load(b[column, {start}:{stop}], id="load_b_{i}")',
            f'a32_{i} = lm.cast(a_{i}, to="fp32", id="cast_a_{i}")',
            f'b32_{i} = lm.cast(b_{i}, to="fp32", id="cast_b_{i}")',
            f'products_{i} = a32_{i} * b32_{i}',
            f'sum_{i} = lm.reduce(products_{i}, op="sum", axis=0, scope="cta", across_loop=False, id="sum_{i}")',
        ])
    total = 'sum_0'
    if len(row_spans(k)) > 1:
        body.append('total = '+' + '.join(f'sum_{i}' for i in range(len(row_spans(k)))))
        total = 'total'
    body.extend([f'rounded = lm.cast({total}, to="fp16", id="round_out")',
                 'lm.store(out[row, column], rounded, coalesced=False, id="store_out")'])
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'+(', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}", backend="{BACKENDS[backend_for_target(workload.target)]["route"]}",\n'
            f'               entry_point="cake_fib_gemm")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(execution_groups=[0])\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            '    column = lm.program(b, axis=1, dimension=0, tile=1)\n'
            '    with compute:\n        '+'\n        '.join(body)+'\n')


def _fib008_arguments(workload: WorkloadContract, case_id: str):
    validate_contract(workload.document)
    shape = workload.case(case_id)["shape"]
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n4096_k14336"][0]):
        raise ValueError("column reuse is scoped to FlashInfer GEMM 008 on sm_103a")
    args = workload.tensor_abi(case_id)
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"' +
                    (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return shape, declarations


def column_reuse_source(workload: WorkloadContract, case_id: str = "primary", *,
                        columns_per_program: int = 4, k_tile: int = 256) -> str:
    """Explicit B300 008 candidate: reuse one A tile across output columns.

    The reduction is carried across K tiles. This changes the reduction order, so
    the candidate still needs the external oracle and device measurement before use.
    """
    shape, declarations = _fib008_arguments(workload, case_id)
    for name, value, extent in (("columns_per_program", columns_per_program, shape["N"]),
                                ("k_tile", k_tile, shape["K"])):
        if type(value) is not int or value <= 0 or value & (value - 1) or extent % value:
            raise ValueError(f"{name} must be a power of two that divides its axis")
    if columns_per_program not in (2, 4, 8) or k_tile not in (128, 256, 512):
        raise ValueError("column reuse candidate requires 2/4/8 columns and 128/256/512 K tiles")
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}-column-reuse-c{columns_per_program}-k{k_tile}", '
            f'target="{workload.target}", backend="triton", entry_point="cake_fib008_column_reuse")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            f'    column = lm.program(b, axis=1, dimension=0, tile={columns_per_program})\n'
            f'    for k in lm.range(a, name="k_loop", dimension=1, tile={k_tile}, num_stages=1):\n'
            '        with compute:\n'
            '            av = lm.load(a[row, k], id="load_a")\n'
            '            bv = lm.load(b[column, k], id="load_b")\n'
            '            a32 = lm.cast(av, to="fp32", id="cast_a")\n'
            '            b32 = lm.cast(bv, to="fp32", id="cast_b")\n'
            '            products = b32 * lm.broadcast(a32, axis=1)\n'
            '            accum = lm.reduce(products, op="sum", axis=1, scope="cta", id="sum_k")\n'
            '    with compute:\n'
            '        rounded = lm.cast(accum, to="fp16", id="round_out")\n'
            '        lm.store(out[row, column], rounded, coalesced=False, id="store_out")\n')


def column_reuse_unrolled_source(workload: WorkloadContract,
                                 case_id: str = "primary", *,
                                 columns_per_program: int = 2) -> str:
    """FIB 008 column reuse with three static K reductions, without a tile loop."""
    shape, declarations = _fib008_arguments(workload, case_id)
    if type(columns_per_program) is not int or columns_per_program not in (2, 4):
        raise ValueError("static column reuse admits two or four columns per program")
    body = []
    spans = row_spans(shape["K"])
    for index, (start, stop) in enumerate(spans):
        body.extend([
            f'a_{index} = lm.load(a[row, {start}:{stop}], id="load_a_{index}")',
            f'b_{index} = lm.load(b[column, {start}:{stop}], id="load_b_{index}")',
            f'a32_{index} = lm.cast(a_{index}, to="fp32", id="cast_a_{index}")',
            f'b32_{index} = lm.cast(b_{index}, to="fp32", id="cast_b_{index}")',
            f'products_{index} = b32_{index} * lm.broadcast(a32_{index}, axis=1)',
            f'sum_{index} = lm.reduce(products_{index}, op="sum", axis=1, '
            f'scope="cta", across_loop=False, id="sum_{index}")',
        ])
    body.extend([
        'total = ' + ' + '.join(f'sum_{index}' for index in range(len(spans))),
        'rounded = lm.cast(total, to="fp16", id="round_out")',
        'lm.store(out[row, column], rounded, coalesced=False, id="store_out")',
    ])
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}-column-reuse-unrolled-c{columns_per_program}", '
            f'target="{workload.target}", backend="triton", '
            f'entry_point="cake_fib008_column_reuse_unrolled_c{columns_per_program}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            f'    column = lm.program(b, axis=1, dimension=0, tile={columns_per_program})\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
