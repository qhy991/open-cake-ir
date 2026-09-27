"""B300 column-tiled GEMV candidates for FlashInfer GEMM task 004."""
from __future__ import annotations

from open_cake_ir.compiler import Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, starter_source, validate_contract


_SMALL_M = frozenset((1, 2, 4, 5, 6, 8))


def _small_m_arguments(workload: WorkloadContract, case_id: str):
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n128_k2048"][0]):
        raise ValueError("B300 GEMV mapping requires FIB 004 on exact B300")
    args = workload.tensor_abi(case_id)
    rows = args[0].shape[0]
    if (rows not in _SMALL_M or args[0].shape != (rows, 2048)
            or args[1].shape != (128, 2048) or args[2].shape != (rows, 128)):
        raise ValueError("B300 GEMV mapping requires an official small M, N=128, K=2048")
    return args


def _m1_arguments(workload: WorkloadContract, case_id: str):
    args = _small_m_arguments(workload, case_id)
    if args[0].shape[0] != 1:
        raise ValueError("column-tiled GEMV requires official M=1")
    return args


def column_tiled_source(workload: WorkloadContract, case_id: str = "primary", *,
                        columns_per_cta: int = 2, execution_groups: int = 4) -> str:
    """Reuse one A row across adjacent B rows in a single CTA.

    Each CTA computes ``columns_per_cta`` output columns with separate FP32
    K reductions and one final FP16 conversion. This mapping is scoped to the
    official M=1 task shape; the generic starter remains the correctness path
    for other batch extents.
    """
    args = _m1_arguments(workload, case_id)
    if type(columns_per_cta) is not int or columns_per_cta not in (2, 4, 8):
        raise ValueError("column-tiled GEMV admits 2, 4 or 8 columns per CTA")
    if type(execution_groups) is not int or execution_groups not in (4, 8):
        raise ValueError("column-tiled GEMV admits 4 or 8 execution groups")
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')')
                    for arg in args]
    body = [
        'stored_a = lm.load(a[row, :], id="load_a")',
        'stored_b = lm.load(b[column, :], id="load_b")',
        'a32 = lm.cast(stored_a, to="fp32", id="widen_a")',
        'b32 = lm.cast(stored_b, to="fp32", id="widen_b")',
        'products = b32 * lm.broadcast(a32, axis=1)',
        'totals = lm.reduce(products, op="sum", axis=1, scope="cta", '
        'across_loop=False, id="sum_k")',
        'rounded = lm.cast(totals, to="fp16", id="round_out")',
        'lm.store(out[row, column], rounded, coalesced=False, id="store_out")',
    ]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}-gemv-n{columns_per_cta}-w{execution_groups}", '
            f'target="{workload.target}", backend="triton", '
            f'entry_point="cake_fib004_gemv_n{columns_per_cta}_w{execution_groups}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            f'    compute = lm.role(execution_groups={list(range(execution_groups))!r})\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            f'    column = lm.program(b, axis=1, dimension=0, tile={columns_per_cta})\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')


def scalar_warp_program(compiler, workload: WorkloadContract,
                        case_id: str = "primary", *, num_warps: int) -> Program:
    """Keep one output per CTA and specialize the starter's execution width.

    This retains M*128 independent CTAs, the starter operation graph and every
    access map. The qualified Compiler pass owns the warp-count legality;
    on-device comparison owns the performance decision.
    """
    _small_m_arguments(workload, case_id)
    starter = parse(starter_source(workload, case_id), filename="fib004_starter.py").document
    program = Program.from_schedule(starter)
    result = compiler.rewrite_program(program, "specialize_triton_warps", {
        "stage": program.stages[0].name,
        "num_warps": num_warps,
        "schedule_id": f"{workload.workload_id}-scalar-w{num_warps}",
        "entry_point": f"cake_fib004_scalar_w{num_warps}",
    })
    if not result.applied or result.program is None:
        raise ValueError(f"FIB 004 scalar warp candidate refused: {result.reason}: {result.message}")
    return result.program
