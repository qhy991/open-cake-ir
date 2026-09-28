"""Exact NVIDIA CTA-width candidates for FlashInfer fused add RMSNorm 001/002."""
from __future__ import annotations

from open_cake_ir.compiler import Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract

from .authoring import starter_source
from .workload import TASKS, validate_solx_fib_contract


TASK_NAMES = ("fib_fused_add_rmsnorm_h2048", "fib_fused_add_rmsnorm_h4096")
OPERATORS = frozenset(TASKS[name][0] for name in TASK_NAMES)
TARGETS = frozenset(("sm_100a", "sm_103a"))


def candidate_program(compiler, workload: WorkloadContract,
                      case_id: str = "primary", *, num_warps: int) -> Program:
    """Apply the qualified Compiler rewrite to a complete, exact Workload Program.

    The pass retains the Schedule's graph and access maps. Its applicability and
    lowering gates, rather than this task adapter, own the warp-count decision.
    """
    validate_solx_fib_contract(workload.document)
    if workload.target not in TARGETS or workload.document["operator"] not in OPERATORS:
        raise ValueError("fused add RMSNorm width candidate requires an exact B200 or B300 Workload")
    starter = parse(starter_source(workload, case_id), filename="fib_add_rmsnorm_starter.py").document
    program = Program.from_schedule(starter)
    width = workload.tensor_abi(case_id)[0].shape[-1]
    result = compiler.rewrite_program(program, "specialize_triton_warps", {
        "stage": program.stages[0].name,
        "num_warps": num_warps,
        "schedule_id": f"{workload.workload_id}-cta-w{num_warps}",
        "entry_point": f"cake_fib_add_rmsnorm_h{width}_w{num_warps}",
    })
    if not result.applied or result.program is None:
        raise ValueError(f"fused add RMSNorm CTA-width candidate refused: {result.reason}: {result.message}")
    return result.program
