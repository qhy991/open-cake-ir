"""Prepare matched IR/native baselines through Compiler; never execute target code."""

from __future__ import annotations
from open_cake_ir.tasks.workloads import load_workload

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks.tiles.workload import source_schedule_path
from open_cake_ir.evaluation.workload import WorkloadContract

ROOT = Path(__file__).resolve().parents[2]


def baseline_schedule(workload: WorkloadContract, case_id: str, *, project_root: Path = ROOT) -> dict:
    """Specialize the visible corpus seed; Workload ABI owns every global dimension.

    The small per-seed edits below describe its existing register tiles and reduction
    constant. They are Schedule authoring decisions, not Compiler routing or another
    Workload shape table. There is no copy of the Triton emitter here.
    """

    document = workload.document
    source_path = source_schedule_path(document)
    root = project_root.resolve(strict=True)
    source = (root / source_path).resolve(strict=True)
    if not source.is_relative_to(root / "corpus/schedules"):
        raise ValueError("baseline source must be a corpus Schedule")
    schedule = json.loads(source.read_text(encoding="utf-8"))
    primary_abi = workload.tensor_abi(document["validation"]["primary_case"])
    globals_ = [buffer for buffer in schedule["buffers"] if buffer["space"] == "global"]
    if [(b["name"], tuple(b["shape"]), b["dtype"], b["mode"]) for b in globals_] != [
        (arg.name, arg.shape, arg.dtype, arg.mode) for arg in primary_abi
    ]:
        raise ValueError("source Schedule does not match the Workload primary ABI")
    if schedule["target"] != document["semantics"]["target"] or schedule["lowering"]["backend"] != "triton":
        raise ValueError("baseline target or lowering backend differs")
    abi = workload.tensor_abi(case_id)
    for buffer, arg in zip(globals_, abi):
        buffer["shape"] = list(arg.shape)
    operator = document["operator"]
    buffers = {buffer["name"]: buffer for buffer in schedule["buffers"]}
    operations = {operation["id"]: operation for operation in schedule["operations"]}
    if operator == "rmsnorm_fp32":
        width = abi[0].shape[-1]
        for name in ("x_tile", "sq", "gamma_tile", "normed", "y_tile"):
            buffers[name]["shape"][-1] = width
        operations["mean"]["parameters"]["scalar"] = 1.0 / width
        operations["shift"]["parameters"]["scalar"] = document["semantics"]["epsilon"]
    elif operator == "indexed_gather_bf16":
        slots = abi[1].shape[-1]
        width = abi[0].shape[-1]
        for name in ("expert_id_tile", "row_id_tile"):
            buffers[name]["shape"] = [slots]
        buffers["gathered_tile"]["shape"] = [slots, width]
        if document["semantics"]["indexing"]["invalid"] != "positive_zero":
            raise ValueError("seed only implements zero-masked invalid indices")
    elif operator != "gemm_bias_bf16_fp32":
        raise ValueError("no existing baseline for this Workload")
    schedule["schedule_id"] = f"{workload.workload_id}-{case_id}-baseline"
    schedule["metadata"]["workload_contract_sha256"] = workload.canonical_sha256
    return schedule


def prepare_baseline(workload: WorkloadContract, case_id: str, output_root: Path, *, project_root: Path = ROOT, compiler_revision: Path | None = None) -> dict:
    """Create Schedule, exact Compiler output, and a source-only preparation record.

    The native source is a projection of this Schedule, not a second implementation.
    Compilation/sealing and common Evaluation must occur later via their existing APIs.
    """

    root = project_root.resolve(strict=True)
    output = output_root.resolve()
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("baseline preparation output must be outside every checkout")
    compiler = Compiler.load(root, compiler_revision or root / "compiler/revision.lock.json")
    assessment = compiler.assess(baseline_schedule(workload, case_id, project_root=root))
    if not assessment.accepted or not assessment.lowering_eligible:
        findings = "; ".join(f"{finding.code}: {finding.path}: {finding.message}" for finding in assessment.findings)
        raise ValueError(f"baseline refused by Compiler: {findings}")
    lowering = compiler.lower(assessment)
    if not lowering.generated:
        raise ValueError("baseline must use generated Compiler lowering")
    # Python syntax only: imports and target code are never executed here.
    compile(lowering.source, "baseline.triton.py", "exec")
    record = {
        "workload_id": workload.workload_id,
        "workload_path": str(workload.source_path) if workload.source_path else None,
        "case_id": case_id,
        "abi": [asdict(arg) for arg in workload.tensor_abi(case_id)],
        "compiler_revision_id": lowering.compiler_revision_id,
        "schedule_id": lowering.schedule_id,
        "entry_point": lowering.route.entry_point,
        "schedule_file": "baseline.schedule.json",
        "native_source_file": "baseline.triton.py",
        "toolchain_requirements": dict(lowering.toolchain_requirements),
        "source_map": dict(lowering.source_map),
        "status": "source_only",
        "target_compilation": "not_run",
        "gpu_correctness": "R2_pending",
        "timing_and_profiler": "R2_pending",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.schedule.json").write_bytes(assessment.schedule_bytes)
    (output / "baseline.triton.py").write_text(lowering.source, encoding="utf-8")
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--case", default="primary")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compiler-revision", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_baseline(load_workload(args.workload), args.case, args.output_root, compiler_revision=args.compiler_revision), indent=2))
