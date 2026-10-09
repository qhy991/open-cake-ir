"""Prepare L1/069 using the frozen CAKE Compiler and its isolated native compiler.

This program performs no GPU work. Keep the output outside the Compiler checkout.
Raw Bench definitions are read in place and are not copied into the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

COMMIT = "5bb474c6df2948d1cc50b2d46f9ab828525be3c2"
TASK = "L1/069_rms_norm"


def source_for(rows: int, width: int, eps: float) -> str:
    return f'''from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="bench_l1_069_r{rows}_h{width}", target="xcore1002",
               backend="triton", entry_point="cake_bench_l1_069")
def candidate(lm, x: cake.Tensor(({rows}, {width}), "bf16"),
              residual: cake.Tensor(({rows}, {width}), "bf16"),
              weight: cake.Tensor(({width},), "bf16"),
              out: cake.Tensor(({rows}, {width}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        stored_x = lm.load(x[row, :], id="load_x")
        x_fp32 = lm.cast(stored_x, to="fp32", id="widen_x")
        stored_residual = lm.load(residual[row, :], id="load_residual")
        residual_fp32 = lm.cast(stored_residual, to="fp32", id="widen_residual")
        sum_fp32 = x_fp32 + residual_fp32
        sum_bf16 = lm.cast(sum_fp32, to="bf16", id="round_residual_sum")
        values = lm.cast(sum_bf16, to="fp32", id="widen_rounded_sum")
        squares = lm.square(values, id="square")
        square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")
        mean_square = square_sum / {float(width)!r}
        inverse = lm.rsqrt(mean_square + {eps!r}, id="inverse")
        normalized = values * inverse
        normalized_bf16 = lm.cast(normalized, to="bf16", id="round_before_weight")
        normalized_fp32 = lm.cast(normalized_bf16, to="fp32", id="widen_normalized")
        stored_weight = lm.load(weight[:], id="load_weight")
        weights = lm.cast(stored_weight, to="fp32", id="widen_weight")
        weighted = normalized_fp32 * weights
        narrowed = lm.cast(weighted, to="bf16", id="round_output")
        lm.store(out[row, :], narrowed, coalesced=False, id="store_out")
'''


def read_cases(bench: Path) -> list[dict]:
    task_root = bench / ".data/benchmark" / TASK
    definition = json.loads((task_root / "definition.json").read_text())
    if (definition["inputs"]["hidden_states"]["dtype"] != "bfloat16"
            or definition["inputs"]["residual"]["dtype"] != "bfloat16"
            or definition["inputs"]["weight"]["dtype"] != "bfloat16"
            or definition["outputs"]["output"]["dtype"] != "bfloat16"):
        raise ValueError("The fixed L1/069 tensor types differ")
    constants = {name: item["value"] for name, item in definition["axes"].items()
                 if item["type"] == "const"}
    cases = []
    for line in (task_root / "workload.jsonl").read_text().splitlines():
        row = json.loads(line)
        axes = {**constants, **row["axes"]}
        shape = [axes[dim] for dim in definition["inputs"]["hidden_states"]["shape"]]
        scalar = row["inputs"]["eps"]
        if scalar["type"] != "scalar" or len(shape) != 3:
            raise ValueError("The fixed L1/069 scalar or tensor ABI differs")
        cases.append({"uuid": row["uuid"], "shape": shape, "eps": scalar["value"]})
    if len(cases) != 16 or len({row["uuid"] for row in cases}) != 16:
        raise ValueError("The fixed Bench task must contain 16 distinct original workloads")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assess-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.compiler_root / "src"))
    from open_cake_ir.compiler.core import Compiler
    from open_cake_ir.compiler.frontend import parse
    from open_cake_ir.compiler.toolchain import project_triton_kernel
    from open_cake_ir.compiler.metax_toolchain import device_image, native_pointer_parameters
    from open_cake_ir.source_identity import checkout_commit

    compiler = Compiler.load(args.compiler_root)
    if compiler.commit != COMMIT:
        raise ValueError("The Compiler checkout must be clean and fixed at C1 5bb474c6")
    adapter_commit = checkout_commit(Path(__file__).resolve().parents[3])
    args.output.mkdir(parents=True, exist_ok=False)
    cases = read_cases(args.bench_root)
    index = {"task": TASK, "compiler_commit": COMMIT, "adapter_commit": adapter_commit,
             "target": "xcore1002",
             "cases": cases, "variants": {}, "status": "preparing",
             "semantics": "BF16 residual sum; FP32 RMSNorm; BF16 normalized value; BF16 weight product"}
    (args.output / "index.json").write_text(json.dumps(index, indent=2))
    isolated = None
    if not args.assess_only:
        sys.path.insert(0, str(args.compiler_root / "tools"))
        from launch_task import _triton_toolchain_config
        from open_cake_ir.lab.executor import ExecutorRevision
        from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
        executor = ExecutorRevision.for_target(args.compiler_root, "xcore1002")
        admission = executor.admit_host()
        (args.output / "host-admission.json").write_text(json.dumps(admission, indent=2))
        index["runtime_library"] = admission["runtime_library"]
        isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
        isolated.check_executor(executor, author_workspace=args.output)
    for case in cases:
        batch, seq, width = case["shape"]
        rows = batch * seq
        variant_id = f"r{rows}_h{width}_eps{case['eps']}"
        case["variant"] = variant_id
        if variant_id in index["variants"]:
            continue
        directory = args.output / variant_id
        directory.mkdir()
        source = source_for(rows, width, case["eps"])
        (directory / "candidate.cake.py").write_text(source)
        document = parse(source).document
        (directory / "schedule.json").write_text(json.dumps(document, indent=2))
        assessment = compiler.assess(document)
        (directory / "assessment.json").write_text(json.dumps({
            "accepted": assessment.accepted, "lowering_eligible": assessment.lowering_eligible,
            "findings": [{"code": row.code, "message": row.message,
                          "blocks_lowering": row.blocks_lowering} for row in assessment.findings]}, indent=2))
        if not assessment.lowering_eligible:
            raise ValueError(f"C1 refused {variant_id}; see retained assessment")
        lowered = compiler.lower(assessment)
        (directory / "lowered.triton.py").write_text(lowered.source)
        requirements = dict(lowered.toolchain_requirements)
        (directory / "requirements.json").write_text(json.dumps(requirements, indent=2))
        kernel = project_triton_kernel(lowered.source.encode(), requirements)
        (directory / "kernel.triton.py").write_bytes(kernel)
        entry = {"rows": rows, "width": width, "eps": case["eps"],
                 "kernel_name": requirements["kernel_entry_point"], "grid": requirements["grid"]}
        if isolated is not None:
            try:
                compiled = isolated.compile(kernel, requirements)
            except Exception as error:
                (directory / "compile-failure.txt").write_text(f"{type(error).__name__}: {error}\n")
                for name, payload in getattr(error, "artifact_payloads", {}).items():
                    if name in {"toolchain_stdout", "toolchain_stderr"}:
                        (directory / name).write_bytes(payload)
                index["status"] = "compile_failed"
                index["failed_variant"] = variant_id
                (args.output / "index.json").write_text(json.dumps(index, indent=2))
                raise
            for role, payload in compiled.artifacts.items():
                (directory / role).write_bytes(payload)
            hidden = native_pointer_parameters(compiled.artifacts["mcfatbin"],
                requirements["codegen_arch"], compiled.entry_point) - 4
            if hidden not in (0, 2):
                raise ValueError("The compiled pointer ABI has an unqualified hidden parameter count")
            native = device_image(compiled.artifacts["mcfatbin"], requirements["codegen_arch"])
            (directory / "native.elf").write_bytes(native)
            entry.update({"hidden_null_pointers": hidden,
                          "block": [compiled.threads_per_cta, 1, 1],
                          "dynamic_shared_bytes": compiled.dynamic_shared_bytes,
                          "compiler_version": compiled.compiler_version})
        index["variants"][variant_id] = entry
        (args.output / "index.json").write_text(json.dumps(index, indent=2))
        print(f"prepared {variant_id}", flush=True)
    index["status"] = "assessed" if args.assess_only else "compiled"
    (args.output / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
