#!/usr/bin/env python3
"""Development comparison of an unchanged external RMSNorm and a Cake starter.

No provider calls, kernel edits, Campaign promotion or inherited upstream scores.
Inputs are staged by GPU Infra; its broker owns the exclusive allocation.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.gpuq import observe_allocation
from open_cake_ir.evaluation.paired import paired_protocol, paired_summary
from open_cake_ir.evaluation.timing import summarize_cohort
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.lab.pairing import bind_baseline
from open_cake_ir.tasks.evaluate import _fresh_tile_cohort
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def admit_judge_source(commit, task, stage_id):
    """The frozen Infra stage, not its directory name, binds the judge commit."""
    stages = [stage for stage in task.get("stages", []) if stage.get("id") == stage_id]
    if (len(stages) != 1
            or stages[0].get("judge", {}).get("identity") != "open-cake-ir@" + commit):
        raise ValueError("comparison judge source differs from the frozen Infra task")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def input_spec(root):
    document = json.loads((root / "comparison.json").read_bytes())
    if (not isinstance(document, dict) or set(document) != {"task", "rows", "columns", "target"}
            or document["task"] != "fib_rmsnorm_h2048"
            or document["target"] != "sm_103a" or document["columns"] != 2048
            or type(document["rows"]) is not int or document["rows"] <= 0):
        raise ValueError("this comparison binds the existing h2048 BF16 B300 task")
    for name in ("submission.py", "kernel.py"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("reference requires regular submission.py and kernel.py")
    return document


def cake_candidate(root, workload, default_source):
    """Admit an optional authored Cake candidate through the existing ABI owner."""
    path = root / "candidate.py"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Cake candidate must be a regular source file")
    source = path.read_text(encoding="utf-8") if path.is_file() else default_source
    schedule = bind_baseline(frontend.parse(source).document, workload, "primary", backend="triton")
    return source, schedule, "authored Cake candidate" if path.is_file() else "generated Cake starter"


class LoadedCallable:
    """Use the common fresh-output cohort checker with task-owned Python launchers."""
    def __init__(self, workload, values, launch, torch):
        from array import array
        self.launch_function = launch
        self.torch = torch
        self.abi = workload.tensor_abi("primary")
        dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16}
        outputs = [a for a in self.abi if a.mode == 'output']
        if (len(outputs) != 1 or outputs[0].name != 'out'
                or any(a.dtype not in dtypes for a in self.abi)):
            raise ValueError('external callable requires a supported single-output ABI')
        self.output_dtype = dtypes[outputs[0].dtype]
        self.inputs = {a.name: torch.tensor(values[a.name], dtype=dtypes[a.dtype],
            device="cuda:0").reshape(a.shape) for a in self.abi if a.mode == "input"}
        self.output_shape = outputs[0].shape
        self.validation_inputs = {name:array('d', value) for name,value in values.items()}

    def fresh_argument_sets(self, count):
        return [{"inputs": {k: v.clone() for k, v in self.inputs.items()},
                 "out": self.torch.full(self.output_shape, float("nan"),
                     dtype=self.output_dtype, device="cuda:0"), "result": None}
                for _ in range(count)]

    def launch(self, arguments):
        result = self.launch_function(arguments["inputs"], arguments["out"])
        if (not isinstance(result, self.torch.Tensor) or result.dtype != self.output_dtype
                or tuple(result.shape) != tuple(self.output_shape) or not result.is_contiguous()
                or result.device.type != "cuda" or result.device.index != 0):
            raise ValueError("candidate output ABI differs")
        if any(result.data_ptr() == value.data_ptr() for value in arguments["inputs"].values()):
            raise ValueError("output aliases a public input")
        arguments["result"] = result

    def snapshot(self, arguments):
        from array import array
        observed = {"out": arguments["result"].cpu().flatten().tolist()}
        after = {k: array('d',v.cpu().flatten().tolist()) for k, v in arguments["inputs"].items()}
        return observed, after


def main():
    output = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    inputs_root = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    report = {"scope": "development_comparison_not_Campaign_or_promotion",
              "provider_calls": 0, "promotion_disposition": "No promotion",
              "roles": {"candidate": "generated Cake starter", "baseline": "unchanged external reference"},
              "cases": [], "measurements": []}
    try:
        config = input_spec(inputs_root)
        report.update(input=config, source_commit=checkout_commit(ROOT))
        admit_judge_source(report["source_commit"],
            json.loads(Path(os.environ["KERNELINFRA_TASK"]).read_bytes()), os.environ["KERNELINFRA_STAGE_ID"])
        report["allocation"] = observe_allocation(config["target"])
        # Observe exclusive exact hardware before importing Torch initializes CUDA.
        admission = observe_exclusive_cuda(config["target"])
        report["device"] = {"name": admission.device_name, "job_id": admission.broker_job_id}
        import torch
        import triton
        import flashinfer.testing as timer
        torch.set_num_threads(8)
        sys.dont_write_bytecode = True
        document, source = create_task(config["task"], backend="triton-b300",
            rows=config["rows"], columns=config["columns"])
        workload = WorkloadContract(document)
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        gate = compiler.check_corpus()
        if not gate.passed:
            raise ValueError("Corpus Gate failed; no candidate launch")
        source, schedule, role = cake_candidate(inputs_root, workload, source)
        report["roles"]["candidate"] = role
        lowering = compiler.lower(compiler.assess(schedule))
        write(output / "workload.json", document)
        (output / "candidate.cake.py").write_text(source)
        write(output / "schedule.json", schedule)
        (output / "lowered.py").write_text(lowering.source)
        cake = getattr(load_module("cake_comparison", output / "lowered.py"), lowering.route.entry_point)
        sys.path.insert(0, str(inputs_root))
        reference = load_module("external_submission", inputs_root / "submission.py").run
        report["runtime"] = {"torch": torch.__version__, "triton": triton.__version__}
        launches = {"candidate": lambda v, out: cake(**v, out=out),
                    "baseline": lambda v, out: reference(v["x"], v["weight"])}
        primary = None
        def check_cases(phase):
            nonlocal primary
            for case_id in workload.case_ids:
                values = materialize_case(workload, case_id)
                expected = reference_outputs(workload, case_id, values)
                for role, launch in launches.items():
                    loaded = LoadedCallable(workload, values, launch, torch)
                    arguments = loaded.fresh_argument_sets(1)[0]
                    loaded.launch(arguments)
                    torch.cuda.synchronize()
                    observed, after = loaded.snapshot(arguments)
                    correct, metrics = compare_tile_outputs(workload, values, expected, observed, after)
                    row = {"phase": phase, "case_id": case_id, "role": role,
                           "passed": correct, "metrics": metrics}
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
                    if not correct:
                        raise ArithmeticError(f"{role} failed {case_id}; no tolerance changes")
                if case_id == "primary":
                    primary = values, expected
        check_cases("preflight")
        autotuner = getattr(sys.modules.get("kernel"), "_rmsnorm_kernel", None)
        selected = getattr(autotuner, "best_config", None)
        report["reference_autotune"] = {"selection": str(selected),
            "kwargs": dict(selected.kwargs) if selected is not None else None,
            "num_warps": getattr(selected, "num_warps", None),
            "num_stages": getattr(selected, "num_stages", None),
            "scope": "original autotuner runs before authoritative timing; its scores are not reported"}
        policy = evaluation_policy(workload)
        protocol = paired_protocol(policy)
        report["evaluation_protocol"] = policy
        values, expected = primary
        loaded = {role: LoadedCallable(workload, values, launch, torch) for role, launch in launches.items()}
        strict = StrictCuptiBenchmark(timer)
        for index, order in enumerate(protocol.pair_order):
            pair = {"pair_index": index, "order": list(order), "arms": {}}
            for position, role in enumerate(order):
                samples, check = _fresh_tile_cohort(loaded[role], strict, workload, values, expected,
                    samples_per_cohort=protocol.samples_per_cohort,
                    route_calls_per_cohort=protocol.route_calls_per_cohort)
                if not check["passed"]:
                    raise ArithmeticError("timed output validation failed")
                pair["arms"][role] = {"position": position, "samples_ms": samples,
                    "summary": summarize_cohort(samples), "route_calls": check["checked_launches"],
                    "output_check": check}
            report["measurements"].append(pair)
            print(json.dumps({"pair": index, "medians": {r: v["summary"]["median_ms"]
                  for r, v in pair["arms"].items()}}), flush=True)
        check_cases("postflight")
        report["timing"] = paired_summary(report)
        report["timed_interval"] = "CUPTI device kernel interval; cold L2 each sample; no CUDA-event/graph fallback; no host allocation or autotune cost claim"
        report["status"] = "passed"
        validity = "valid"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        validity = "invalid" if isinstance(error, ArithmeticError) else "unknown"
    write(output / "comparison-report.json", report)
    # Keep Lab acceptance and GPU Infra frontier separate. No generic timing row
    # is projected from this development test into a promotion decision.
    write(result_path, {"schema": "kernelinfra.stage-result.v1",
        "status": "passed" if validity == "valid" else "failed", "validity": validity,
        "summary": report.get("error", "Both implementations checked under the existing task"),
        "artifacts": {"comparison": "comparison-report.json"}})
    return 0 if validity != "unknown" else 1


if __name__ == "__main__":
    raise SystemExit(main())
