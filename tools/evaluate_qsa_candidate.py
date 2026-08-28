#!/usr/bin/env python3
"""Judge one immutable QSA Program candidate inside a GPU Infra staged run."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, Schedule  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    NCU_ATTRIBUTION_METRICS,
    ProgramContract,
    WorkloadContract,
    audit_qsa_output,
    build_ncu_attribution_profile,
    materialize_qsa_case,
    ncu_attribution_feedback,
    reference_qsa_output,
)
from open_cake_ir.evaluation.portfolio_runtime import (  # noqa: E402
    StrictCuptiBenchmark,
)
from open_cake_ir.evaluation.qsa_cuda import (  # noqa: E402
    LoadedQsaProgram,
    QsaProgramArtifact,
    qsa_program_tensors,
)
from open_cake_ir.lab import (  # noqa: E402
    BuildRequest,
    ExecutorRevision,
    TritonToolchainBuilder,
)

_STAGE_SCHEMA = "kernelinfra.stage-result.v1"
_WORKLOAD_ID = "qsa-prefill-t32768"
_PROGRAM_PATH = "contracts/programs/qsa-prefill-t32768-v1.json"
_WORKLOAD_PATH = "contracts/workloads/qsa-prefill-t32768-v1.json"
_DIRECT_SOURCE = "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.cu"
_DIRECT_MANIFEST = "src/open_cake_ir/evaluation/assets/qsa_direct_reference_v1.json"
_OPEN_CAKE_ORDER = ("pool", "layernorm", "score_topk", "expand", "attention")
_DIRECT_ORDER = ("pool_layernorm", "score_topk", "expand", "attention")


class _BaselineCompileError(RuntimeError):
    """The task-owned fixed reference failed before candidate attribution."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _owned_file(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} is unsafe")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} custody differs")
    return path


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is missing")
    return Path(value).resolve(strict=True)


def _required_output_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is missing")
    unresolved = Path(value).absolute()
    parent = unresolved.parent.resolve(strict=True)
    return parent / unresolved.name


def _write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _stage_result(
    result_path: Path,
    *,
    status: str,
    validity: str,
    summary: str,
    workloads: list[dict[str, object]] | None = None,
    artifacts: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
) -> int:
    document: dict[str, object] = {
        "schema": _STAGE_SCHEMA,
        "status": status,
        "validity": validity,
        "summary": summary,
        "workloads": workloads or [],
    }
    if artifacts:
        document["artifacts"] = dict(artifacts)
    if metrics:
        document["metrics"] = dict(metrics)
    _write_json(result_path, document)
    return 0 if status == "passed" else 1


def _executor(root: Path) -> ExecutorRevision:
    inventory = json.loads(
        (root / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
    )
    current = _object(inventory["current"], "Executor inventory current")
    executor = ExecutorRevision.load(root, root / str(current["path"]))
    if (
        executor.executor_id != current["executor_id"]
        or executor.canonical_sha256 != current["canonical_sha256"]
    ):
        raise ValueError("current Executor Revision differs")
    task = json.loads(_required_environment_path("KERNELINFRA_TASK").read_text())
    stage_id = os.environ.get("KERNELINFRA_STAGE_ID")
    stages = [stage for stage in task["stages"] if stage["id"] == stage_id]
    expected_identity = f"{executor.executor_id}@{executor.canonical_sha256}"
    if len(stages) != 1 or stages[0]["judge"]["identity"] != expected_identity:
        raise ValueError("GPU Infra task does not bind the current Executor Revision")
    return executor


def _candidate_document(candidate_root: Path) -> Mapping[str, object]:
    source = _owned_file(candidate_root, "candidate.json", "candidate descriptor")
    document = _object(json.loads(source.read_text(encoding="utf-8")), "candidate")
    arm = document.get("arm")
    expected = (
        {"schema_version", "arm", "nodes"}
        if arm == "open_cake"
        else {"schema_version", "arm", "source", "launch_manifest"}
    )
    if set(document) != expected or document.get("schema_version") != 1:
        raise ValueError("QSA candidate fields differ")
    return document


def _global_abi(schedule: Schedule) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        buffer.name: (buffer.shape, buffer.dtype.value)
        for buffer in schedule.buffers
        if buffer.space.value == "global"
    }


def _compile_open_cake(
    root: Path,
    candidate_root: Path,
    candidate: Mapping[str, object],
    output: Path,
) -> None:
    compiler = Compiler.load(root, root / "compiler/revision.lock.json")
    if not compiler.check_corpus().passed:
        raise RuntimeError("released Compiler Corpus Gate no longer passes")
    program = ProgramContract.load(root, root / _PROGRAM_PATH, compiler)
    raw_nodes = candidate.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Open Cake candidate nodes differ")
    by_id: dict[str, Path] = {}
    for index, raw in enumerate(raw_nodes):
        node = _object(raw, f"candidate.nodes[{index}]")
        if set(node) != {"id", "schedule"} or node.get("id") in by_id:
            raise ValueError(f"candidate.nodes[{index}] differs")
        by_id[str(node["id"])] = _owned_file(
            candidate_root,
            node["schedule"],
            f"candidate.nodes[{index}].schedule",
        )
    if tuple(by_id) != _OPEN_CAKE_ORDER:
        raise ValueError("Open Cake candidate node order differs")
    output.mkdir(parents=True, exist_ok=False)
    manifest_kernels: list[dict[str, object]] = []
    toolchain = TritonToolchainBuilder()
    canonical_nodes = {node.node_id: node for node in program.nodes}
    for node_id in _OPEN_CAKE_ORDER:
        schedule_path = by_id[node_id]
        candidate_schedule = Schedule.load(schedule_path)
        canonical_schedule = Schedule.load(canonical_nodes[node_id].schedule_path)
        if _global_abi(candidate_schedule) != _global_abi(canonical_schedule):
            raise ValueError(f"Open Cake node {node_id!r} changes the Program ABI")
        assessment = compiler.assess_file(schedule_path)
        if not assessment.accepted or not assessment.lowering_eligible:
            codes = ",".join(finding.code for finding in assessment.findings)
            raise ValueError(f"Open Cake node {node_id!r} rejected: {codes}")
        lowering = compiler.lower(assessment)
        request = BuildRequest(
            candidate_sha256=lowering.schedule_sha256,
            source=lowering.source.encode("utf-8"),
            source_role="lowered_source",
            source_sha256=lowering.source_sha256,
            target=lowering.target,
            entry_point=lowering.route.entry_point,
            toolchain_requirements=lowering.toolchain_requirements,
        )
        try:
            launchable = toolchain.build(request)
        except Exception as error:
            raise ValueError(
                f"Open Cake node {node_id!r} toolchain rejected the lowering: {error}"
            ) from error
        node_root = output / node_id
        node_root.mkdir()
        for role in ("lowered_source", "ptx", "cubin"):
            (node_root / role).write_bytes(launchable.artifact_payloads[role])
        launch = json.loads(launchable.artifact_payloads["launch_manifest"])
        manifest_kernels.append(
            {
                "id": node_id,
                "cubin": f"candidate/{node_id}/cubin",
                "kernel_name": launch["kernel_name"],
                "grid": launch["grid"],
                "block": launch["block"],
                "dynamic_shared_memory_bytes": launch[
                    "dynamic_shared_memory_bytes"
                ],
                "arguments": list(lowering.toolchain_requirements["signature"]),
                "hidden_null_pointer_parameters": 2,
            }
        )
    _write_json(
        output / "program.json",
        {
            "schema_version": 1,
            "abi": program.abi,
            "arm": "open_cake",
            "kernels": manifest_kernels,
        },
    )


def _checked_direct_manifest(path: Path) -> list[Mapping[str, object]]:
    document = _object(json.loads(path.read_text(encoding="utf-8")), "direct manifest")
    if (
        set(document) != {"schema_version", "abi", "kernels"}
        or document.get("schema_version") != 1
        or document.get("abi") != "qsa_prefill_task_geometry_v1"
        or not isinstance(document.get("kernels"), list)
    ):
        raise ValueError("direct QSA launch manifest fields differ")
    kernels = cast(list[Mapping[str, object]], document["kernels"])
    if tuple(str(item.get("id")) for item in kernels) != _DIRECT_ORDER:
        raise ValueError("direct QSA launch order differs")
    expected_arguments = {
        "pool_layernorm": ["index_k", "k_norm_weight", "normalized_keys"],
        "score_topk": ["index_q", "normalized_keys", "block_indices"],
        "expand": ["block_indices", "token_indices"],
        "attention": ["q", "k", "v", "token_indices", "output"],
    }
    for index, kernel in enumerate(kernels):
        if set(kernel) != {
            "id",
            "kernel_name",
            "grid",
            "block",
            "dynamic_shared_memory_bytes",
            "arguments",
        } or kernel.get("arguments") != expected_arguments[kernel["id"]]:
            raise ValueError(f"direct QSA kernel {index} fields differ")
    return kernels


def _run_tool(
    command: list[str],
    *,
    output: Path,
    name: str,
    timeout: int = 900,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=timeout,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
        },
    )
    (output / f"{name}.stdout.log").write_bytes(completed.stdout)
    (output / f"{name}.stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _compile_direct(
    source: Path,
    launch_manifest: Path,
    output: Path,
    *,
    nvcc: Path,
    cuobjdump: Path,
    arm: str,
) -> None:
    kernels = _checked_direct_manifest(launch_manifest)
    output.mkdir(parents=True, exist_ok=False)
    cubin = output / "program.cubin"
    ptx = output / "program.ptx"
    common = [str(nvcc), "-std=c++17", "-O3", "-arch=sm_100a", "-lineinfo"]
    cubin_run = _run_tool(
        common + ["--cubin", str(source), "-o", str(cubin)],
        output=output,
        name="nvcc-cubin",
    )
    ptx_run = _run_tool(
        common + ["--ptx", str(source), "-o", str(ptx)],
        output=output,
        name="nvcc-ptx",
    )
    sass_run = _run_tool(
        [str(cuobjdump), "--dump-sass", str(cubin)],
        output=output,
        name="cuobjdump-sass",
    )
    (output / "program.sass").write_bytes(sass_run.stdout)
    (output / "compile.stdout.log").write_bytes(cubin_run.stdout + ptx_run.stdout)
    (output / "compile.stderr.log").write_bytes(cubin_run.stderr + ptx_run.stderr)
    if not cubin.read_bytes().startswith(b"\x7fELF"):
        raise ValueError("NVCC did not produce an sm_100a CUBIN")
    _write_json(
        output / "program.json",
        {
            "schema_version": 1,
            "abi": "qsa_prefill_task_geometry_v1",
            "arm": arm,
            "kernels": [
                {
                    **dict(kernel),
                    "cubin": f"{output.name}/program.cubin",
                    "hidden_null_pointer_parameters": 0,
                }
                for kernel in kernels
            ],
        },
    )


def _compile_stage(
    root: Path,
    candidate_root: Path,
    build_root: Path,
    *,
    nvcc: Path,
    cuobjdump: Path,
) -> tuple[str, Mapping[str, object]]:
    candidate = _candidate_document(candidate_root)
    baseline = build_root / "baseline"
    try:
        _compile_direct(
            root / _DIRECT_SOURCE,
            root / _DIRECT_MANIFEST,
            baseline,
            nvcc=nvcc,
            cuobjdump=cuobjdump,
            arm="direct_cuda",
        )
    except Exception as error:
        diagnostic = (
            error.stderr.decode("utf-8", errors="replace")[-2048:]
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise _BaselineCompileError(diagnostic) from error
    arm = str(candidate["arm"])
    candidate_output = build_root / "candidate"
    if arm == "open_cake":
        _compile_open_cake(root, candidate_root, candidate, candidate_output)
    elif arm == "direct_cuda":
        _compile_direct(
            _owned_file(candidate_root, candidate["source"], "direct candidate source"),
            _owned_file(
                candidate_root,
                candidate["launch_manifest"],
                "direct candidate launch manifest",
            ),
            candidate_output,
            nvcc=nvcc,
            cuobjdump=cuobjdump,
            arm="direct_cuda",
        )
    else:
        raise ValueError("QSA candidate arm differs")
    return arm, {
        "candidate_program": "qsa-build/candidate/program.json",
        "baseline_program": "qsa-build/baseline/program.json",
    }


def _admit_gpu() -> object:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        os.environ.get("KERNELINFRA_STAGE_KIND")
        not in {"correctness", "benchmark", "profile"}
        or not os.environ.get("KERNELINFRA_RUN_ID")
        or not visible
        or "," in visible
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA B200"
        or torch.cuda.get_device_capability(0) != (10, 0)
    ):
        raise ValueError("broker-visible QSA device differs")
    return {
        "device_name": "NVIDIA B200",
        "compute_capability": [10, 0],
        "visible_device": visible,
    }


def _load_program(build_root: Path, name: str) -> LoadedQsaProgram:
    artifact_root = build_root / name
    artifact = QsaProgramArtifact.load(build_root, artifact_root / "program.json")
    return LoadedQsaProgram(artifact)


def _case(root: Path):
    workload = WorkloadContract.load(root / _WORKLOAD_PATH)
    inputs = materialize_qsa_case(workload, "target_t32768", "cuda")
    return workload, inputs


def _correctness_stage(root: Path, build_root: Path, stage_dir: Path) -> tuple[bool, dict[str, object]]:
    import torch

    _admit_gpu()
    workload, inputs = _case(root)
    output = torch.empty(
        (1, 32768, 32, 128), dtype=torch.bfloat16, device="cuda"
    )
    tensors = qsa_program_tensors(inputs, output[0])
    program = _load_program(build_root, "candidate")
    try:
        with torch.no_grad():
            program.launch(tensors, stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            expected = reference_qsa_output(workload, inputs)
            observation = audit_qsa_output(workload, output, expected)
        metrics = observation.as_dict()
        _write_json(stage_dir / "correctness.json", metrics)
        return observation.passed, metrics
    finally:
        program.close(synchronize=torch.cuda.synchronize)


def _cohort_cv(samples: list[float]) -> float:
    mean = statistics.mean(samples)
    return statistics.pstdev(samples) / mean


def _benchmark_stage(
    root: Path,
    build_root: Path,
    stage_dir: Path,
    executor: ExecutorRevision,
) -> dict[str, object]:
    import torch

    _admit_gpu()
    helper = executor.admit_host()
    cupti = StrictCuptiBenchmark(helper)
    _, inputs = _case(root)
    candidate_output = torch.empty(
        (1, 32768, 32, 128), dtype=torch.bfloat16, device="cuda"
    )
    baseline_output = torch.empty_like(candidate_output)
    candidate_tensors = qsa_program_tensors(inputs, candidate_output[0])
    baseline_tensors = qsa_program_tensors(inputs, baseline_output[0])
    candidate = _load_program(build_root, "candidate")
    baseline = _load_program(build_root, "baseline")
    stream = torch.cuda.current_stream().cuda_stream

    def candidate_launch() -> None:
        candidate.launch(candidate_tensors, stream=stream)

    def baseline_launch() -> None:
        baseline.launch(baseline_tensors, stream=stream)

    cohorts: list[dict[str, object]] = []
    samples: dict[str, list[float]] = {"candidate": [], "baseline": []}
    try:
        for pair_index in range(5):
            order = (
                ("candidate", candidate_launch), ("baseline", baseline_launch)
            ) if pair_index % 2 == 0 else (
                ("baseline", baseline_launch), ("candidate", candidate_launch)
            )
            pair: dict[str, object] = {"pair_index": pair_index, "order": []}
            for name, launch in order:
                values = [
                    float(value)
                    for value in cupti(
                        launch,
                        dry_run_iters=3,
                        repeat_iters=25,
                        cold_l2_cache=True,
                        use_cuda_graph=False,
                    )
                ]
                if len(values) != 25:
                    raise ValueError("QSA CUPTI cohort sample count differs")
                samples[name].extend(values)
                cast(list[str], pair["order"]).append(name)
                pair[name] = values
            cohorts.append(pair)
        document = {"cohorts": cohorts}
        _write_json(stage_dir / "cupti-samples.json", document)
        return {
            "candidate_samples": samples["candidate"],
            "baseline_samples": samples["baseline"],
            "candidate_median": statistics.median(samples["candidate"]),
            "baseline_median": statistics.median(samples["baseline"]),
            "baseline_cv": _cohort_cv(samples["baseline"]),
        }
    finally:
        candidate.close(synchronize=torch.cuda.synchronize)
        baseline.close(synchronize=torch.cuda.synchronize)


def _profile_child(root: Path, build_root: Path) -> int:
    import torch

    _admit_gpu()
    _, inputs = _case(root)
    output = torch.empty(
        (1, 32768, 32, 128), dtype=torch.bfloat16, device="cuda"
    )
    tensors = qsa_program_tensors(inputs, output[0])
    program = _load_program(build_root, "candidate")
    try:
        program.launch(tensors, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return 0
    finally:
        program.close(synchronize=torch.cuda.synchronize)


def _profile_stage(
    root: Path,
    run_dir: Path,
    build_root: Path,
    stage_dir: Path,
    executor: ExecutorRevision,
    *,
    nvcc: Path,
    cuobjdump: Path,
) -> Mapping[str, object]:
    artifact_root = build_root / "candidate"
    artifact = QsaProgramArtifact.load(build_root, artifact_root / "program.json")
    score = next(
        (kernel for kernel in artifact.kernels if kernel.kernel_id == "score_topk"),
        None,
    )
    if score is None:
        raise ValueError("QSA Program has no score_topk attribution target")
    profiler = executor.admit_profiler()
    command = [
        str(profiler["path"]),
        "--csv",
        "--metrics",
        ",".join(NCU_ATTRIBUTION_METRICS),
        "--target-processes",
        "all",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        score.kernel_name,
        sys.executable,
        str(Path(__file__).resolve()),
        "--project-root",
        str(root),
        "--nvcc",
        str(nvcc),
        "--cuobjdump",
        str(cuobjdump),
        "--profile-child",
        "--build-root",
        str(build_root),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=1800,
        env=os.environ.copy(),
    )
    (stage_dir / "ncu.stdout.log").write_bytes(completed.stdout)
    (stage_dir / "ncu.stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-4096:]
        raise RuntimeError(f"NCU attribution failed: {diagnostic}")
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    profile = build_ncu_attribution_profile(
        candidate_sha256=str(request["candidate_sha256"]),
        case_id="target_t32768",
        kernel_name=score.kernel_name,
        ncu_version=str(profiler["version"]),
        ncu_executable_sha256=str(profiler["sha256"]),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    (stage_dir / "ncu-profile.json").write_bytes(profile)
    loaded = json.loads(profile)
    return ncu_attribution_feedback(loaded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.project_root.resolve(strict=True)
    if root != ROOT:
        raise ValueError("QSA evaluator project root differs from its source root")
    if arguments.profile_child:
        if arguments.build_root is None:
            raise ValueError("QSA profile child build root is missing")
        return _profile_child(root, arguments.build_root.resolve(strict=True))
    result_path = _required_output_path("KERNELINFRA_RESULT")
    stage_dir = _required_environment_path("KERNELINFRA_STAGE_DIR")
    run_dir = _required_environment_path("KERNELINFRA_RUN_DIR")
    candidate_root = _required_environment_path("KERNELINFRA_CANDIDATE_DIR")
    stage_kind = os.environ.get("KERNELINFRA_STAGE_KIND")
    executor = _executor(root)
    build_root = run_dir / "qsa-build"
    try:
        if stage_kind == "compile":
            build_root.mkdir(parents=True, exist_ok=False)
            try:
                arm, artifacts = _compile_stage(
                    root,
                    candidate_root,
                    build_root,
                    nvcc=arguments.nvcc.resolve(strict=True),
                    cuobjdump=arguments.cuobjdump.resolve(strict=True),
                )
            except _BaselineCompileError as error:
                return _stage_result(
                    result_path,
                    status="failed",
                    validity="unknown",
                    summary=f"fixed QSA CUDA baseline compile failed: {error}",
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
                return _stage_result(
                    result_path,
                    status="failed",
                    validity="invalid",
                    summary=f"QSA candidate compile rejected: {type(error).__name__}: {error}",
                )
            return _stage_result(
                result_path,
                status="passed",
                validity="unknown",
                summary=f"QSA {arm} candidate and fixed CUDA baseline compiled",
                artifacts=artifacts,
            )
        if not (build_root / "candidate/program.json").is_file() or not (
            build_root / "baseline/program.json"
        ).is_file():
            raise RuntimeError("QSA compile-stage artifacts are missing")
        if stage_kind == "correctness":
            passed, metrics = _correctness_stage(root, build_root, stage_dir)
            return _stage_result(
                result_path,
                status="passed" if passed else "failed",
                validity="valid" if passed else "invalid",
                summary=(
                    "QSA target_t32768 external-oracle correctness passed"
                    if passed
                    else "QSA target_t32768 external-oracle correctness failed"
                ),
                workloads=[
                    {
                        "id": _WORKLOAD_ID,
                        "correct": passed,
                        "notes": (
                            f"match_fraction={metrics['match_fraction']:.9g} "
                            f"max_abs_diff={metrics['max_abs_diff']:.9g}"
                        ),
                    }
                ],
                artifacts={"correctness": "correctness.json"},
                metrics=metrics,
            )
        if stage_kind == "benchmark":
            timing = _benchmark_stage(root, build_root, stage_dir, executor)
            candidate_ms = float(timing["candidate_median"])
            baseline_ms = float(timing["baseline_median"])
            stable = float(timing["baseline_cv"]) <= 0.05
            return _stage_result(
                result_path,
                status="passed",
                validity="valid",
                summary="QSA balanced CUPTI cold-L2 timing completed",
                workloads=[
                    {
                        "id": _WORKLOAD_ID,
                        "correct": True,
                        "candidate_ms": candidate_ms,
                        "baseline_ms": baseline_ms,
                        "candidate_samples_ms": timing["candidate_samples"],
                        "baseline_samples_ms": timing["baseline_samples"],
                        "stable": stable,
                        "speedup": baseline_ms / candidate_ms,
                        "notes": f"baseline_cv={timing['baseline_cv']:.9g}",
                    }
                ],
                artifacts={"cupti_samples": "cupti-samples.json"},
            )
        if stage_kind == "profile":
            feedback = _profile_stage(
                root,
                run_dir,
                build_root,
                stage_dir,
                executor,
                nvcc=arguments.nvcc.resolve(strict=True),
                cuobjdump=arguments.cuobjdump.resolve(strict=True),
            )
            return _stage_result(
                result_path,
                status="passed",
                validity="valid",
                summary="QSA score_topk NCU attribution completed",
                artifacts={
                    "profile": "ncu-profile.json",
                    "ncu_stdout": "ncu.stdout.log",
                    "ncu_stderr": "ncu.stderr.log",
                },
                metrics=feedback,
            )
        return _stage_result(
            result_path,
            status="failed",
            validity="unknown",
            summary=f"unsupported QSA stage kind: {stage_kind}",
        )
    except Exception as error:
        if not result_path.exists():
            return _stage_result(
                result_path,
                status="failed",
                validity="unknown",
                summary=f"QSA evaluator error: {type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
