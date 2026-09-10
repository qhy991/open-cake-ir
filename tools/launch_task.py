#!/usr/bin/env python3
"""Prepare and execute one task through the existing TaskLab/Ralph composition."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import grp
import json
import os
import platform
from pathlib import Path
import pwd
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler
from open_cake_ir.lab.bindings import external_file, resolve_executor, CURRENT_RELEASE_BINDING
from open_cake_ir.lab.environments import CandidateSubmission
from open_cake_ir.lab.build import TritonToolchainBuilder
from open_cake_ir.lab.metal_build import MetalArchiveHost, MetalToolchainBuilder
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from open_cake_ir.lab.providers import ProviderQualificationReceipt
from open_cake_ir.tasks.compose import execute_matched_from_config
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
from open_cake_ir.tasks.normalization.study import OUTPUT_SCHEMA, canonical, study_template
from open_cake_ir.tasks.devices import BACKENDS as DEVICE_BACKENDS, admit_cohort_payload
from open_cake_ir.tasks.activation.workload import TASKS as _ACTIVATION_TASKS
from open_cake_ir.tasks.rowwise.workload import TASKS as _ROWWISE_TASKS
from open_cake_ir.tasks.reductions.workload import TASKS as _REDUCTION_TASKS
from open_cake_ir.tasks.optimizers.workload import TASKS as _OPTIMIZER_TASKS
from open_cake_ir.tasks.contraction.workload import TASKS as _CONTRACTION_TASKS
from open_cake_ir.tasks.normalization.workload import BACKENDS
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import create_task, load_workload
from open_cake_ir.evaluation.paired import candidate_identity

# The launcher offers whatever the activation family registers, so a migrated AKA
# parent becomes launchable by being added to that one table.
ACTIVATION_TASKS = tuple(_ACTIVATION_TASKS)
ROWWISE_TASKS = tuple(_ROWWISE_TASKS)
REDUCTION_TASKS = tuple(_REDUCTION_TASKS)
OPTIMIZER_TASKS = tuple(_OPTIMIZER_TASKS)
# The arithmetic-bound family declares a K extent, as the legacy GEMM task does.
CONTRACTION_TASKS = tuple(_CONTRACTION_TASKS)


def _provider_executable(harness: str, requested: Path | None) -> Path:
    name = str(requested) if requested else {"codex": "codex", "claude-code": "claude"}[harness]
    discovered = shutil.which(name)
    if discovered is None:
        raise ValueError(f"requested harness executable is unavailable: {name}")
    executable = Path(discovered).resolve(strict=True)
    if harness != "codex" or executable.suffix != ".js":
        return executable
    package_root = executable.parent.parent
    metadata_path = package_root / "package.json"
    metadata = json.loads(metadata_path.read_bytes()) if metadata_path.is_file() else {}
    if (metadata.get("name") != "@openai/codex" or metadata.get("bin", {}).get("codex") != "bin/codex.js"
            or executable != package_root / "bin/codex.js"):
        raise ValueError("unsupported Codex wrapper; pass its native executable explicitly")
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise ValueError("the Metal launcher resolves Codex npm packages only on Apple Silicon")
    target, dependency = "aarch64-apple-darwin", "@openai/codex-darwin-arm64"
    node = shutil.which("node")
    if node is None:
        raise ValueError("resolving the installed Codex npm wrapper requires its existing Node runtime")
    # Resolve from this entrypoint exactly as the installed shim does; no package code executes.
    script = ('const {createRequire}=require("node:module");'
              'try{process.stdout.write(createRequire(process.argv[1]).resolve(process.argv[2]+"/package.json"));}'
              'catch(e){if(e.code==="MODULE_NOT_FOUND")process.exit(44);throw e;}')
    result = subprocess.run([node, "-e", script, str(executable), dependency],
                            capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        vendor = Path(result.stdout.strip()).resolve(strict=True).parent / "vendor"
    elif result.returncode == 44:
        vendor = package_root / "vendor"
    else:
        raise ValueError("installed Codex package resolution failed; no alternative installation selected")
    native = vendor / target / "bin/codex"
    if not native.is_file() or not os.access(native, os.X_OK):
        raise ValueError("this Codex installation has no usable native executable; no installation attempted")
    return native.resolve(strict=True)


def _new_workspace(value: Path) -> Path:
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError("--workspace requires an absolute external path")
    resolved = value.resolve()
    if value != resolved or any((parent / ".git").exists() for parent in resolved.parents):
        raise ValueError("--workspace must be canonical and outside every Git checkout")
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError("--workspace already exists; retained task state is never reset")
    return resolved


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


# A Metal host names its kind. The CUDA host schema predates that field and carries none,
# so a route that is not Metal is defined by the absence of that name rather than by a
# marker of its own. Stated as "is metal" / "is not metal" so a host the capture tool has
# never produced still lands on the right side (F-2026-09-10-012 records why one checkout
# cannot hold both hosts at once).


def _route_of(backend: str) -> str:
    return DEVICE_BACKENDS[backend]["route"]


def _admit_stack(root: Path, workspace: Path, target: str, route: str = "metal"):
    compiler = Compiler.load(root, root / "compiler/revision.lock.json")
    gate = compiler.check_corpus()
    _write(workspace / "compiler-gate.json", canonical(asdict(gate)))
    if compiler.state != "released" or not gate.passed:
        raise ValueError("task launch requires a released Compiler and passing full Corpus Gate")
    try:
        executor = resolve_executor(root, CURRENT_RELEASE_BINDING, "task.execution", template=True)
    except ValueError as error:
        raise ValueError(f"task launch requires a released Metal Executor matching this source; {error}") from error
    is_metal_host = executor.document["host_environment"].get("kind") == "metal"
    if route == "metal" and not is_metal_host:
        raise ValueError("task launch requires an actually released Metal Executor")
    if route != "metal" and is_metal_host:
        raise ValueError(f"the {route!r} route requires a released Executor bound to a GPU host; "
                         "the current one is Metal, and no host is substituted for another")
    host = None
    if route == "metal":
        released = executor.document["host_environment"].get("host", {}).get("target")
        if released != target:
            raise ValueError(f"released Metal Executor is bound to {released!r}, not the requested "
                             f"{target!r}; no other Apple GPU is substituted")
        host = MetalArchiveHost.from_executor(executor)
    return compiler, executor, host, {"path": "compiler/revision.lock.json",
        "revision_id": gate.compiler_revision_id, "canonical_sha256": gate.compiler_revision_sha256}


def _triton_runtime_roots(interpreter: Path) -> list[str]:
    """Directories the jail must carry for the admitted interpreter to run inside it.

    The interpreter the Executor pins may be a venv whose `bin/python` is a symlink into
    another prefix, so mounting the venv alone leaves the jail without the real binary or
    its standard library. Both the named prefix and the resolved one are mounted, and the
    system directories that hold the ELF interpreter and shared libraries alongside them.
    Duplicates and nested paths are dropped so bwrap is not handed the same mount twice.
    """
    roots = [Path("/usr"), Path("/lib"), Path("/lib64"), Path("/opt")]
    for candidate in (interpreter, Path(os.path.realpath(interpreter))):
        # `.../prefix/bin/python` -> `.../prefix`
        roots.append(candidate.parents[1])
    admitted: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if any(root == kept or kept in root.parents for kept in admitted):
            continue
        admitted = [kept for kept in admitted if root not in kept.parents]
        admitted.append(root)
    return [str(root) for root in admitted]


def _triton_builder(executor, workload):
    """Bind the isolated compiler to the exact runtime the Executor host admits."""
    host = executor.document["host_environment"]
    interpreter = Path(str(host["python"]["invocation_path"]))
    return TritonToolchainBuilder(
        workload=workload, case_id="primary",
        isolated_compiler=IsolatedTritonCompiler(
            python=str(interpreter), bubblewrap="/usr/bin/bwrap",
            runtime_roots=_triton_runtime_roots(interpreter),
            triton_version=host["packages"]["triton"]))


def _prepare_baseline(root, workspace, compiler, executor, host, workload, study, source,
                      compiler_reference, route="metal"):
    builder = (MetalToolchainBuilder(workload=workload, case_id="primary",
                                     output_root=workspace / "builds", host=host, project_root=root,
                                     compiler_reference=compiler_reference)
               if route == "metal" else _triton_builder(executor, workload))
    environment = TaskOpenCakeEnvironment(compiler, builder, authority_document=study["arms"]["open_cake"],
                                         workload=workload, case_id="primary", executor=executor)
    submission = CandidateSubmission.seal(environment.media_type, canonical({"python_source": source}))
    result = environment.build(submission)
    _write(workspace / "baseline-feedback.json", canonical(dict(result.feedback)))
    if result.launchable is None:
        raise ValueError("baseline preparation refused; see baseline-feedback.json")
    directory = workspace / "baseline"
    directory.mkdir()
    paths = {}
    for role, payload in result.launchable.artifact_payloads.items():
        if not role.isidentifier():
            raise ValueError("baseline artifact role cannot be a file name")
        filename = role + ".bin"
        _write(directory / filename, payload)
        paths[role] = filename
    bundle = directory / "candidate.json"
    _write(bundle, canonical({"candidate": candidate_identity(result.launchable), "artifact_paths": paths}))
    return bundle


def _qualify(root, workspace, args, executable, source_path):
    if args.qualification is not None:
        receipt_path = external_file(root, str(args.qualification), "provider qualification")
        anchor_path = external_file(root, str(args.qualification_anchor), "provider qualification anchor")
        return receipt_path, anchor_path
    version = subprocess.run([str(executable), "--version"], check=True, capture_output=True, text=True, timeout=30)
    if not version.stdout.strip():
        raise ValueError("provider executable reported no version")
    revision = args.provider_revision or f"{args.harness}:{version.stdout.strip()}"
    receipt, anchor = workspace / "provider-qualification.json", workspace / "provider-anchor.json"
    command = [sys.executable, str(root / "tools/qualify_codex_provider.py"),
               "--harness", args.harness, "--executable", str(executable), "--provider-revision", revision,
               "--model", args.model, "--reasoning-effort", args.effort,
               "--output-schema", str(root / OUTPUT_SCHEMA), "--python-source", str(source_path),
               "--feature-policy", "provider_defaults_optimization", "--maximum-candidates-per-turn", str(args.max_candidates),
               "--workspace", str(workspace / "qualification-workspace"), "--receipt-output", str(receipt),
               "--anchor-output", str(anchor), "--evidence-root", str(workspace / "qualification-evidence"),
               "--run-id", "task-provider-qualification"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=args.wall_seconds)
    _write(workspace / "qualification.stdout", completed.stdout.encode())
    _write(workspace / "qualification.stderr", completed.stderr.encode())
    if completed.returncode:
        raise ValueError("provider qualification refused; see qualification.stderr")
    return receipt, anchor



def _campaign_exit_code(report) -> int:
    """CLI success describes an intact protocol outcome, including negative results."""
    valid = (report.campaign_complete and report.archive_integrity_passed
             and report.filesystem_custody_verified and report.semantic_replay_passed
             and bool(report.run_audits)
             and all(audit.protocol_adherence == "adhered" for audit in report.run_audits))
    return 0 if valid else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rmsnorm", "layernorm", "residual_rmsnorm", "softmax",
                                          *ACTIVATION_TASKS, *ROWWISE_TASKS, *REDUCTION_TASKS,
                                          *OPTIMIZER_TASKS, *CONTRACTION_TASKS,
                                          "gemm_bias"), required=True)
    parser.add_argument("--backend", choices=tuple(DEVICE_BACKENDS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", choices=("codex", "claude-code"), required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--depth", type=int,
                        help="contracted K extent; only a contraction task declares one")
    parser.add_argument("--case", choices=("primary",), default="primary", help="timing case; all five input cases remain required")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=150000)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--searches-per-turn", type=int, default=2)
    parser.add_argument("--dispatches-per-sample", type=int, default=64,
                        help="dispatches encoded in each timed command buffer; amortizes fixed command overhead")
    parser.add_argument("--wall-seconds", type=int, default=14400)
    parser.add_argument("--provider-executable", type=Path)
    parser.add_argument("--provider-revision")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualification-anchor", type=Path)
    parser.add_argument("--fixed-baseline-bundle", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="stop after baseline preparation, qualification and Campaign preflight")
    args = parser.parse_args(argv)
    if (args.qualification is None) != (args.qualification_anchor is None):
        parser.error("--qualification and --qualification-anchor must be supplied together")
    workspace = _new_workspace(args.workspace)
    document, source = create_task(args.task, backend=args.backend, rows=args.rows, columns=args.columns,
                                   depth=args.depth, case_id=args.case)
    executable = _provider_executable(args.harness, args.provider_executable)
    workspace.mkdir(mode=0o750, parents=True)
    workload_path, source_path = workspace / "workload.json", workspace / "starter.py"
    _write(workload_path, canonical(document))
    _write(source_path, source.encode())
    workload = load_workload(workload_path)
    # F-2026-09-10-002: the native observer refuses an oversized snapshot cohort before it
    # dispatches anything, so a shape that exceeds the bound dies at the first evaluation
    # with the campaign's authoring tokens already spent. Check the same arithmetic here.
    admit_cohort_payload(workload, args.case,
                         study_template.__globals__["_ROUTE_CALLS_PER_COHORT"])
    study = study_template(ROOT, workload, workload_path, source_path, harness=args.harness,
        model=args.model, effort=args.effort, turns=args.turns, token_budget=args.token_budget,
        maximum_candidates=args.max_candidates, searches_per_turn=args.searches_per_turn, wall_seconds=args.wall_seconds,
        dispatches_per_sample=args.dispatches_per_sample)
    study_path = workspace / "study.json"
    _write(study_path, canonical(study))
    route = _route_of(args.backend)
    compiler, executor, host, compiler_reference = _admit_stack(ROOT, workspace, workload.target, route)
    baseline_path = (external_file(ROOT, str(args.fixed_baseline_bundle), "fixed baseline bundle")
                     if args.fixed_baseline_bundle else _prepare_baseline(
                         ROOT, workspace, compiler, executor, host, workload, study, source,
                         compiler_reference, route))
    receipt_path, anchor_path = _qualify(ROOT, workspace, args, executable, source_path)
    receipt = ProviderQualificationReceipt.load(receipt_path)
    if not receipt.qualified or receipt.scope != "live_two_turn_tool_rich_provider":
        raise ValueError("task execution requires an actual live artifact-optimization provider qualification")
    from open_cake_ir.evaluation.source_bootstrap import module_command
    runtime = {"schema_version": 1,
        "provider": {"executable": str(executable), "workspace_root": str(workspace / "actors")},
        "toolchain": {"output_root": str(workspace / "builds")},
        "broker": {"command": module_command(executor.document["host_environment"]["python"]["invocation_path"],
                    "open_cake_ir.evaluation.local_broker", "--worker-module", "open_cake_ir.tasks.evaluate"),
                   "cwd": str(ROOT), "timeout_seconds": 1800,
                   "service_user": pwd.getpwuid(os.getuid()).pw_name,
                   "service_group": grp.getgrgid(os.getgid()).gr_name}}
    runtime_path = workspace / "runtime.json"
    _write(runtime_path, canonical(runtime))
    bindings_path = workspace / "execution-bindings.json"
    _write(bindings_path, canonical({"schema_version": 1, "qualification_path": str(receipt_path),
        "qualification_anchor_path": str(anchor_path), "runtime_config_path": str(runtime_path),
        "fixed_baseline_bundle_path": str(baseline_path)}))
    lock = TaskLab(ROOT).preflight(study_path, execution_bindings_path=bindings_path)
    _write(workspace / "campaign-lock.json", canonical(lock.document))
    if args.preflight_only:
        print(workspace / "campaign-lock.json")
        return 0
    campaign = execute_matched_from_config(ROOT, lock, runtime_path, workspace / "campaign-evidence")
    print(campaign.evidence_root)
    return _campaign_exit_code(TaskLab(ROOT).audit(campaign))


if __name__ == "__main__":
    raise SystemExit(main())
