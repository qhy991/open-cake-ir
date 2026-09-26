"""Create-only CPU NVCC build for one exact sealed B300 ranked-tile source."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess

from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    NativeRankedTileLowering,
)
from open_cake_ir.evaluation.ranked_tile_manifest import (
    RankedTileCandidate, RankedTileLaunchManifest,
)
from open_cake_ir.evaluation.workload import WorkloadContract


_LIBRARY = re.compile(r'[A-Za-z0-9_]+\Z')


class RankedTileBuildError(ValueError):
    """The create-only build directory retains the exact failure stage."""


@dataclass(frozen=True)
class RankedTileBuildProducts:
    output_dir: Path
    source: Path
    library: Path
    report: Path
    candidate: RankedTileCandidate


def _route(lowered: NativeRankedTileLowering):
    req = lowered.toolchain_requirements
    flags = req.get('nvcc_flags')
    libraries = req.get('link_libraries')
    target = lowered.local_program.target
    if (req.get('source_language') != 'cuda_cpp'
            or target != 'sm_103a'
            or not isinstance(flags, list) or not flags
            or any(not isinstance(flag, str) or not flag for flag in flags)
            or flags.count('--gpu-architecture=compute_103a') != 1
            or flags.count('--gpu-code=sm_103a') != 1
            or '-std=c++17' not in flags or '--fmad=false' not in flags
            or any(flag == '-arch' or flag.startswith('-arch=') for flag in flags)
            or libraries != ['cuda', 'cudart']
            or any(_LIBRARY.fullmatch(name) is None for name in libraries)):
        raise ValueError('ranked tile NVCC exact target or linkage differs')
    return tuple(flags), tuple('-l'+name for name in libraries)


def compile_ranked_tile_candidate(lowered: NativeRankedTileLowering,
        manifest: RankedTileLaunchManifest, workload: WorkloadContract,
        case_id: str, *, nvcc: str | Path, output_dir: str | Path,
        timeout_seconds: int = 240) -> RankedTileBuildProducts:
    """Compile without a GPU lease, retain failures and return sealed bytes."""
    manifest.check_lowered(lowered)
    manifest.check_workload(workload, case_id, lowered)
    flags, libraries = _route(lowered)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError('ranked tile build timeout must be positive seconds')
    compiler = Path(nvcc).resolve(strict=True)
    if not compiler.is_file() or not os.access(compiler,os.X_OK):
        raise ValueError('ranked tile NVCC must be an executable regular file')
    output = Path(output_dir).expanduser().absolute()
    if any((parent/'.git').exists() for parent in (output,*output.parents)):
        raise ValueError('ranked tile build output must be outside a checkout')
    if output.exists():
        raise ValueError('ranked tile build output must be create-only')
    output.mkdir(parents=True)
    source = output/'ranked_tile.cu'
    library = output/'libranked_tile.so'
    source.write_bytes(lowered.source.encode('utf-8'))
    (output/'launch_manifest.json').write_bytes(manifest._bytes)
    command = [str(compiler),*flags,'-shared','-Xcompiler','-fPIC',
               str(source),'-o',str(library),*libraries]
    (output/'build_plan.json').write_text(json.dumps({
        'schema_version':1,'compiler_commit':
            lowered.toolchain_requirements['compiler_commit'],
        'target':lowered.local_program.target,'argv':command,
    },indent=2)+'\n')
    log = output/'compile.log'
    try:
        result = subprocess.run(command,cwd=output,capture_output=True,
                                text=True,timeout=timeout_seconds,check=False)
    except (OSError,subprocess.TimeoutExpired) as error:
        stdout = error.stdout if isinstance(error,subprocess.TimeoutExpired) else ''
        stderr = error.stderr if isinstance(error,subprocess.TimeoutExpired) else ''
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
        log.write_text(decoded(stdout)+decoded(stderr)+
                       f'\n{type(error).__name__}: {error}\n')
        (output/'build_failure.json').write_text(json.dumps({
            'reason':type(error).__name__},indent=2)+'\n')
        raise RankedTileBuildError(f'ranked tile NVCC did not finish; retained {output}') from error
    log.write_text(result.stdout+result.stderr or '(no compiler output)\n')
    if result.returncode:
        (output/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RankedTileBuildError(f'ranked tile NVCC rejected source; retained {output}')
    if not library.is_file() or library.read_bytes()[:4] != b'\x7fELF':
        (output/'build_failure.json').write_text(json.dumps({
            'reason':'missing or non-ELF library'},indent=2)+'\n')
        raise RankedTileBuildError(f'ranked tile library differs; retained {output}')
    candidate = RankedTileCandidate.seal(
        lowered,manifest,workload=workload,case_id=case_id,
        library=library.read_bytes())
    report = output/'build_report.json'
    report.write_text(json.dumps({
        'schema_version':1,'target':lowered.local_program.target,
        'compiler_commit':lowered.toolchain_requirements['compiler_commit'],
        'exit_code':0,
        'artifact_bytes':{'source':source.stat().st_size,
                          'library':library.stat().st_size},
        'scope':'CPU-only exact B300 NVCC build and byte-bearing candidate; no GPU validation',
    },indent=2)+'\n')
    return RankedTileBuildProducts(output,source,library,report,candidate)
