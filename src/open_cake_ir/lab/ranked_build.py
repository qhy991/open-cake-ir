"""Create-only CPU nvcc build of one pre-sealed ranked mailbox source.

The output remains development evidence, not a LaunchableCandidate. Device
admission, broker allocation, oracle and multi-GPU timing are separate gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess

from open_cake_ir.compiler.program import LoweredRankedMailbox
from open_cake_ir.evaluation.ranked_manifest import RankedMailboxLaunchManifest
from open_cake_ir.serialization import canonical_json_bytes


_LIBRARY = re.compile(r'[A-Za-z0-9_]+\Z')


class RankedBuildError(ValueError):
    """CPU compilation failed; the create-only output directory is retained."""


@dataclass(frozen=True)
class RankedBuildProducts:
    manifest: RankedMailboxLaunchManifest
    output_dir: Path
    source: Path
    host_wrapper: Path
    cubin: Path
    report: Path


def _flags(lowered):
    req = lowered.toolchain_requirements
    flags = req.get('nvcc_flags')
    libraries = req.get('link_libraries')
    target = lowered.local_program.target
    if (req.get('source_language') != 'cuda_cpp'
            or not isinstance(flags, list) or not flags
            or any(not isinstance(flag, str) or not flag for flag in flags)
            or flags.count('--gpu-architecture=' + target.replace('sm_', 'compute_', 1)) != 1
            or flags.count('--gpu-code=' + target) != 1
            or any(flag == '-arch' or flag.startswith('-arch=') for flag in flags)
            or '-std=c++17' not in flags or '--fmad=false' not in flags
            or not isinstance(libraries, list) or libraries != ['cudart']
            or any(not isinstance(name, str) or _LIBRARY.fullmatch(name) is None
                   for name in libraries)):
        raise ValueError('ranked nvcc route, exact architecture or linkage differs')
    return tuple(flags), tuple('-l' + name for name in libraries)


def compile_ranked_development(lowered: LoweredRankedMailbox,
        manifest: RankedMailboxLaunchManifest, workload, case_id: str, *,
        nvcc: str | Path, output_dir: str | Path,
        timeout_seconds: int = 180) -> RankedBuildProducts:
    """Compile exact source to ELF host wrapper and CUBIN, retaining failures.

    This standalone step owns no GPU and never overwrites prior evidence.
    The caller supplies the qualified nvcc path and an unused output path.
    """
    manifest.check_lowered(lowered)
    manifest.check_workload(workload, case_id)
    flags, libraries = _flags(lowered)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError('ranked nvcc timeout must be positive seconds')
    compiler = Path(nvcc).resolve(strict=True)
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise ValueError('ranked nvcc must be an executable regular file')
    output = Path(output_dir).expanduser().absolute()
    if output.exists():
        raise ValueError('ranked build output must be create-only')
    output.mkdir()
    source = output / 'ranked.cu'
    host = output / 'ranked.so'
    cubin = output / 'ranked.cubin'
    source.write_text(lowered.source)
    (output / 'manifest.json').write_bytes(canonical_json_bytes(manifest.as_dict()))
    (output / 'requirements.json').write_bytes(
        canonical_json_bytes(dict(lowered.toolchain_requirements)))
    base = (str(compiler), *flags)
    commands = {
        'host_wrapper': (*base, '-Xcompiler=-fPIC', '-shared', str(source),
                         '-o', str(host), *libraries),
        'cubin': (*base, '--cubin', str(source), '-o', str(cubin)),
    }
    (output / 'build_plan.json').write_text(json.dumps({
        'schema_version': 1, 'compiler_commit': manifest.as_dict()['compiler_commit'],
        'target': lowered.local_program.target,
        'commands': {name: list(argv) for name, argv in commands.items()}},
        indent=2) + '\n')
    results = {}
    for name, argv in commands.items():
        log = output / f'{name}.compile.log'
        try:
            completed = subprocess.run(argv, cwd=output, capture_output=True,
                                       text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return (value.decode(errors='replace') if isinstance(value, bytes)
                        else value or '')
            log.write_text(decoded(error.stdout) + decoded(error.stderr)
                           + f'\nTimeoutExpired: {error}\n')
            (output / 'build_failure.json').write_text(json.dumps({
                'stage': name, 'reason': 'timeout',
                'completed_stages': results}, indent=2) + '\n')
            raise RankedBuildError(f'ranked {name} nvcc timed out; retained {output}') from error
        except OSError as error:
            log.write_text(f'{type(error).__name__}: {error}\n')
            (output / 'build_failure.json').write_text(json.dumps({
                'stage': name, 'reason': type(error).__name__,
                'completed_stages': results}, indent=2) + '\n')
            raise RankedBuildError(f'ranked {name} nvcc invocation failed; retained {output}') from error
        log.write_text(completed.stdout + completed.stderr or '(no compiler output)\n')
        results[name] = completed.returncode
        if completed.returncode != 0:
            (output / 'build_failure.json').write_text(json.dumps({
                'stage': name, 'exit_code': completed.returncode,
                'completed_stages': results}, indent=2) + '\n')
            raise RankedBuildError(f'ranked {name} nvcc rejected source; retained {output}')
    for name, artifact in (('host_wrapper', host), ('cubin', cubin)):
        magic = None
        if artifact.is_file():
            with artifact.open('rb') as stream:
                magic = stream.read(4)
        if magic != b'\x7fELF':
            (output / 'build_failure.json').write_text(json.dumps({
                'stage': name, 'reason': 'missing or non-ELF compiled artifact',
                'completed_stages': results}, indent=2) + '\n')
            raise RankedBuildError(f'ranked {name} build artifact differs; retained {output}')
    report = output / 'build_report.json'
    report.write_text(json.dumps({
        'schema_version': 1, 'target': lowered.local_program.target,
        'compiler_commit': manifest.as_dict()['compiler_commit'],
        'exit_codes': results,
        'artifact_bytes': {'host_wrapper': host.stat().st_size,
                           'cubin': cubin.stat().st_size},
        'scope': 'CPU nvcc build only; no GPU or candidate qualification'},
        indent=2) + '\n')
    return RankedBuildProducts(manifest, output, source, host, cubin, report)
