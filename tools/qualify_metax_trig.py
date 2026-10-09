#!/usr/bin/env python3
"""Build isolated MACA FP32 sin/cos probes without admitting a Target or using a GPU.

The synthetic in-memory contract declaration is limited to this qualification
probe. The committed Target and production Compiler admission remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]


def probe_source(op):
    if op not in {'sin', 'cos'}:
        raise ValueError('The MACA trig probe supports only FP32 sin/cos')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="maca_{op}_probe", target="xcore1002", backend="triton", entry_point="maca_{op}_probe")
def candidate(lm, x: cake.Tensor((4, 256), "fp32"), out: cake.Tensor((4, 256), "fp32", mode="output")):
    row = lm.program(x, axis=0, dimension=0, tile=1)
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[row, :], id="load")
        result = lm.{op}(values, instruction={{"contract": "maca.{op}.f32"}}, id="trig")
        lm.store(out[row, :], result, coalesced=False, id="store")
'''


def run(output, native):
    from open_cake_ir.compiler import Compiler, Target, frontend
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.toolchain import project_triton_kernel
    from open_cake_ir.compiler.verifier import verify
    from open_cake_ir.source_identity import checkout_commit
    commit = checkout_commit(ROOT)
    if commit is None:
        raise ValueError('Math qualification requires a clean committed source')
    output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'target': 'xcore1002', 'passed': False,
              'scope': 'isolated native compilation only' if native else 'software emission only',
              'target_admission_changed': False, 'device_execution': False, 'operations': {}}
    try:
        compiler = Compiler.load(ROOT)
        real = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        if {'maca.sin.f32', 'maca.cos.f32'} & real.instruction_contracts:
            raise ValueError('This pre-admission probe expects the committed Target to remain unqualified')
        isolated = None
        if native:
            from launch_task import _triton_toolchain_config
            from open_cake_ir.lab.executor import ExecutorRevision
            from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
            executor = ExecutorRevision.for_target(ROOT, 'xcore1002')
            host = executor.admit_host()
            (output / 'host-admission.json').write_text(json.dumps(host, indent=2))
            isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
            isolated.check_executor(executor, author_workspace=output)
        for op in ('sin', 'cos'):
            directory = output / op
            directory.mkdir()
            source = probe_source(op)
            (directory / 'candidate.cake.py').write_text(source)
            schedule = Schedule.from_dict(frontend.parse(source).document)
            production = compiler.assess(frontend.parse(source).document)
            if production.lowering_eligible:
                raise ValueError('The committed Target unexpectedly admits unqualified trig')
            probe = replace(real, instruction_contracts=real.instruction_contracts | {f'maca.{op}.f32'})
            findings = (*verify(schedule, probe), *preflight(schedule, probe))
            (directory / 'software-findings.json').write_text(json.dumps([asdict(item) for item in findings], indent=2))
            if any(item.blocks_lowering for item in findings):
                raise ValueError('The synthetic math-contract probe has a blocking software finding')
            lowered = emit(schedule, probe)
            kernel = project_triton_kernel(lowered.source.encode(), lowered.toolchain)
            (directory / 'kernel.triton.py').write_bytes(kernel)
            (directory / 'requirements.json').write_text(json.dumps(lowered.toolchain, indent=2))
            row = {'source_projected': True, 'production_refusal': [f.code for f in production.findings if f.blocks_lowering],
                   'native_compiled': False}
            result['operations'][op] = row
            if isolated is not None:
                try:
                    built = isolated.compile(kernel, lowered.toolchain)
                except Exception as error:
                    for name, data in getattr(error, 'artifact_payloads', {}).items():
                        if isinstance(name, str) and name.isidentifier() and isinstance(data, bytes):
                            (directory / name).write_bytes(data)
                    raise
                for role, payload in built.artifacts.items():
                    (directory / role).write_bytes(payload)
                row.update(native_compiled=True, entry_point=built.entry_point,
                           threads_per_cta=built.threads_per_cta,
                           dynamic_shared_bytes=built.dynamic_shared_bytes,
                           code_object=built.code_object, compiler_version=built.compiler_version)
        result['passed'] = True
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        (output / 'failure.txt').write_text(traceback.format_exc())
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Qualification outputs must remain outside the checkout')
    result = run(output, args.native)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
