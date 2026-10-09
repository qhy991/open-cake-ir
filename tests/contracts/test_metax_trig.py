"""MACA math lowering is separate from exact-target device qualification."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.ir.instruction_contracts import contract
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def source(op):
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


class MetaxTrig(unittest.TestCase):
    def test_registered_types_emit_only_under_explicit_maca_probe_admission(self):
        real = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        for op in ('sin', 'cos'):
            record = contract(f'maca.{op}.f32')
            self.assertIsNotNone(record)
            self.assertEqual(record.elementwise_op.value, op)
            self.assertEqual(record.elementwise_dtype.value, 'fp32')
            schedule = Schedule.from_dict(frontend.parse(source(op)).document)
            # This synthetic target is only a software probe, never a target file.
            probe = replace(real, instruction_contracts=real.instruction_contracts | {record.name})
            self.assertFalse([f for f in verify(schedule, probe) if f.blocks_lowering])
            self.assertFalse([f for f in preflight(schedule, probe) if f.blocks_lowering])
            emitted = emit(schedule, probe)
            projected = project_triton_kernel(emitted.source.encode(), emitted.toolchain)
            self.assertIn(f'libdevice.{op}(values)'.encode(), projected)
            validate_triton_kernel(projected, emitted.toolchain)
            self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED',
                          {f.code for f in Compiler.load(ROOT).assess(frontend.parse(source(op)).document).findings})

    def test_maca_contract_cannot_borrow_another_code_object_or_dtype(self):
        for op in ('sin', 'cos'):
            for target_id in ('gfx938', 'sm_103a'):
                schedule = Schedule.from_dict(frontend.parse(source(op).replace('xcore1002', target_id)).document)
                target = Target.load(ROOT / f'compiler/targets/{target_id}.json')
                target = replace(target, instruction_contracts=target.instruction_contracts | {f'maca.{op}.f32'})
                self.assertIn('TRITON_ELEMENTWISE_UNSUPPORTED', {f.code for f in preflight(schedule, target)})
                with self.assertRaises(EmitError): emit(schedule, target)
            for dtype in ('bf16', 'fp16', 'int32'):
                with self.assertRaisesRegex(Exception, 'no implicit dtype promotion'):
                    frontend.parse(source(op).replace('"fp32"', f'"{dtype}"'))

    def test_native_source_boundary_accepts_only_pure_unary_calls(self):
        native = ('import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n'
                  '@triton.jit\ndef kernel(a,b):\n    x=tl.load(a)\n    y=libdevice.sin(x)\n    tl.store(b,y)\n')
        req = {'kernel_entry_point': 'kernel', 'signature': {'a': '*fp32', 'b': '*fp32'},
               'compile_constants': {}, 'code_object': 'mcfatbin'}
        validate_triton_kernel(native.encode(), req)
        for expr in ('libdevice.sin', 'libdevice.sin(*x)', 'libdevice.sin(x,x)',
                     'libdevice.sin(x=x)', 'libdevice.erf(x)', 'libdevice.fma(x,x,x)'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                validate_triton_kernel(native.replace('libdevice.sin(x)', expr).encode(), req)
        for code_object in ('cubin', None):
            with self.assertRaises(ValueError):
                validate_triton_kernel(native.encode(), {**req, 'code_object': code_object})

    def test_qualification_binds_complete_compile_fields_before_native_build(self):
        from open_cake_ir.compiler.toolchain import triton_route
        spec = importlib.util.spec_from_file_location('qualify_metax_trig', ROOT / 'tools/qualify_metax_trig.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            result = module.run(output, False)
            self.assertTrue(result['passed'], result)
            self.assertFalse(result['target_admission_changed'])
            for op in ('sin', 'cos'):
                requirements = json.loads((output / op / 'requirements.json').read_text())
                self.assertEqual(requirements['compiler'], 'triton')
                self.assertEqual(requirements['source_language'], 'python')
                self.assertEqual(requirements['target'], 'xcore1002')
                self.assertEqual(triton_route(requirements).gpu_backend, 'maca')
                self.assertFalse(result['operations'][op]['native_compiled'])


if __name__ == '__main__':
    unittest.main()
