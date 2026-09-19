"""Kernel-only source admission; these checks neither compile nor execute a GPU kernel."""

import ast
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from open_cake_ir.tasks.workloads import create_task


ROOT = Path(__file__).resolve().parents[2]
LIBDEVICE_IMPORT = "from triton.language.extra import libdevice\n"
SOURCE = (
    "import triton\nimport triton.language as tl\n" + LIBDEVICE_IMPORT
    + "@triton.jit\ndef kernel(a, b):\n"
    "    x = tl.load(a)\n"
    "    y = libdevice.tanh(x).to(tl.float32)\n"
    "    tl.store(b, y)\n"
)
REQUIREMENTS = {
    "kernel_entry_point": "kernel",
    "signature": {"a": "*fp32", "b": "*fp32"},
    "compile_constants": {},
}


class TritonToolchainAdmissionTests(unittest.TestCase):
    def test_fma_and_loop_max_lowering_reach_public_source_admission(self):
        from tests.contracts.test_triton_loop_scopes import _reduction

        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        for target in ("sm_100a", "sm_103a"):
            source = f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="fma-admission", target="{target}", backend="triton", entry_point="kernel")
def candidate(lm, x: cake.Tensor((2, 8), "fp32"), out: cake.Tensor((2, 8), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load")
        result = lm.fma(values, values, values,
                        instruction={{"contract": "ptx.fma.rn.f32"}}, id="fma")
        lm.store(out[row, :], result, coalesced=False, id="store")
'''
            maximum = _reduction("max")
            maximum["target"] = target
            for label, document in (("fma", frontend.parse(source).document), ("max", maximum)):
                with self.subTest(target=target, primitive=label):
                    assessment = compiler.assess(document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    lowering = compiler.lower(assessment)
                    projected = project_triton_kernel(lowering.source.encode(), lowering.toolchain_requirements)
                    original = next(n for n in ast.parse(lowering.source).body
                                    if isinstance(n, ast.FunctionDef)
                                    and n.name == lowering.toolchain_requirements["kernel_entry_point"])
                    self.assertEqual(ast.dump(ast.parse(projected).body[-1]), ast.dump(original))
                    self.assertIn(b"fma.rn.f32" if label == "fma" else b'float("-inf")', projected)

    def test_fma_admission_requires_exact_instruction_and_direct_callee(self):
        expression = ('tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", '
                      'constraints="=f,f,f,f", args=[x, x, x], dtype=tl.float32, is_pure=True, pack=1)')
        source = SOURCE.replace(LIBDEVICE_IMPORT, "").replace("libdevice.tanh(x)", expression)
        # PTX inline assembly is a fact of the cubin route: the contract is admitted by
        # the code object the compile contract names, not by which id names it.
        requirements = {**REQUIREMENTS, "target": "sm_103a", "code_object": "cubin"}
        validate_triton_kernel(source.encode(), requirements)
        changes = (
            ("fma.rn.f32", "fma.rn.ftz.f32"),
            ("fma.rn.f32 $0, $1, $2, $3;", "mov.b32 $0, $1;"),
            ('constraints="=f,f,f,f"', 'constraints="=r,r,r,r"'),
            ("args=[x, x, x]", "args=[x, x]"),
            ("args=[x, x, x]", "args=[*x, x, x]"),
            ("dtype=tl.float32", "dtype=tl.float16"),
            ("is_pure=True", "is_pure=False"),
            ("pack=1", "pack=True"),
            ("pack=1", "pack=2"),
            ("pack=1", "pack=1, pack=1"),
            ("pack=1", "pack=1, unknown=1"),
            (", pack=1", ""),
        )
        for old, new in changes:
            with self.subTest(change=new):
                with self.assertRaisesRegex(ValueError, "exact FP32 FMA contract"):
                    validate_triton_kernel(source.replace(old, new).encode(), requirements)
        for code_object in (None, "hsaco", "metal_binary_archive", "CUBIN"):
            with self.subTest(code_object=code_object):
                with self.assertRaisesRegex(ValueError, "exact FP32 FMA contract"):
                    validate_triton_kernel(
                        source.encode(), {**requirements, "code_object": code_object})
        with self.assertRaisesRegex(ValueError, "exact FP32 FMA contract"):
            validate_triton_kernel(
                source.encode(), {k: v for k, v in requirements.items() if k != "code_object"})
        for value in ("tl.inline_asm_elementwise", "[tl.inline_asm_elementwise]",
                      "tl.inline_asm_elementwise.to(x)"):
            with self.subTest(escape=value):
                escaped = source.replace("    y =", f"    escape = {value}\n    y =")
                with self.assertRaisesRegex(ValueError, "exact FP32 FMA contract"):
                    validate_triton_kernel(escaped.encode(), requirements)
        callback = source.replace("args=[x, x, x]", "args=[callback(x), x, x]")
        with self.assertRaisesRegex(ValueError, "unsupported call"):
            validate_triton_kernel(callback.encode(), requirements)

    def test_infinity_admission_never_allows_dynamic_conversion_or_name_escape(self):
        source = SOURCE.replace(LIBDEVICE_IMPORT, "")
        for literal in ('float("inf")', 'float("-inf")'):
            validate_triton_kernel(source.replace("libdevice.tanh(x)", literal).encode(), REQUIREMENTS)
        for expression in ('float("nan")', "float(x)", "float(1)", "float(*x)",
                           'float(x="inf")', 'float("inf", "-inf")', "float", "[float]", "float.to(x)"):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, "direct infinity literal"):
                    validate_triton_kernel(source.replace("libdevice.tanh(x)", expression).encode(), REQUIREMENTS)
        source = source.replace("libdevice.tanh(x)", 'float("-inf")')
        with self.assertRaisesRegex(ValueError, "reserved name"):
            validate_triton_kernel(source.replace("    y =", "    float = x\n    y =").encode(), REQUIREMENTS)
        with self.assertRaisesRegex(ValueError, "parameter annotations or names"):
            validate_triton_kernel(source.replace("kernel(a, b)", "kernel(float, b)").encode(),
                                   {**REQUIREMENTS, "signature": {"float": "*fp32", "b": "*fp32"}})

    def test_compiler_gelu_projection_preserves_dependency_and_kernel_body(self):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        for task in ("gelu_tanh", "gelu_tanh_backward"):
            for backend in ("triton-b200", "triton-b300"):
                with self.subTest(task=task, backend=backend):
                    _, authored = create_task(task, backend=backend, rows=16, columns=64)
                    assessment = compiler.assess(frontend.parse(authored).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    lowering = compiler.lower(assessment)
                    projected = project_triton_kernel(
                        lowering.source.encode(), lowering.toolchain_requirements,
                    )
                    validate_triton_kernel(projected, lowering.toolchain_requirements)
                    self.assertIn(LIBDEVICE_IMPORT.encode(), projected)
                    self.assertIn(b"libdevice.tanh(", projected)
                    original_kernel = next(
                        node for node in ast.parse(lowering.source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == lowering.toolchain_requirements["kernel_entry_point"]
                    )
                    projected_tree = ast.parse(projected)
                    self.assertEqual(ast.dump(projected_tree.body[-1]), ast.dump(original_kernel))
                    self.assertEqual(len(projected_tree.body), 4)
                    self.assertNotIn(b"import torch", projected)

    def test_native_tanh_and_result_cast_are_admitted(self):
        validate_triton_kernel(SOURCE.encode(), REQUIREMENTS)

    def test_libdevice_module_and_function_values_cannot_escape_to_tensor_methods(self):
        sources = (
            SOURCE.replace("libdevice.tanh(x)", "libdevice.tanh.to(x)"),
            SOURCE.replace("    y =", "    helper = libdevice\n    y =")
                  .replace("libdevice.tanh(x)", "helper.to(x)"),
            SOURCE.replace("    y =", "    helper = libdevice.tanh\n    y =")
                  .replace("libdevice.tanh(x)", "helper.to(x)"),
        )
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "libdevice requires a direct tanh call"):
                    validate_triton_kernel(source.encode(), REQUIREMENTS)

    def test_existing_module_without_libdevice_keeps_its_projection(self):
        source = SOURCE.replace(LIBDEVICE_IMPORT, "").replace("libdevice.tanh(x)", "tl.exp(x)")
        validate_triton_kernel(source.encode(), REQUIREMENTS)
        projected = project_triton_kernel(source.encode(), REQUIREMENTS)
        self.assertNotIn(b"libdevice", projected)
        self.assertEqual(ast.dump(ast.parse(projected)), ast.dump(ast.parse(source)))

    def test_tanh_requires_the_exact_fixed_import(self):
        changes = (
            ("", "unsupported call"),
            ("import os as libdevice\n", "fixed imports"),
            ("from triton.language.extra import libdevice as other\n", "fixed imports"),
            ("from triton.language.extra import libdevice, cuda\n", "fixed imports"),
            ("from triton.language.extra.cuda import libdevice\n", "fixed imports"),
            (LIBDEVICE_IMPORT + "import os\n", "fixed imports"),
        )
        for replacement, refusal in changes:
            with self.subTest(replacement=replacement):
                source = SOURCE.replace(LIBDEVICE_IMPORT, replacement)
                with self.assertRaisesRegex(ValueError, refusal):
                    validate_triton_kernel(source.encode(), REQUIREMENTS)

    def test_other_libdevice_calls_remain_outside_admission(self):
        for expression in (
            "libdevice.sin(x)", "libdevice.exp(x)", "libdevice.to(x)",
            "libdevice(x)", "libdevice.tanh.load(x)", "libdevice.__getattribute__('tanh')(x)",
        ):
            with self.subTest(expression=expression):
                source = SOURCE.replace("libdevice.tanh(x)", expression)
                with self.assertRaisesRegex(ValueError, "unsupported call"):
                    validate_triton_kernel(source.encode(), REQUIREMENTS)

    def test_libdevice_parameter_and_kernel_shadowing_are_refused(self):
        source = SOURCE.replace("kernel(a, b)", "kernel(libdevice, b)")
        requirements = {**REQUIREMENTS, "signature": {"libdevice": "*fp32", "b": "*fp32"}}
        with self.assertRaisesRegex(ValueError, "parameter annotations or names"):
            validate_triton_kernel(source.encode(), requirements)
        source = SOURCE.replace("def kernel(", "def libdevice(")
        with self.assertRaisesRegex(ValueError, "kernel entry point"):
            validate_triton_kernel(source.encode(), {**REQUIREMENTS, "kernel_entry_point": "libdevice"})

    def test_local_libdevice_bindings_and_attribute_writes_are_refused(self):
        changes = (
            ("    libdevice = x\n", "reserved name"),
            ("    libdevice, q = x, x\n", "reserved name"),
            ("    libdevice: tl.constexpr = 0\n", "reserved name"),
            ("    libdevice += x\n", "reserved name"),
            ("    for libdevice in range(1):\n        pass\n", "reserved name"),
            ("    libdevice.tanh = x\n", "assignment target"),
            ("    import os as libdevice\n", "unsupported host form"),
        )
        for statement, refusal in changes:
            with self.subTest(statement=statement):
                source = SOURCE.replace("    y =", statement + "    y =")
                with self.assertRaisesRegex(ValueError, refusal):
                    validate_triton_kernel(source.encode(), REQUIREMENTS)


if __name__ == "__main__":
    unittest.main()
