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
