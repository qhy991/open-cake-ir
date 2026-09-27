"""Dormant instruction registration and the exact native MACA source boundary."""
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends.metax import DIRECTED_FMA_FUNCTIONS
from open_cake_ir.compiler.ir import DType, ElementwiseOp
from open_cake_ir.compiler.ir.instruction_contracts import ContractKind, contract
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import validate_triton_kernel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''import triton
import triton.language as tl
from triton.language.extra import libdevice
@triton.jit
def kernel(a, b):
    x = tl.load(a)
    result = libdevice.FUNCTION(x, x, x)
    tl.store(b, result)
'''
REQUIREMENTS = {"kernel_entry_point": "kernel", "signature": {"a": "*fp32", "b": "*fp32"},
                "compile_constants": {}, "code_object": "mcfatbin"}


class MetaxDirectedFmaAdmission(unittest.TestCase):
    def test_registered_contracts_keep_the_existing_typed_ternary_primitive(self):
        for name in DIRECTED_FMA_FUNCTIONS:
            record = contract(name)
            self.assertIs(record.kind, ContractKind.ELEMENTWISE)
            self.assertIs(record.elementwise_op, ElementwiseOp.FMA)
            self.assertIs(record.elementwise_dtype, DType.FP32)
        for path in (ROOT / "compiler/targets").glob("*.json"):
            self.assertFalse(set(DIRECTED_FMA_FUNCTIONS) & Target.load(path).instruction_contracts)

    def test_direct_ternary_calls_are_admitted_only_by_the_mcfatbin_route(self):
        for function in DIRECTED_FMA_FUNCTIONS.values():
            source = SOURCE.replace("FUNCTION", function).encode()
            validate_triton_kernel(source, REQUIREMENTS)
            for code_object in ("cubin", "hsaco", "metal_binary_archive", None):
                with self.subTest(function=function, code_object=code_object), self.assertRaises(ValueError):
                    validate_triton_kernel(source, {**REQUIREMENTS, "code_object": code_object})

    def test_the_call_guard_refuses_other_functions_and_nonternary_arguments(self):
        for expression in ("libdevice.fma_rz(x, x)", "libdevice.fma_rz(x, x, x, x)",
                           "libdevice.fma_rz(x, x, c=x)", "libdevice.fma_rz(*x)",
                           "libdevice.add_ru(x, x)", "libdevice.fma_rn(x, x, x)",
                           "libdevice.fma(x, x, x)"):
            source = SOURCE.replace("libdevice.FUNCTION(x, x, x)", expression)
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "unsupported call"):
                validate_triton_kernel(source.encode(), REQUIREMENTS)

    def test_function_aliases_cannot_escape_the_direct_call_guard(self):
        source = SOURCE.replace("FUNCTION", "fma_rz")
        source = source.replace("    result = libdevice.fma_rz(x, x, x)",
                                "    fn = libdevice.fma_rz\n    result = fn(x, x, x)")
        with self.assertRaisesRegex(ValueError, "libdevice requires a direct tanh call"):
            validate_triton_kernel(source.encode(), REQUIREMENTS)
