"""The native source boundary admits a dtype constant without admitting host calls."""
import unittest

from open_cake_ir.compiler.toolchain import validate_triton_kernel


SOURCE = b'''import triton
import triton.language as tl
@triton.jit
def convert(x, out):
    i = tl.arange(0, 256)
    value = tl.load(x + i)
    tl.store(out + i, value.to(tl.float8e4nv))
'''
REQUIREMENTS = {'kernel_entry_point': 'convert', 'signature': {'x': '*fp32', 'out': '*fp8e4nv'},
                'compile_constants': {}}


class Fp8TypeSpelling(unittest.TestCase):
    def test_existing_fp8_pointer_type_can_be_named_as_a_cast_destination(self):
        # Source admission only: the target/toolchain must still qualify this cast.
        validate_triton_kernel(SOURCE, REQUIREMENTS)

    def test_dtype_admission_does_not_allow_calls_attributes_or_other_formats(self):
        for replacement in (b'tl.float8e4nv()', b'tl.float8e4nv.__class__', b'tl.float8e4unknown'):
            with self.subTest(expression=replacement), self.assertRaises(ValueError):
                validate_triton_kernel(SOURCE.replace(b'tl.float8e4nv', replacement), REQUIREMENTS)
