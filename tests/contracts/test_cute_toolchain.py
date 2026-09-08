"""CPU source/ABI fixtures; these do not claim device execution or correctness."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from open_cake_ir.compiler.cute_toolchain import (
    CuTeCompilation, validate_cute_requirements, validate_cute_kernel, validate_cute_compilation,
    _launcher_source, _ptx_entry, _cubin_parameters, _deny_cuda_calls,
)


SOURCE = b'''import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

@cute.kernel
def fixture(a: cute.Pointer, b: cute.Pointer, bias: cute.Pointer, c: cute.Pointer):
    tid, ty, tz = cute.arch.thread_idx()
    layout = cute.make_layout((1,))
    output = cute.make_tensor(c, layout)
    if tid == 0:
        output[0] = cutlass.Float32(0.0)
'''


def requirements():
    return {"compiler": "cutlass_cute_dsl", "source_language": "python", "target": "sm_103a",
        "kernel_entry_point": "fixture", "signature": [
            {"name": name, "dtype": dtype} for name, dtype in
            (("a", "bf16"), ("b", "bf16"), ("bias", "fp32"), ("c", "fp32"))],
        "grid": [32, 32, 1], "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0}


NAME = "kernel_cutlass_fixture_ptrbf16gmem_ptrbf16gmem_ptrf32gmem_ptrf32gmem_0"


def ptx_fixture():
    params = ',\n'.join(f".param .u64 .ptr .global .align {alignment} {NAME}_param_{i}"
                        for i, alignment in enumerate((2, 2, 4, 4)))
    return f".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry {NAME}(\n{params}\n)\n.reqntid 32, 1, 1\n{{ ret; }}\n".encode()


def elf_fixture():
    # Match the observable cuobjdump 13.1 primary .nv.info format, not binary hashes.
    params = ''.join(f"\tAttribute: EIATTR_KPARAM_INFO\n\tFormat: EIFMT_SVAL\n"
        f"\tValue: Index : 0x0 Ordinal : {i:#x} Offset : {8*i:#x} Size : 0x8\n"
        f"\t\tPointee's logAlignment : {1 if i < 2 else 2:#x} Space : 0x4 cbank : 0x1f\n"
        for i in reversed(range(4)))
    return f"64-bit ELF: type=ET_EXEC, ABI=8, sm=103a, toolkit=12.9\n\n.nv.info.{NAME}\n" + params + (
        "\tAttribute: EIATTR_CBANK_PARAM_SIZE\n\tFormat: EIFMT_HVAL\n\tValue: 0x20\n"
        "\tAttribute: EIATTR_REQNTID\n\tFormat: EIFMT_SVAL\n\tValue: 0x20 0x1 0x1\n\n.nv.compat\n")


def compilation_fixture(source=SOURCE, request=None):
    request = requirements() if request is None else request
    resource = f"\nFunction {NAME}:\n REG:14 STACK:0 SHARED:0 LOCAL:0\n"
    elf = elf_fixture()
    report = {"compiler_version": "4.5.2", "target": "sm_103a", "entry_point": NAME,
        "resources": {"registers_per_thread": 14, "stack_bytes": 0, "static_shared_bytes": 0, "local_bytes": 0},
        "resource_report": resource, "elf_report": elf,
        "device_parameters": _cubin_parameters(elf, NAME, requirements()["signature"]),
        "denied_cuda_calls": ["cuda.bindings.driver.cuInit"], "jit_engine_created": False}
    return CuTeCompilation(source, "sm_103a", NAME, {
        "source": _launcher_source(source, request), "ptx": ptx_fixture(),
        "cubin": b"\x7fELFcpu-fixture-not-executable", "toolchain_resource_report": json.dumps(report).encode()},
        32, 0, "4.5.2")


class CuTeSourceAdmissionTests(unittest.TestCase):
    def test_ordered_signature_survives_canonical_json_and_raw_source_is_validated(self):
        request = json.loads(json.dumps(requirements(), sort_keys=True))
        validate_cute_kernel(SOURCE, request)
        self.assertEqual([row["name"] for row in request["signature"]], ["a", "b", "bias", "c"])
        with self.assertRaisesRegex(ValueError, "positional pointer"):
            validate_cute_kernel(SOURCE.replace(b"fixture(a: cute.Pointer, b:", b"fixture(b: cute.Pointer, a:"), request)

    def test_input_declaration_order_controls_pointer_types_and_alignment(self):
        request = requirements()
        request['signature'] = [request['signature'][index] for index in (2, 1, 0, 3)]
        source = SOURCE.replace(b'fixture(a: cute.Pointer, b: cute.Pointer, bias: cute.Pointer, c:',
                                b'fixture(bias: cute.Pointer, b: cute.Pointer, a: cute.Pointer, c:')
        validate_cute_kernel(source, request)
        name = 'kernel_cutlass_fixture_ptrf32gmem_ptrbf16gmem_ptrbf16gmem_ptrf32gmem_0'
        ptx = ptx_fixture().replace(NAME.encode(), name.encode())
        ptx = ptx.replace(b'.align 2 ' + name.encode() + b'_param_0', b'.align 4 ' + name.encode() + b'_param_0')
        ptx = ptx.replace(b'.align 4 ' + name.encode() + b'_param_2', b'.align 2 ' + name.encode() + b'_param_2')
        self.assertEqual(_ptx_entry(ptx, request), name)
        with self.assertRaises(ValueError):
            _ptx_entry(ptx_fixture(), request)

    def test_closed_requirements_refuse_scalar_arguments_targets_and_launch_options(self):
        mutations = {"target": "sm_100a", "compiler": "cutedsl", "block": [64, 1, 1],
                     "grid": [True, 1, 1], "dynamic_shared_memory_bytes": True,
                     "signature": {"a": "bf16"}}
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cute_requirements({**requirements(), key: value})
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_cute_requirements({**requirements(), "options": "--enable-tvm-ffi"})
        request = requirements(); request["signature"][0]["dtype"] = "fp32"
        with self.assertRaisesRegex(ValueError, "order or types"):
            validate_cute_requirements(request)

    def test_host_escape_forms_never_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'MUST_NOT_EXIST'
            effect = f'open({str(marker)!r}, "w").write("bad")'.encode()
            candidates = [effect + b'\n' + SOURCE, SOURCE + effect,
                SOURCE.replace(b'@cute.kernel', b'@' + effect),
                SOURCE.replace(b'a: cute.Pointer', b'a: ' + effect),
                SOURCE.replace(b'    tid,', b'    import os\n    tid,'),
                SOURCE.replace(b'    tid,', b'    def helper():\n        pass\n    tid,'),
                SOURCE.replace(b'    tid,', b'    while True:\n        pass\n    tid,'),
                SOURCE.replace(b'cute.arch.thread_idx()', b'cute.__dict__["arch"].thread_idx()'),
                SOURCE.replace(b'cute.arch.thread_idx()', b'getattr(cute, "arch").thread_idx()'),
                SOURCE.replace(b'cute.arch.thread_idx()', b'cute.compile()'),
                SOURCE.replace(b'cute.arch.thread_idx()', b'output.to()'),
                SOURCE.replace(b'    tid,', b'    cute = 1\n    tid,'),
                SOURCE.replace(b'    tid,', b'    function = cute.arch.thread_idx\n    function()\n    tid,'),
                SOURCE.replace(b'    tid,', b'    value = (lambda: 1)()\n    tid,'),
                SOURCE.replace(b'    tid,', b'    output.launch(grid=(1,1,1))\n    tid,')]
            for source in candidates:
                with self.subTest(source=source[:65]), self.assertRaises(ValueError):
                    validate_cute_kernel(source, requirements())
                self.assertFalse(marker.exists())

    def test_cache_copy_and_mma_forms_are_in_the_admitted_kernel_interface(self):
        body = b'''    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
    thread = mma.get_slice(tid)
    coords = thread.partition_A(cute.make_identity_tensor((16, 16)))
    fragment = cute.make_rmem_tensor(coords.shape, cutlass.BFloat16)
    fragment.fill(0)
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=16)
    for i in cutlass.range_constexpr(cute.size(fragment)):
        fragment[i] = cutlass.BFloat16(0)
'''
        validate_cute_kernel(SOURCE + body, requirements())


class CuTeArtifactTests(unittest.TestCase):
    def test_ptx_and_cubin_prove_ordered_four_pointer_abi_and_one_warp(self):
        self.assertEqual(_ptx_entry(ptx_fixture(), requirements()), NAME)
        self.assertEqual([row["offset"] for row in _cubin_parameters(elf_fixture(), NAME, requirements()["signature"])], [0, 8, 16, 24])
        validate_cute_compilation(compilation_fixture(), SOURCE, requirements())

    def test_ptx_rejects_scalar_hidden_target_entry_and_block_mismatches(self):
        ptx = ptx_fixture()
        bad = [ptx.replace(b'.ptr .global .align 2 ', b'', 1),
               ptx.replace(b'.target sm_103a', b'.target sm_100a'),
               ptx.replace(b'.reqntid 32,', b'.reqntid 64,'),
               ptx.replace(b'\n)\n', b', .param .u64 hidden\n)\n'), ptx + ptx,
               ptx.replace(NAME.encode(), b'wrong_entry')]
        for value in bad:
            with self.subTest(ptx=value[:80]), self.assertRaises(ValueError):
                _ptx_entry(value, requirements())

    def test_cubin_rejects_mismatched_binary_parameter_metadata(self):
        elf = elf_fixture()
        bad = [elf.replace('sm=103a', 'sm=100a'), elf.replace('Size : 0x8', 'Size : 0x4', 1),
               elf.replace('Offset : 0x18', 'Offset : 0x20'), elf.replace('Space : 0x4', 'Space : 0x0', 1),
               elf.replace('Ordinal : 0x3', 'Ordinal : 0x2'), elf.replace('Value: 0x20 0x1', 'Value: 0x40 0x1'),
               elf.replace('Value: 0x20\n', 'Value: 0x28\n'), elf + '\n.nv.info.extra\n\tValue: 1\n']
        for value in bad:
            with self.subTest(elf=value[:80]), self.assertRaises(ValueError):
                _cubin_parameters(value, NAME, requirements()["signature"])

    def test_receipt_checks_both_code_and_observed_resource_evidence(self):
        compilation = compilation_fixture()
        for fields in ({"target": "sm_100a"}, {"entry_point": "wrong"}, {"threads_per_cta": True},
                       {"dynamic_shared_bytes": 4}, {"compiler_version": "4.5.1"}, {"source": b"wrong"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_cute_compilation(replace(compilation, **fields), SOURCE, requirements())
        for key, value in (("source", b"wrong"), ("cubin", b"not-elf"), ("ptx", b"wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cute_compilation(replace(compilation, artifacts={**compilation.artifacts, key: value}), SOURCE, requirements())
        report = json.loads(compilation.artifacts['toolchain_resource_report'])
        for key, value in (("jit_engine_created", True), ("denied_cuda_calls", ["cuda.bindings.driver.cuLaunchKernel"]),
                           ("resources", {}), ("device_parameters", [])):
            artifacts = {**compilation.artifacts, 'toolchain_resource_report': json.dumps({**report, key: value}).encode()}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cute_compilation(replace(compilation, artifacts=artifacts), SOURCE, requirements())

    def test_driver_guard_denies_functions_preserves_enum_types_and_retains_attempts(self):
        cuda = ModuleType('cuda'); bindings = ModuleType('cuda.bindings')
        driver = ModuleType('cuda.bindings.driver'); runtime = ModuleType('cuda.bindings.runtime')
        driver.cuInit = mock.Mock(); driver.cuLaunchKernel = mock.Mock()
        runtime.cudaGetDevice = mock.Mock(); runtime.cudaError_t = type('cudaError_t', (), {})
        cuda.bindings = bindings; bindings.driver = driver; bindings.runtime = runtime
        original = driver.cuInit
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {
            'cuda': cuda, 'cuda.bindings': bindings, 'cuda.bindings.driver': driver,
            'cuda.bindings.runtime': runtime}):
            audit = Path(directory) / 'guard.json'
            with _deny_cuda_calls(audit) as calls:
                for fn in (driver.cuInit, driver.cuLaunchKernel, runtime.cudaGetDevice):
                    with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
                        fn(0)
                self.assertIsInstance(runtime.cudaError_t, type)
            self.assertIs(driver.cuInit, original); original.assert_not_called()
            self.assertEqual(json.loads(audit.read_text())['denied_cuda_calls'], calls)
            self.assertEqual(len(calls), 3)

    def test_worker_guard_remains_active_for_late_diagnostic_queries(self):
        cuda = ModuleType('cuda'); bindings = ModuleType('cuda.bindings')
        driver = ModuleType('cuda.bindings.driver'); runtime = ModuleType('cuda.bindings.runtime')
        original = mock.Mock(); driver.cuInit = original
        cuda.bindings = bindings; bindings.driver = driver; bindings.runtime = runtime
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {
            'cuda': cuda, 'cuda.bindings': bindings, 'cuda.bindings.driver': driver,
            'cuda.bindings.runtime': runtime}):
            audit = Path(directory) / 'guard.json'
            with _deny_cuda_calls(audit, retain=True):
                pass
            with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
                driver.cuInit(0)
            original.assert_not_called()
            self.assertEqual(json.loads(audit.read_text())['denied_cuda_calls'], ['cuda.bindings.driver.cuInit'])


if __name__ == '__main__':
    unittest.main()
