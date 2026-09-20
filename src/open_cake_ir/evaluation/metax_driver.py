"""Retain and launch the native image of a sealed MACA kernel bundle.

Only the requested native ELF is loaded. The bitcode member is not handed to the
runtime, so module admission cannot silently become another compilation.
"""

from __future__ import annotations

import ctypes
from typing import Callable, Sequence

from open_cake_ir.compiler.metax_toolchain import device_image
from open_cake_ir.compiler.target import CodeObject, declared_target
from .loaders import LifecycleError, check_candidate_authority


def _call(api, name: str, *arguments) -> None:
    status = getattr(api, name)(*arguments)
    if status:
        raise RuntimeError(f"MACA {name} failed with status {status}")


def load_runtime(path: str):
    # Host admission proved this absolute file is the runtime torch already mapped.
    # Do not resolve the soname again under a potentially different search path.
    api = ctypes.CDLL(path)
    signatures = {
        "mcModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
        "mcModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
        "mcModuleUnload": [ctypes.c_void_p],
        "mcFuncGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p],
        "mcModuleLaunchKernel": [ctypes.c_void_p, *([ctypes.c_uint] * 7), ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(api, name)
        function.restype = ctypes.c_int
        function.argtypes = arguments
    return api


class LoadedMetaxCandidate:
    """One retained native kernel, with the common tensor-tile lifecycle interface."""

    def __init__(self, candidate, manifest, api, module, function, image, resources):
        self.candidate, self.manifest = candidate, manifest
        self._api, self._module, self._function = api, module, function
        self._image = image
        self.closed = False
        self.launch_calls = 0
        self.resources = resources

    @classmethod
    def load(cls, candidate, manifest, admission, *, api=None):
        payload = candidate.artifact_payloads["mcfatbin"]
        check_candidate_authority(candidate, payload, "mcfatbin", manifest)
        target = declared_target(candidate.target)
        if (target.code_object is not CodeObject.MCFATBIN or admission.target != target.target_id
                or admission.device_name not in target.device_names
                or admission.device_arch != target.target_id
                or admission.warp_size != target.warp_size
                or manifest.hidden_null_pointer_parameters != 0):
            raise ValueError("MACA device admission differs from the sealed target")
        image = ctypes.create_string_buffer(device_image(payload, target.architecture))
        runtime = load_runtime(admission.runtime_library) if api is None else api
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        _call(runtime, "mcModuleLoadData", ctypes.byref(module), image)
        try:
            _call(runtime, "mcModuleGetFunction", ctypes.byref(function), module,
                  manifest.kernel_name.encode("ascii"))
            resources = {"dynamic_shared_bytes": manifest.dynamic_shared_memory_bytes}
            # MACA 3.5.3 mcFuncGetAttribute enum values, compiled from the SDK header:
            # MC_FUNC_ATTRIBUTE_NUM_REGS=4, MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES=3.
            for field, attribute in (("registers_per_thread", 4), ("local_bytes", 3)):
                value = ctypes.c_int()
                _call(runtime, "mcFuncGetAttribute", ctypes.byref(value), attribute, function)
                if value.value < 0:
                    raise ValueError(f"MACA function reports negative {field}")
                resources[field] = value.value
        except BaseException as primary:
            try:
                _call(runtime, "mcModuleUnload", module)
            except BaseException as teardown:
                raise LifecycleError(primary, teardown) from primary
            raise
        return cls(candidate, manifest, runtime, module, function, image, resources)

    def launch(self, arguments: Sequence[object], *, tensor_contract, stream: int = 0) -> None:
        if self.closed:
            raise RuntimeError("MACA module is closed")
        if tensor_contract is not self.manifest or len(arguments) != len(self.manifest.tensor_abi):
            raise ValueError("MACA launch tensor contract differs")
        dtype_names = {"fp32": "torch.float32", "fp16": "torch.float16",
                       "bf16": "torch.bfloat16", "int32": "torch.int32",
                       "fp8_e4m3": "torch.float8_e4m3fn"}
        pointers = []
        for (name, shape, dtype, _mode), argument in zip(self.manifest.tensor_abi, arguments, strict=True):
            if (not argument.is_contiguous() or tuple(argument.shape) != tuple(shape)
                    or str(argument.dtype) != dtype_names.get(dtype)
                    or argument.device.type != "cuda" or argument.device.index != 0):
                raise ValueError(f"MACA tensor {name!r} differs from its sealed ABI")
            if dtype == "fp8_e4m3":
                from ..compiler.ir import DType
                if argument.element_size() != DType.FP8_E4M3.itemsize:
                    raise ValueError(f"MACA tensor {name!r} differs from its sealed FP8 storage width")
            pointer = argument.data_ptr()
            if type(pointer) is not int or pointer <= 0:
                raise ValueError(f"MACA tensor {name!r} has no device address")
            pointers.append(ctypes.c_void_p(pointer))
        from .launch_manifest import check_pointer_alignments
        check_pointer_alignments({row[0]: pointer.value for row, pointer
                                  in zip(self.manifest.tensor_abi, pointers, strict=True)},
                                 getattr(self.manifest, 'pointer_alignments', {}))
        pointers.extend(ctypes.c_void_p(0) for _ in range(self.manifest.hidden_null_pointer_parameters))
        slots = (ctypes.c_void_p * len(pointers))(
            *(ctypes.cast(ctypes.pointer(pointer), ctypes.c_void_p) for pointer in pointers))
        _call(self._api, "mcModuleLaunchKernel", self._function,
              *(ctypes.c_uint(n) for n in (*self.manifest.grid, *self.manifest.block)),
              ctypes.c_uint(self.manifest.dynamic_shared_memory_bytes),
              ctypes.c_void_p(stream), slots, None)
        self.launch_calls += 1

    def close(self, *, synchronize: Callable[[], None] | None = None,
              primary: BaseException | None = None) -> None:
        failures = [] if primary is None else [primary]
        if not self.closed:
            self.closed = True
            if synchronize is not None:
                try:
                    synchronize()
                except BaseException as error:
                    failures.append(error)
            try:
                _call(self._api, "mcModuleUnload", self._module)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise LifecycleError(failures[0], *failures[1:])
