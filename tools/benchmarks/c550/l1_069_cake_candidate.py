"""L1/069 return-value adapter for precompiled CAKE C1 native MACA kernels.

Torch supplies storage, tensor metadata, and the current stream. All arithmetic
runs in native ELF images produced from the retained CAKE Schedules before this
module is evaluated. This module never invokes a compiler or a reference.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
from pathlib import Path

import torch

from open_cake_ir.evaluation.metax_driver import load_runtime, _call
from open_cake_ir.evaluation.triton_metax import observe_local_metax

_ROOT = Path(os.environ.get("CAKE_BENCH_ARTIFACTS", str(Path(__file__).parent / "compiled")))
_INDEX = json.loads((_ROOT / "index.json").read_text())
if (_INDEX["status"] != "compiled" or _INDEX["target"] != "xcore1002"
        or _INDEX["compiler_commit"] != "5bb474c6df2948d1cc50b2d46f9ab828525be3c2"):
    raise ValueError("This adapter requires the fully compiled frozen C1 artifacts")
_CASES = {(tuple(case["shape"]), float(case["eps"])): case["variant"]
          for case in _INDEX["cases"]}
_LOADED = {}
_ADMISSION = None
_API = None


def _load(variant):
    global _API, _ADMISSION
    if variant in _LOADED:
        return _LOADED[variant]
    if _ADMISSION is None:
        _ADMISSION = observe_local_metax("xcore1002", runtime_library=_INDEX["runtime_library"])
        _API = load_runtime(_ADMISSION.runtime_library)
    record = _INDEX["variants"][variant]
    image = ctypes.create_string_buffer((_ROOT / variant / "native.elf").read_bytes())
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    _call(_API, "mcModuleLoadData", ctypes.byref(module), image)
    try:
        _call(_API, "mcModuleGetFunction", ctypes.byref(function), module,
              record["kernel_name"].encode("ascii"))
    except BaseException:
        _call(_API, "mcModuleUnload", module)
        raise
    _LOADED[variant] = (record, image, module, function)
    return _LOADED[variant]


@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    shape = tuple(hidden_states.shape)
    variant = _CASES.get((shape, float(eps)))
    if variant is None:
        raise ValueError("The input shape and epsilon are outside the 16 fixed workloads")
    if (tuple(residual.shape) != shape or tuple(weight.shape) != (shape[-1],)
            or any(t.dtype != torch.bfloat16 or not t.is_contiguous()
                   or t.device.type != "cuda" or t.device.index != 0
                   for t in (hidden_states, residual, weight))):
        raise ValueError("L1/069 expects contiguous BF16 tensors on the one leased device")
    record, image, module, function = _load(variant)
    out = torch.empty_like(hidden_states)
    tensors = (hidden_states, residual, weight, out)
    pointers = [ctypes.c_void_p(tensor.data_ptr()) for tensor in tensors]
    pointers.extend(ctypes.c_void_p(0) for _ in range(record["hidden_null_pointers"]))
    slots = (ctypes.c_void_p * len(pointers))(
        *(ctypes.cast(ctypes.pointer(pointer), ctypes.c_void_p) for pointer in pointers))
    _call(_API, "mcModuleLaunchKernel", function,
          *(ctypes.c_uint(n) for n in (*record["grid"], *record["block"])),
          ctypes.c_uint(record["dynamic_shared_bytes"]),
          ctypes.c_void_p(torch.cuda.current_stream(0).cuda_stream), slots, None)
    return out


@atexit.register
def _close():
    if _LOADED:
        torch.cuda.synchronize(0)
        for record, image, module, function in _LOADED.values():
            _call(_API, "mcModuleUnload", module)
        _LOADED.clear()
