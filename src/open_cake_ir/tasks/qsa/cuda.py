"""Exact-CUBIN launch boundary for the multi-kernel QSA Program.

The Workload owns semantics and the candidate owns implementation bytes.  This module
owns only the common launch seam: a closed kernel list, exact tensor bindings, one CUDA
context, and deterministic module teardown.  It deliberately does not compile source or
decide correctness.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence, cast

_PROGRAM_FIELDS = {"schema_version", "abi", "arm", "kernels"}
_KERNEL_FIELDS = {
    "id",
    "cubin",
    "kernel_name",
    "grid",
    "block",
    "dynamic_shared_memory_bytes",
    "arguments",
    "hidden_null_pointer_parameters",
}
_EXPECTED_ARGUMENTS = {
    "pool": ("index_k", "pooled"),
    "layernorm": ("pooled", "k_norm_weight", "normalized_keys"),
    "score_topk": ("index_q", "normalized_keys", "block_indices"),
    "pool_layernorm": ("index_k", "k_norm_weight", "normalized_keys"),
    "expand": ("block_indices", "token_indices"),
    "attention": ("q", "k", "v", "token_indices", "output"),
}
_EXPECTED_ORDER = {
    "open_cake": ("pool", "layernorm", "score_topk", "expand", "attention"),
    "direct_cuda": ("pool_layernorm", "score_topk", "expand", "attention"),
}
_MAX_DYNAMIC_SHARED_MEMORY_BYTES = 232_448


def _dimensions(value: object, context: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{context} differs")
    parsed = cast(tuple[int, int, int], tuple(value))
    if parsed[0] > 2_147_483_647 or parsed[1] > 65_535 or parsed[2] > 65_535:
        raise ValueError(f"{context} exceeds CUDA grid limits")
    return parsed


def _owned_file(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} is unsafe")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} custody differs")
    return path


@dataclass(frozen=True)
class QsaKernelArtifact:
    kernel_id: str
    cubin_path: Path
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    arguments: tuple[str, ...]
    hidden_null_pointer_parameters: int


@dataclass(frozen=True)
class QsaProgramArtifact:
    """Checked launch metadata and exact CUBIN paths for one QSA candidate."""

    arm: str
    kernels: tuple[QsaKernelArtifact, ...]

    @classmethod
    def load(cls, build_root: str | Path, path: str | Path) -> "QsaProgramArtifact":
        root = Path(build_root).resolve(strict=True)
        source = Path(path).resolve(strict=True)
        if root not in source.parents or source.is_symlink():
            raise ValueError("QSA Program artifact custody differs")
        document = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(document, Mapping)
            or set(document) != _PROGRAM_FIELDS
            or document.get("schema_version") != 1
            or document.get("abi") != "qsa_prefill_task_geometry_v1"
            or document.get("arm") not in _EXPECTED_ORDER
            or not isinstance(document.get("kernels"), list)
        ):
            raise ValueError("QSA Program artifact fields differ")
        arm = cast(str, document["arm"])
        kernels: list[QsaKernelArtifact] = []
        for index, raw in enumerate(cast(list[object], document["kernels"])):
            if not isinstance(raw, Mapping) or set(raw) != _KERNEL_FIELDS:
                raise ValueError(f"QSA Program kernel {index} fields differ")
            kernel_id = raw.get("id")
            arguments = raw.get("arguments")
            hidden = raw.get("hidden_null_pointer_parameters")
            dynamic = raw.get("dynamic_shared_memory_bytes")
            if (
                not isinstance(kernel_id, str)
                or kernel_id not in _EXPECTED_ARGUMENTS
                or not isinstance(raw.get("kernel_name"), str)
                or not raw["kernel_name"]
                or not isinstance(arguments, list)
                or tuple(arguments) != _EXPECTED_ARGUMENTS[kernel_id]
                or not isinstance(hidden, int)
                or isinstance(hidden, bool)
                or hidden not in {0, 2}
                or not isinstance(dynamic, int)
                or isinstance(dynamic, bool)
                or not 0 <= dynamic <= _MAX_DYNAMIC_SHARED_MEMORY_BYTES
            ):
                raise ValueError(f"QSA Program kernel {index} contract differs")
            block = _dimensions(raw["block"], f"QSA Program kernel {index} block")
            if block[0] * block[1] * block[2] > 1024:
                raise ValueError(f"QSA Program kernel {index} block exceeds 1024 threads")
            cubin = _owned_file(root, raw["cubin"], f"QSA Program kernel {index} cubin")
            if not cubin.read_bytes().startswith(b"\x7fELF"):
                raise ValueError(f"QSA Program kernel {index} CUBIN differs")
            kernels.append(
                QsaKernelArtifact(
                    kernel_id,
                    cubin,
                    cast(str, raw["kernel_name"]),
                    _dimensions(raw["grid"], f"QSA Program kernel {index} grid"),
                    block,
                    dynamic,
                    cast(tuple[str, ...], tuple(arguments)),
                    hidden,
                )
            )
        if tuple(item.kernel_id for item in kernels) != _EXPECTED_ORDER[arm]:
            raise ValueError("QSA Program launch order differs")
        return cls(arm, tuple(kernels))


def _driver_call(api: object, name: str, *arguments: object, outputs: int) -> tuple[object, ...]:
    function = getattr(api, name, None)
    if not callable(function):
        raise RuntimeError(f"CUDA Driver lacks {name}")
    result = function(*arguments)
    values = result if isinstance(result, tuple) else (result,)
    if len(values) != outputs + 1 or int(values[0]) != 0:
        raise RuntimeError(f"CUDA Driver {name} failed")
    return values[1:]


def _handle(value: object) -> object:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class LoadedQsaProgram:
    """Load every QSA kernel once and launch the closed Program on one stream."""

    def __init__(
        self,
        artifact: QsaProgramArtifact,
        *,
        driver: object | None = None,
    ) -> None:
        if driver is None:
            from cuda.bindings import driver as cuda_driver

            driver = cuda_driver
        self.artifact = artifact
        self._api = driver
        (context,) = _driver_call(driver, "cuCtxGetCurrent", outputs=1)
        if not context:
            raise RuntimeError("QSA Program requires a current CUDA context")
        self._context = _handle(context)
        self._modules: list[object] = []
        self._functions: dict[str, object] = {}
        modules_by_path: dict[Path, object] = {}
        try:
            for kernel in artifact.kernels:
                module = modules_by_path.get(kernel.cubin_path)
                if module is None:
                    (module,) = _driver_call(
                        driver,
                        "cuModuleLoadData",
                        kernel.cubin_path.read_bytes(),
                        outputs=1,
                    )
                    modules_by_path[kernel.cubin_path] = module
                    self._modules.append(module)
                (function,) = _driver_call(
                    driver,
                    "cuModuleGetFunction",
                    module,
                    kernel.kernel_name.encode("ascii"),
                    outputs=1,
                )
                if kernel.dynamic_shared_memory_bytes > 49_152:
                    enum = getattr(driver, "CUfunction_attribute")
                    _driver_call(
                        driver,
                        "cuFuncSetAttribute",
                        function,
                        getattr(
                            enum,
                            "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
                        ),
                        kernel.dynamic_shared_memory_bytes,
                        outputs=0,
                    )
                self._functions[kernel.kernel_id] = function
        except BaseException:
            for module in reversed(self._modules):
                try:
                    _driver_call(driver, "cuModuleUnload", module, outputs=0)
                except BaseException:
                    pass
            raise
        self.launch_calls = 0
        self.closed = False

    def _checked_stream(self, stream: object) -> object:
        if self.closed:
            raise ValueError("QSA Program is closed")
        (context,) = _driver_call(self._api, "cuCtxGetCurrent", outputs=1)
        if _handle(context) != self._context:
            raise ValueError("QSA Program CUDA context changed")
        stream_type = getattr(self._api, "CUstream", None)
        return (
            stream_type(stream)
            if isinstance(stream, int) and callable(stream_type)
            else stream
        )

    def _launch_artifact(
        self,
        kernel: QsaKernelArtifact,
        tensors: Mapping[str, object],
        launch_stream: object,
    ) -> None:
        pointers: list[int] = []
        for name in kernel.arguments:
            tensor = tensors.get(name)
            pointer = getattr(tensor, "data_ptr", lambda: 0)()
            if (
                not isinstance(pointer, int)
                or isinstance(pointer, bool)
                or pointer <= 0
                or not bool(getattr(tensor, "is_cuda", False))
                or not bool(getattr(tensor, "is_contiguous", lambda: False)())
            ):
                raise ValueError(f"QSA Program tensor {name!r} differs")
            pointers.append(pointer)
        if len(set(pointers)) != len(pointers):
            raise ValueError(f"QSA Program kernel {kernel.kernel_id!r} aliases tensors")
        values = [ctypes.c_void_p(pointer) for pointer in pointers]
        values.extend(
            ctypes.c_void_p(0)
            for _ in range(kernel.hidden_null_pointer_parameters)
        )
        parameters = (ctypes.c_void_p * len(values))(
            *[
                ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
                for value in values
            ]
        )
        _driver_call(
            self._api,
            "cuLaunchKernel",
            self._functions[kernel.kernel_id],
            *kernel.grid,
            *kernel.block,
            kernel.dynamic_shared_memory_bytes,
            launch_stream,
            parameters,
            0,
            outputs=0,
        )
        self.launch_calls += 1

    def launch(
        self,
        tensors: Mapping[str, object],
        *,
        stream: object,
        boundary: Callable[[str, str], None] | None = None,
    ) -> None:
        """Launch in contract order, optionally marking each queued node boundary."""

        launch_stream = self._checked_stream(stream)
        for kernel in self.artifact.kernels:
            if boundary is not None:
                boundary(kernel.kernel_id, "before")
            self._launch_artifact(kernel, tensors, launch_stream)
            if boundary is not None:
                boundary(kernel.kernel_id, "after")

    def close(self, *, synchronize: Callable[[], None]) -> None:
        if self.closed:
            raise ValueError("QSA Program is already closed")
        synchronize()
        for module in reversed(self._modules):
            _driver_call(self._api, "cuModuleUnload", module, outputs=0)
        self.closed = True


def qsa_program_tensors(inputs: Mapping[str, object], output: object) -> dict[str, object]:
    """Allocate the four Program-owned intermediates on the input device."""

    import torch

    device = getattr(inputs["q"], "device")
    return {
        **inputs,
        "pooled": torch.empty((8192, 128), dtype=torch.float32, device=device),
        "normalized_keys": torch.empty(
            (8192, 128), dtype=torch.float32, device=device
        ),
        "block_indices": torch.empty(
            (32768, 512), dtype=torch.int32, device=device
        ),
        "token_indices": torch.empty(
            (32768, 2048), dtype=torch.int32, device=device
        ),
        "output": output,
    }
