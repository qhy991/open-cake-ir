"""Host-owned exact-CUBIN CUDA Driver launch with no fallback."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Protocol, Sequence, cast

from open_cake_ir.compiler.target import CodeObject, Target, declared_target

from .attempts import valid_job_mode
from .core import LaunchableCandidate
from .loaders import LifecycleError, check_launch_authority
from .platforms import platform_for

_ATTRIBUTES = (
    "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
    "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
)
_DYNAMIC_SHARED_OPT_IN_THRESHOLD = 49_152


def _cubin_target(target_id: str) -> Target:
    """The declared Target a CUDA Driver launch is checked against.

    A launch here loads a cubin, so a Target that produces another object is refused by
    that object's name rather than admitted to a check written for a compute capability
    it does not declare.
    """
    target = declared_target(target_id)
    if target.code_object is not CodeObject.CUBIN:
        raise ValueError(
            f"CUDA Driver launch requires a cubin target; {target_id!r} declares "
            f"{target.code_object.value}"
        )
    return target


def _binary_version(target: Target) -> int:
    """CU_FUNC_ATTRIBUTE_BINARY_VERSION as the declared capability pair encodes it."""
    major, minor = cast(tuple[int, int], target.compute_capability)
    return major * 10 + minor


# The shared lifecycle exception under the name every retained test still raises it by.
CudaLifecycleError = LifecycleError


class TensorLike(Protocol):
    shape: object
    dtype: object
    is_cuda: bool
    device: object

    def is_contiguous(self) -> bool: ...

    def data_ptr(self) -> int: ...


def _driver_call(driver: object, name: str, *arguments: object, outputs: int) -> tuple[object, ...]:
    function = getattr(driver, name, None)
    if function is None or not callable(function):
        raise RuntimeError(f"CUDA Driver lacks {name}")
    result = cast(Callable[..., object], function)(*arguments)
    values = result if isinstance(result, tuple) else (result,)
    if len(values) != outputs + 1:
        raise RuntimeError(f"CUDA Driver {name} result arity differs")
    try:
        code = int(values[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"CUDA Driver {name} returned a non-integer code") from error
    if code != 0:
        raise RuntimeError(f"CUDA Driver {name} failed with code {code}")
    return values[1:]


def _is_null(value: object) -> bool:
    if value is None:
        return True
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return not bool(value)


def _handle_identity(value: object) -> object:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class CudaModules:
    """Own loaded module handles in one current context, without owning execution.

    Failed unloads remain owned for a later close attempt. Once teardown starts, no
    caller may launch or load again, even if a failed unload leaves a handle resident.
    ``closed`` means every owned handle was successfully unloaded.
    """

    def __init__(self, api: object) -> None:
        self.api = api
        (context,) = _driver_call(api, "cuCtxGetCurrent", outputs=1)
        if _is_null(context):
            raise RuntimeError("CUDA Driver requires a current CUDA context")
        self._context = _handle_identity(context)
        self._modules: list[object] = []
        self._teardown_started = False

    @property
    def closed(self) -> bool:
        return self._teardown_started and not self._modules

    def check_context(self) -> None:
        (context,) = _driver_call(self.api, "cuCtxGetCurrent", outputs=1)
        if _is_null(context) or _handle_identity(context) != self._context:
            raise ValueError("CUDA context changed")

    def check_open(self) -> None:
        if self._teardown_started:
            raise ValueError("CUDA module teardown has started")
        self.check_context()

    def load(self, cubin: bytes) -> object:
        self.check_open()
        (module,) = _driver_call(self.api, "cuModuleLoadData", cubin, outputs=1)
        if _is_null(module):
            raise RuntimeError("CUDA Driver loaded a null module")
        self._modules.append(module)
        return module

    def function(self, module: object, name: str) -> object:
        self.check_open()
        (function,) = _driver_call(
            self.api, "cuModuleGetFunction", module, name.encode("ascii"), outputs=1
        )
        if _is_null(function):
            raise RuntimeError("CUDA Driver resolved a null function")
        return function

    def close(
        self, *, synchronize: Callable[[], None] | None = None,
        primary: BaseException | None = None,
    ) -> None:
        """Attempt every unload in reverse order and retain all observed failures."""
        if self.closed:
            raise ValueError("CUDA modules are already closed")
        errors = [] if primary is None else [primary]
        try:
            self.check_context()
        except BaseException as error:
            errors.append(error)
        else:
            self._teardown_started = True
            if synchronize is not None:
                try:
                    synchronize()
                except BaseException as error:
                    errors.append(error)
            for index in range(len(self._modules) - 1, -1, -1):
                try:
                    self.check_context()
                except BaseException as error:
                    # No module belongs to the replacement context. Leave the remaining
                    # handles owned so cleanup can resume when the caller restores it.
                    errors.append(error)
                    break
                try:
                    _driver_call(self.api, "cuModuleUnload", self._modules[index], outputs=0)
                except BaseException as error:
                    errors.append(error)
                else:
                    del self._modules[index]
        if len(errors) > 1:
            raise LifecycleError(errors[0], errors[1], *errors[2:]) from errors[0]
        if errors:
            raise errors[0]


def _tensor_contract(
    arguments: Sequence[TensorLike], contract: TensorContract
) -> tuple[dict[str, object], tuple[int, ...]]:
    if len(arguments) != len(contract.tensors):
        raise ValueError("CUDA Driver launch tensor count differs from the task contract")
    observed: dict[str, object] = {}
    pointers: list[int] = []
    device: str | None = None
    for tensor, (name, expected_shape, expected_dtype) in zip(
        arguments, contract.tensors, strict=True
    ):
        shape = tuple(cast(Sequence[int], tensor.shape))
        tensor_device = str(tensor.device)
        pointer = tensor.data_ptr()
        if shape != expected_shape:
            raise ValueError(f"CUDA Driver {name} shape differs")
        if str(tensor.dtype) != expected_dtype:
            raise ValueError(f"CUDA Driver {name} dtype differs")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"CUDA Driver {name} must be a contiguous CUDA tensor")
        if not isinstance(pointer, int) or isinstance(pointer, bool) or pointer <= 0:
            raise ValueError(f"CUDA Driver {name} pointer is invalid")
        if device is None:
            device = tensor_device
        elif tensor_device != device:
            raise ValueError("CUDA Driver tensor devices differ")
        pointers.append(pointer)
        observed[name] = {
            "shape": list(shape),
            "dtype": str(tensor.dtype),
            "device": tensor_device,
            "contiguous": True,
        }
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA Driver tensor pointers alias")
    return observed, tuple(pointers)


def _attribute(driver: object, name: str, function: object) -> int:
    enum_type = getattr(driver, "CUfunction_attribute", None)
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(f"CUDA Driver lacks function attribute {name}")
    (value,) = _driver_call(
        driver,
        "cuFuncGetAttribute",
        getattr(enum_type, name),
        function,
        outputs=1,
    )
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"CUDA function attribute {name} is not an integer") from error


def _function_resources(
    api: object,
    function: object,
    manifest: LaunchManifest,
) -> dict[str, int]:
    target = _cubin_target(manifest.target)
    resources = {name: _attribute(api, name, function) for name in _ATTRIBUTES}
    if resources["CU_FUNC_ATTRIBUTE_BINARY_VERSION"] != _binary_version(target):
        raise ValueError("CUDA Driver function binary version differs")
    if manifest.block_threads > resources["CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]:
        raise ValueError("CUDA Driver manifest block exceeds function maximum")
    if (
        resources["CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"]
        + manifest.dynamic_shared_memory_bytes
        > target.resource_limits.maximum_shared_memory_bytes
    ):
        raise ValueError("CUDA Driver static plus dynamic shared memory exceeds Target limit")
    cluster_names = (
        "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET",
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
    )
    if any(resources[name] != 0 for name in cluster_names):
        raise ValueError("CUDA Driver function requires a forbidden cluster launch")
    if manifest.dynamic_shared_memory_bytes > _DYNAMIC_SHARED_OPT_IN_THRESHOLD:
        enum_value = getattr(
            getattr(api, "CUfunction_attribute"),
            "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
        )
        _driver_call(
            api,
            "cuFuncSetAttribute",
            function,
            enum_value,
            manifest.dynamic_shared_memory_bytes,
            outputs=0,
        )
    return resources


def _load_cuda_driver() -> object:
    from cuda.bindings import driver

    return driver


@dataclass(frozen=True)
class CudaDeviceAdmission:
    """Broker-bound exact device observation required before persistent load."""

    device_name: str
    compute_capability: tuple[int, int]
    gpu_uuid: str
    broker_job_id: str
    mode: str

    def __post_init__(self) -> None:
        if (not isinstance(self.compute_capability, tuple) or len(self.compute_capability) != 2
            or any(type(value) is not int for value in self.compute_capability)):
            raise ValueError("CUDA device compute capability differs")
        target = _cubin_target(self.target)
        # Either of the cubin row's allocators, in that allocator's mode: the cluster
        # lease or the local broker (D6). Which of the two a timed assay may use is the
        # paired policy's rule, not this record's; another family's local job is not one
        # of this device's allocations at all.
        row = platform_for(target)
        prefix = self.broker_job_id.split("-", 1)[0] if isinstance(self.broker_job_id, str) else None
        if (
            self.device_name not in target.device_names
            or self.compute_capability != target.compute_capability
            or not isinstance(self.gpu_uuid, str) or not self.gpu_uuid
            or prefix not in {row.exclusive_job_prefix, row.local_job_prefix}
            or not valid_job_mode(self.broker_job_id, self.mode)
        ):
            raise ValueError("CUDA device admission differs")

    @property
    def target(self) -> str:
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}a"


class LoadedCudaCandidate:
    """One exact admitted module reused across preflight, cohorts and postflight."""

    def __init__(
        self,
        *,
        candidate: LaunchableCandidate,
        cubin: bytes,
        manifest: LaunchManifest,
        admission: CudaDeviceAdmission,
        modules: CudaModules,
        function: object,
        resources: dict[str, int],
    ) -> None:
        self.candidate = candidate
        self.cubin = cubin
        self.manifest = manifest
        self.admission = admission
        self._api = modules.api
        self._modules = modules
        self._function = function
        self.resources = resources
        self.launch_calls = 0

    @property
    def closed(self) -> bool:
        return self._modules.closed

    @classmethod
    def load(
        cls,
        candidate: LaunchableCandidate,
        cubin: bytes,
        manifest: LaunchManifest,
        admission: CudaDeviceAdmission,
        *,
        driver: object | None = None,
    ) -> "LoadedCudaCandidate":
        """Validate all immutable authority before retaining one loaded module."""

        check_launch_authority(candidate, cubin, "cubin", manifest)
        if candidate.target != admission.target:
            raise ValueError("persistent candidate launch authority differs")
        api = _load_cuda_driver() if driver is None else driver
        modules = CudaModules(api)
        try:
            module = modules.load(cubin)
            function = modules.function(module, manifest.kernel_name)
            resources = _function_resources(api, function, manifest)
            return cls(
                candidate=candidate,
                cubin=cubin,
                manifest=manifest,
                admission=admission,
                modules=modules,
                function=function,
                resources=resources,
            )
        except BaseException as primary:
            modules.close(primary=primary)
            raise

    def launch(
        self,
        arguments: Sequence[TensorLike],
        *,
        tensor_contract: TensorContract,
        stream: object,
    ) -> None:
        """Validate the shape-bound tensor contract before one Driver launch."""

        if self.closed or sha256(self.cubin).hexdigest() != self.candidate.artifact_roles["cubin"]:
            raise ValueError("persistent candidate is closed or its CUBIN changed")
        if tensor_contract.target != self.candidate.target:
            raise ValueError("persistent candidate tensor Target differs")
        if (getattr(self.manifest, 'pointer_alignments', {})
                and tensor_contract is not self.manifest
                and getattr(tensor_contract, 'canonical_sha256', None) != self.manifest.canonical_sha256):
            raise ValueError('aligned kernel tensor contract differs from its sealed manifest')
        observed, pointers = _tensor_contract(arguments, tensor_contract)
        from .launch_manifest import check_pointer_alignments
        check_pointer_alignments(dict(zip((row[0] for row in tensor_contract.tensors), pointers, strict=True)),
                                 getattr(self.manifest, 'pointer_alignments', {}))
        self._modules.check_open()
        devices = {
            str(cast(Mapping[str, object], value)["device"])
            for value in observed.values()
        }
        if devices not in ({"cuda"}, {"cuda:0"}):
            raise ValueError("persistent candidate tensors are outside the admitted logical GPU")
        argument_values = [ctypes.c_void_p(pointer) for pointer in pointers] + [
            ctypes.c_void_p(0)
            for _ in range(self.manifest.hidden_null_pointer_parameters)
        ]
        kernel_parameters = (ctypes.c_void_p * len(argument_values))(
            *[
                ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
                for value in argument_values
            ]
        )
        stream_type = getattr(self._api, "CUstream", None)
        launch_stream = (
            stream_type(stream) if isinstance(stream, int) and callable(stream_type) else stream
        )
        _driver_call(
            self._api,
            "cuLaunchKernel",
            self._function,
            *self.manifest.grid,
            *self.manifest.block,
            self.manifest.dynamic_shared_memory_bytes,
            launch_stream,
            kernel_parameters,
            0,
            outputs=0,
        )
        self.launch_calls += 1

    def close(self, *, synchronize: Callable[[], None]) -> None:
        """Synchronize then unload exactly once."""

        if self.closed:
            raise ValueError("persistent candidate is already closed")
        self._modules.close(synchronize=synchronize)


class TensorContract(Protocol):
    target: str
    tensors: tuple[tuple[str, tuple[int, ...], str], ...]


class LaunchManifest(Protocol):
    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    hidden_null_pointer_parameters: int
    block_threads: int
    canonical_sha256: str
