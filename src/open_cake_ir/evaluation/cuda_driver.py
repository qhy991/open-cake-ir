"""Host-owned exact-CUBIN CUDA Driver launch with no fallback."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Protocol, Sequence, cast

from open_cake_ir.compiler.target import cuda_architecture, cuda_target

from .core import LaunchableCandidate
from .cuda_manifest import MAX_DYNAMIC_SHARED_MEMORY_BYTES

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


class CudaLifecycleError(RuntimeError):
    """Preserve both the primary CUDA failure and teardown failure."""

    def __init__(self, primary: BaseException, teardown: BaseException) -> None:
        super().__init__(
            f"CUDA lifecycle failed in {type(primary).__name__} and teardown {type(teardown).__name__}"
        )
        self.primary = primary
        self.teardown = teardown


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
    resources = {name: _attribute(api, name, function) for name in _ATTRIBUTES}
    if resources["CU_FUNC_ATTRIBUTE_BINARY_VERSION"] != cuda_architecture(manifest.target):
        raise ValueError("CUDA Driver function binary version differs")
    if manifest.block_threads > resources["CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]:
        raise ValueError("CUDA Driver manifest block exceeds function maximum")
    if (
        resources["CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"]
        + manifest.dynamic_shared_memory_bytes
        > cuda_target(manifest.target).resource_limits.maximum_shared_memory_bytes
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
class CudaDriverLaunchReceipt:
    """Host-owned launch lifecycle observation."""

    cubin_sha256: str
    manifest_sha256: str
    kernel_calls: int
    fallback_calls: int
    synchronized: bool
    module_unloaded: bool
    same_cubin: bool
    resources: dict[str, int]
    tensor_contract: dict[str, object]


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
        target = cuda_target(self.target)
        if (
            self.device_name not in target.device_names
            or self.compute_capability != target.compute_capability
            or not isinstance(self.gpu_uuid, str) or not self.gpu_uuid
            or not isinstance(self.broker_job_id, str) or not self.broker_job_id.startswith("gpuq-")
            or self.mode != "exclusive"
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
        api: object,
        module: object,
        function: object,
        resources: dict[str, int],
        context: object,
    ) -> None:
        self.candidate = candidate
        self.cubin = cubin
        self.manifest = manifest
        self.admission = admission
        self._api = api
        self._module = module
        self._function = function
        self._context = _handle_identity(context)
        self.resources = resources
        self.launch_calls = 0
        self.closed = False

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

        cubin_sha256 = sha256(cubin).hexdigest()
        if (
            not cubin.startswith(b"\x7fELF")
            or candidate.target != manifest.target
            or candidate.target != admission.target
            or candidate.entry_point != manifest.kernel_name
            or candidate.launch_spec_sha256 != manifest.canonical_sha256
            or candidate.artifact_roles.get("cubin") != cubin_sha256
        ):
            raise ValueError("persistent candidate launch authority differs")
        api = _load_cuda_driver() if driver is None else driver
        (context,) = _driver_call(api, "cuCtxGetCurrent", outputs=1)
        if _is_null(context):
            raise RuntimeError("CUDA Driver requires a current CUDA context")
        module: object | None = None
        try:
            (module,) = _driver_call(api, "cuModuleLoadData", cubin, outputs=1)
            if _is_null(module):
                raise RuntimeError("CUDA Driver loaded a null module")
            (function,) = _driver_call(
                api,
                "cuModuleGetFunction",
                module,
                manifest.kernel_name.encode("ascii"),
                outputs=1,
            )
            if _is_null(function):
                raise RuntimeError("CUDA Driver resolved a null function")
            resources = _function_resources(api, function, manifest)
            return cls(
                candidate=candidate,
                cubin=cubin,
                manifest=manifest,
                admission=admission,
                api=api,
                module=module,
                function=function,
                resources=resources,
                context=context,
            )
        except BaseException as primary:
            if module is not None:
                try:
                    _driver_call(api, "cuModuleUnload", module, outputs=0)
                except BaseException as teardown:
                    raise CudaLifecycleError(primary, teardown) from primary
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
        observed, pointers = _tensor_contract(arguments, tensor_contract)
        (current_context,) = _driver_call(self._api, "cuCtxGetCurrent", outputs=1)
        if _is_null(current_context) or _handle_identity(current_context) != self._context:
            raise ValueError("persistent candidate CUDA context changed")
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
        (current_context,) = _driver_call(self._api, "cuCtxGetCurrent", outputs=1)
        if _is_null(current_context) or _handle_identity(current_context) != self._context:
            raise ValueError("persistent candidate CUDA context changed before teardown")
        primary: BaseException | None = None
        try:
            synchronize()
        except BaseException as error:
            primary = error
        try:
            _driver_call(self._api, "cuModuleUnload", self._module, outputs=0)
            self.closed = True
        except BaseException as teardown:
            if primary is not None:
                raise CudaLifecycleError(primary, teardown) from primary
            raise
        if primary is not None:
            raise primary


def launch_cubin_once(
    cubin: bytes,
    expected_cubin_sha256: str,
    manifest: LaunchManifest,
    arguments: Sequence[TensorLike],
    *,
    tensor_contract: TensorContract,
    stream: object,
    synchronize: Callable[[], None],
    driver: object | None = None,
) -> CudaDriverLaunchReceipt:
    """Load, admit, launch and unload one exact CUBIN synchronously."""

    if not isinstance(cubin, bytes) or not cubin.startswith(b"\x7fELF"):
        raise ValueError("CUDA Driver CUBIN must be exact ELF bytes")
    before = sha256(cubin).hexdigest()
    if expected_cubin_sha256 != before:
        raise ValueError("CUDA Driver CUBIN SHA256 differs before load")
    if manifest.target != tensor_contract.target:
        raise ValueError("CUDA Driver manifest and tensor Target differ")
    observed_tensor_contract, pointers = _tensor_contract(arguments, tensor_contract)
    api = _load_cuda_driver() if driver is None else driver
    (context,) = _driver_call(api, "cuCtxGetCurrent", outputs=1)
    if _is_null(context):
        raise RuntimeError("CUDA Driver requires a current CUDA context")

    module: object | None = None
    synchronized = False
    unloaded = False
    launch_calls = 0
    resources: dict[str, int] = {}
    primary_error: BaseException | None = None
    try:
        (module,) = _driver_call(api, "cuModuleLoadData", cubin, outputs=1)
        if _is_null(module):
            raise RuntimeError("CUDA Driver loaded a null module")
        if sha256(cubin).hexdigest() != before:
            raise ValueError("CUDA Driver CUBIN SHA256 changed after load")
        (function,) = _driver_call(
            api,
            "cuModuleGetFunction",
            module,
            manifest.kernel_name.encode("ascii"),
            outputs=1,
        )
        if _is_null(function):
            raise RuntimeError("CUDA Driver resolved a null function")
        resources = {name: _attribute(api, name, function) for name in _ATTRIBUTES}
        if resources["CU_FUNC_ATTRIBUTE_BINARY_VERSION"] != 100:
            raise ValueError("CUDA Driver function binary version differs")
        if manifest.block_threads > resources["CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"]:
            raise ValueError("CUDA Driver manifest block exceeds function maximum")
        if (
            resources["CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"]
            + manifest.dynamic_shared_memory_bytes
            > MAX_DYNAMIC_SHARED_MEMORY_BYTES
        ):
            raise ValueError("CUDA Driver static plus dynamic shared memory exceeds B200 limit")
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

        argument_values = [ctypes.c_void_p(pointer) for pointer in pointers] + [
            ctypes.c_void_p(0)
            for _ in range(manifest.hidden_null_pointer_parameters)
        ]
        kernel_parameters = (ctypes.c_void_p * len(argument_values))(
            *[
                ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
                for value in argument_values
            ]
        )
        stream_type = getattr(api, "CUstream", None)
        launch_stream = stream_type(stream) if isinstance(stream, int) and callable(stream_type) else stream
        _driver_call(
            api,
            "cuLaunchKernel",
            function,
            *manifest.grid,
            *manifest.block,
            manifest.dynamic_shared_memory_bytes,
            launch_stream,
            kernel_parameters,
            0,
            outputs=0,
        )
        launch_calls += 1
        synchronize()
        synchronized = True
        if sha256(cubin).hexdigest() != before:
            raise ValueError("CUDA Driver CUBIN SHA256 changed after synchronization")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if module is not None:
            try:
                _driver_call(api, "cuModuleUnload", module, outputs=0)
                unloaded = True
            except BaseException as teardown:
                if primary_error is not None:
                    raise CudaLifecycleError(primary_error, teardown) from primary_error
                raise
    if not unloaded:
        raise RuntimeError("CUDA Driver module was not unloaded")
    return CudaDriverLaunchReceipt(
        cubin_sha256=before,
        manifest_sha256=manifest.canonical_sha256,
        kernel_calls=launch_calls,
        fallback_calls=0,
        synchronized=synchronized,
        module_unloaded=unloaded,
        same_cubin=True,
        resources=resources,
        tensor_contract=observed_tensor_contract,
    )


def launch_candidate_once(
    candidate: LaunchableCandidate,
    cubin: bytes,
    manifest: LaunchManifest,
    arguments: Sequence[TensorLike],
    *,
    tensor_contract: TensorContract,
    stream: object,
    synchronize: Callable[[], None],
    driver: object | None = None,
) -> CudaDriverLaunchReceipt:
    """Bind Candidate, manifest, CUBIN and exact shape before any module load."""

    cubin_sha256 = sha256(cubin).hexdigest()
    if (
        candidate.target != manifest.target
        or candidate.entry_point != manifest.kernel_name
        or candidate.launch_spec_sha256 != manifest.canonical_sha256
        or candidate.artifact_roles.get("cubin") != cubin_sha256
    ):
        raise ValueError("shape-bound candidate launch authority differs")
    return launch_cubin_once(
        cubin,
        cubin_sha256,
        manifest,
        arguments,
        tensor_contract=tensor_contract,
        stream=stream,
        synchronize=synchronize,
        driver=driver,
    )


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
