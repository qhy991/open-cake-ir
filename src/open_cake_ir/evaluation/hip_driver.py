"""Host-owned exact-HSACO HIP module launch with no fallback.

The AMDGCN peer of `cuda_driver.py`, and deliberately smaller than it. That module owns
CUDA's cluster attributes, its dynamic-shared opt-in threshold and its binary-version
check; none of those is a fact about a code object that carries the `amdgcn-amd-amdhsa--`
triple. What is shared is the part that matters: the same immutable authority is checked
before a module is retained, and there is no fallback path when a check fails.

Two deliberate absences:

* No soname is named here. `admit_exact_hip` has already imported a ROCm PyTorch, so the
  HIP runtime is loaded into this process -- a Hygon DTK host loads `libgalaxyhip.so.5`
  and a ROCm host `libamdhip64.so.5`, and resolving through the global symbol table asks
  for neither by name. A list of sonames here would be one more per-vendor enumeration in
  shared code (F-2026-09-15-004).
* No `hipFuncGetAttribute`. Register and scratch counts for an AMDGCN kernel are already
  carried by its own `.amdgpu_metadata`, which `triton_hip.amdgcn_resource_record` reads
  from the assembly this candidate seals beside its HSACO. Asking the driver would make
  the same fact have two owners, and the enumeration values behind that call differ
  between HIP versions in a way nothing here has measured.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .core import LaunchableCandidate
from .loaders import LifecycleError, check_launch_authority

# hipError_t 0. Every other value is reported with its number and, when the runtime can
# name it, its string -- never mapped onto a CUDA error here.
_HIP_SUCCESS = 0
_REQUIRED_SYMBOLS = (
    "hipModuleLoadData",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
    "hipModuleUnload",
    "hipDeviceSynchronize",
)


# The shared lifecycle exception under the name every retained test still raises it by.
HipLifecycleError = LifecycleError


def _exports_every_symbol(library: object) -> bool:
    return all(hasattr(library, name) for name in _REQUIRED_SYMBOLS)


def _mapped_shared_objects() -> list[str]:
    """Every shared object this process has mapped, in first-mapped order.

    The process's own memory map is the vendor-neutral way to find the runtime torch
    loaded: a Hygon DTK host maps `libgalaxyhip.so.5` and a ROCm host `libamdhip64.so.5`,
    and neither name appears here.
    """
    seen: list[str] = []
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return seen
    for line in lines:
        path = line.split(" ", 5)[-1].strip() if line.count(" ") >= 5 else ""
        if path.startswith("/") and (path.endswith(".so") or ".so." in path):
            if path not in seen:
                seen.append(path)
    return seen


def load_hip_runtime(library: object | None = None) -> object:
    """Resolve the HIP runtime already loaded by the admitted ROCm PyTorch.

    Nothing is opened by name. `admit_exact_hip` imports torch and refuses a build whose
    `torch.version.hip` is absent, so by the time anything here runs the process has the
    runtime this host installs mapped -- whichever vendor's fork that is.

    Finding it is not as simple as the global symbol table, which is what this tried
    first and what the DCU refused: torch loads its extensions with RTLD_LOCAL, so the
    HIP entry points are mapped but not globally visible, and `CDLL(None)` reports every
    one of them unresolved. The process's own memory map names the file, and `dlopen` of
    an already-mapped library is a reference-count bump rather than a second load.
    """
    if library is not None:
        api = library
    else:
        api = ctypes.CDLL(None)
        if not _exports_every_symbol(api):
            for path in _mapped_shared_objects():
                try:
                    candidate = ctypes.CDLL(path)
                except OSError:
                    continue
                if _exports_every_symbol(candidate):
                    api = candidate
                    break
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(api, name)]
    if missing:
        raise RuntimeError(
            "the HIP runtime is not loaded in this process; "
            + ", ".join(missing)
            + " are unresolved in the global symbol table and in every shared object this "
            "process has mapped. An admitted ROCm PyTorch loads it before this point."
        )
    return api


def _hip_call(api: object, name: str, *arguments: object) -> None:
    function = getattr(api, name, None)
    if function is None:
        raise RuntimeError(f"the HIP runtime lacks {name}")
    status = function(*arguments)
    if status == _HIP_SUCCESS:
        return
    detail = ""
    describe = getattr(api, "hipGetErrorString", None)
    if describe is not None:
        describe.restype = ctypes.c_char_p
        text = describe(ctypes.c_int(status))
        if isinstance(text, bytes) and text:
            detail = f" ({text.decode('utf-8', 'replace')})"
    raise RuntimeError(f"HIP {name} failed with status {status}{detail}")


class HipModules:
    """Every module this loader opened, unloaded exactly once and in reverse order."""

    def __init__(self, api: object) -> None:
        self._api = api
        self._modules: list[ctypes.c_void_p] = []
        self.closed = False

    @property
    def api(self) -> object:
        return self._api

    def load(self, hsaco: bytes) -> ctypes.c_void_p:
        if self.closed:
            raise RuntimeError("HIP modules are closed")
        module = ctypes.c_void_p()
        _hip_call(self._api, "hipModuleLoadData", ctypes.byref(module), hsaco)
        self._modules.append(module)
        return module

    def function(self, module: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        if self.closed:
            raise RuntimeError("HIP modules are closed")
        function = ctypes.c_void_p()
        _hip_call(self._api, "hipModuleGetFunction", ctypes.byref(function), module,
                  name.encode("ascii"))
        return function

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        failures: list[BaseException] = []
        for module in reversed(self._modules):
            try:
                _hip_call(self._api, "hipModuleUnload", module)
            except BaseException as error:  # every teardown failure is reported
                failures.append(error)
        self._modules.clear()
        if failures:
            raise LifecycleError(failures[0], *failures[1:]) if len(failures) > 1 else failures[0]


class LoadedHipModuleCandidate:
    """One exact admitted HSACO module reused across preflight and postflight."""

    def __init__(self, *, candidate: LaunchableCandidate, hsaco: bytes, manifest: object,
                 modules: HipModules, function: ctypes.c_void_p,
                 resources: Mapping[str, object]) -> None:
        self.candidate = candidate
        self.hsaco = hsaco
        self.manifest = manifest
        self._api = modules.api
        self._modules = modules
        self._function = function
        self.resources = dict(resources)
        self.launch_calls = 0
        # The kernel-parameter array holds addresses of these, so they outlive the call
        # only if something keeps a reference. Declared here rather than appearing on the
        # instance mid-launch.
        self._arguments: list[ctypes.c_void_p] = []

    @property
    def closed(self) -> bool:
        return self._modules.closed

    @classmethod
    def load(cls, candidate: LaunchableCandidate, hsaco: bytes, manifest: object,
             device_arch: str, *, api: object | None = None) -> "LoadedHipModuleCandidate":
        """Validate all immutable authority before retaining one loaded module."""
        from .triton_hip import amdgcn_resource_record, device_arch_matches

        check_launch_authority(candidate, hsaco, "hsaco", manifest)
        if not device_arch_matches(device_arch, candidate.target):
            raise ValueError(
                f"admitted device reports {device_arch!r}, which is not the "
                f"{candidate.target!r} this candidate was built for"
            )
        assembly = candidate.artifact_payloads.get("amdgcn")
        if not assembly:
            raise ValueError(
                "an AMDGCN candidate seals its assembly beside its HSACO; its resource "
                "record is read from that, not from the driver"
            )
        resources = amdgcn_resource_record(assembly)
        runtime = load_hip_runtime() if api is None else api
        modules = HipModules(runtime)
        try:
            module = modules.load(hsaco)
            function = modules.function(module, manifest.kernel_name)
        except BaseException as primary:
            try:
                modules.close()
            except BaseException as teardown:
                raise LifecycleError(primary, teardown) from primary
            raise
        return cls(candidate=candidate, hsaco=hsaco, manifest=manifest, modules=modules,
                   function=function, resources=resources)

    def launch(self, arguments: Sequence[object], *, tensor_contract: object,
               stream: int = 0) -> None:
        """Launch the admitted kernel over this manifest's exact tensor ABI."""
        if self.closed:
            raise RuntimeError("HIP modules are closed")
        if tensor_contract is not self.manifest:
            raise ValueError("launch tensor contract differs from the admitted manifest")
        abi = self.manifest.tensor_abi
        if len(arguments) != len(abi):
            raise ValueError("launch argument count differs from the sealed tensor ABI")
        pointers: list[ctypes.c_void_p] = []
        for (name, shape, _dtype, _mode), argument in zip(abi, arguments, strict=True):
            if not argument.is_contiguous():
                raise ValueError(f"tensor {name!r} must be contiguous at launch")
            if tuple(argument.shape) != tuple(shape):
                raise ValueError(f"tensor {name!r} shape differs from the sealed ABI")
            pointers.append(ctypes.c_void_p(argument.data_ptr()))
        from .launch_manifest import check_pointer_alignments
        check_pointer_alignments({row[0]: pointer.value for row, pointer in zip(abi, pointers, strict=True)},
                                 getattr(self.manifest, 'pointer_alignments', {}))
        # Triton appends one hidden null pointer per scratch buffer its options declare;
        # the manifest carries how many, sealed from the route rather than assumed.
        pointers.extend(
            ctypes.c_void_p(0)
            for _ in range(self.manifest.hidden_null_pointer_parameters)
        )
        slots = (ctypes.c_void_p * len(pointers))(
            *(ctypes.cast(ctypes.pointer(pointer), ctypes.c_void_p) for pointer in pointers)
        )
        self._arguments = pointers  # the array holds their addresses, not their values
        grid, block = self.manifest.grid, self.manifest.block
        _hip_call(
            self._api, "hipModuleLaunchKernel", self._function,
            ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]), ctypes.c_uint(block[1]), ctypes.c_uint(block[2]),
            ctypes.c_uint(self.manifest.dynamic_shared_memory_bytes),
            ctypes.c_void_p(stream), slots, None,
        )
        self.launch_calls += 1

    def synchronize(self) -> None:
        _hip_call(self._api, "hipDeviceSynchronize")

    def close(self, *, synchronize: Callable[[], None] | None = None,
              primary: BaseException | None = None) -> None:
        """Drain the device, then unload, keeping every failure observed on the way.

        The signature is the CUDA loader's because the one caller -- the shared
        tensor-tile lifecycle -- closes both through it. Draining first is the point: an
        unload while a launch is still in flight is what `synchronize` exists to prevent,
        and the caller passes `torch.cuda.synchronize`, which a ROCm build maps onto HIP.
        """
        failures: list[BaseException] = [] if primary is None else [primary]
        if synchronize is not None and not self.closed:
            try:
                synchronize()
            except BaseException as error:
                failures.append(error)
        try:
            self._modules.close()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise LifecycleError(failures[0], *failures[1:])
