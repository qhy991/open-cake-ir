"""The one table the Lab consults per lowering backend to bind a toolchain.

A Schedule declares `lowering.backend`; that declaration selects which fields a runtime
document's `toolchain` section carries, which comparison arm lowers through the same
toolchain, whether the backend is admitted as the one arm of an artifact-optimization
Study, and how the identity-bearing toolchain object is bound to a runtime document and
an admitted Executor. Before this table those facts were re-spelled as a four-way
`toolchain_kind` ladder in runtime_config.py, an `if backend == "metal"` in bindings.py,
a `{"metal", "triton"}` literal in tasks/compose.py and lab/preflight.py, and a
`route == "metal"` pair in tools/launch_task.py. Here each backend is one row, and a
backend no row names is refused by name.

The row binds the toolchain; it does not build candidates. Builders stay with the
modules that own them (`build.py`, `cute_build.py`, `metal_build.py`), and the
`native_cuda` row binds nothing here because the direct-CUDA arm owns its nvcc builder
under tasks/flash_kmeans (D12) and this layer must not import task code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from open_cake_ir.compiler.ir.vocabulary import LoweringBackend

from .pairing import NativeBackend, native_backend


def _bind_isolated_triton(config, executor, author_workspace):
    from .triton_build import IsolatedTritonCompiler
    toolchain = IsolatedTritonCompiler(**config)
    toolchain.check_executor(executor, author_workspace=author_workspace)
    return toolchain


def _bind_isolated_cute(config, executor, author_workspace):
    from .cute_build import IsolatedCuTeCompiler
    toolchain = IsolatedCuTeCompiler(**config)
    toolchain.check_executor(executor, author_workspace=author_workspace)
    return toolchain


def _bind_metal_archive_host(config, executor, author_workspace):
    # The archive helper is the Executor's own admitted binary; the runtime document
    # only says where builds are retained, and the author workspace is never mounted
    # because nothing here runs in a jail.
    from .metal_build import MetalArchiveHost
    return MetalArchiveHost.from_executor(executor)


@dataclass(frozen=True)
class Toolchain:
    backend: LoweringBackend
    # The `toolchain_kind` spelling runtime documents and their callers use. `nvcc` is
    # the historical spelling for the native_cuda backend and every released runtime
    # document under runtime/ carries it, so it is a row fact rather than a renamed one.
    runtime_kind: str
    # Exactly the fields the runtime document's `toolchain` section carries.
    runtime_fields: frozenset[str]
    # The comparison arm that lowers through this toolchain in a two-arm Study; None for
    # a backend no comparison arm authors in.
    arm: str | None
    # Whether the Study Contract admits this backend as the one arm of an
    # artifact_optimization_only Study (lab/preflight.py quotes this set back).
    single_environment: bool
    # Binds the identity-bearing toolchain -- the object whose canonical_sha256 the
    # Campaign Lock pins -- to a parsed runtime `toolchain` section and an admitted
    # Executor. None means another layer owns the binding (D12).
    bind_toolchain: Callable[[Mapping[str, object], object, object], object] | None

    @property
    def native(self) -> NativeBackend | None:
        """The same-backend native policy, when this toolchain's arm has one."""
        return native_backend(self.arm)

    def bind(self, config: Mapping[str, object], executor, *, author_workspace) -> object:
        if self.bind_toolchain is None:
            raise ValueError(
                f"the {self.backend.value!r} backend binds its toolchain in the arm that "
                "owns it, not through the Lab toolchain table"
            )
        return self.bind_toolchain(config, executor, author_workspace)


_ROWS = (
    Toolchain(
        backend=LoweringBackend.TRITON,
        runtime_kind="triton",
        runtime_fields=frozenset({"python", "bubblewrap", "runtime_roots", "build_environment",
                                  "triton_version", "timeout_seconds"}),
        arm="native_triton",
        single_environment=True,
        bind_toolchain=_bind_isolated_triton,
    ),
    Toolchain(
        backend=LoweringBackend.CUTLASS_CUTE_DSL,
        runtime_kind="cutlass_cute_dsl",
        runtime_fields=frozenset({"python", "bubblewrap", "runtime_roots", "cuobjdump",
                                  "cutlass_version", "timeout_seconds"}),
        arm="native_cute_dsl",
        # No Study Contract admits a single-environment CuTe route: the preflight in
        # lab/preflight.py quotes this flag, and no such Study has been frozen. Turning
        # it on is a Study Contract change, not a composition one.
        single_environment=False,
        bind_toolchain=_bind_isolated_cute,
    ),
    Toolchain(
        backend=LoweringBackend.METAL,
        runtime_kind="metal",
        runtime_fields=frozenset({"output_root"}),
        arm=None,
        single_environment=True,
        bind_toolchain=_bind_metal_archive_host,
    ),
    Toolchain(
        backend=LoweringBackend.NATIVE_CUDA,
        runtime_kind="nvcc",
        runtime_fields=frozenset({"nvcc", "cuobjdump"}),
        arm="direct_cuda",
        single_environment=False,
        bind_toolchain=None,
    ),
)

TOOLCHAINS: Mapping[LoweringBackend, Toolchain] = MappingProxyType(
    {row.backend: row for row in _ROWS}
)


def toolchain_for(backend: object) -> Toolchain:
    """The row for one backend, by LoweringBackend, its value, or its runtime spelling."""
    if isinstance(backend, LoweringBackend):
        return TOOLCHAINS[backend]
    for row in _ROWS:
        if backend in (row.backend.value, row.runtime_kind):
            return row
    raise ValueError(f"runtime toolchain kind {backend!r} is unsupported")


def toolchain_for_arm(comparison: str) -> Toolchain:
    """The row the named comparison arm lowers through."""
    for row in _ROWS:
        if row.arm is not None and row.arm == comparison:
            return row
    raise ValueError(f"no toolchain row serves the {comparison!r} comparison arm")


def single_environment_backends() -> frozenset[str]:
    return frozenset(row.backend.value for row in _ROWS if row.single_environment)


def single_environment_toolchain(backend: object) -> Toolchain:
    """The row for the one arm of an artifact-optimization Study, or a by-name refusal."""
    row = toolchain_for(backend)
    if not row.single_environment:
        raise ValueError(
            f"the {row.backend.value!r} backend is not admitted as a single-environment "
            f"route; the Study Contract admits {sorted(single_environment_backends())}"
        )
    return row
