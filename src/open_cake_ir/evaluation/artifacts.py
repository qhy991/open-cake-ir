"""Backend build products; authoring-arm policy remains in the Lab."""
from open_cake_ir.compiler.target import cuda_architecture

METAL_TARGETS = frozenset({"apple_gpu_family7", "apple_gpu_family8", "apple_gpu_family9"})
_BUILD_ROLES = {
    "metal": frozenset({"metal_binary_archive", "metal_build_report", "launch_manifest"}),
    "triton": frozenset({"compiler_expanded_source", "ptx", "cubin", "launch_manifest"}),
    "cuda": frozenset({"ptx", "cubin", "sass", "launch_manifest"}),
}
_CUDA_ALLOWED = frozenset({"authored_source", "lowered_source", "compiler_expanded_source",
    "ttir", "ttgir", "llir", "ptx", "cubin", "sass", "toolchain_resource_report", "launch_manifest"})


def executable_role(target: str) -> str:
    if target in METAL_TARGETS:
        return "metal_binary_archive"
    cuda_architecture(target)
    return "cubin"


def required_build_roles(backend: str) -> frozenset[str]:
    """Complete build products, excluding the arm-owned source provenance role."""
    try:
        return _BUILD_ROLES[backend]
    except KeyError as error:
        raise ValueError(f"unsupported build backend: {backend}") from error


def allowed_artifact_roles(target: str) -> frozenset[str]:
    if executable_role(target) == "metal_binary_archive":
        return _BUILD_ROLES["metal"] | {"authored_source", "lowered_source"}
    return _CUDA_ALLOWED
