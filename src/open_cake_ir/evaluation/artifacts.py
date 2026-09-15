"""Backend build products; authoring-arm policy remains in the Lab."""

METAL_TARGETS = frozenset({"apple_gpu_family7", "apple_gpu_family8", "apple_gpu_family9"})
_CUBIN_TARGETS = frozenset({"sm_100a", "sm_103a"})
_BUILD_ROLES = {
    "metal": frozenset({"metal_binary_archive", "metal_build_report", "launch_manifest"}),
    "triton": frozenset({"compiler_expanded_source", "ptx", "cubin", "launch_manifest"}),
    "cuda": frozenset({"ptx", "cubin", "sass", "launch_manifest"}),
}
_CUDA_ALLOWED = frozenset({"authored_source", "lowered_source", "compiler_expanded_source",
    "ttir", "ttgir", "llir", "ptx", "cubin", "sass", "toolchain_resource_report", "launch_manifest"})


def executable_role(target: str) -> str:
    """Name the executable this Evaluation layer builds for one exact target.

    This is the whole layer's target-to-executable map, so it refuses in the class it
    owns. It used to fall through an Apple id set into `cuda_architecture`, which reported
    an AMD target, an unimplemented Apple family and an unimplemented NVIDIA target alike
    as `unsupported exact CUDA target` -- a refusal in a vendor's words that the caller
    never named.
    """
    if target in METAL_TARGETS:
        return "metal_binary_archive"
    if target in _CUBIN_TARGETS:
        return "cubin"
    raise ValueError(
        f"target {target!r} has no executable role in this Evaluation layer; it builds "
        f"{', '.join(sorted(METAL_TARGETS))} and {', '.join(sorted(_CUBIN_TARGETS))}"
    )


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
