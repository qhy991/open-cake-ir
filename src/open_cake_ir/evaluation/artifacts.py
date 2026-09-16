"""Backend build products; authoring-arm policy remains in the Lab."""

from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError, declared_target

_BUILD_ROLES = {
    "metal": frozenset({"metal_binary_archive", "metal_build_report", "launch_manifest"}),
    "triton": frozenset({"compiler_expanded_source", "ptx", "cubin", "launch_manifest"}),
    "cuda": frozenset({"ptx", "cubin", "sass", "launch_manifest"}),
    # Triton on AMDGCN emits its own roles: assembly and an ELF HSACO where the CUDA
    # route has PTX and a CUBIN. evaluation/triton_hip.ARTIFACT_ROLES is the producer.
    "triton_amdgcn": frozenset({"compiler_expanded_source", "amdgcn", "hsaco",
                                "launch_manifest"}),
}
_CUDA_ALLOWED = frozenset({"authored_source", "lowered_source", "compiler_expanded_source",
    "ttir", "ttgir", "llir", "ptx", "cubin", "sass", "toolchain_resource_report", "launch_manifest"})


def _declared(target: object) -> Target:
    """Resolve an exact target id, or accept a Target the caller already holds."""
    return target if isinstance(target, Target) else declared_target(target)


def executable_role(target: object) -> str:
    """Name the executable this Evaluation layer builds for one exact target.

    The Target declares which object its toolchain produces, so this layer reads that
    fact instead of keeping its own partition of target ids: an eighth target is a
    document, not an edit here (F-2026-09-15-004). It still refuses in the class it owns
    -- a target no document declares gets this layer's words, never a vendor's.
    """
    try:
        return _declared(target).code_object.value
    except (TargetParseError, ValueError) as error:
        raise ValueError(
            f"target {target!r} has no executable role in this Evaluation layer; each "
            "declared Target document names the object its toolchain produces"
        ) from error


def builds_metal_archive(target: object) -> bool:
    """Whether this exact target declares a Metal binary archive as its object.

    The Metal host, build and observation paths own Metal's class and may name it. What
    they may not own is a list of which targets are Apple's, so they ask the declared
    fact; a target no document declares is simply not one of them.
    """
    try:
        return _declared(target).code_object is CodeObject.METAL_BINARY_ARCHIVE
    except (TargetParseError, ValueError):
        return False


def required_build_roles(backend: str) -> frozenset[str]:
    """Complete build products, excluding the arm-owned source provenance role."""
    try:
        return _BUILD_ROLES[backend]
    except KeyError as error:
        raise ValueError(f"unsupported build backend: {backend}") from error


def allowed_artifact_roles(target: object) -> frozenset[str]:
    """The roles a candidate for this target may carry, keyed by its declared object.

    Every branch must admit the target's own executable: `LaunchableCandidate` requires
    the executable role to be present and every carried role to be allowed, so an AMDGCN
    candidate whose roles were read from the CUDA set would be refused for carrying the
    hsaco its own route emits.
    """
    role = executable_role(target)
    if role == "metal_binary_archive":
        return _BUILD_ROLES["metal"] | {"authored_source", "lowered_source"}
    if role == "hsaco":
        return _BUILD_ROLES["triton_amdgcn"] | {"authored_source", "lowered_source",
                                                "ttir", "ttgir", "llir",
                                                "toolchain_resource_report"}
    return _CUDA_ALLOWED
