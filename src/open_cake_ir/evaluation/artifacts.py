"""Backend build products; authoring-arm policy remains in the Lab.

Every answer here is read off the execution platform row the Target's declared code
object selects. This module used to keep its own build-role sets keyed by backend name
and a CUDA role set reached by falling past the other two objects; now the row states
both, and a target no document declares is refused in this layer's words.
"""

from open_cake_ir.compiler.target import CodeObject

from .platforms import ExecutionPlatform, platform_for


def _platform(target: object) -> ExecutionPlatform:
    """The row for an exact target id, a Target the caller holds, or a code object."""
    try:
        return platform_for(target)
    except ValueError as error:
        raise ValueError(
            f"target {target!r} has no executable role in this Evaluation layer; each "
            "declared Target document names the object its toolchain produces"
        ) from error


def executable_role(target: object) -> str:
    """Name the executable this Evaluation layer builds for one exact target.

    The Target declares which object its toolchain produces, so this layer reads that
    fact instead of keeping its own partition of target ids: an eighth target is a
    document, not an edit here (F-2026-09-15-004). It still refuses in the class it owns
    -- a target no document declares gets this layer's words, never a vendor's.
    """
    return _platform(target).code_object.value


def builds_metal_archive(target: object) -> bool:
    """Whether this exact target declares a Metal binary archive as its object.

    The Metal host, build and observation paths own Metal's class and may name it. What
    they may not own is a list of which targets are Apple's, so they ask the declared
    fact; a target no document declares is simply not one of them.
    """
    try:
        return _platform(target).code_object is CodeObject.METAL_BINARY_ARCHIVE
    except ValueError:
        return False


def required_build_roles(target: object) -> frozenset[str]:
    """Complete build products for one target's object, excluding the arm-owned source role."""
    return _platform(target).build_roles


def allowed_artifact_roles(target: object) -> frozenset[str]:
    """The roles a candidate for this target may carry, keyed by its declared object.

    Every row admits the target's own executable: `LaunchableCandidate` requires the
    executable role to be present and every carried role to be allowed, so an AMDGCN
    candidate whose roles were read from the CUDA set would be refused for carrying the
    hsaco its own route emits.
    """
    return _platform(target).allowed_artifact_roles
