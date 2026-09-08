"""Errors shared by Compiler admission and its public facade."""


class CompilerError(ValueError):
    """Raised when compiler authority or Schedule syntax cannot be interpreted."""


class LoweringRefusedError(CompilerError):
    """A backend's controlled refusal to derive source from a Schedule."""

    code = "LOWERING_UNDETERMINED"
