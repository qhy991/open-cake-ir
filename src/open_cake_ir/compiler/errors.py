"""Errors shared by Compiler admission and its public facade."""


class CompilerError(ValueError):
    """Raised when compiler authority or Schedule syntax cannot be interpreted."""
