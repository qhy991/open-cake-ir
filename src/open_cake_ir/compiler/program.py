"""Lowered static Programs, independent of storage allocation and execution."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
from .ir.program import Program
if TYPE_CHECKING:
    from .core import Lowering

@dataclass(frozen=True)
class LoweredProgram:
    program: Program
    compiler_revision_id: str
    lowerings: tuple[Lowering, ...]

    def validate_binding(self) -> None:
        """Check the executable handoff, before loading any stage's code.

        Existing Schedule identities bind code to ordered stages. This single
        boundary check prevents substituting a different lowering or revision.
        """
        Program.from_dict(self.program.document)
        if len(self.lowerings) != len(self.program.stages):
            raise ValueError('lowered Program stage count differs')
        for stage, lowering in zip(self.program.stages, self.lowerings, strict=True):
            schedule = stage.schedule
            if (lowering.compiler_revision_id != self.compiler_revision_id
                or lowering.schedule_id != schedule.schedule_id
                or lowering.schedule_sha256 != sha256(stage.schedule_bytes).hexdigest()
                or lowering.target != self.program.target
                or lowering.route != schedule.lowering or not lowering.generated):
                raise ValueError(f'lowered Program stage {stage.name!r} code binding differs')
