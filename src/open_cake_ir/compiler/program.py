"""Lowered Programs, independent of storage allocation and execution."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Mapping
from .ir.program import Program
from .ir.ranked_mailbox import RankedMailboxAnalysis, RankedMailboxEffects
from .ir.schedule import Schedule
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


@dataclass(frozen=True)
class LoweredWorkerProgram:
    """One generated kernel for a complete worker Program.

    The owning backend supplies its source and internal-state requirements;
    Evaluation allocates the declared state and mathematical tensors.
    """

    program: Program
    compiler_revision_id: str
    source: str
    toolchain_requirements: Mapping[str, object]

    def validate_binding(self) -> None:
        replayed = Program.from_dict(self.program.document)
        if (replayed != self.program or self.program.execution is None
                or self.toolchain_requirements.get('target') != self.program.target
                or self.toolchain_requirements.get('entry_point')
                != self.program.execution.lowering.entry_point):
            raise ValueError('lowered worker Program binding differs')


@dataclass(frozen=True)
class LoweredRankedMailbox:
    """One source binding complete local/combine math to ranked effects."""

    local_program: Program
    combine_schedule: Schedule
    effects: RankedMailboxEffects
    analysis: RankedMailboxAnalysis
    compiler_revision_id: str
    source: str
    source_map: Mapping[str, tuple[int, int]]
    toolchain_requirements: Mapping[str, object]

    def validate_binding(self) -> None:
        if (Program.from_dict(self.local_program.document) != self.local_program
                or RankedMailboxEffects.from_dict(self.effects.document) != self.effects
                or self.effects.analyze(self.local_program, self.combine_schedule)
                != self.analysis):
            raise ValueError('lowered ranked mailbox typed binding differs')
        requirements = self.toolchain_requirements
        if (requirements.get('target') != self.local_program.target
                or requirements.get('entry_point') != self.effects.lowering.entry_point
                or requirements.get('world_size') != self.analysis.world_size
                or requirements.get('tokens_per_rank') != self.analysis.items_per_rank
                or requirements.get('routes_per_token') != self.analysis.routes_per_item
                or requirements.get('payload_capacity')
                != self.analysis.remote_payload_capacity_per_rank
                or requirements.get('task_capacity')
                != self.analysis.compute_task_capacity_per_rank
                or requirements.get('return_slots')
                != self.analysis.return_slots_per_rank):
            raise ValueError('lowered ranked mailbox route or capacity differs')
        expected_math = {f'{stage.name}.{op.op_id}'
                         for stage in self.local_program.stages
                         for op in stage.schedule.operations}
        expected_math.update(f'combine.{op.op_id}' for op in self.combine_schedule.operations)
        if not expected_math <= set(self.source_map):
            raise ValueError('lowered ranked mailbox source map omits math')
        if any(f'// CAKE_OP: {name}' not in self.source for name in expected_math):
            raise ValueError('lowered ranked mailbox source omits mapped math')
