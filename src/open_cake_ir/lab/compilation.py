"""Run-owned compilation permits and observations at the native call boundary."""
from dataclasses import dataclass

from .faults import CompilationBudgetExceeded


@dataclass(frozen=True)
class CompilationRecorder:
    ralph: object
    ledger: object
    turn: int
    candidate_sha256: str

    def __call__(self, request, *, compiler, variant, operation):
        if request.candidate_sha256 != self.candidate_sha256:
            raise ValueError('compilation candidate differs from the Run build')
        identity = {'turn':self.turn,'candidate_sha256':self.candidate_sha256,
                    'compiler':compiler,'target':request.target,
                    'entry_point':request.toolchain_requirements.get('kernel_entry_point',request.entry_point),'variant':variant}
        try:
            number = self.ralph.record_compilation()
        except CompilationBudgetExceeded:
            self.ledger.append('compilation_refused', {**identity,'reason':'compilation_budget'})
            raise
        self.ledger.append('compilation_started', {**identity,'number':number,
            'elapsed_wall_seconds':round(self.ralph.elapsed_wall_seconds,6)})
        outcome = 'raised'
        try:
            result = operation()
            outcome = 'returned'
            return result
        finally:
            self.ledger.append('compilation_completed', {'turn':self.turn,'candidate_sha256':self.candidate_sha256,
                'number':number,'outcome':outcome,'elapsed_wall_seconds':round(self.ralph.elapsed_wall_seconds,6)})
