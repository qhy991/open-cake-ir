"""One pre-confirmation choice from completed search observations."""
from dataclasses import dataclass

from .checkpoints import TurnObservation


def nominate(observations, *, provider_token_limit):
    """Fixed search rule; equal measurements prefer the earliest completed turn."""
    eligible = [row for row in observations if row.search_qualified
                and (provider_token_limit is None or row.cumulative_provider_tokens <= provider_token_limit)]
    return min(eligible, key=lambda row: (row.search_latency_ms, row.turn)) if eligible else None


def nomination_document(observation, candidate):
    return {'source_turn': observation.turn if observation else None,
            'candidate_sha256': observation.candidate_sha256 if observation else None,
            'candidate_record_sha256': candidate.canonical_sha256 if candidate else None,
            'reason': 'lowest_qualified_search_latency' if observation else 'no_qualified_search_candidate'}


@dataclass(frozen=True)
class FinalConfirmation:
    source_turn: int
    candidate_sha256: str
    provider_tokens: int
    qualified: bool
    confirmed_latency_ms: float | None

    def __post_init__(self):
        # The same numeric/digest validation as search, with independent meaning.
        TurnObservation(self.source_turn, self.provider_tokens, self.candidate_sha256,
                        self.qualified, self.confirmed_latency_ms)
