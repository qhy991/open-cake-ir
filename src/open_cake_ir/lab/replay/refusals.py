"""Located replay refusals: where the ledger stopped reconstructing, and what differed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Mapping, NoReturn


_ABSENT = object()


def _plain(value: object) -> object:
    """A JSON-shaped copy of a compared value; sets sort so two reports compare equal."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


class ReplayRefusal(ValueError):
    """One located reason a retained ledger did not reconstruct from its raw sources.

    `location` is a path into the replayed Run: an event named by kind and its
    identifying key (`provider_turn_completed[turn=2].payload.thread_id`), a row of
    a retained document (`candidate_set_filtered[turn=1].order[3].cost`) or a lock
    field. `observed` and `expected` are carried only when the check compared two
    small values; a seal or a byte comparison names the object and nothing more.
    A ValueError so the reporting boundary that already catches the live path's
    refusals catches this one too.
    """

    def __init__(self, location: str, message: str, *, observed: object = _ABSENT,
                 expected: object = _ABSENT) -> None:
        self.location = location
        self.message = message
        self.observed = observed
        self.expected = expected
        super().__init__(str(self))

    def __str__(self) -> str:
        sides = [
            f"{name} {_plain(value)!r}"
            for name, value in (("expected", self.expected), ("observed", self.observed))
            if value is not _ABSENT
        ]
        text = f"{self.location}: {self.message}"
        return text + (f" ({', '.join(sides)})" if sides else "")

    @property
    def document(self) -> Mapping[str, object]:
        document: dict[str, object] = {"location": self.location, "message": self.message}
        if self.observed is not _ABSENT:
            document["observed"] = _plain(self.observed)
        if self.expected is not _ABSENT:
            document["expected"] = _plain(self.expected)
        return document


def refuse(location: str, message: str, *, observed: object = _ABSENT,
           expected: object = _ABSENT) -> NoReturn:
    """Stop this replay at `location`; `replay_matched_run` collects the refusal."""
    raise ReplayRefusal(location, message, observed=observed, expected=expected)


def event_location(kind: str, **keys: object) -> str:
    """`kind[key=value,...]`: an event named by its kind and identifying fields."""
    inside = ",".join(f"{key}={value}" for key, value in keys.items())
    return f"{kind}[{inside}]" if inside else kind


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of one Run's semantic replay: passed, or the refusals that stopped it.

    Truthy exactly when the Run replayed, so a caller that reads the result as the
    boolean it used to be keeps its meaning; `refusals` is the located account.
    """

    run_id: str
    refusals: tuple[ReplayRefusal, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.refusals

    def __bool__(self) -> bool:
        return self.passed

    @property
    def document(self) -> Mapping[str, object]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "refusals": [refusal.document for refusal in self.refusals],
        }
