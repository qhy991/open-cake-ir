"""Provider records and strict candidate-envelope byte admission."""

from __future__ import annotations

import json, math, os, re, stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ._documents import _canonical_json_bytes


def required_live_provider_qualification_scope(claim_scope: str) -> str:
    """Return the one live provider capability authorized by a matched Claim Scope."""

    if claim_scope == "artifact_optimization_only":
        return "live_two_turn_tool_rich_provider"
    if claim_scope in {"scientific_matched_search", "system_qualification_only"}:
        return "live_two_turn_current_provider"
    raise ValueError("matched Claim Scope has no live provider qualification")


def _read_candidate_nofollow(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_CANDIDATE_BYTES
        ):
            raise ValueError("provider candidate owner, type, or size differs")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("provider candidate read ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(descriptor).st_size != metadata.st_size:
            raise ValueError("provider candidate changed while sealing")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("provider feedback object keys must be strings")
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("provider feedback contains a non-JSON value")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("provider candidate-set envelope contains duplicate JSON keys")
        document[key] = value
    return document


def _finite_json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("provider candidate-set envelope contains a non-finite JSON number")
    return value


def _reject_json_constant(text: str) -> object:
    raise ValueError("provider candidate-set envelope contains a non-finite JSON number")


def _project_candidate_submission(
    payload: bytes,
    *,
    submission_contract: str,
    arm: str | None,
    maximum_candidates_per_turn: int,
) -> tuple[bytes, ...]:
    """Project one sealed provider file into the ordered semantic Candidate set."""

    if (
        not isinstance(maximum_candidates_per_turn, int)
        or isinstance(maximum_candidates_per_turn, bool)
        or maximum_candidates_per_turn <= 0
    ):
        raise ValueError("provider maximum candidates per Turn differs")
    if submission_contract != CANDIDATE_SET_ENVELOPE_V1 or arm not in {
        "open_cake",
        "direct_cuda",
        "native_triton",
        "native_cute_dsl",
    }:
        raise ValueError("provider submission contract differs")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provider candidate-set envelope is not JSON") from error
    if (
        not isinstance(document, Mapping)
        or set(document) != {"schema_version", "arm", "candidates"}
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("arm") != arm
    ):
        raise ValueError("provider candidate-set envelope fields differ")
    candidates = document.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > maximum_candidates_per_turn
    ):
        raise ValueError("provider candidate-set envelope count differs")
    if arm in {"open_cake", "native_triton", "native_cute_dsl"}:
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise ValueError("Open Cake candidate-set member is not a Schedule object")
        projected = tuple(_canonical_json_bytes(candidate) for candidate in candidates)
    else:
        if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
            raise ValueError("direct CUDA candidate-set member is not non-empty source")
        projected = tuple(candidate.encode("utf-8") for candidate in candidates)
    if len({sha256(candidate).hexdigest() for candidate in projected}) != len(projected):
        raise ValueError("provider candidate-set contains duplicate Candidate bytes")
    return projected


@dataclass(frozen=True)
class ProviderInvocation:
    """Complete immutable provider process request."""

    argv: tuple[str, ...]
    cwd: Path
    sandbox: str
    provider_revision: str
    removed_environment: tuple[str, ...]
    thread_id: str | None


@dataclass(frozen=True)
class ProviderTurn:
    """Strict projection of one provider JSONL Turn and its sealed candidate."""

    thread_id: str
    provider_tokens: int
    candidates: tuple[bytes, ...]
    """Every candidate this Turn wrote, in the order the provider wrote them.

    A tuple rather than one, because the paper's loop generates structurally distinct
    candidates and ranks them before spending GPU time. One candidate leaves the ranking
    stage with nothing to rank. How many a Turn may write is declared by the Study
    Contract and granted identically to both arms (`docs/adr/0006`).
    """

    candidate_sha256s: tuple[str, ...]
    raw_submission: bytes
    """Exact candidate envelope bytes from this Turn's no-follow file read."""

    raw_events: bytes
    raw_events_sha256: str
    terminal_message: str
    terminal_message_count: int
    normalization: str
    tool_activity: tuple["ProviderAuxiliaryActivity", ...] = ()
    reference_bundle: bytes | None = None
    """Exact rendered reference bytes embedded in this Turn, when one exists."""


@dataclass(frozen=True)
class ProviderAuxiliaryActivity:
    """One provider-emitted non-submission item retained from raw JSONL."""

    item_id: str
    item_type: str
    status: str
    server: str | None = None
    tool: str | None = None

    @property
    def document(self) -> Mapping[str, object]:
        """Return the deterministic replay projection of this auxiliary item."""

        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "status": self.status,
            "server": self.server,
            "tool": self.tool,
        }


@dataclass(frozen=True)
class ParsedCodexTurnEvents:
    """Filesystem-independent meaning of one admitted Codex JSONL Turn."""

    thread_id: str
    provider_tokens: int
    candidate_path: str | None
    change_kind: str | None
    terminal_message_count: int
    normalization: str
    tool_activity: tuple[ProviderAuxiliaryActivity, ...] = ()


@dataclass(frozen=True)
class ProviderQualificationReceipt:
    """Non-secret capability gate for a provider revision."""

    provider_revision: str
    executable_sha256: str
    configuration_sha256: str
    initial_and_resume_equivalent: bool
    file_lifecycle_observed: bool
    usage_observed: bool
    qualified: bool
    scope: str

    def __post_init__(self) -> None:
        if (
            not self.provider_revision
            or len(self.executable_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.executable_sha256)
            or len(self.configuration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.configuration_sha256
            )
            or self.scope
            not in {
                "zero_gpu_contract_fixture_only",
                "live_two_turn_current_provider",
                "live_two_turn_tool_rich_provider",
            }
        ):
            raise ValueError("provider qualification identity differs")

    @classmethod
    def load(cls, path: str | Path) -> "ProviderQualificationReceipt":
        """Load the closed non-secret provider capability receipt."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or set(document) != {
            "schema_version",
            "provider_revision",
            "executable_sha256",
            "configuration_sha256",
            "initial_and_resume_equivalent",
            "file_lifecycle_observed",
            "usage_observed",
            "qualified",
            "scope",
        } or document.get("schema_version") != 1:
            raise ValueError("provider qualification fields differ")
        return cls(
            provider_revision=str(document["provider_revision"]),
            executable_sha256=str(document["executable_sha256"]),
            configuration_sha256=str(document["configuration_sha256"]),
            initial_and_resume_equivalent=document["initial_and_resume_equivalent"] is True,
            file_lifecycle_observed=document["file_lifecycle_observed"] is True,
            usage_observed=document["usage_observed"] is True,
            qualified=document["qualified"] is True,
            scope=str(document["scope"]),
        )

    @property
    def document(self) -> Mapping[str, object]:
        """Return the canonical non-secret receipt document."""

        return {
            "schema_version": 1,
            "provider_revision": self.provider_revision,
            "executable_sha256": self.executable_sha256,
            "configuration_sha256": self.configuration_sha256,
            "initial_and_resume_equivalent": self.initial_and_resume_equivalent,
            "file_lifecycle_observed": self.file_lifecycle_observed,
            "usage_observed": self.usage_observed,
            "qualified": self.qualified,
            "scope": self.scope,
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


_THREAD_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


_MAX_CANDIDATE_BYTES = 64 * 1024 * 1024


CANDIDATE_SET_ENVELOPE_V1 = "candidate_set_envelope_v1"


CODEX_DISABLED_FEATURES = (
    "apps",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_mcp_dependency_install",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "workspace_dependencies",
)
