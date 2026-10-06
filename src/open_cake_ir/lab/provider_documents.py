"""Provider records and strict authored-file byte admission."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json, math, os, re, stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from ._documents import _canonical_json_bytes
from .native_skills import NativeSkillPackage


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
    environment_kind: str = "open_cake",
    maximum_candidates_per_turn: int,
) -> tuple[bytes, ...]:
    """Project one sealed provider file into the ordered semantic Candidate set."""

    if (
        not isinstance(maximum_candidates_per_turn, int)
        or isinstance(maximum_candidates_per_turn, bool)
        or maximum_candidates_per_turn <= 0
    ):
        raise ValueError("provider maximum candidates per Turn differs")
    if submission_contract == PYTHON_SOURCE_FILE_V1:
        if (environment_kind != 'open_cake' or not isinstance(arm, str) or not arm
            or maximum_candidates_per_turn != 1 or type(payload) is not bytes or not payload):
            raise ValueError('Python source-file submission contract differs')
        try:
            source = payload.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError('provider Python source file is not UTF-8') from error
        return (canonical_json_bytes({'python_source': source}),)
    if submission_contract == PYTHON_CANDIDATE_BUNDLE_V1:
        if environment_kind != 'open_cake' or not isinstance(arm, str) or not arm:
            raise ValueError('Python candidate-bundle submission contract differs')
        from .python_candidate_bundle import project_python_candidate_bundle
        try:
            return project_python_candidate_bundle(payload,
                maximum_candidates_per_turn=maximum_candidates_per_turn)
        except ValueError as error:
            # The native Turn/session/file/usage were valid; the authored file
            # was not. Preserve its exact bytes in raw_submission and reject
            # the whole bundle as one invalid submission, without compiling or
            # truncating excess proposals. The next Turn can repair its source.
            return (canonical_json_bytes({'python_bundle_error':str(error)}),)
    if submission_contract != CANDIDATE_SET_ENVELOPE_V1 or not isinstance(arm, str) or not arm or environment_kind not in {
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
    if environment_kind in {"open_cake", "native_triton", "native_cute_dsl"}:
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
    codex_home: Path | None = None
    system_skills_snapshot: tuple[tuple[str, int, str], ...] | None = None
    user_home: Path | None = None
    native_skill_package: "NativeSkillPackage | None" = None


def invocation_document(invocation: ProviderInvocation, *, include_prompt: bool = True) -> dict:
    """One invocation projection; Run skill binding explicitly omits the final prompt."""
    document = {
        'argv' if include_prompt else 'argv_without_prompt': list(
            invocation.argv if include_prompt else invocation.argv[:-1]),
        'cwd': str(invocation.cwd), 'sandbox': invocation.sandbox,
        'provider_revision': invocation.provider_revision,
        'removed_environment': list(invocation.removed_environment),
        'thread_id': invocation.thread_id,
    }
    if invocation.codex_home is not None: document['codex_home'] = str(invocation.codex_home)
    if invocation.user_home is not None: document['user_home'] = str(invocation.user_home)
    if invocation.native_skill_package is not None:
        document['native_skill_package'] = invocation.native_skill_package.reference
    return document


@dataclass(frozen=True)
class ProviderTurn:
    """Strict projection of one provider JSONL Turn and its sealed candidate."""

    thread_id: str
    provider_tokens: int
    """Native counter at adapter output; QualifiedRunProvider returns an invocation delta."""
    candidates: tuple[bytes, ...]
    """Every candidate this Turn wrote, in the order the provider wrote them.

    A tuple rather than one, because the paper's loop generates structurally distinct
    candidates and ranks them before spending GPU time. One candidate leaves the ranking
    stage with nothing to rank. How many a Turn may write is declared by the Study
    Contract and granted identically to both arms (`docs/adr/0006`).
    """

    candidate_sha256s: tuple[str, ...]
    raw_submission: bytes
    """Exact author file bytes from this Turn's no-follow read, before projection."""

    raw_events: bytes
    raw_events_sha256: str
    terminal_message: str
    terminal_message_count: int
    normalization: str
    tool_activity: tuple["ProviderAuxiliaryActivity", ...] = ()
    reference_bundle: bytes | None = None
    """Exact rendered reference bytes embedded in this Turn, when one exists."""

    native_skill_input: bytes | None = None
    """Bounded same-invocation native skill frames; not a qualification receipt."""

    native_skill_binding: bytes | None = None
    """Executor-owned Run/turn/invocation binding, never authored by the provider."""


@dataclass(frozen=True)
class ProviderAuxiliaryActivity:
    """One native item or indexed transport notice retained from raw JSONL.

    For transport_reconnect, item_id is the zero-based jsonl:N location rather
    than a native item identifier. Its full text remains in the original stream.
    """

    item_id: str
    item_type: str
    status: str
    server: str | None = None
    tool: str | None = None
    model: str | None = None
    provider_tokens: int | None = None

    def __post_init__(self) -> None:
        if (self.model is None) != (self.provider_tokens is None) or self.model is not None and (
                not isinstance(self.model, str) or not self.model
                or type(self.provider_tokens) is not int or self.provider_tokens < 0):
            raise ValueError("provider auxiliary model usage differs")

    @property
    def document(self) -> Mapping[str, object]:
        """Return the deterministic replay projection of this auxiliary item."""

        result = {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "status": self.status,
            "server": self.server,
            "tool": self.tool,
        }

        if self.model is not None:
            result.update(model=self.model, provider_tokens=self.provider_tokens)
        return result


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


NATIVE_SKILL_QUALIFICATION_V1 = "native_skill_qualification_v1"


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
    system_skills_sha256: str | None = None
    native_skill_input_contract: str | None = None

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
            or (self.system_skills_sha256 is not None and (
                len(self.system_skills_sha256) != 64
                or any(char not in '0123456789abcdef' for char in self.system_skills_sha256)))
        ):
            raise ValueError("provider qualification identity differs")
        if self.native_skill_input_contract is not None and (
            self.native_skill_input_contract != NATIVE_SKILL_QUALIFICATION_V1
            or self.system_skills_sha256 is None):
            raise ValueError('provider native skill qualification capability differs')


    @classmethod
    def load(cls, path: str | Path) -> "ProviderQualificationReceipt":
        """Load the closed non-secret provider capability receipt."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {
            "schema_version",
            "provider_revision",
            "executable_sha256",
            "configuration_sha256",
            "initial_and_resume_equivalent",
            "file_lifecycle_observed",
            "usage_observed",
            "qualified",
            "scope",
        }
        version = document.get('schema_version') if isinstance(document, Mapping) else None
        if (not isinstance(document, Mapping) or type(version) is not int or version not in {1, 2, 3}
            or set(document) != fields | ({'system_skills_sha256'} if version in {2, 3} else set())
                | ({'native_skill_input_contract'} if version == 3 else set())):
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
            system_skills_sha256=(str(document['system_skills_sha256']) if version in {2, 3} else None),
            native_skill_input_contract=(str(document['native_skill_input_contract']) if version == 3 else None),
        )

    @property
    def document(self) -> Mapping[str, object]:
        """Return the canonical non-secret receipt document."""

        return {
            "schema_version": 3 if self.native_skill_input_contract is not None else (
                2 if self.system_skills_sha256 is not None else 1),
            "provider_revision": self.provider_revision,
            "executable_sha256": self.executable_sha256,
            "configuration_sha256": self.configuration_sha256,
            "initial_and_resume_equivalent": self.initial_and_resume_equivalent,
            "file_lifecycle_observed": self.file_lifecycle_observed,
            "usage_observed": self.usage_observed,
            "qualified": self.qualified,
            "scope": self.scope,
            **({'system_skills_sha256': self.system_skills_sha256}
               if self.system_skills_sha256 is not None else {}),
            **({'native_skill_input_contract': self.native_skill_input_contract}
               if self.native_skill_input_contract is not None else {}),
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            canonical_json_bytes(self.document)
        ).hexdigest()


_THREAD_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


_MAX_CANDIDATE_BYTES = 64 * 1024 * 1024


CANDIDATE_SET_ENVELOPE_V1 = "candidate_set_envelope_v1"
PYTHON_SOURCE_FILE_V1 = "python_source_file_v1"
PYTHON_CANDIDATE_BUNDLE_V1 = "python_candidate_bundle_v1"


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



def expected_codex_disabled_features(event_contract: str, author_home_policy: str | None) -> tuple[str, ...]:
    """The declared tool surface and author-home boundary jointly own CLI features.

    Native task skills remain available in the explicitly projected user HOME.
    Isolated homes cannot import account plugins or connector integrations on
    startup; their existing file and native-input checks still decide admission.
    This is configuration, not proof that a CLI actually honored the restriction.
    """
    if event_contract == 'closed_file_change_v1':
        return CODEX_DISABLED_FEATURES
    if event_contract == 'tool_rich_candidate_v1':
        return () if author_home_policy is None else ('apps', 'plugins', 'remote_plugin')
    raise ValueError('Codex feature and event contracts differ')
