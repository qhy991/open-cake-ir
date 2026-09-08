"""Shared credential-shape detection for Evidence, export and tracked-file admission."""

import re

_SECRET_MARKERS = (
    b"OPENAI_API_KEY=",
    b"CODEX_ACCESS_TOKEN=",
    b"INFINI_API_KEY=",
)
_SECRET_PATTERNS = (
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY"),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\."),
    re.compile(rb"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"(?i)authorization\s*:\s*bearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(rb"(?i)bearer\s+sk-[a-z0-9_-]{8,}"),
    re.compile(
        rb"(?i)(?:openai|anthropic|github|gitlab|azure|aws|codex|infini)"
        rb"[a-z0-9_-]{0,24}(?:key|token|secret)\s*[:=]\s*['\"]?[a-z0-9._~+/=-]{8,}"
    ),
    re.compile(rb"\b(?:ghp[-_]|github_pat[-_]|sk-)[a-zA-Z0-9_-]{8,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def contains_forbidden_secret(payload: bytes) -> bool:
    """Detect bounded credential shapes without returning or logging matched bytes.

    This is a shared refusal rule, not a guarantee that arbitrary secret formats
    are recognizable. Callers retain their own file/CAS admission boundary.
    """
    return any(marker in payload for marker in _SECRET_MARKERS) or any(
        pattern.search(payload) is not None for pattern in _SECRET_PATTERNS
    )


