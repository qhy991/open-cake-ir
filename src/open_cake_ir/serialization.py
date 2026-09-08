"""Canonical UTF-8 JSON bytes shared by runtime authority boundaries."""

import json


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON value without ASCII escaping or non-finite numbers.

    Callers own validation and conversion of their domain objects to JSON values.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
