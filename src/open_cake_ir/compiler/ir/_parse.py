"""Shared structural parsing checks and the Schedule parse error."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence


class ScheduleParseError(ValueError):
    """One Schedule document is not structurally admissible."""


def _strict_object(
    value: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleParseError(f"{context} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing:
        raise ScheduleParseError(f"{context} missing fields: {sorted(missing)}")
    if extra:
        raise ScheduleParseError(f"{context} unknown fields: {sorted(extra)}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleParseError(f"{context} must be a non-empty string")
    return value


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleParseError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScheduleParseError(f"{context} must be a non-negative integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ScheduleParseError(f"{context} must be a boolean")
    return value


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScheduleParseError(f"{context} must be a list of strings")
    return tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        admitted = ", ".join(sorted(member.value for member in enum_type))
        raise ScheduleParseError(
            f"{context} is unsupported; admitted values are {admitted}"
        ) from error


def _object_list(value: Any, context: str, *, allow_empty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        raise ScheduleParseError(f"{context} must be a list")
    if not value and not allow_empty:
        raise ScheduleParseError(f"{context} must not be empty")
    return value
