"""Reusable Workload Contract independent of study and provider policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, cast

_FIELDS = {
    "schema_version",
    "workload_id",
    "revision",
    "state",
    "provenance",
    "operator",
    "cases",
    "tensors",
    "semantics",
    "oracle",
    "validation",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


@dataclass(frozen=True)
class TensorABI:
    """One ordered argument, resolved only from Workload tensor/case declarations."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    mode: str


class WorkloadContract:
    """Canonical operator semantics, cases and correctness authority."""

    def __init__(self, document: Mapping[str, object], source_path: Path | None = None) -> None:
        self._document = json.loads(_canonical_json_bytes(document))
        self.source_path = source_path
        self.workload_id = _name(document.get("workload_id"), "workload.workload_id")
        self.canonical_sha256 = sha256(_canonical_json_bytes(document)).hexdigest()
        raw_cases = document.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("workload.cases must be a non-empty list")
        cases: dict[str, Mapping[str, object]] = {}
        for index, value in enumerate(raw_cases):
            case = _object(value, f"workload.cases[{index}]")
            if not {"case_id", "shape", "seed", "mode"} <= set(case) <= {
                "case_id",
                "shape",
                "seed",
                "mode",
                "materialized",
            }:
                raise ValueError(f"workload.cases[{index}] fields differ")
            case_id = _name(case.get("case_id"), f"workload.cases[{index}].case_id")
            if case_id in cases:
                raise ValueError(f"workload case {case_id!r} is duplicated")
            shape = _object(case.get("shape"), f"workload.cases[{index}].shape")
            if not shape or any(
                not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
                for extent in shape.values()
            ):
                raise ValueError(f"workload.cases[{index}].shape is invalid")
            seed = case.get("seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
                raise ValueError(f"workload.cases[{index}].seed is invalid")
            _name(case.get("mode"), f"workload.cases[{index}].mode")
            materialized = case.get("materialized")
            if materialized is not None:
                objects = _object(materialized, f"workload.cases[{index}].materialized")
                for name, raw_receipt in objects.items():
                    receipt = _object(raw_receipt, f"workload.cases[{index}].materialized.{name}")
                    if set(receipt) != {"sha256", "size_bytes"}:
                        raise ValueError(f"workload materialized receipt {name!r} fields differ")
                    digest = receipt.get("sha256")
                    size = receipt.get("size_bytes")
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or not isinstance(size, int)
                        or isinstance(size, bool)
                        or size <= 0
                    ):
                        raise ValueError(f"workload materialized receipt {name!r} is invalid")
            cases[case_id] = case
        self._cases = cases

    @classmethod
    def from_document(cls, value: object, source: Path, *, validate: Callable[[Mapping[str, object]], None]) -> "WorkloadContract":
        """Load and validate one frozen Workload Contract."""

        document = _object(value, "workload")
        if set(document) != _FIELDS or document.get("schema_version") != 1:
            raise ValueError("workload root fields or schema_version differ")
        if document.get("state") != "frozen":
            raise ValueError("workload must be frozen")
        _name(document.get("revision"), "workload.revision")
        _name(document.get("operator"), "workload.operator")
        provenance = document.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError("workload provenance must be a non-empty list")
        for field in ("tensors", "semantics", "oracle", "validation"):
            _object(document.get(field), f"workload.{field}")
        validate(document)
        return cls(document, source)

    @property
    def document(self) -> dict[str, object]:
        """Return a detached JSON projection of the frozen contract."""

        return cast(dict[str, object], json.loads(_canonical_json_bytes(self._document)))

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(self._cases)

    def case(self, case_id: str) -> dict[str, object]:
        """Return one detached workload case by canonical ID."""

        try:
            value = self._cases[case_id]
        except KeyError as error:
            raise KeyError(f"unknown workload case {case_id!r}") from error
        return cast(dict[str, object], json.loads(_canonical_json_bytes(value)))

    @property
    def target(self) -> str:
        return _name(self._document["semantics"].get("target"), "workload target")

    def tensor_abi(self, case_id: str) -> tuple[TensorABI, ...]:
        """Resolve the explicitly ordered input/output ABI, without operator dispatch.

        A shape component is a positive integer or an exact case-dimension name;
        expressions are not evaluated. Historical contracts without this declaration
        retain their existing admission and do not acquire an inferred ABI.
        """

        shape = _object(self.case(case_id)["shape"], "workload case shape")
        tensors = _object(self._document["tensors"], "workload tensors")
        semantics = _object(self._document["semantics"], "workload semantics")
        abi = _object(semantics.get("candidate_abi"), "explicit workload tensor ABI")
        if set(abi) != {"inputs", "outputs"}:
            raise ValueError("workload tensor ABI must declare inputs and outputs")
        result: list[TensorABI] = []
        for mode in ("input", "output"):
            names = abi[mode + "s"]
            if not isinstance(names, list) or not names:
                raise ValueError(f"workload tensor ABI {mode}s must be non-empty")
            for name in names:
                name = _name(name, "workload tensor ABI name")
                tensor = _object(tensors.get(name), f"workload tensor {name}")
                dimensions = tensor.get("shape")
                if not isinstance(dimensions, list) or not dimensions:
                    raise ValueError(f"workload tensor {name} must have a shape")
                extents = tuple(
                    shape.get(dimension) if isinstance(dimension, str) else dimension
                    for dimension in dimensions
                )
                if any(
                    not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
                    for extent in extents
                ):
                    raise ValueError(f"workload tensor {name} has an unresolved shape")
                dtype = tensor.get("dtype")
                if not isinstance(dtype, str) or dtype not in {"fp32", "bf16", "int32"}:
                    raise ValueError(f"workload tensor {name} dtype is unsupported")
                if tensor.get("layout") != "contiguous_row_major":
                    raise ValueError(f"workload tensor {name} must be contiguous row major")
                result.append(TensorABI(name, cast(tuple[int, ...], extents), dtype, mode))
        names = [tensor.name for tensor in result]
        if len(names) != len(set(names)) or set(names) != set(tensors):
            raise ValueError("workload tensor ABI must cover each tensor exactly once")
        return tuple(result)
