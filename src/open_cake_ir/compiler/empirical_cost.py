"""CPU interpolation over explicitly supplied, revision-bound calibration curves.

The Compiler checks coverage and arithmetic, not the authenticity of the external
measurements. This advisory input never changes released ranking qualification.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .ir import Schedule


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} fields differ")
    return value


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _number(value: object, label: str) -> float:
    try:
        numeric = float(value) if type(value) in (int, float) else math.nan
    except OverflowError:
        numeric = math.nan
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class _Curve:
    template: bytes
    dimensions: tuple[tuple[str, int], ...]
    multiple: int
    points: tuple[tuple[int, float], ...]
    envelope: float

    @classmethod
    def parse(cls, value: object, target: str) -> _Curve:
        row = _object(value, {"template", "varying_dimensions", "extent_multiple", "points", "relative_error_envelope"}, "curve")
        template = row["template"]
        if not isinstance(template, dict):
            raise ValueError("curve template must be a Schedule object")
        Schedule.from_dict(template)
        if template["target"] != target:
            raise ValueError("curve template target differs from model")
        buffers = {buffer["name"]: buffer for buffer in template["buffers"]}
        bindings = row["varying_dimensions"]
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("curve requires varying buffer dimensions")
        dimensions: list[tuple[str, int]] = []
        extents: list[int] = []
        for item in bindings:
            item = _object(item, {"buffer", "dimension"}, "varying dimension")
            name, axis = item["buffer"], item["dimension"]
            if not isinstance(name, str) or name not in buffers or type(axis) is not int or not 0 <= axis < len(buffers[name]["shape"]):
                raise ValueError("varying dimension is not a template buffer axis")
            if (name, axis) in dimensions:
                raise ValueError("varying dimension is duplicated")
            dimensions.append((name, axis))
            extents.append(_integer(buffers[name]["shape"][axis], "template extent"))
        if len(set(extents)) != 1:
            raise ValueError("varying dimensions must share one scalar extent")
        multiple = _integer(row["extent_multiple"], "extent_multiple")
        if not isinstance(row["points"], list) or len(row["points"]) < 2:
            raise ValueError("curve requires at least two measured points")
        points = []
        for item in row["points"]:
            item = _object(item, {"extent", "kernel_us"}, "curve point")
            extent = _integer(item["extent"], "point extent")
            duration = _number(item["kernel_us"], "point kernel_us")
            if duration <= 0 or extent % multiple or (points and extent <= points[-1][0]):
                raise ValueError("curve points must be positive, aligned and strictly ordered by extent")
            points.append((extent, duration))
        envelope = _number(row["relative_error_envelope"], "relative_error_envelope")
        if not 0 <= envelope < 1:
            raise ValueError("relative_error_envelope must be in [0, 1)")
        if any(not math.isfinite(duration * (1 + envelope)) for _, duration in points):
            raise ValueError("curve empirical interval is not finite")
        return cls(_canonical(template), tuple(dimensions), multiple, tuple(points), envelope)

    def predict(self, query: dict, query_bytes: bytes) -> tuple[float, tuple[float, float]] | None:
        expected = json.loads(self.template)
        buffers = {buffer["name"]: buffer for buffer in query["buffers"]}
        if any(name not in buffers or axis >= len(buffers[name]["shape"]) for name, axis in self.dimensions):
            return None
        extents = [buffers[name]["shape"][axis] for name, axis in self.dimensions]
        if any(type(extent) is not int or extent <= 0 for extent in extents) or len(set(extents)) != 1:
            return None
        extent = extents[0]
        if extent % self.multiple or not self.points[0][0] <= extent <= self.points[-1][0]:
            return None
        expected["schedule_id"] = query["schedule_id"]
        for buffer in expected["buffers"]:
            for name, axis in self.dimensions:
                if buffer["name"] == name:
                    buffer["shape"][axis] = extent
        if _canonical(expected) != query_bytes:
            return None
        upper = bisect.bisect_left([point[0] for point in self.points], extent)
        if self.points[upper][0] == extent:
            duration = self.points[upper][1]
        else:
            low, high = self.points[upper - 1], self.points[upper]
            fraction = (extent - low[0]) / (high[0] - low[0])
            duration = low[1] * (1 - fraction) + high[1] * fraction
        return duration, (duration * (1 - self.envelope), duration * (1 + self.envelope))


@dataclass(frozen=True, init=False)
class EmpiricalCostModel:
    """An immutable external model; context/evidence remain supplier declarations."""

    model_id: str
    compiler_revision_id: str
    compiler_revision_sha256: str
    target: str
    _details: bytes
    _curves: tuple[_Curve, ...]

    def __init__(self, document: object) -> None:
        row = _object(document, {"schema_version", "model_id", "compiler_revision_id", "compiler_revision_sha256", "target", "context", "reported_evidence", "curves"}, "empirical cost model")
        if type(row["schema_version"]) is not int or row["schema_version"] != 2:
            raise ValueError("empirical cost model requires schema_version 2 with Compiler content identity")
        for key in ("model_id", "compiler_revision_id", "target"):
            object.__setattr__(self, key, _name(row[key], key))
        identity = row["compiler_revision_sha256"]
        if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("model Compiler content identity differs")
        object.__setattr__(self, "compiler_revision_sha256", identity)
        context = _object(row["context"], {"timer", "cache_protocol", "runtime", "input_scope"}, "model context")
        for key in ("timer", "cache_protocol", "input_scope"):
            _name(context[key], key)
        if not isinstance(context["runtime"], dict) or not context["runtime"]:
            raise ValueError("model context requires declared runtime versions")
        _name(context["runtime"].get("compiler_version"), "runtime compiler_version")
        for key, value in context["runtime"].items():
            _name(key, "runtime key")
            _name(value, "runtime value")
        if not isinstance(row["reported_evidence"], dict) or not row["reported_evidence"]:
            raise ValueError("model requires reported evidence/provenance")
        object.__setattr__(self, "_details", _canonical({"context": context, "reported_evidence": row["reported_evidence"]}))
        if not isinstance(row["curves"], list) or not row["curves"]:
            raise ValueError("model requires at least one curve")
        object.__setattr__(self, "_curves", tuple(_Curve.parse(curve, self.target) for curve in row["curves"]))

    @classmethod
    def load(cls, path: str | Path) -> EmpiricalCostModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def estimate(self, schedule: dict, *, compiler_revision_id: str, compiler_revision_sha256: str, target: str,
                 compiled_compiler_version: str | None = None) -> dict[str, object]:
        """Estimate an assessed Schedule, preserving all metadata on abstention."""
        result = {"kind": "external_empirical_cost", "model_id": self.model_id,
                  "model_compiler_revision_id": self.compiler_revision_id,
                  "model_compiler_revision_sha256": self.compiler_revision_sha256,
                  "target": self.target, "covered": False, "predicted_kernel_us": None,
                  "empirical_range_us": None, **json.loads(self._details)}
        if (compiler_revision_id != self.compiler_revision_id
                or compiler_revision_sha256 != self.compiler_revision_sha256 or target != self.target):
            result["reason"] = "model Compiler Revision content identity or target differs"
            return result
        if compiled_compiler_version is not None and compiled_compiler_version != result["context"]["runtime"]["compiler_version"]:
            result["reason"] = "compiled artifact compiler version differs from model context"
            return result
        # Public callers get the same construction/type checks as the Compiler.
        Schedule.from_dict(schedule)
        query_bytes = _canonical(schedule)
        predictions = [prediction for curve in self._curves if (prediction := curve.predict(schedule, query_bytes)) is not None]
        if len(predictions) != 1:
            result["reason"] = "ambiguous overlapping cost curves" if predictions else "no curve covers this exact Schedule and extent"
            return result
        duration, interval = predictions[0]
        result.update(covered=True, predicted_kernel_us=duration, empirical_range_us=list(interval), reason=None)
        return result
