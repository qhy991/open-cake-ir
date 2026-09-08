"""Typed exact-shape specialization from a frozen kernel seed."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from open_cake_ir.compiler import Assessment, Compiler, CompilerError, Lowering




def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompilerError(f"{context} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True, order=True)
class ExactShape:
    """One exact Flash-KMeans semantic key."""

    batch: int
    tokens: int
    centroids: int
    features: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.batch, self.tokens, self.centroids, self.features)
        ):
            raise CompilerError("exact-shape axes must be positive integers")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExactShape":
        """Build from the Workload Contract's closed B/N/K/D case shape."""

        if set(value) != {"B", "N", "K", "D"}:
            raise CompilerError("exact-shape fields differ")
        return cls(
            cast(int, value["B"]),
            cast(int, value["N"]),
            cast(int, value["K"]),
            cast(int, value["D"]),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.batch, self.tokens, self.centroids, self.features

    @classmethod
    def from_case(cls, case: Mapping[str, object]) -> "ExactShape":
        shape = case.get("shape")
        if not isinstance(shape, Mapping):
            raise ValueError("Workload case shape differs")
        return cls.from_mapping(shape)

    def as_dict(self) -> dict[str, int]:
        return {"B": self.batch, "N": self.tokens, "K": self.centroids, "D": self.features}


@dataclass(frozen=True)
class KernelSeed:
    """Frozen schedule choices retained before held-out cases are observed."""

    seed_id: str
    workload_sha256: str
    base_schedule_path: str
    base_schedule_sha256: str
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    source_revision: str
    source_path: str
    source_sha256: str
    canonical_sha256: str
    _base_schedule: Mapping[str, object]

    @classmethod
    def load(cls, project_root: str | Path, path: str | Path) -> "KernelSeed":
        """Load and verify one seed plus its semantic Schedule authority."""

        root = Path(project_root).resolve(strict=True)
        source = Path(path).resolve(strict=True)
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "schema_version",
            "seed_id",
            "workload_sha256",
            "base_schedule",
            "configuration",
            "provenance",
        } or document.get("schema_version") != 1:
            raise CompilerError("Kernel Seed fields or schema_version differ")
        seed_id = document.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id:
            raise CompilerError("Kernel Seed identity differs")
        base = document.get("base_schedule")
        configuration = document.get("configuration")
        provenance = document.get("provenance")
        if not isinstance(base, dict) or set(base) != {"path", "canonical_sha256"}:
            raise CompilerError("Kernel Seed base Schedule reference differs")
        if not isinstance(configuration, dict) or set(configuration) != {
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
        }:
            raise CompilerError("Kernel Seed configuration differs")
        if not isinstance(provenance, dict) or set(provenance) != {
            "legacy_revision",
            "legacy_path",
            "raw_sha256",
            "selection",
        } or provenance.get("selection") != "pre_heldout_first_valid_candidate":
            raise CompilerError("Kernel Seed provenance differs")
        relative = base.get("path")
        if not isinstance(relative, str) or not relative:
            raise CompilerError("Kernel Seed base Schedule path differs")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
            raise CompilerError("Kernel Seed base Schedule path is unsafe")
        schedule_path = (root / relative).resolve(strict=True)
        if root not in schedule_path.parents:
            raise CompilerError("Kernel Seed base Schedule escapes project root")
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        schedule_sha256 = sha256(_canonical_json_bytes(schedule)).hexdigest()
        if base.get("canonical_sha256") != schedule_sha256:
            raise CompilerError("Kernel Seed base Schedule bytes differ")

        def positive(name: str) -> int:
            value = configuration.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CompilerError(f"Kernel Seed {name} differs")
            return value

        return cls(
            seed_id=seed_id,
            workload_sha256=_digest(document.get("workload_sha256"), "Kernel Seed workload"),
            base_schedule_path=relative,
            base_schedule_sha256=schedule_sha256,
            block_n=positive("BLOCK_N"),
            block_k=positive("BLOCK_K"),
            num_warps=positive("num_warps"),
            num_stages=positive("num_stages"),
            source_revision=cast(str, provenance.get("legacy_revision")),
            source_path=cast(str, provenance.get("legacy_path")),
            source_sha256=_digest(provenance.get("raw_sha256"), "Kernel Seed source"),
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            _base_schedule=cast(Mapping[str, object], schedule),
        )

    def schedule_for(self, case_id: str, shape: ExactShape) -> Mapping[str, object]:
        """Bind the seed to one exact shape without observing evaluation outcomes."""

        if not case_id or shape.features != 128:
            raise CompilerError("specialist case identity or feature extent differs")
        if shape.tokens % self.block_n or shape.centroids % self.block_k:
            raise CompilerError("specialist shape is not exactly tiled by the frozen seed")
        schedule = cast(dict[str, object], json.loads(_canonical_json_bytes(self._base_schedule)))
        schedule["schedule_id"] = f"{self.seed_id}-{case_id}"
        roles = cast(list[dict[str, object]], schedule["roles"])
        roles[0]["warps"] = list(range(self.num_warps))
        axes = cast(list[dict[str, object]], cast(dict[str, object], schedule["program_map"])["axes"])
        for axis in axes:
            if axis["name"] == "token_block":
                axis["tile"] = self.block_n
        loop = cast(list[dict[str, object]], schedule["tile_loops"])[0]
        loop["tile"] = self.block_k
        cast(dict[str, object], loop["range_options"])["num_stages"] = self.num_stages
        buffers = {
            cast(str, buffer["name"]): buffer
            for buffer in cast(list[dict[str, object]], schedule["buffers"])
        }
        buffers["tokens"]["shape"] = [shape.batch, shape.tokens, shape.features]
        buffers["centroids"]["shape"] = [shape.batch, shape.centroids, shape.features]
        buffers["centroid_sq"]["shape"] = [shape.batch, shape.centroids]
        buffers["assignments"]["shape"] = [shape.batch, shape.tokens]
        buffers["token_tile"]["shape"] = [self.block_n, shape.features]
        buffers["centroid_tile"]["shape"] = [self.block_k, shape.features]
        buffers["distance_tile"]["shape"] = [self.block_n, self.block_k]
        buffers["best_index_tile"]["shape"] = [self.block_n]
        # The distance is composed now, so the intermediates the composition names are
        # tiled by the same seed. Listing them here is the same hand-kept coupling the
        # accumulator note below describes; the verifier catches a member left behind.
        for name in ("cross", "scaled_cross"):
            if name in buffers:
                buffers[name]["shape"] = [self.block_n, self.block_k]
        if "norm_tile" in buffers:
            buffers["norm_tile"]["shape"] = [self.block_k]
        # The MMA tile is the accumulator's shape, so a seed that retiles the buffers has
        # to retile it too. Keeping the two in step by hand across this boundary is the
        # coupling that made a specialist silently disagree with its own accumulator.
        for operation in cast(list[dict[str, object]], schedule["operations"]):
            if operation.get("kind") != "mma":
                continue
            parameters = cast(dict[str, object], operation["parameters"])
            if "tile_shape" in parameters:
                parameters["tile_shape"] = [
                    self.block_n,
                    self.block_k,
                    shape.features,
                ]
        return schedule


@dataclass(frozen=True)
class SpecialistLowering:
    """One independently sealed exact-shape Compiler result."""

    case_id: str
    shape: ExactShape
    assessment: Assessment
    lowering: Lowering


def lower_specialists(
    compiler: Compiler,
    seed: KernelSeed,
    cases: Mapping[str, Mapping[str, object]],
) -> tuple[SpecialistLowering, ...]:
    """Lower an ordered case mapping through the same public Compiler path."""

    result: list[SpecialistLowering] = []
    for case_id, shape_value in cases.items():
        shape = ExactShape.from_mapping(shape_value)
        assessment = compiler.assess(seed.schedule_for(case_id, shape))
        lowering = compiler.lower(assessment)
        result.append(SpecialistLowering(case_id, shape, assessment, lowering))
    if len({item.lowering.source_sha256 for item in result}) != len(result):
        raise CompilerError("specialist lowerings must have distinct source identities")
    return tuple(result)
