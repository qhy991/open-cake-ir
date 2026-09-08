"""Immutable multi-kernel program contract over independently compiled Schedules."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes
from open_cake_ir.tasks.workloads import load_workload

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from open_cake_ir.compiler import Compiler, Schedule

from open_cake_ir.evaluation.workload import WorkloadContract




def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _project_path(root: Path, value: object, context: str) -> Path:
    relative = PurePosixPath(_name(value, context))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} escapes project root")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    if root not in path.parents or path.is_symlink():
        raise ValueError(f"{context} custody differs")
    return path


@dataclass(frozen=True)
class ProgramNode:
    """One independently analyzable Schedule in ordered launch dependency order."""

    node_id: str
    schedule_path: Path
    schedule_sha256: str
    lowering_source_sha256: str
    entry_point: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...]
    views: Mapping[str, str]


@dataclass(frozen=True)
class ProgramContract:
    """Logical ABI, intermediates, and exact Schedule nodes for one candidate program."""

    source_path: Path
    program_id: str
    canonical_sha256: str
    workload: WorkloadContract
    abi: str
    public_inputs: tuple[str, ...]
    public_outputs: tuple[str, ...]
    intermediates: Mapping[str, Mapping[str, object]]
    nodes: tuple[ProgramNode, ...]
    reference_visibility: str

    @classmethod
    def load(
        cls,
        project_root: str | Path,
        path: str | Path,
        compiler: Compiler,
    ) -> "ProgramContract":
        root = Path(project_root).resolve(strict=True)
        source = Path(path).resolve(strict=True)
        document = _object(json.loads(source.read_text(encoding="utf-8")), "program")
        if set(document) != {
            "schema_version",
            "program_id",
            "state",
            "workload",
            "abi",
            "public_inputs",
            "public_outputs",
            "intermediates",
            "nodes",
            "reference_visibility",
        } or document.get("schema_version") != 1:
            raise ValueError("program root fields or schema_version differ")
        if document.get("state") != "frozen":
            raise ValueError("program must be frozen")
        workload_ref = _object(document["workload"], "program.workload")
        if set(workload_ref) != {"path", "canonical_sha256"}:
            raise ValueError("program Workload reference fields differ")
        workload_path = _project_path(root, workload_ref["path"], "program.workload.path")
        workload = load_workload(workload_path)
        if workload.canonical_sha256 != workload_ref["canonical_sha256"]:
            raise ValueError("program Workload bytes differ")

        public_inputs = tuple(cast(list[str], document["public_inputs"]))
        public_outputs = tuple(cast(list[str], document["public_outputs"]))
        tensors = _object(workload.document["tensors"], "workload.tensors")
        if (
            not public_inputs
            or not public_outputs
            or len(set(public_inputs + public_outputs)) != len(public_inputs + public_outputs)
            or any(name not in tensors for name in public_inputs + public_outputs)
        ):
            raise ValueError("program public ABI differs from Workload tensors")
        intermediates = _object(document["intermediates"], "program.intermediates")
        available = set(public_inputs)
        writers: dict[str, str] = {}
        observed_nodes: set[str] = set()
        parsed_nodes: list[ProgramNode] = []
        raw_nodes = document["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("program nodes must be a non-empty list")
        target_shape = cast(Mapping[str, int], workload.case("target_t32768")["shape"])

        for index, raw in enumerate(raw_nodes):
            node = _object(raw, f"program.nodes[{index}]")
            if set(node) != {
                "id",
                "schedule",
                "schedule_sha256",
                "lowering_source_sha256",
                "inputs",
                "outputs",
                "views",
                "depends_on",
            }:
                raise ValueError(f"program.nodes[{index}] fields differ")
            node_id = _name(node["id"], f"program.nodes[{index}].id")
            if node_id in observed_nodes:
                raise ValueError(f"program node {node_id!r} is duplicated")
            inputs = tuple(cast(list[str], node["inputs"]))
            outputs = tuple(cast(list[str], node["outputs"]))
            dependencies = tuple(cast(list[str], node["depends_on"]))
            views = _object(node["views"], f"program.nodes[{index}].views")
            if (
                not inputs
                or not outputs
                or set(views) != set(inputs + outputs)
                or any(name not in available for name in inputs)
                or any(name in writers or name in public_inputs for name in outputs)
                or any(dependency not in observed_nodes for dependency in dependencies)
            ):
                raise ValueError(f"program node {node_id!r} dataflow differs")
            schedule_path = _project_path(
                root,
                node["schedule"],
                f"program.nodes[{index}].schedule",
            )
            assessment = compiler.assess_file(schedule_path)
            if (
                not assessment.accepted
                or not assessment.lowering_eligible
                or assessment.schedule_sha256 != node["schedule_sha256"]
            ):
                raise ValueError(f"program node {node_id!r} Schedule differs")
            lowering = compiler.lower(assessment)
            if lowering.source_sha256 != node["lowering_source_sha256"]:
                raise ValueError(f"program node {node_id!r} lowering differs")
            schedule = Schedule.load(schedule_path)
            globals_by_name = {
                buffer.name: buffer
                for buffer in schedule.buffers
                if buffer.space.value == "global"
            }
            if set(globals_by_name) != set(inputs + outputs):
                raise ValueError(f"program node {node_id!r} external buffers differ")
            for name in inputs + outputs:
                spec = tensors.get(name) if name in tensors else intermediates.get(name)
                if spec is None:
                    raise ValueError(f"program tensor {name!r} has no owner")
                expected_shape = cls._schedule_shape(
                    cast(Mapping[str, object], spec),
                    target_shape,
                    _name(views[name], f"program.nodes[{index}].views.{name}"),
                )
                buffer = globals_by_name[name]
                if buffer.shape != expected_shape or buffer.dtype.value != spec["dtype"]:
                    raise ValueError(f"program node {node_id!r} tensor {name!r} differs")
            for name in outputs:
                writers[name] = node_id
                available.add(name)
            observed_nodes.add(node_id)
            parsed_nodes.append(
                ProgramNode(
                    node_id,
                    schedule_path,
                    cast(str, node["schedule_sha256"]),
                    cast(str, node["lowering_source_sha256"]),
                    schedule.lowering.entry_point,
                    inputs,
                    outputs,
                    dependencies,
                    dict(cast(Mapping[str, str], views)),
                )
            )
        if any(name not in writers for name in public_outputs):
            raise ValueError("program public output has no writer")
        return cls(
            source,
            _name(document["program_id"], "program.program_id"),
            sha256(_canonical_json_bytes(document)).hexdigest(),
            workload,
            _name(document["abi"], "program.abi"),
            public_inputs,
            public_outputs,
            cast(Mapping[str, Mapping[str, object]], intermediates),
            tuple(parsed_nodes),
            _name(document["reference_visibility"], "program.reference_visibility"),
        )

    @staticmethod
    def _schedule_shape(
        spec: Mapping[str, object],
        target_shape: Mapping[str, int],
        view: str,
    ) -> tuple[int, ...]:
        raw_shape = spec.get("shape")
        if not isinstance(raw_shape, list):
            raise ValueError("program tensor shape differs")
        shape = tuple(
            target_shape[item] if isinstance(item, str) else cast(int, item)
            for item in raw_shape
        )
        if view == "identity":
            return shape
        if view == "squeeze_batch" and shape[:1] == (1,):
            return shape[1:]
        if (
            view == "squeeze_batch_and_singleton_head"
            and len(shape) >= 3
            and shape[0] == 1
            and shape[2] == 1
        ):
            return shape[1:2] + shape[3:]
        if view == "unsqueeze_batch" and shape[:1] == (1,):
            return shape[1:]
        raise ValueError(f"program tensor view {view!r} differs")
