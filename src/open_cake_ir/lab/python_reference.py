"""Bind task-owned Python starters without introducing another Schedule language."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

from open_cake_ir.compiler import frontend


def parse_skeleton(source: str, *, filename: str) -> dict:
    """Select the existing Program parser only for an explicit static declaration."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        location = frontend.SourceLocation(filename, error.lineno or 1, error.offset or 1,
            error.end_lineno or error.lineno or 1, error.end_offset or error.offset or 1)
        raise frontend.FrontendError(error.msg, location) from error
    program = any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'cake' and node.value.func.attr == 'program'
        for node in tree.body)
    if program:
        from open_cake_ir.compiler.program_frontend import parse_program
        return parse_program(source, filename=filename).document
    return frontend.parse(source, filename=filename).document


def read_skeleton(path: Path) -> dict:
    """Read a static Schedule or Program without executing authored Python."""
    path = Path(path)
    payload = path.read_text(encoding='utf-8')
    return parse_skeleton(payload, filename=str(path)) if path.suffix == '.py' else json.loads(payload)


def skeleton_route(document: Mapping[str, object]) -> dict:
    """All Program stages share a backend; the first entry is a Schedule default."""
    if 'program_id' not in document:
        return dict(document['lowering'])
    from open_cake_ir.compiler.ir import Program
    program = Program.from_dict(document)
    backends = {stage.schedule.lowering.backend for stage in program.stages}
    if len(backends) != 1:
        raise ValueError('Program starter stages require one authoring backend')
    return dict(program.document['stages'][0]['schedule']['lowering'])


def lower_skeleton(compiler, document):
    """Lower the complete bound starter through its existing Compiler entry."""
    if 'program_id' in document:
        from open_cake_ir.compiler.ir import Program
        return compiler.lower_program(Program.from_dict(document))
    assessment = compiler.assess(document)
    if not assessment.lowering_eligible:
        raise ValueError('task optimization baseline is not lowerable')
    return compiler.lower(assessment)


def bind_python_reference(source: str, prepared: Mapping[str, object], *, filename: str) -> bytes:
    """Keep the author's Python untouched while Lab owns its Workload binding.

    Shape or arithmetic specialization belongs in the task-owned Python starter.
    The prepared Schedule carries the Lab's binding; the author does not copy
    its content hash into a decorator. Other metadata must still match.
    """
    original = parse_skeleton(source, filename=filename)
    if "program_id" in original:
        from open_cake_ir.compiler.ir import Program
        # Compare graph and every Schedule; only the existing metadata binding may differ.
        authored = Program.from_dict(original).document
        bound = Program.from_dict(prepared).document
        if len(authored["stages"]) != len(bound["stages"]):
            raise ValueError("Python Program starter stage count differs")
        for left, right in zip(authored["stages"], bound["stages"], strict=True):
            _check_schedule_binding(left["schedule"], right["schedule"])
            left["schedule"] = right["schedule"]
        if authored != bound:
            raise ValueError("Python Program starter must already match the prepared Program")
        return source.encode("utf-8")
    _check_schedule_binding(original, prepared)
    return source.encode("utf-8")


def _check_schedule_binding(original, prepared):
    if {k: v for k, v in original.items() if k != "metadata"} != {
        k: v for k, v in prepared.items() if k != "metadata"
    }:
        raise ValueError("Python starter must already match the prepared Schedule; only metadata may be bound")
    authored_metadata = original["metadata"]
    prepared_metadata = prepared["metadata"]
    if (not isinstance(prepared_metadata, Mapping)
            or {key: value for key, value in authored_metadata.items()
                if key != "workload_contract_sha256"}
            != {key: value for key, value in prepared_metadata.items()
                if key != "workload_contract_sha256"}
            or ("workload_contract_sha256" in authored_metadata
                and authored_metadata["workload_contract_sha256"]
                != prepared_metadata.get("workload_contract_sha256"))):
        raise ValueError("Python starter metadata differs from the prepared Schedule")


def read_skeleton_reference(project_root, reference, context='Schedule skeleton'):
    """Read and seal the actual delivered source against its frozen reference."""
    import json
    from hashlib import sha256
    from .bindings import source_reference_path
    from ._documents import _object, _digest, _canonical_json_bytes, differs

    reference = _object(reference, context)
    if set(reference) != {'path', 'canonical_sha256'}:
        raise ValueError(f'{context} reference fields differ')
    _, path = source_reference_path(project_root, reference['path'], context)
    payload = path.read_bytes()
    document = (parse_skeleton(payload.decode('utf-8'), filename=str(path))
                if path.suffix == '.py' else json.loads(payload))
    expected = _digest(reference['canonical_sha256'], context)
    observed = sha256(_canonical_json_bytes(document)).hexdigest()
    if expected != observed:
        raise differs(f'{context} bytes differ', expected=expected, observed=observed)
    return payload, document
