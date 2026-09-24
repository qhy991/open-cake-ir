"""Nonexecuting Python authoring for ordered, complete Cake Programs."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .frontend import parse as parse_schedule, schedule_function_source
from .ir import Program
from .ir.vocabulary import MemorySpace


@dataclass(frozen=True)
class ProgramSource:
    program: Program

    @property
    def document(self):
        return self.program.document


def _call(node, name: str, fields: set[str]):
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
        or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'cake'
        or node.func.attr != name or node.args or len(node.keywords) != len(fields)
        or any(key.arg is None for key in node.keywords)
        or {key.arg for key in node.keywords} != fields):
        raise ValueError(f'Python Program {name} declaration differs')
    return {key.arg: key.value for key in node.keywords}


def _literal(node):
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ValueError('Python Program declarations cannot unpack mappings')
            name = _literal(key)
            if not isinstance(name, str) or name in result:
                raise ValueError('Python Program keys must be unique strings')
            result[name] = _literal(value)
        return result
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool, type(None))):
        return node.value
    raise ValueError('Python Program declarations require static literals')


def _schedule_source(source: str, node: ast.FunctionDef) -> str:
    if len(node.decorator_list) != 1:
        raise ValueError('Python Program stage needs one Cake schedule decorator')
    decorator = node.decorator_list[0]
    if (not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute)
        or not isinstance(decorator.func.value, ast.Name)
        or decorator.func.value.id != 'cake' or decorator.func.attr != 'schedule'):
        raise ValueError('Python Program stage needs one Cake schedule decorator')
    return schedule_function_source(source, node)


def _binding(node):
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'cake' and node.func.attr == 'singleton_view'
            and len(node.args) == 1 and not node.keywords):
            tensor = _literal(node.args[0])
        else:
            arguments = _call(node, 'singleton_view', {'tensor'})
            tensor = _literal(arguments['tensor'])
        if not isinstance(tensor, str):
            raise ValueError('Python Program singleton view tensor must be text')
        return {'tensor': tensor, 'view': 'singleton_axes'}
    tensor = _literal(node)
    if not isinstance(tensor, str):
        raise ValueError('Python Program binding tensor must be text')
    return tensor


def _program_document(source: str, declaration: ast.Call,
                      functions: dict[str, ast.FunctionDef], filename: str):
    required = {'program_id', 'inputs', 'outputs', 'stages'}
    declared = {key.arg for key in declaration.keywords}
    fields = _call(declaration, 'program', required | ({'tensors'} if 'tensors' in declared else set()))
    program_id = _literal(fields['program_id'])
    inputs = _literal(fields['inputs'])
    outputs = _literal(fields['outputs'])
    if (not isinstance(program_id, str) or not program_id
        or not isinstance(inputs, list) or not isinstance(outputs, list)
        or any(not isinstance(name, str) for name in inputs + outputs)):
        raise ValueError('Python Program identity or public tensors differ')
    raw_stages = fields['stages']
    if not isinstance(raw_stages, (ast.Tuple, ast.List)) or not raw_stages.elts:
        raise ValueError('Python Program requires ordered stages')
    tensors = {}
    if 'tensors' in fields:
        raw_tensors = fields['tensors']
        if not isinstance(raw_tensors, ast.Dict):
            raise ValueError('Python Program explicit tensors must be a mapping')
        for name_node, spec_node in zip(raw_tensors.keys, raw_tensors.values, strict=True):
            if name_node is None:
                raise ValueError('Python Program tensors cannot unpack mappings')
            name = _literal(name_node)
            if not isinstance(name, str) or name in tensors:
                raise ValueError('Python Program tensor names must be unique strings')
            if (not isinstance(spec_node, ast.Call) or not isinstance(spec_node.func, ast.Attribute)
                or not isinstance(spec_node.func.value, ast.Name)
                or spec_node.func.value.id != 'cake' or spec_node.func.attr != 'Tensor'
                or len(spec_node.args) != 2 or spec_node.keywords):
                raise ValueError('Python Program explicit tensor needs cake.Tensor(shape, dtype)')
            shape, dtype = (_literal(item) for item in spec_node.args)
            if (not isinstance(shape, list) or not shape
                or any(type(extent) is not int or extent <= 0 for extent in shape)
                or not isinstance(dtype, str)):
                raise ValueError('Python Program explicit tensor shape or dtype differs')
            tensors[name] = {'shape': shape, 'dtype': dtype}
    stages = []
    targets = set()
    for raw_stage in raw_stages.elts:
        stage_fields = _call(raw_stage, 'stage', {'name', 'schedule', 'bindings'})
        name = _literal(stage_fields['name'])
        schedule_name = stage_fields['schedule']
        bindings_node = stage_fields['bindings']
        if (not isinstance(name, str) or not name
            or not isinstance(schedule_name, ast.Name) or schedule_name.id not in functions
            or not isinstance(bindings_node, ast.Dict)):
            raise ValueError('Python Program stage name, Schedule or bindings differ')
        bindings = {}
        for local_node, tensor_node in zip(bindings_node.keys, bindings_node.values, strict=True):
            if local_node is None:
                raise ValueError('Python Program bindings cannot unpack mappings')
            local = _literal(local_node)
            if not isinstance(local, str) or local in bindings:
                raise ValueError('Python Program stage binding names must be unique strings')
            bindings[local] = _binding(tensor_node)
        schedule = parse_schedule(_schedule_source(source, functions[schedule_name.id]),
                                  filename=filename).document
        targets.add(schedule['target'])
        for buffer in schedule['buffers']:
            if buffer['space'] != MemorySpace.GLOBAL.value:
                continue
            binding = bindings.get(buffer['name'])
            if isinstance(binding, str):
                tensors.setdefault(binding, {'shape': list(buffer['shape']),
                                             'dtype': buffer['dtype']})
        stages.append({'name': name, 'schedule': schedule, 'bindings': bindings})
    if len(targets) != 1:
        raise ValueError('Python Program stages require one exact Target')
    return {'schema_version': 1, 'program_id': program_id, 'target': targets.pop(),
            'inputs': inputs, 'outputs': outputs, 'tensors': tensors, 'stages': stages}


def parse_program(source: str, *, filename: str = 'candidate-set.py',
                  program_id: str | None = None) -> ProgramSource:
    """Parse static composition and reuse Program.from_dict for SSA/ABI legality."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise ValueError(f'{filename}:{error.lineno}: Python Program syntax differs') from error
    first = tree.body[0] if tree.body else None
    if (not isinstance(first, ast.ImportFrom) or first.module != 'open_cake_ir.compiler'
        or first.level != 0 or len(first.names) != 1
        or first.names[0].name != 'frontend' or first.names[0].asname != 'cake'):
        raise ValueError('Python Program requires the Cake frontend import')
    functions = {}
    declarations = []
    for node in tree.body[1:]:
        if isinstance(node, ast.FunctionDef):
            if node.name in functions:
                raise ValueError('Python Program stage functions must be unique')
            functions[node.name] = node
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == 'cake'
              and node.value.func.attr == 'program'):
            declarations.append(node.value)
        else:
            raise ValueError('Python Program source has unsupported top-level code')
    documents = [_program_document(source, declaration, functions, filename)
                 for declaration in declarations]
    selected = [document for document in documents
                if program_id is None or document['program_id'] == program_id]
    if len(selected) != 1:
        raise ValueError('Python Program source must select exactly one declaration')
    return ProgramSource(Program.from_dict(selected[0]))


def read_program(path: str | Path, *, program_id: str | None = None) -> ProgramSource:
    path = Path(path).resolve(strict=True)
    return parse_program(path.read_text(encoding='utf-8'), filename=str(path),
                         program_id=program_id)
