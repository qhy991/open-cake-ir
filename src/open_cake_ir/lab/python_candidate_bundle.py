"""Project ordered Cake Python author source into complete candidate actions.

This is a nonexecuting author transport parser, not another Schedule frontend.
Each decorated function goes through the existing Compiler frontend when the Lab
builds it. A static transform declaration goes through the existing action owner.
"""
from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping

from open_cake_ir.serialization import canonical_json_bytes


def _json_literal(node: ast.AST):
    def reject_duplicate_keys(item: ast.AST):
        if isinstance(item, ast.Dict):
            seen = set()
            for key, value in zip(item.keys, item.values, strict=True):
                if key is None:
                    raise ValueError('Python transform parameters cannot unpack mappings')
                try:
                    name = ast.literal_eval(key)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
                    raise ValueError('Python transform parameter keys must be static strings') from error
                if not isinstance(name, str) or name in seen:
                    raise ValueError('Python transform parameter keys must be unique strings')
                seen.add(name)
                reject_duplicate_keys(value)
        elif isinstance(item, (ast.List, ast.Tuple)):
            for value in item.elts:
                reject_duplicate_keys(value)

    reject_duplicate_keys(node)
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise ValueError('Python transform arguments must be static literals') from error

    def plain(item):
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('Python transform parameter keys must be strings')
            return {key: plain(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(value) for value in item]
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        raise ValueError('Python transform parameters must be JSON-compatible literals')

    return plain(value)


def project_python_candidate_bundle(payload: bytes, *, maximum_candidates_per_turn: int) -> tuple[bytes, ...]:
    if type(payload) is not bytes or not payload:
        raise ValueError('Python candidate bundle must be nonempty UTF-8 bytes')
    try:
        source = payload.decode('utf-8')
        tree = ast.parse(source, filename='candidate-set.py')
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError('Python candidate bundle is not valid UTF-8 Python') from error
    body = tree.body
    first = body[0] if body else None
    if (not isinstance(first, ast.ImportFrom) or first.module != 'open_cake_ir.compiler'
        or first.level != 0 or len(first.names) != 1
        or first.names[0].name != 'frontend' or first.names[0].asname != 'cake'):
        raise ValueError('Python candidate bundle requires the Cake frontend import')
    lines = source.split('\n')
    program_nodes = [node for node in body[1:]
                     if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Attribute)
                         and isinstance(node.value.func.value, ast.Name)
                         and node.value.func.value.id == 'cake'
                         and node.value.func.attr == 'program')]
    program_source = None
    referenced_stages = set()
    program_ids = {}
    if program_nodes:
        from open_cake_ir.compiler.program_frontend import parse_program
        sanitized = list(lines)
        for node in body[1:]:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'cake'
                and node.value.func.attr == 'transform'):
                sanitized[node.lineno - 1:node.end_lineno] = [''] * (node.end_lineno - node.lineno + 1)
        program_source = '\n'.join(sanitized)
        for node in program_nodes:
            fields = {key.arg: key.value for key in node.value.keywords if key.arg is not None}
            if (node.value.args or len(fields) != len(node.value.keywords)
                or 'program_id' not in fields):
                raise ValueError('Python Program declaration fields differ')
            program_id = _json_literal(fields['program_id'])
            if not isinstance(program_id, str) or program_id in program_ids:
                raise ValueError('Python Program ids must be unique strings')
            parse_program(program_source, filename='candidate-set.py', program_id=program_id)
            program_ids[id(node)] = program_id
            for call in ast.walk(node.value):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'cake' and call.func.attr == 'stage'):
                    referenced_stages.update(
                        key.value.id for key in call.keywords
                        if key.arg == 'schedule' and isinstance(key.value, ast.Name))
    candidates = []
    names = set()
    for node in body[1:]:
        if isinstance(node, ast.FunctionDef):
            decorator = node.decorator_list
            if (node.name in names or len(decorator) != 1
                or not isinstance(decorator[0], ast.Call)
                or not isinstance(decorator[0].func, ast.Attribute)
                or not isinstance(decorator[0].func.value, ast.Name)
                or decorator[0].func.value.id != 'cake'
                or decorator[0].func.attr != 'schedule'):
                raise ValueError('Python candidate function must have one Cake schedule decorator')
            names.add(node.name)
            if node.name in referenced_stages:
                continue
            from open_cake_ir.compiler.frontend import schedule_function_source
            projected_source = schedule_function_source(source, node)
            candidates.append(canonical_json_bytes({'python_source': projected_source}))
        elif id(node) in program_ids:
            candidates.append(canonical_json_bytes({
                'python_program_source': program_source, 'program_id': program_ids[id(node)]}))
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == 'cake'
              and node.value.func.attr == 'transform'):
            call = node.value
            if (call.args or len(call.keywords) != 3 or any(key.arg is None for key in call.keywords)
                or {key.arg for key in call.keywords} != {'parent', 'transformation', 'parameters'}):
                raise ValueError('Python transform declaration fields differ')
            values = {key.arg: _json_literal(key.value) for key in call.keywords}
            if (not isinstance(values['parent'], str) or not values['parent']
                or not isinstance(values['transformation'], str) or not values['transformation']
                or not isinstance(values['parameters'], dict)):
                raise ValueError('Python transform declaration values differ')
            candidates.append(canonical_json_bytes({'action': 'transform', **values}))
        else:
            raise ValueError('Python candidate bundle admits only Schedules and static transforms')
        if len(candidates) > maximum_candidates_per_turn:
            raise ValueError('Python candidate bundle exceeds the Turn budget')
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Python candidate bundle is empty or duplicates a candidate')
    return tuple(candidates)
