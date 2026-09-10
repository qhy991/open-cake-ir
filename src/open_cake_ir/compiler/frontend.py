"""A restricted Python authoring surface for the existing canonical Schedule.

Reading a source file never imports or executes it. The AST is elaborated into the
same document consumed by Schedule.from_dict and Compiler.assess; locations are a
separate presentation projection, not another semantic representation.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .ir import (
    Allocation, Barrier, Buffer, ElementwiseOp, OperationKind, Pipeline,
    ProgramAxis, Role, Schedule, ScheduleParseError,
)

from .ir.operations import elementwise_result_dtype


@dataclass(frozen=True)
class SourceLocation:
    filename: str
    line: int
    column: int
    end_line: int
    end_column: int


class FrontendError(ValueError):
    """An unsupported source construct or a localized Schedule construction error."""

    def __init__(self, message: str, location: SourceLocation, code: str = "PYTHON_SYNTAX", *, canonical_path: str | None = None):
        self.location = location
        self.code = code
        self.canonical_path = canonical_path
        super().__init__(f"{location.filename}:{location.line}:{location.column}: {message}")


@dataclass(frozen=True)
class Tensor:
    """Function-argument annotation for the optional Python decorator entry point."""

    shape: tuple[int, ...]
    dtype: str
    mode: str = "input"


@dataclass(frozen=True)
class ScheduleSource:
    schedule_bytes: bytes
    locations: Mapping[str, SourceLocation]

    @property
    def document(self) -> dict[str, Any]:
        """Return a fresh projection; editing it cannot invalidate source locations."""
        return json.loads(self.schedule_bytes)

    def location_for(self, path: str) -> SourceLocation | None:
        path = path.removeprefix("schedule.")
        if path in self.locations:
            return self.locations[path]
        elements = [key for key in self.locations if re.fullmatch(re.escape(path) + r"\[\d+\]", key)]
        if elements:
            return self.locations[max(elements, key=lambda key: int(key.rsplit("[", 1)[1][:-1]))]
        matches = [
            key for key in self.locations
            if not key or path == key or path.startswith(key + ".") or path.startswith(key + "[")
        ]
        return self.locations[max(matches, key=len)] if matches else None


def read_schedule(path: str | Path) -> ScheduleSource:
    """Read JSON or elaborate Python without executing authored code."""
    path = Path(path).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return parse(source, filename=str(path))
    return ScheduleSource(source.encode("utf-8"), MappingProxyType({}))


def schedule(**options):
    """Compile a module-level decorated function without running its body.

    File-based consumers should use read_schedule: it also refuses executable
    module-level statements. Normal Python imports retain normal Python semantics.
    """
    def decorate(function):
        lines, first_line = inspect.getsourcelines(function)
        return parse(
            textwrap.dedent("".join(lines)),
            filename=inspect.getsourcefile(function) or "<python>",
            line_offset=first_line - 1,
        )
    return decorate


def _encode(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class _Ref:
    collection: str
    name: str


@dataclass(frozen=True)
class _Access:
    buffer: _Ref
    indices: list[dict[str, Any]]


@dataclass(frozen=True)
class _Broadcast:
    buffer: _Ref
    axis: int


_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
_MATH = {op.value for op in ElementwiseOp}
_OPERATIONS = {op.value for op in OperationKind} - {"elementwise"}
_DECLARATIONS = dict(roles=Role, allocations=Allocation, buffers=Buffer,
                     pipelines=Pipeline, barriers=Barrier)
_RANGE_DEFAULTS = dict(num_stages=1, loop_unroll_factor=1, flatten=False,
                       warp_specialize=False, disallow_acc_multi_buffer=False, disable_licm=False)


class _Builder:
    def __init__(self, source: str, filename: str, line_offset: int):
        self.source = source
        self.lines = source.splitlines()
        self.filename = filename
        self.line_offset = line_offset
        self.locations: dict[str, SourceLocation] = {}
        self.symbols: dict[str, Any] = {}
        self.role: str | None = None
        self.loops: list[dict[str, Any]] = []
        self.writers: dict[str, str] = {}
        self.readers: dict[str, list[str]] = {}
        self.ancestors: dict[str, set[str]] = {}
        self.temporary = 0
        self.lm = "lm"
        self.document: dict[str, Any] = dict(
            schema_version=1, roles=[], allocations=[], buffers=[], pipelines=[],
            barriers=[], tile_loops=[], access_maps=[], operations=[], outputs=[], metadata={},
        )

    def location(self, node: ast.AST) -> SourceLocation:
        line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", line)
        def column(row, offset):
            raw = self.lines[row - 1].encode("utf-8") if 0 < row <= len(self.lines) else b""
            return len(raw[:offset].decode("utf-8")) + 1
        return SourceLocation(self.filename, line + self.line_offset,
                              column(line, getattr(node, "col_offset", 0)),
                              end_line + self.line_offset,
                              column(end_line, getattr(node, "end_col_offset", 0)))

    def fail(self, node, message, code="PYTHON_SYNTAX", *, canonical_path=None):
        raise FrontendError(message, self.location(node), code, canonical_path=canonical_path)

    def mark(self, path, node):
        self.locations[path] = self.location(node)

    def literal(self, node):
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None or isinstance(value, (str, bool, int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    self.fail(node, "constants must be finite")
                if isinstance(value, str):
                    try:
                        value.encode("utf-8")
                    except UnicodeEncodeError:
                        self.fail(node, "string literals must be UTF-8 encodable")
                return value
        if isinstance(node, ast.Name) and node.id in self.symbols:
            value = self.symbols[node.id]
            return value.name if isinstance(value, _Ref) else value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.literal(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    self.fail(node, "dictionary unpacking is not supported")
                key = self.literal(key)
                if not isinstance(key, str) or key in result:
                    self.fail(node, "dictionary keys must be unique strings")
                result[key] = self.literal(value)
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.literal(node.operand)
            if type(value) in (int, float):
                return -value if isinstance(node.op, ast.USub) else value
        self.fail(node, "expected a literal or a previously declared name")

    def keywords(self, call, *, deferred=()):
        result = {}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in result:
                self.fail(keyword, "keyword unpacking and duplicate keywords are not supported")
            result[keyword.arg] = keyword.value if keyword.arg in deferred else self.literal(keyword.value)
        return result

    def named_call(self, node, owner, method):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == owner and node.func.attr == method)

    def reference(self, value, node, collection="buffers"):
        if not isinstance(value, _Ref) or value.collection != collection:
            self.fail(node, f"expected a declared {collection} name; indexed views cannot be used here")
        return value

    def record(self, ref):
        if ref.collection == "program":
            items = self.document["program_map"]["axes"]
        elif ref.collection == "loop":
            return next(item for item in self.document["tile_loops"] if item["iterator"] == ref.name)
        else:
            items = self.document[ref.collection]
        return next(item for item in items if item["name"] == ref.name)

    def buffer(self, value, node):
        ref = self.reference(value, node)
        index = next(i for i, b in enumerate(self.document["buffers"]) if b["name"] == ref.name)
        return Buffer.from_dict(self.record(ref), f"schedule.buffers[{index}]")

    def declare(self, collection, name, fields, node):
        if name is None or name in self.symbols:
            self.fail(node, "declarations and results require a new, single-assignment variable")
        if "name" in fields:
            self.fail(node, "the assignment owns the declaration name")
        index = len(self.document[collection])
        self.document[collection].append(dict(name=name, **fields))
        self.mark(f"{collection}[{index}]", node)
        _DECLARATIONS[collection].from_dict(self.document[collection][-1], f"schedule.{collection}[{index}]")
        ref = self.symbols[name] = _Ref(collection, name)
        if collection == "buffers" and fields.get("mode") == "output":
            self.mark(f"outputs[{len(self.document['outputs'])}]", node)
            self.document["outputs"].append(name)
        return ref

    def fresh(self):
        while True:
            name = f"temporary_{self.temporary}"
            self.temporary += 1
            if name not in self.symbols:
                return name

    def access(self, node):
        ref = self.reference(self.value(node.value), node)
        buffer = self.buffer(ref, node)
        components = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(components) != len(buffer.shape):
            self.fail(node, "indexing must name every buffer dimension")
        indices = []
        index_shape = None
        for dimension, component in enumerate(components):
            if isinstance(component, ast.Slice):
                if component.step is not None:
                    self.fail(component, "strided slices are not in the current AccessMap contract")
                start = 0 if component.lower is None else self.literal(component.lower)
                stop = buffer.shape[dimension] if component.upper is None else self.literal(component.upper)
                if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= buffer.shape[dimension]:
                    self.fail(component, "slice bounds must form a nonempty static subrange")
                item = dict(source="dimension", dimension=dimension)
                if start:
                    item["offset"] = start
                if stop != buffer.shape[dimension]:
                    item["extent"] = stop - start
            else:
                index = self.value(component)
                if isinstance(index, _Ref) and index.collection == "buffers":
                    index_buffer = self.buffer(index, component)
                    if (index_buffer.dtype.value != "int32" or index_buffer.space.value != "register"
                        or len(index_buffer.shape) != 1):
                        self.fail(component, "gather indices require rank-one INT32 register buffers")
                    if buffer.space.value != "global":
                        self.fail(node, "buffer-indexed loads require global source storage")
                    if index_shape is not None and index_buffer.shape != index_shape:
                        self.fail(component, "gather index buffers require the same zipped shape")
                    index_shape = index_buffer.shape
                    indices.append(dict(source="buffer", name=index.name))
                    continue
                if not isinstance(index, _Ref) or index.collection not in {"program", "loop"}:
                    self.fail(component, "indices must be program coordinates, loop tiles or static slices")
                source = "loop_tile" if index.collection == "loop" else (
                    "program" if self.record(index)["tile"] == 1 else "program_tile")
                item = dict(source=source, name=index.name)
            indices.append(item)
        if any(item['source'] == 'buffer' for item in indices):
            tiled = [(item['source'], item['name']) for item in indices
                     if item['source'] in {'program_tile', 'loop_tile'}]
            if len(tiled) != len(set(tiled)):
                self.fail(node, "buffer-indexed loads cannot repeat a tiled coordinate")
        return _Access(ref, indices)

    def load_shape(self, access, node):
        if not isinstance(access, _Access):
            self.fail(node, "a load without an explicit destination requires indexed source coordinates")
        source = self.buffer(access.buffer, node)
        shape = []
        index_shape = None
        for item in access.indices:
            if item["source"] == "buffer":
                current = self.buffer(self.symbols[item["name"]], node).shape
                if index_shape is None:
                    index_shape = current
                    shape.extend(current)
            elif item["source"] == "dimension":
                shape.append(item.get("extent", source.shape[item["dimension"]] - item.get("offset", 0)))
            elif item["source"] != "program":
                shape.append(self.record(self.symbols[item["name"]])["tile"])
        return shape or [1]

    def value(self, node, target=None):
        if isinstance(node, ast.Name):
            if node.id not in self.symbols:
                self.fail(node, f"unknown name {node.id!r}")
            return self.symbols[node.id]
        if isinstance(node, ast.Subscript):
            return self.access(node)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return self.operation("elementwise", [self.value(node.left), self.value(node.right)],
                                  {"op": _BINARY[type(node.op)]}, {}, target, node)
        if isinstance(node, ast.Call):
            return self.call(node, target)
        return self.literal(node)

    def call(self, node, target):
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            self.fail(node, "only declared Schedule operations can be called")
        owner, method = node.func.value.id, node.func.attr
        if owner != self.lm:
            allocation = self.symbols.get(owner)
            if method != "view" or not isinstance(allocation, _Ref) or allocation.collection != "allocations":
                self.fail(node, "only allocation.view is supported outside the Schedule namespace")
            if node.args:
                self.fail(node, "view uses named shape, dtype and storage commitments")
            fields = self.keywords(node)
            if "space" in fields or "allocation" in fields:
                self.fail(node, "a view's allocation owns its memory space and storage")
            fields.setdefault("mode", "scratch")
            fields["space"] = self.record(allocation)["space"]
            fields["allocation"] = allocation.name
            return self.declare("buffers", target, fields, node)
        if method in {"role", "pipeline", "barrier", "buffer", "smem", "tmem"}:
            fields = self.keywords(node)
            collection = {"role": "roles", "pipeline": "pipelines", "barrier": "barriers",
                          "buffer": "buffers", "smem": "allocations", "tmem": "allocations"}[method]
            if node.args:
                if method not in {"smem", "tmem"} or len(node.args) != 1 or "size_bytes" in fields:
                    self.fail(node, "resource declarations use named fields; memory pools take one byte extent")
                fields["size_bytes"] = self.literal(node.args[0])
            if method in {"smem", "tmem"}:
                if "space" in fields:
                    self.fail(node, "the memory declaration owns its space")
                fields["space"] = "shared" if method == "smem" else "tensor"
            if method == "buffer":
                fields.setdefault("space", "register")
                fields.setdefault("mode", "scratch")
            return self.declare(collection, target, fields, node)
        if method == "program":
            if target is None or target in self.symbols or len(node.args) != 1 or "grid" in self.document:
                self.fail(node, "program needs a fresh name and one buffer, and cannot coexist with grid")
            ref = self.reference(self.value(node.args[0]), node)
            fields = self.keywords(node)
            if "buffer" in fields or "name" in fields:
                self.fail(node, "program buffer and name are owned by its argument and assignment")
            axes = self.document.setdefault("program_map", {"axes": []})["axes"]
            self.mark(f"program_map.axes[{len(axes)}]", node)
            axes.append(dict(name=target, buffer=ref.name, **fields))
            ProgramAxis.from_dict(axes[-1], f"schedule.program_map.axes[{len(axes) - 1}]")
            result = self.symbols[target] = _Ref("program", target)
            return result
        if method == "broadcast":
            if len(node.args) != 1 or set(k.arg for k in node.keywords) != {"axis"}:
                self.fail(node, "broadcast takes one value and an explicit axis")
            ref = self.reference(self.value(node.args[0]), node)
            return _Broadcast(ref, self.keywords(node)["axis"])
        if method not in _MATH | _OPERATIONS:
            self.fail(node, f"unsupported Schedule operation {method!r}")
        fields = self.keywords(node, deferred={"out"})
        controls = {key: fields.pop(key) for key in ("out", "id", "waits", "signals", "pipeline", "depends_on") if key in fields}
        if "out" in controls:
            value = controls["out"]
            controls["out"] = ([self.value(item) for item in value.elts]
                               if isinstance(value, (ast.List, ast.Tuple)) else [self.value(value)])
        values = [self.value(arg) for arg in node.args]
        if method in _MATH:
            if "op" in fields:
                self.fail(node, "the called arithmetic primitive owns op")
            fields["op"] = method
            if method == "fma":
                fields.setdefault("instruction", {"contract": "ptx.fma.rn.f32"})
            kind = "elementwise"
        else:
            kind = method
        if kind == "store":
            if len(values) != 2 or "out" in controls:
                self.fail(node, "store takes destination coordinates followed by one value")
            controls["out"], values = [values[0]], [values[1]]
            fields.setdefault("coalesced", True)
        if kind == "load":
            fields.setdefault("movement", "global")
        if kind == "mma":
            fields.setdefault("accumulator", "fp32")
        if kind in {"reduce", "online_softmax"}:
            fields.setdefault("scope", "cta")
        return self.operation(kind, values, fields, controls, target, node)

    def operation(self, kind, values, parameters, controls, target, node):
        if self.role is None:
            self.fail(node, "operations require a with-role scope")
        reads, accesses = [], []
        for position, value in enumerate(values):
            if isinstance(value, _Broadcast):
                if kind != "elementwise" or position != 1 or "broadcast_axis" in parameters:
                    self.fail(node, "broadcast applies once to the second arithmetic operand")
                parameters["broadcast_axis"], value = value.axis, value.buffer
            if type(value) in (int, float):
                if kind != "elementwise" or position != 1 or len(values) != 2 or parameters["op"] == "fma" or "scalar" in parameters:
                    self.fail(node, "a literal is supported only as the second binary arithmetic operand")
                parameters["scalar"] = value
                continue
            ref = self.reference(value.buffer if isinstance(value, _Access) else value, node)
            if isinstance(value, _Access):
                accesses.append(value)
            reads.append(ref)
        if kind == "load":
            for access in accesses:
                for item in access.indices:
                    if item["source"] == "buffer":
                        index = self.symbols[item["name"]]
                        if index not in reads:
                            reads.append(index)
        outputs = controls.pop("out", None)
        if outputs is None:
            if not reads:
                self.fail(node, "a computed result requires an input value")
            first = self.buffer(reads[0], node)
            shape, dtype = list(first.shape), first.dtype.value
            if kind == "elementwise":
                operands = [self.buffer(ref, node) for ref in reads]
                promoted = elementwise_result_dtype(operand.dtype for operand in operands)
                if promoted is None:
                    self.fail(node, f"{parameters['op']} has no implicit dtype promotion for "
                        f"{[operand.dtype.value for operand in operands]}; use lm.cast(..., to=...) explicitly",
                        "ELEMENTWISE_DTYPE_UNSUPPORTED", canonical_path=f"operations[{len(self.document['operations'])}].reads")
                dtype = promoted.value
                non_scalar = [operand for operand in operands if not operand.is_scalar]
                shape = list(max((operand.shape for operand in (non_scalar or operands)), key=len))
            if kind == "load":
                shape = self.load_shape(values[0], node)
            elif kind == "reduce":
                axis = parameters.get("axis")
                if type(axis) is not int or not 0 <= axis < len(shape):
                    self.fail(node, "reduce requires a valid static axis")
                shape = shape[:axis] + shape[axis + 1:] or [1]
            elif kind == "cast":
                if "to" not in parameters:
                    self.fail(node, "lm.cast requires to=...", "SCHEDULE_STRUCTURE",
                              canonical_path=f"operations[{len(self.document['operations'])}].parameters")
                dtype = parameters["to"]
            elif kind == "mma":
                if len(reads) != 2 or len(first.shape) != 2:
                    self.fail(node, "automatic MMA results require two rank-two operands")
                right = self.buffer(reads[1], node)
                shape, dtype = [first.shape[0], right.shape[0]], "fp32"
            elif kind not in {"elementwise", "scan"}:
                self.fail(node, f"{kind} requires explicit result buffers via out")
            result_name = target or self.fresh()
            outputs = [self.declare("buffers", result_name,
                                   dict(space="register", dtype=dtype, shape=shape, mode="scratch"), node)]
        elif target is not None:
            self.fail(node, "an operation with out updates its declared buffer; do not rebind it")
        if not outputs:
            self.fail(node, "out requires at least one result buffer")
        writes = [self.reference(value.buffer if isinstance(value, _Access) else value, node) for value in outputs]
        accesses.extend(value for value in outputs if isinstance(value, _Access))
        if any(isinstance(value, _Access) and any(item['source'] == 'buffer' for item in value.indices)
               for value in outputs):
            self.fail(node, "buffer-indexed destinations are not supported by this frontend")
        if kind != 'load' and any(item['source'] == 'buffer' for access in accesses for item in access.indices):
            self.fail(node, "buffer-indexed views are supported only by load")
        op_id = controls.pop("id", target or f"{kind}_{writes[0].name}")
        if not isinstance(op_id, str) or op_id in self.ancestors:
            self.fail(node, "operation ids must be unique strings; use id for repeated destination writes")
        dependencies = []
        for ref in reads + writes:
            producer = self.writers.get(ref.name)
            if producer is not None and producer not in dependencies:
                dependencies.append(producer)
        for ref in writes:
            dependencies.extend(reader for reader in self.readers.get(ref.name, []) if reader not in dependencies)
        explicit = controls.pop("depends_on", [])
        if not isinstance(explicit, list) or any(not isinstance(item, str) or item not in self.ancestors for item in explicit):
            self.fail(node, "depends_on must name preceding operation ids")
        dependencies.extend(item for item in explicit if item not in dependencies)
        # Keep only immediate data/effect prerequisites, as in the canonical corpus.
        dependencies = [item for item in dependencies
                        if not any(item in self.ancestors[other] for other in dependencies if item != other)]
        operation = dict(id=op_id, kind=kind, role=self.role,
                         reads=[ref.name for ref in reads], writes=[ref.name for ref in writes], parameters=parameters)
        operation.update(controls)
        if dependencies:
            operation["depends_on"] = dependencies
        index = len(self.document["operations"])
        self.document["operations"].append(operation)
        self.mark(f"operations[{index}]", node)
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                part = keyword.arg if keyword.arg in controls else "parameters." + str(keyword.arg)
                self.mark(f"operations[{index}].{part}", keyword.value)
        for access in accesses:
            self.mark(f"access_maps[{len(self.document['access_maps'])}]", node)
            self.document["access_maps"].append(dict(operation=op_id, buffer=access.buffer.name,
                                                       indices=access.indices, boundary="mask_tiled_axes"))
        if self.loops:
            self.loops[-1]["body"].append(op_id)
        self.ancestors[op_id] = set(dependencies).union(*(self.ancestors[d] for d in dependencies))
        for ref in writes:
            self.writers[ref.name] = op_id
            self.readers[ref.name] = []
        for ref in reads:
            if ref not in writes:
                self.readers.setdefault(ref.name, []).append(op_id)
        return writes[0] if len(writes) == 1 else tuple(writes)

    def statements(self, statements):
        for node in statements:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    self.fail(node, "assignments bind one new variable")
                name = node.targets[0].id
                if name in self.symbols or name == self.lm:
                    self.fail(node, "variables are single-assignment; use an explicit out buffer for effects")
                result = self.value(node.value, name)
                if name not in self.symbols:
                    if isinstance(result, (_Ref, _Access, _Broadcast)):
                        self.fail(node, "assign an operation result or a declaration, not an alias")
                    self.symbols[name] = result
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.value(node.value)
            elif isinstance(node, ast.With):
                if len(node.items) != 1 or node.items[0].optional_vars is not None or self.role is not None:
                    self.fail(node, "with selects one role, without nested roles or an as binding")
                role = self.reference(self.value(node.items[0].context_expr), node, "roles")
                self.role = role.name
                self.statements(node.body)
                self.role = None
            elif isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name) or not self.named_call(node.iter, self.lm, "range") or node.orelse:
                    self.fail(node, "for must bind one tile from lm.range, without else")
                iterator = node.target.id
                if iterator in self.symbols or len(node.iter.args) != 1:
                    self.fail(node, "range needs a fresh iterator and one buffer")
                ref = self.reference(self.value(node.iter.args[0]), node)
                fields = self.keywords(node.iter)
                name = fields.pop("name", "loop_" + iterator)
                if any(loop["name"] == name for loop in self.document["tile_loops"]):
                    self.fail(node, "loop names must be unique")
                if "dimension" not in fields or "tile" not in fields:
                    self.fail(node, "range requires dimension and tile")
                loop = dict(name=name, iterator=iterator, buffer=ref.name,
                            dimension=fields.pop("dimension"), tile=fields.pop("tile"), body=[])
                if (type(loop["tile"]) is not int or loop["tile"] <= 0
                        or type(loop["dimension"]) is not int or loop["dimension"] < 0):
                    self.fail(node.iter, "range requires a positive integer tile and nonnegative integer dimension")
                if "stop" in fields:
                    loop["stop"] = fields.pop("stop")
                loop["range_options"] = dict(_RANGE_DEFAULTS, **fields)
                self.mark(f"tile_loops[{len(self.document['tile_loops'])}]", node)
                self.document["tile_loops"].append(loop)
                if self.loops:
                    self.loops[-1]["body"].append(name)
                self.symbols[iterator] = _Ref("loop", iterator)
                self.loops.append(loop)
                self.statements(node.body)
                self.loops.pop()
                del self.symbols[iterator]
            else:
                self.fail(node, f"unsupported statement {type(node).__name__}")

    def build(self, tree):
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "open_cake_ir.compiler" and node.level == 0:
                if [(item.name, item.asname) for item in node.names] != [("frontend", "cake")]:
                    self.fail(node, "use from open_cake_ir.compiler import frontend as cake")
            elif (isinstance(node, ast.ImportFrom) and node.module == "__future__" and node.level == 0
                  and [(a.name, a.asname) for a in node.names] == [("annotations", None)]):
                continue
            elif isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                self.fail(node, "a source file contains only the frontend import and one schedule function")
        if len(functions) != 1:
            self.fail(tree, "a source file must define exactly one schedule function")
        function = functions[0]
        self.mark("", function)
        if len(function.decorator_list) != 1 or not self.named_call(function.decorator_list[0], "cake", "schedule"):
            self.fail(function, "the function needs exactly one @cake.schedule(...) decorator")
        decorator = function.decorator_list[0]
        if decorator.args:
            self.fail(decorator, "schedule options must be named")
        options = self.keywords(decorator)
        spellings = {"name": "schedule_id", "target": "target", "backend": "lowering.backend",
                     "entry_point": "lowering.entry_point"}
        for keyword in decorator.keywords:
            if keyword.arg in spellings:
                self.mark(spellings[keyword.arg], keyword)
        if "target" not in options or "backend" not in options:
            self.fail(decorator, "schedule requires an exact target and backend")
        self.document.update(schedule_id=options.pop("name", function.name), target=options.pop("target"),
                             lowering=dict(backend=options.pop("backend"), entry_point=options.pop("entry_point", function.name)))
        for key in ("grid", "residency", "metadata"):
            if key in options:
                self.document[key] = options.pop(key)
                self.mark(key, decorator)
        if options:
            self.fail(decorator, f"unknown schedule options: {', '.join(options)}")
        args = function.args
        if (not args.args or args.posonlyargs or args.kwonlyargs or args.defaults
                or args.vararg or args.kwarg or function.returns is not None
                or getattr(function, "type_params", [])):
            self.fail(function, "use an unannotated Schedule context followed by annotated tensor arguments")
        self.lm = args.args[0].arg
        if args.args[0].annotation is not None:
            self.fail(args.args[0], "the Schedule context is not a tensor argument")
        for arg in args.args[1:]:
            if arg.arg == self.lm:
                self.fail(arg, "the Schedule context cannot also name a tensor parameter")
            annotation = arg.annotation
            if not self.named_call(annotation, "cake", "Tensor") or len(annotation.args) != 2:
                self.fail(arg, "tensor arguments use cake.Tensor(shape, dtype, mode=...)")
            fields = self.keywords(annotation)
            if "shape" in fields or "dtype" in fields or "space" in fields:
                self.fail(annotation, "Tensor takes positional shape and dtype and always declares global memory")
            fields.setdefault("mode", "input")
            fields.update(shape=self.literal(annotation.args[0]), dtype=self.literal(annotation.args[1]), space="global")
            self.declare("buffers", arg.arg, fields, arg)
        self.statements(function.body)
        if "grid" in self.document and not self.document["access_maps"]:
            del self.document["access_maps"]
        typed = Schedule.from_dict(self.document)
        return ScheduleSource(_encode(typed.canonical_document(self.document)), MappingProxyType(dict(self.locations)))


def parse(source: str, *, filename: str = "<python>", line_offset: int = 0) -> ScheduleSource:
    """Elaborate one supported Python function; never eval, exec or import its text."""
    builder = _Builder(source, filename, line_offset)
    try:
        tree = ast.parse(source, filename=filename)
    except UnicodeEncodeError as error:
        raise FrontendError("source must be UTF-8 encodable",
                            SourceLocation(filename, 1 + line_offset, 1, 1 + line_offset, 1)) from error
    except SyntaxError as error:
        location = SourceLocation(filename, (error.lineno or 1) + line_offset, error.offset or 1,
                                  (error.end_lineno or error.lineno or 1) + line_offset,
                                  error.end_offset or error.offset or 1)
        raise FrontendError(error.msg, location) from error
    try:
        return builder.build(tree)
    except ScheduleParseError as error:
        source = ScheduleSource(b"{}", builder.locations)
        # Structural parser messages start with their canonical schedule path.
        path = str(error).split(" ", 1)[0]
        location = source.location_for(path) or builder.location(tree)
        normalized = path.removeprefix("schedule.")
        spelling = {"target": "@cake.schedule(target=...)", "lowering.backend": "@cake.schedule(backend=...)",
                    "lowering.entry_point": "@cake.schedule(entry_point=...)", "schedule_id": "@cake.schedule(name=...)"}.get(normalized)
        match = re.match(r"(buffers|roles|allocations|pipelines|barriers|operations|outputs|tile_loops)\[(\d+)\](.*)", normalized)
        if spelling is None and match:
            collection, index, suffix = match.groups()
            items = builder.document[collection]
            if int(index) < len(items):
                item = items[int(index)]
                if collection == "operations":
                    method = item["parameters"].get("op", "elementwise") if item["kind"] == "elementwise" else item["kind"]
                    spelling = f"lm.{method}(...)" + suffix.removeprefix(".parameters")
                elif collection == "outputs":
                    spelling = item + suffix
                else:
                    spelling = item.get("iterator", item.get("name", collection)) + suffix
        message = (spelling or normalized) + str(error)[len(path):]
        raise FrontendError(message, location, "SCHEDULE_STRUCTURE", canonical_path=path) from error
