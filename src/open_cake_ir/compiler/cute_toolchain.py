"""Explicit-target CuTe compilation for ordered register-MMA and FP32 SIMT interfaces.

Importing this module needs no SDK. Native source admission is not a sandbox;
the Lab invokes compilation only inside its filesystem-isolated CPU worker.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Mapping

from .toolchain import _parse_cuobjdump_resources


@dataclass(frozen=True)
class CuTeCompilation:
    source: bytes
    target: str
    entry_point: str
    artifacts: Mapping[str, bytes]
    threads_per_cta: int
    dynamic_shared_bytes: int
    compiler_version: str


_IMPORTS = ("import cutlass", "import cutlass.cute as cute",
            "from cutlass.cute.nvgpu import warp")
_CUTE_CALLS = frozenset({
    "make_layout", "make_tensor", "local_tile", "make_identity_tensor", "make_tiled_mma",
    "make_rmem_tensor", "make_copy_atom", "copy", "gemm", "size", "rank", "cosize",
    "arch.thread_idx", "arch.block_idx", "nvgpu.CopyG2ROp", "nvgpu.CopyR2GOp",
    "arch.shuffle_sync", "arch.warp_reduction_sum", "arch.warp_reduction_max", "arch.fmax",
    "math.exp", "math.exp2", "math.rsqrt", "math.tanh",
})
_CUTE_VALUES = frozenset({
    "Pointer", "nvgpu.LoadCacheMode.ALWAYS", "nvgpu.LoadCacheMode.GLOBAL",
    "nvgpu.LoadCacheMode.STREAMING", "nvgpu.CacheEvictionPriority.EVICT_NORMAL",
    "nvgpu.CacheEvictionPriority.EVICT_FIRST", "nvgpu.CacheEvictionPriority.EVICT_LAST",
})
_CUTLASS_CALLS = frozenset({"range", "range_constexpr", "BFloat16", "Float32", "Boolean", "Int32"})
_METHODS = frozenset({"get_slice", "partition_A", "partition_B", "partition_C",
                      "make_fragment_A", "make_fragment_B", "fill", "load", "store"})
_PROPERTIES = frozenset({"shape", "layout", "element_type"})
_RESERVED = frozenset({"cutlass", "cute", "warp"})


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def validate_cute_requirements(requirements: Mapping[str, object]) -> None:
    """One closed ordered-pointer compile contract; options are host-owned."""
    fields = {"compiler", "source_language", "target", "kernel_entry_point", "signature",
              "grid", "block", "dynamic_shared_memory_bytes"}
    if not isinstance(requirements, Mapping) or set(requirements) != fields:
        raise ValueError("CuTe compile requirement fields differ")
    if (requirements["compiler"] != "cutlass_cute_dsl"
        or requirements["source_language"] != "python" or requirements["target"] not in {"sm_100a", "sm_103a"}):
        raise ValueError("CuTe compilation requires an exact supported CUDA target and backend")
    name = requirements["kernel_entry_point"]
    if (not isinstance(name, str) or not name.isidentifier() or "__" in name
        or name in _RESERVED or name == "open_cake_cute_launch"):
        raise ValueError("CuTe kernel entry point differs")
    signature = requirements["signature"]
    if not isinstance(signature, list) or not signature:
        raise ValueError("CuTe requires a nonempty ordered pointer signature")
    names = []
    for row in signature:
        if (not isinstance(row, Mapping) or set(row) != {"name", "dtype"}
            or not isinstance(row["name"], str) or not row["name"].isidentifier()
            or "__" in row["name"] or row["name"] in _RESERVED
            or row["name"] in {name, "open_cake_cute_launch"}
            or row["dtype"] not in {"bf16", "fp32"}):
            raise ValueError("CuTe pointer signature row differs")
        names.append(row["name"])
    simt = all(row["dtype"] == "fp32" for row in signature)
    register = (requirements["target"] == "sm_103a" and len(signature) == 4
                and sorted(row["dtype"] for row in signature[:3]) == ["bf16", "bf16", "fp32"]
                and signature[-1]["dtype"] == "fp32")
    if len(set(names)) != len(names) or not (simt or register):
        raise ValueError("CuTe pointer signature order or types differ")
    grid = requirements["grid"]
    if (not isinstance(grid, list) or len(grid) != 3
        or any(type(value) is not int or value <= 0 for value in grid)
        or requirements["block"] != [32, 1, 1]
        or any(type(value) is not int for value in requirements["block"])
        or type(requirements["dynamic_shared_memory_bytes"]) is not int
        or requirements["dynamic_shared_memory_bytes"] != 0):
        raise ValueError("CuTe launch requires one warp and zero dynamic shared memory")


def validate_cute_kernel(source: bytes, requirements: Mapping[str, object]) -> None:
    """Admit the entire single-kernel module before evaluating any source."""
    validate_cute_requirements(requirements)
    try:
        tree = ast.parse(source.decode("utf-8"), filename="candidate.cute.py")
    except (UnicodeError, SyntaxError) as error:
        raise ValueError(f"native CuTe syntax: {error}") from error
    imports = [ast.dump(ast.parse(line).body[0]) for line in _IMPORTS]
    if len(tree.body) != 4 or [ast.dump(node) for node in tree.body[:3]] != imports:
        raise ValueError("native CuTe permits only fixed imports and one kernel definition")
    kernel = tree.body[3]
    if not isinstance(kernel, ast.FunctionDef) or kernel.name != requirements["kernel_entry_point"]:
        raise ValueError("native CuTe kernel entry point differs")
    args = kernel.args
    if (len(kernel.decorator_list) != 1 or _attribute_path(kernel.decorator_list[0]) != "cute.kernel"
        or kernel.returns is not None or kernel.type_comment is not None
        or getattr(kernel, "type_params", []) or args.defaults or args.kw_defaults
        or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs):
        raise ValueError("native CuTe host decorators, defaults and return annotations are forbidden")
    if ([arg.arg for arg in args.args] != [row["name"] for row in requirements["signature"]]
        or any(_attribute_path(arg.annotation) != "cute.Pointer" or arg.type_comment for arg in args.args)):
        raise ValueError("native CuTe positional pointer signature differs")
    forbidden = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                 ast.Lambda, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
                 ast.Delete, ast.Await, ast.Yield, ast.YieldFrom, ast.ListComp, ast.SetComp,
                 ast.DictComp, ast.GeneratorExp, ast.NamedExpr, ast.While)
    float_shadowed = (kernel.name == 'float' or any(arg.arg == 'float' for arg in args.args)
                      or any(isinstance(node, ast.Name) and node.id == 'float'
                             and isinstance(node.ctx, ast.Store) for node in ast.walk(kernel)))
    for statement in kernel.body:
        for node in ast.walk(statement):
            if isinstance(node, forbidden):
                raise ValueError(f"native CuTe unsupported host form at line {node.lineno}: {type(node).__name__}")
            if isinstance(node, ast.stmt) and not isinstance(node, (
                ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr, ast.Pass, ast.Break, ast.Continue,
            )):
                raise ValueError(f"native CuTe unsupported statement at line {node.lineno}")
            if isinstance(node, ast.Name) and ("__" in node.id or
                isinstance(node.ctx, ast.Store) and (node.id in _RESERVED or node.id == kernel.name)):
                raise ValueError(f"native CuTe reserved name at line {node.lineno}")
            if isinstance(node, ast.Attribute):
                path = _attribute_path(node)
                allowed = False
                if path and path.startswith("cute."):
                    suffix = path[5:]
                    allowed = suffix in _CUTE_CALLS | _CUTE_VALUES or any(
                        value.startswith(suffix + ".") for value in _CUTE_CALLS | _CUTE_VALUES)
                elif path and path.startswith("cutlass."):
                    allowed = path[8:] in _CUTLASS_CALLS
                elif path and path.startswith("warp."):
                    allowed = path == "warp.MmaF16BF16Op"
                else:
                    allowed = node.attr in _METHODS | _PROPERTIES
                if not allowed or not isinstance(node.ctx, ast.Load):
                    raise ValueError(f"native CuTe unsupported attribute at line {node.lineno}: {node.attr}")
            if isinstance(node, ast.Call):
                path = _attribute_path(node.func)
                allowed = (path in {f"cute.{v}" for v in _CUTE_CALLS}
                           or path in {f"cutlass.{v}" for v in _CUTLASS_CALLS}
                           or path == "warp.MmaF16BF16Op"
                           or (isinstance(node.func, ast.Name) and node.func.id == "float"
                               and not float_shadowed
                               and len(node.args) == 1 and not node.keywords
                               and isinstance(node.args[0], ast.Constant) and node.args[0].value in {"-inf", "inf"})
                           or isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS
                           and not (path and path.split('.')[0] in _RESERVED))
                if not allowed or any(kw.arg is None for kw in node.keywords):
                    raise ValueError(f"native CuTe unsupported call at line {node.lineno}")
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, (ast.Name, ast.Tuple, ast.Subscript)):
                        raise ValueError(f"native CuTe assignment target at line {node.lineno} differs")


def _launcher_source(source: bytes, requirements: Mapping[str, object]) -> bytes:
    names = ", ".join(row["name"] for row in requirements["signature"])
    annotations = ", ".join(row["name"] + ": cute.Pointer" for row in requirements["signature"])
    return source.rstrip() + (f"\n\n@cute.jit\ndef open_cake_cute_launch({annotations}):\n"
        f"    {requirements['kernel_entry_point']}({names}).launch("
        f"grid={requirements['grid']!r}, block={requirements['block']!r}, smem=0)\n").encode()


@contextmanager
def _deny_cuda_calls(audit_path: Path, *, retain: bool = False):
    """Intercept CUDA Python driver/runtime calls before importing Cutlass.

    CuTe 4.5.2 attempts cuInit during eager target detection, then catches
    RuntimeError. These are denied attempts, never successful GPU observations.
    Filesystem isolation separately keeps GPU device nodes unavailable to native code.
    """
    import cuda.bindings.driver as driver
    import cuda.bindings.runtime as runtime
    replaced = []
    denied: list[str] = []
    for module in (driver, runtime):
        for name, value in vars(module).copy().items():
            if name.startswith("cu") and callable(value) and not isinstance(value, type):
                label = f"{module.__name__}.{name}"
                def refuse(*args, _label=label, **kwargs):
                    denied.append(_label)
                    audit_path.write_text(json.dumps({"denied_cuda_calls": denied}, sort_keys=True))
                    raise RuntimeError("CPU-only CuTe compilation denied " + _label)
                replaced.append((module, name, value))
                setattr(module, name, refuse)
    try:
        yield denied
    finally:
        audit_path.write_text(json.dumps({"denied_cuda_calls": denied}, sort_keys=True))
        if not retain:
            for module, name, value in replaced:
                setattr(module, name, value)


def _ptx_entry(ptx: bytes, requirements: Mapping[str, object]) -> str:
    """Verify target and the complete, scalar-free PTX device parameter list."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx.decode("utf-8"), flags=re.DOTALL)
    if re.findall(r"(?m)^\s*\.target\s+([^\s,]+)", text) != [requirements["target"]]:
        raise ValueError("CuTe PTX target differs")
    entries = re.findall(r"\.entry\s+(\w+)\s*\((.*?)\)", text, re.DOTALL)
    if len(entries) != 1:
        raise ValueError("CuTe PTX must contain exactly one kernel entry")
    name, parameters = entries[0]
    rows = [row.strip() for row in parameters.split(',') if row.strip()]
    patterns = [rf"\.param\s+\.u64\s+\.ptr\s+\.global\s+\.align\s+{2 if row['dtype'] == 'bf16' else 4}\s+\w+"
                for row in requirements["signature"]]
    if len(rows) != len(patterns) or any(re.fullmatch(pattern, row) is None for pattern, row in zip(patterns, rows)):
        raise ValueError("CuTe PTX raw 64-bit pointer parameters must match the complete ordered signature")
    pointer_types = {"bf16": "ptrbf16gmem", "fp32": "ptrf32gmem"}
    expected = "kernel_cutlass_" + requirements["kernel_entry_point"] + "_" + "_".join(
        pointer_types[row["dtype"]] for row in requirements["signature"]) + "_0"
    if name != expected:
        raise ValueError("CuTe PTX entry differs from the requested kernel")
    if re.findall(r"\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text) != [("32", "1", "1")]:
        raise ValueError("CuTe PTX launch dimensions differ")
    return name


def _cubin_parameters(report: str, entry_point: str, signature: list[Mapping[str, str]], *, target: str = "sm_103a") -> list[dict[str, int]]:
    """Read the actual device ABI from cuobjdump's primary .nv.info section."""
    if re.findall(r"(?m)^64-bit ELF:.*?\bsm=([^,\s]+)", report) != [target.removeprefix("sm_")]:
        raise ValueError("CuTe CUBIN target differs")
    sections = re.findall(r"(?m)^\.nv\.info\.(\w+)\s*$\n(.*?)(?=^\S|\Z)", report, re.DOTALL)
    if len(sections) != 1 or sections[0][0] != entry_point:
        raise ValueError("CuTe CUBIN must contain one matching kernel ABI section")
    body = sections[0][1]
    blocks = re.findall(r"Attribute:\s*EIATTR_KPARAM_INFO\s+Format:\s*EIFMT_SVAL\s+"
        r"Value:\s*Index\s*:\s*(0x[0-9a-f]+)\s+Ordinal\s*:\s*(0x[0-9a-f]+)\s+"
        r"Offset\s*:\s*(0x[0-9a-f]+)\s+Size\s*:\s*(0x[0-9a-f]+)\s+"
        r"Pointee's logAlignment\s*:\s*(0x[0-9a-f]+)\s+Space\s*:\s*(0x[0-9a-f]+)", body)
    rows = sorted((dict(zip(("index", "ordinal", "offset", "size", "log_alignment", "space"),
                            (int(value, 16) for value in block))) for block in blocks),
                  key=lambda row: row["ordinal"])
    expected = [dict(index=0, ordinal=index, offset=8 * index, size=8,
                     log_alignment=1 if row["dtype"] == "bf16" else 2, space=4)
                for index, row in enumerate(signature)]
    if rows != expected or body.count("EIATTR_KPARAM_INFO") != len(signature):
        raise ValueError("CuTe CUBIN parameter offsets, sizes or pointer spaces differ")
    if re.findall(r"Attribute:\s*EIATTR_CBANK_PARAM_SIZE\s+Format:\s*EIFMT_HVAL\s+Value:\s*(0x[0-9a-f]+)", body) != [hex(8 * len(signature))]:
        raise ValueError("CuTe CUBIN parameter bank size differs")
    if re.findall(r"Attribute:\s*EIATTR_REQNTID\s+Format:\s*EIFMT_SVAL\s+Value:\s*(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)", body) != [("0x20", "0x1", "0x1")]:
        raise ValueError("CuTe CUBIN launch dimensions differ")
    return rows


def validate_cute_compilation(compilation: CuTeCompilation, source: bytes,
                              requirements: Mapping[str, object]) -> None:
    """Check a worker receipt before it can become a sealed launchable candidate."""
    artifacts = compilation.artifacts
    if (compilation.source != source or compilation.target != requirements["target"]
        or compilation.compiler_version != "4.5.2"
        or type(compilation.threads_per_cta) is not int or compilation.threads_per_cta != 32
        or type(compilation.dynamic_shared_bytes) is not int or compilation.dynamic_shared_bytes != 0
        or set(artifacts) != {"source", "ptx", "cubin", "toolchain_resource_report"}
        or any(not isinstance(value, bytes) or not value for value in artifacts.values())
        or artifacts["source"] != _launcher_source(source, requirements)
        or not artifacts["cubin"].startswith(b"\x7fELF")):
        raise ValueError("CuTe compilation receipt differs from its request")
    name = _ptx_entry(artifacts["ptx"], requirements)
    if compilation.entry_point != name:
        raise ValueError("CuTe compilation entry point differs")
    report = json.loads(artifacts["toolchain_resource_report"])
    if (not isinstance(report, dict) or set(report) != {"compiler_version", "target", "entry_point",
        "resources", "resource_report", "denied_cuda_calls", "jit_engine_created",
        "device_parameters", "elf_report"}
        or report["compiler_version"] != compilation.compiler_version
        or report["target"] != compilation.target or report["entry_point"] != name
        or not isinstance(report["denied_cuda_calls"], list)
        or any(call != "cuda.bindings.driver.cuInit" for call in report["denied_cuda_calls"])
        or report["jit_engine_created"] is not False
        or not isinstance(report["resource_report"], str) or not isinstance(report["elf_report"], str)
        or report["device_parameters"] != _cubin_parameters(report["elf_report"], name, requirements["signature"], target=requirements["target"])
        or report["resources"] != _parse_cuobjdump_resources(report["resource_report"], name)
        or re.findall(r"(?m)^\s*Function\s+([^\s:]+)\s*:", report["resource_report"]) != [name]):
        raise ValueError("CuTe resource or CPU-only compilation evidence differs")


def compile_cute(source: bytes, requirements: Mapping[str, object], *,
                 output_directory: Path, cuobjdump: str) -> CuTeCompilation:
    """Compile inside the Lab CPU jail, extracting artifacts without an executor."""
    validate_cute_kernel(source, requirements)
    if os.environ.get("CUTE_DSL_ARCH") != requirements["target"]:
        raise ValueError("CuTe compile environment must declare the exact target")
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != "4.5.2":
        raise RuntimeError("CuTe toolchain requires installed SDK 4.5.2")
    expanded = _launcher_source(source, requirements)
    output_directory.mkdir(exist_ok=False)
    module_path = output_directory / "candidate.py"
    module_path.write_bytes(expanded)
    # This process serves one compilation. Keep denial installed through SDK
    # exception formatting and finalizers, which may themselves query CUDA.
    with _deny_cuda_calls(output_directory / "cuda_call_guard.json", retain=True) as denied:
        import cutlass
        import cutlass.cute as cute
        spec = importlib.util.spec_from_file_location("open_cake_cute_candidate", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("CuTe candidate module specification failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        types = {"bf16": cutlass.BFloat16, "fp32": cutlass.Float32}
        pointers = [cute.runtime.make_ptr(types[row["dtype"]], 0, cute.AddressSpace.gmem,
                     assumed_align=2 if row["dtype"] == "bf16" else 4)
                    for row in requirements["signature"]]
        compiled = cute.compile(module.open_cake_cute_launch, *pointers, no_jit_engine=True,
            options=f"--gpu-arch {requirements['target']} --keep-ptx --keep-cubin --dump-dir {output_directory}")
        if compiled.engine is not None:
            raise RuntimeError("CuTe CPU compilation unexpectedly created a JIT engine")
        ptx, cubin = compiled.__ptx__, compiled.__cubin__
        if not isinstance(ptx, str) or not ptx or not isinstance(cubin, bytes) or not cubin.startswith(b"\x7fELF"):
            raise ValueError("CuTe compilation did not produce PTX and ELF CUBIN artifacts")
        ptx = ptx.encode()
        name = _ptx_entry(ptx, requirements)
        if list(compiled.kernel_info) != [name]:
            raise ValueError("CuTe kernel metadata and PTX symbols differ")
        if any(call != "cuda.bindings.driver.cuInit" for call in denied):
            raise RuntimeError("CuTe compiler attempted a non-initialization CUDA operation")
    cubin_path = output_directory / "sealed.cubin"
    cubin_path.write_bytes(cubin)
    resource_text = subprocess.check_output([cuobjdump, "--dump-resource-usage", str(cubin_path)],
                                           text=True, timeout=30)
    resources = _parse_cuobjdump_resources(resource_text, name)
    if re.findall(r"(?m)^\s*Function\s+([^\s:]+)\s*:", resource_text) != [name]:
        raise ValueError("CuTe CUBIN must contain exactly one kernel")
    elf_text = subprocess.check_output([cuobjdump, "--dump-elf", str(cubin_path)], text=True, timeout=30)
    parameters = _cubin_parameters(elf_text, name, requirements["signature"], target=requirements["target"])
    report = json.dumps({"compiler_version": version, "target": requirements["target"],
        "entry_point": name, "resources": resources, "resource_report": resource_text,
        "denied_cuda_calls": denied, "jit_engine_created": False,
        "device_parameters": parameters, "elf_report": elf_text}, sort_keys=True).encode()
    result = CuTeCompilation(source, requirements["target"], name, MappingProxyType({
        "source": expanded, "ptx": ptx, "cubin": cubin, "toolchain_resource_report": report,
    }), 32, 0, version)
    validate_cute_compilation(result, source, requirements)
    return result
