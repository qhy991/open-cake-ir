"""Explicit-target Triton compilation and CUBIN inspection without GPU handles.

Both the Lab builder and the offline profile tool use this path. Importing this module
needs neither Triton nor CUDA; those are required only for an explicit compilation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import importlib.metadata
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping

from .compiled_resources import CompiledResources, parse_cuobjdump_resources


@dataclass(frozen=True)
class TritonCompilation:
    source: bytes
    target: str
    entry_point: str
    artifacts: Mapping[str, bytes]
    threads_per_cta: int
    dynamic_shared_bytes: int
    compiler_version: str


# This is a source admission boundary, not a Python sandbox. Compilation of native
# candidates additionally belongs in the Lab's filesystem-isolated build process.
_TRITON_CALLS = frozenset({
    "arange", "program_id", "num_programs", "load", "store", "full", "zeros",
    "sum", "max", "min", "maximum", "minimum", "where", "exp", "exp2", "log",
    "log2", "sqrt", "rsqrt", "abs", "sigmoid", "dot", "trans", "reshape",
    "broadcast_to", "expand_dims", "cast", "div_rn", "fma", "range", "static_range",
    "cumsum", "cumprod", "gather", "debug_barrier", "multiple_of", "max_contiguous",
})
_TRITON_TYPES = frozenset({"constexpr", "float32", "float16", "bfloat16", "int32",
                           "int64", "uint32", "uint64", "int1"})


def validate_triton_kernel(source: bytes, requirements: Mapping[str, object]) -> None:
    """Admit one kernel-only module without importing or evaluating any source.

    Module imports and @triton.jit have one spelling. Function annotations are only
    tl.constexpr, defaults and arbitrary decorators are forbidden, and kernel calls
    are a closed Triton language subset. No user-supplied host callbacks exist.
    """
    try:
        tree = ast.parse(source.decode("utf-8"), filename="candidate.triton.py")
    except (UnicodeError, SyntaxError) as error:
        raise ValueError(f"native Triton syntax: {error}") from error
    expected_imports = [ast.dump(ast.parse(line).body[0]) for line in
                        ("import triton", "import triton.language as tl")]
    if len(tree.body) != 3 or [ast.dump(n) for n in tree.body[:2]] != expected_imports:
        raise ValueError("native Triton permits only fixed imports and one kernel definition")
    kernel = tree.body[2]
    if not isinstance(kernel, ast.FunctionDef) or kernel.name != requirements.get("kernel_entry_point"):
        raise ValueError("native Triton kernel entry point differs")
    args = kernel.args
    if (len(kernel.decorator_list) != 1
        or ast.dump(kernel.decorator_list[0]) != ast.dump(ast.parse("triton.jit", mode="eval").body)
        or kernel.returns is not None or kernel.type_comment is not None
        or getattr(kernel, "type_params", []) or args.defaults or args.kw_defaults
        or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs):
        raise ValueError("native Triton host decorators, defaults and annotations are forbidden")
    signature = requirements.get("signature")
    constants = requirements.get("compile_constants")
    if not isinstance(signature, Mapping) or not isinstance(constants, Mapping):
        raise ValueError("native Triton signature or constants differ")
    if (len(args.args) != len(signature) + len(constants)
        or [arg.arg for arg in args.args[:len(signature)]] != list(signature)
        or {arg.arg for arg in args.args[len(signature):]} != set(constants)):
        raise ValueError("native Triton positional signature differs from trusted compile metadata")
    for arg in args.args:
        expected = "tl.constexpr" if arg.arg in constants else None
        if ((ast.unparse(arg.annotation) if arg.annotation else None) != expected
            or arg.arg in {"tl", "triton", "range"} or "__" in arg.arg):
            raise ValueError("native Triton parameter annotations or names differ")
    forbidden = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.With,
                 ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Await, ast.Yield,
                 ast.YieldFrom, ast.ListComp, ast.SetComp, ast.DictComp,
                 ast.GeneratorExp, ast.NamedExpr, ast.While)
    for statement in kernel.body:
        for node in ast.walk(statement):
            if isinstance(node, forbidden):
                raise ValueError(f"native Triton unsupported host form at line {node.lineno}: {type(node).__name__}")
            if isinstance(node, ast.stmt) and not isinstance(node, (
                ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.If,
                ast.Expr, ast.Return, ast.Pass, ast.Break, ast.Continue,
            )):
                raise ValueError(f"native Triton unsupported statement at line {node.lineno}")
            if isinstance(node, ast.Name) and ("__" in node.id or
                isinstance(node.ctx, ast.Store) and node.id in {"tl", "triton", "range"}):
                raise ValueError(f"native Triton reserved name at line {node.lineno}")
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "tl":
                    allowed = node.attr in _TRITON_CALLS | _TRITON_TYPES
                else:
                    allowed = node.attr == "to" and isinstance(node.ctx, ast.Load)
                if not allowed:
                    raise ValueError(f"native Triton unsupported attribute at line {node.lineno}: {node.attr}")
            if isinstance(node, ast.Call):
                fn = node.func
                allowed = (isinstance(fn, ast.Name) and fn.id == "range") or (
                    isinstance(fn, ast.Attribute) and (
                        isinstance(fn.value, ast.Name) and fn.value.id == "tl" and fn.attr in _TRITON_CALLS
                        or fn.attr == "to" and not (isinstance(fn.value, ast.Name) and fn.value.id in {"triton", "tl"})
                    ))
                if not allowed or any(kw.arg is None for kw in node.keywords):
                    raise ValueError(f"native Triton unsupported call at line {node.lineno}")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(not isinstance(n, (ast.Name, ast.Tuple)) for n in targets):
                    raise ValueError(f"native Triton assignment target at line {node.lineno} differs")
                if isinstance(node, ast.AnnAssign) and ast.unparse(node.annotation) != "tl.constexpr":
                    raise ValueError("native Triton local annotation differs")


def project_triton_kernel(source: bytes, requirements: Mapping[str, object]) -> bytes:
    """Extract the kernel from trusted Compiler lowering, dropping its host wrapper.

    This may be called only on Compiler-owned lowering. Native submissions instead
    validate their entire module, so host side effects cannot disappear from review.
    """
    text = source.decode("utf-8")
    tree = ast.parse(text)
    kernels = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == requirements.get("kernel_entry_point")]
    if len(kernels) != 1:
        raise ValueError("Compiler lowering kernel entry point differs")
    kernel = kernels[0]
    start = min([kernel.lineno] + [node.lineno for node in kernel.decorator_list])
    result = ("import triton\nimport triton.language as tl\n\n" +
              "\n".join(text.splitlines()[start - 1:kernel.end_lineno]) + "\n").encode()
    validate_triton_kernel(result, requirements)
    return result


def compile_triton(source: bytes, requirements: Mapping[str, object]) -> TritonCompilation:
    """Compile a complete emitted source; never launch it or initialize GPU handles."""
    if (
        requirements.get("compiler") != "triton"
        or requirements.get("source_language") != "python"
        or requirements.get("target") != "sm_100a"
        or not source
    ):
        raise ValueError("offline Triton compilation requires the explicit sm_100a contract")
    name = requirements.get("kernel_entry_point")
    signature = requirements.get("signature")
    constants = requirements.get("compile_constants")
    options = requirements.get("compile_options")
    if not isinstance(name, str) or not name or any(
        not isinstance(item, Mapping) for item in (signature, constants, options)
    ):
        raise ValueError("Triton compile contract differs")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as triton_compile

    with tempfile.TemporaryDirectory(prefix="open-cake-triton-") as directory:
        path = Path(directory) / "lowered.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("open_cake_offline_lowering", path)
        if spec is None or spec.loader is None:
            raise ValueError("Triton lowering module specification failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, name, None)
        if kernel is None:
            raise ValueError("Triton lowering kernel entry point is missing")
        compiled = triton_compile(
            ASTSource(kernel, dict(signature), dict(constants)),
            target=GPUTarget("cuda", 100, 32), options=dict(options),
        )
        artifacts = {}
        for role in ("source", "ttir", "ttgir", "llir", "ptx", "cubin"):
            payload = compiled.asm[role]
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"Triton artifact {role!r} differs")
            artifacts[role] = payload
        # .asm and .metadata are compile-time objects. .run, _init_handles(), n_regs
        # and n_spills require the GPU runtime and must not be used in this path.
        metadata = compiled.metadata
        if metadata.name != name or metadata.num_ctas != 1:
            raise ValueError("offline resource model requires the requested single-CTA-cluster kernel")
        if metadata.global_scratch_size or metadata.profile_scratch_size:
            raise ValueError("offline resource model does not cover auxiliary global scratch")
        if not artifacts["cubin"].startswith(b"\x7fELF") or re.search(
            rb"^\.target\s+sm_100a(?:\s|,|$)", artifacts["ptx"], re.MULTILINE
        ) is None:
            raise ValueError("Triton output does not match the exact sm_100a CUBIN target")
        return TritonCompilation(
            source, "sm_100a", name, MappingProxyType(artifacts),
            int(metadata.num_warps) * 32, int(metadata.shared), importlib.metadata.version("triton"),
        )


def inspect_triton_resources(compilation: TritonCompilation, cuobjdump: str | Path) -> CompiledResources:
    """Inspect compiler output using NVIDIA's binary utility, with no GPU runtime."""
    inspector = str(Path(cuobjdump).resolve(strict=True))
    version = subprocess.check_output([inspector, "--version"], text=True, timeout=20).strip()
    with tempfile.TemporaryDirectory(prefix="open-cake-cubin-") as directory:
        path = Path(directory) / "kernel.cubin"
        path.write_bytes(compilation.artifacts["cubin"])
        report = subprocess.check_output(
            [inspector, "--dump-resource-usage", str(path)], text=True, timeout=20,
        )
    return CompiledResources(
        source_sha256=sha256(compilation.source).hexdigest(),
        cubin_sha256=sha256(compilation.artifacts["cubin"]).hexdigest(),
        target=compilation.target, entry_point=compilation.entry_point,
        threads_per_cta=compilation.threads_per_cta,
        dynamic_shared_bytes=compilation.dynamic_shared_bytes,
        compiler_version=compilation.compiler_version, inspector_version=version,
        **parse_cuobjdump_resources(report, compilation.entry_point),
    )
