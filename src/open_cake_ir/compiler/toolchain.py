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

from .performance.compiled_resources import CompiledResources
from .target import CodeObject


@dataclass(frozen=True)
class TritonCompilation:
    source: bytes
    target: str
    entry_point: str
    artifacts: Mapping[str, bytes]
    threads_per_cta: int
    dynamic_shared_bytes: int
    compiler_version: str
    # Which code object the artifacts are, carried from the compile contract so the
    # post-compile inspectors read it here instead of decoding the target id.
    code_object: str


# This is a source admission boundary, not a Python sandbox. Compilation of native
# candidates additionally belongs in the Lab's filesystem-isolated build process.
_TRITON_CALLS = frozenset({
    "arange", "program_id", "num_programs", "load", "store", "full", "zeros",
    "sum", "max", "min", "maximum", "minimum", "where", "exp", "exp2", "log", "log2",
    "log2", "sqrt", "rsqrt", "abs", "sigmoid", "dot", "trans", "reshape",
    "broadcast_to", "expand_dims", "cast", "div_rn", "fma", "range", "static_range",
    "cumsum", "cumprod", "gather", "debug_barrier", "multiple_of", "max_contiguous",
})
_TRITON_TYPES = frozenset({"constexpr", "float32", "float16", "bfloat16", "int32",
                           "int64", "uint32", "uint64", "int1"})
_TRITON_IMPORTS = ("import triton", "import triton.language as tl")
_LIBDEVICE_IMPORT = "from triton.language.extra import libdevice"
_TRITON_MODULES = frozenset({"triton", "tl", "libdevice"})
_TRITON_RESERVED_NAMES = _TRITON_MODULES | {"range", "float"}


def _infinity_literal(node: ast.AST) -> bool:
    """Only the two constant reduction identities, never Python conversion code."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "float" and not node.keywords and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str) and node.args[0].value in {"inf", "-inf"})


def _fp32_fma_call(node: ast.AST, requirements: Mapping[str, object]) -> bool:
    """Admit the emitter's exact rounding contract, not a general assembly escape.

    PTX inline assembly is a fact of the cubin route, so the contract is admitted by the
    code object the compile contract names and not by which target id names it.
    """
    if (requirements.get("code_object") != CodeObject.CUBIN.value or not isinstance(node, ast.Call)
        or len(node.args) != 1 or not isinstance(node.args[0], ast.Constant)
        or node.args[0].value != "fma.rn.f32 $0, $1, $2, $3;"):
        return False
    keywords = {kw.arg: kw.value for kw in node.keywords}
    if len(keywords) != len(node.keywords) or set(keywords) != {
        "constraints", "args", "dtype", "is_pure", "pack",
    }:
        return False
    operands = keywords["args"]
    return (
        isinstance(operands, ast.List) and len(operands.elts) == 3
        and all(not isinstance(operand, ast.Starred) for operand in operands.elts)
        and ast.dump(keywords["dtype"]) == ast.dump(ast.parse("tl.float32", mode="eval").body)
        and all(isinstance(keywords[key], ast.Constant)
                and type(keywords[key].value) is type(value) and keywords[key].value == value
                for key, value in (("constraints", "=f,f,f,f"), ("is_pure", True), ("pack", 1)))
    )


def validate_triton_kernel(source: bytes, requirements: Mapping[str, object]) -> None:
    """Admit one kernel-only module without importing or evaluating any source.

    Module imports and @triton.jit have one spelling. The optional libdevice import
    admits only tanh. Constant infinity identities and the exact FP32 FMA instruction
    emitted by this Compiler are admitted; arbitrary inline assembly is not.
    Function annotations are only tl.constexpr, defaults and arbitrary
    decorators are forbidden, and kernel calls are a closed Triton language subset.
    No user-supplied host callbacks exist.
    """
    try:
        tree = ast.parse(source.decode("utf-8"), filename="candidate.triton.py")
    except (UnicodeError, SyntaxError) as error:
        raise ValueError(f"native Triton syntax: {error}") from error
    imports = _TRITON_IMPORTS + ((_LIBDEVICE_IMPORT,) if len(tree.body) == 4 else ())
    expected_imports = [ast.dump(ast.parse(line).body[0]) for line in imports]
    if (len(tree.body) != len(imports) + 1
        or [ast.dump(n) for n in tree.body[:-1]] != expected_imports):
        raise ValueError("native Triton permits only fixed imports and one kernel definition")
    has_libdevice = len(imports) == 3
    kernel = tree.body[-1]
    if (not isinstance(kernel, ast.FunctionDef)
        or kernel.name != requirements.get("kernel_entry_point")
        or kernel.name in _TRITON_RESERVED_NAMES):
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
            or arg.arg in _TRITON_RESERVED_NAMES or "__" in arg.arg):
            raise ValueError("native Triton parameter annotations or names differ")
    forbidden = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.With,
                 ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Await, ast.Yield,
                 ast.YieldFrom, ast.ListComp, ast.SetComp, ast.DictComp,
                 ast.GeneratorExp, ast.NamedExpr, ast.While)
    parents = {child: parent for parent in ast.walk(kernel)
               for child in ast.iter_child_nodes(parent)}
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
                isinstance(node.ctx, ast.Store) and node.id in _TRITON_RESERVED_NAMES):
                raise ValueError(f"native Triton reserved name at line {node.lineno}")
            if isinstance(node, ast.Name) and node.id == "float" and isinstance(node.ctx, ast.Load):
                if not _infinity_literal(parents.get(node)):
                    raise ValueError(f"native Triton float requires a direct infinity literal at line {node.lineno}")
            if isinstance(node, ast.Name) and node.id == "libdevice" and isinstance(node.ctx, ast.Load):
                attribute = parents.get(node)
                call = parents.get(attribute)
                if (not isinstance(attribute, ast.Attribute) or attribute.value is not node
                    or attribute.attr != "tanh" or not isinstance(call, ast.Call)
                    or call.func is not attribute):
                    raise ValueError(f"native Triton libdevice requires a direct tanh call at line {node.lineno}")
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "tl":
                    allowed = node.attr in _TRITON_CALLS | _TRITON_TYPES
                    if node.attr == "inline_asm_elementwise":
                        call = parents.get(node)
                        allowed = (isinstance(call, ast.Call) and call.func is node
                                   and _fp32_fma_call(call, requirements))
                        if not allowed:
                            raise ValueError(f"native Triton inline assembly requires the exact FP32 FMA contract at line {node.lineno}")
                elif isinstance(node.value, ast.Name) and node.value.id == "libdevice":
                    allowed = has_libdevice and node.attr == "tanh" and isinstance(node.ctx, ast.Load)
                else:
                    allowed = node.attr == "to" and isinstance(node.ctx, ast.Load)
                if not allowed:
                    raise ValueError(f"native Triton unsupported attribute at line {node.lineno}: {node.attr}")
            if isinstance(node, ast.Call):
                fn = node.func
                allowed = (isinstance(fn, ast.Name) and fn.id == "range") or (
                    isinstance(fn, ast.Attribute) and (
                        isinstance(fn.value, ast.Name) and fn.value.id == "tl" and fn.attr in _TRITON_CALLS
                        or has_libdevice and isinstance(fn.value, ast.Name) and fn.value.id == "libdevice" and fn.attr == "tanh"
                        or fn.attr == "to" and not (isinstance(fn.value, ast.Name) and fn.value.id in _TRITON_MODULES)
                    ))
                if isinstance(fn, ast.Name) and fn.id == "float":
                    allowed = _infinity_literal(node)
                    if not allowed:
                        raise ValueError(f"native Triton float requires a direct infinity literal at line {node.lineno}")
                if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id == "tl" and fn.attr == "inline_asm_elementwise"):
                    allowed = _fp32_fma_call(node, requirements)
                    if not allowed:
                        raise ValueError(f"native Triton inline assembly requires the exact FP32 FMA contract at line {node.lineno}")
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
    imports = _TRITON_IMPORTS
    libdevice_import = ast.dump(ast.parse(_LIBDEVICE_IMPORT).body[0])
    if any(ast.dump(node) == libdevice_import for node in tree.body):
        imports += (_LIBDEVICE_IMPORT,)
    result = ("\n".join(imports) + "\n\n" +
              "\n".join(text.splitlines()[start - 1:kernel.end_lineno]) + "\n").encode()
    validate_triton_kernel(result, requirements)
    return result


@dataclass(frozen=True)
class CodeObjectRoute:
    """What Triton's backend for one code object produces, and what its metadata names.

    Triton reaches every vendor through the same `GPUTarget`, but nothing else about a
    compilation is shared: the artifact roles differ, the text that names the emitted
    target differs, and the metadata object carries different scratch fields. Keying
    those on the code object keeps each commitment visible instead of leaving three
    CUDA assumptions buried in one function, and keys them on the fact that decides
    them rather than on which target id happens to run that object.
    """

    gpu_backend: str
    artifact_roles: tuple[str, ...]
    binary_role: str
    text_role: str
    scratch_fields: tuple[str, ...]

    def target_pattern(self, target: str) -> bytes:
        """The line in the text artifact that names the exact target it was built for."""
        if self.text_role == "amdgcn":
            # `amdhsa.target` adds the feature flags the toolchain selected, which are
            # not the same string the device reports (a BW1101 reports
            # `gfx938:sramecc+:xnack-` and this toolchain emits `gfx938:xnack-`), so the
            # ISA is pinned exactly and only well-formed feature suffixes are admitted
            # after it. The quotes are the emitter's YAML, not part of the target:
            # `gfx938:xnack-` contains a colon so it is quoted, and a bare `gfx1151`
            # with no feature flags is not. Anchoring the end of the line is what keeps
            # `gfx938` from matching a `gfx9380` this toolchain might one day name.
            return (rb"^\s*amdhsa\.target:\s*'?amdgcn-amd-amdhsa--"
                    + target.encode("ascii") + rb"(?::[a-z0-9]+[+-])*'?\s*$")
        return rb"^\.target\s+" + target.encode("ascii") + rb"(?:\s|,|$)"


# One row per code object Triton can emit. Closed on purpose: a new code object is a
# decision (CodeObject is a closed enum), and it lands here as one row beside the
# enum member, never as a target id. Offline compilation never opens a Target
# document in its jail; every fact that varies by target -- which object, which
# architecture string, which lane width -- rides the compile contract the emitter
# produced from the Target it held.
_CODE_OBJECT_ROUTES: Mapping[CodeObject, CodeObjectRoute] = MappingProxyType({
    CodeObject.CUBIN: CodeObjectRoute(
        gpu_backend="cuda",
        artifact_roles=("source", "ttir", "ttgir", "llir", "ptx", "cubin"),
        binary_role="cubin",
        text_role="ptx",
        scratch_fields=("global_scratch_size", "profile_scratch_size"),
    ),
    CodeObject.HSACO: CodeObjectRoute(
        gpu_backend="hip",
        artifact_roles=("source", "ttir", "ttgir", "llir", "amdgcn", "hsaco"),
        binary_role="hsaco",
        text_role="amdgcn",
        # HIPOptions carries no global scratch field at all, so there is nothing to
        # read; `profile_scratch_size` is present and is checked like the CUDA route.
        scratch_fields=("profile_scratch_size",),
    ),
})


def route_for_code_object(code_object: CodeObject | str) -> CodeObjectRoute:
    """The Triton route for one declared code object, or a refusal naming it."""
    try:
        code_object = CodeObject(code_object)
    except ValueError:
        raise ValueError(f"Triton compiles no {code_object!r} code object") from None
    route = _CODE_OBJECT_ROUTES.get(code_object)
    if route is None:
        raise ValueError(f"Triton compiles no {code_object.value!r} code object")
    return route


@dataclass(frozen=True)
class TritonRoute:
    """One exact compilation: the code-object route plus the target's own three facts."""

    gpu_backend: str
    architecture: object
    warp_size: int
    artifact_roles: tuple[str, ...]
    binary_role: str
    text_role: str
    target_pattern: bytes
    scratch_fields: tuple[str, ...]
    code_object: CodeObject


def triton_route(requirements: Mapping[str, object]) -> TritonRoute:
    """Read the route the compile contract declares, or refuse it; nothing is defaulted.

    `code_object`, `triton_arch` and `warp_size` are written by the emitter from the
    Target it held (`backends/triton.py`). A missing key is a contract that differs,
    not a CUDA compilation: reading 32 or `cuda` for a contract that never said so is
    the silent answer these keys exist to stop.
    """
    if not isinstance(requirements, Mapping):
        raise ValueError("Triton compile contract differs")
    target = requirements.get("target")
    code_object = requirements.get("code_object")
    architecture = requirements.get("triton_arch")
    warp_size = requirements.get("warp_size")
    if (not isinstance(target, str) or not target or not isinstance(code_object, str)
            or type(warp_size) is not int or warp_size <= 0):
        raise ValueError("Triton compile contract differs")
    route = route_for_code_object(code_object)
    # The architecture Triton's `GPUTarget` takes is an integer capability for CUDA
    # and the bare ISA name for AMDGPU; either shape on the wrong route is a differing
    # contract, not something to coerce.
    if route.gpu_backend == "cuda":
        if type(architecture) is not int or architecture <= 0:
            raise ValueError("Triton compile contract differs")
    elif not isinstance(architecture, str) or not architecture:
        raise ValueError("Triton compile contract differs")
    return TritonRoute(
        gpu_backend=route.gpu_backend,
        architecture=architecture,
        warp_size=warp_size,
        artifact_roles=route.artifact_roles,
        binary_role=route.binary_role,
        text_role=route.text_role,
        target_pattern=route.target_pattern(target),
        scratch_fields=route.scratch_fields,
        code_object=CodeObject(code_object),
    )


def compile_triton(source: bytes, requirements: Mapping[str, object]) -> TritonCompilation:
    """Compile a complete emitted source; never launch it or initialize GPU handles."""
    route = triton_route(requirements)
    target = requirements["target"]
    if (
        requirements.get("compiler") != "triton"
        or requirements.get("source_language") != "python"
        or not source
    ):
        raise ValueError("offline Triton compilation requires an explicit target contract")
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
            target=GPUTarget(route.gpu_backend, route.architecture, route.warp_size),
            options=dict(options),
        )
        artifacts = {}
        for role in route.artifact_roles:
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
        # Only the fields this route's backend actually defines. Reading an absent one
        # through a default would report "no auxiliary scratch" for a backend that was
        # never asked, which is the failure this names instead.
        if any(getattr(metadata, field) for field in route.scratch_fields):
            raise ValueError("offline resource model does not cover auxiliary global scratch")
        # The lane width is a launch fact the analyses derive thread counts from, so a
        # backend that returns a different one than the Target declared invalidates them.
        if getattr(metadata, "warp_size", route.warp_size) != route.warp_size:
            raise ValueError(f"Triton compiled {target} at a lane width the Target does not declare")
        if not artifacts[route.binary_role].startswith(b"\x7fELF") or re.search(
            route.target_pattern, artifacts[route.text_role], re.MULTILINE
        ) is None:
            raise ValueError(
                f"Triton output does not match the exact {target} "
                f"{route.binary_role.upper()} target"
            )
        return TritonCompilation(
            source, target, name, MappingProxyType(artifacts),
            int(metadata.num_warps) * route.warp_size, int(metadata.shared),
            importlib.metadata.version("triton"), route.code_object.value,
        )


def _parse_amdgcn_resources(assembly: str, entry_point: str) -> dict[str, int]:
    """Read one unambiguous kernel's allocation from the .amdgpu_metadata note.

    The AMDGPU backend writes this note into the assembly it already produced, so there
    is no second binary utility to run and nothing here loads the hsaco. Counts are
    documented at
    https://llvm.org/docs/AMDGPUUsage.html#code-object-v5-metadata.

    Two of them do not line up one-to-one with the CUDA report, and neither is silently
    reshaped to look as though they do:

    - `.vgpr_count` is the per-lane vector register allocation and is the analogue of a
      CUDA per-thread register count. `.sgpr_count` is per wavefront, not per thread, so
      it is not folded into that number.
    - `.private_segment_fixed_size` is one per-lane scratch allocation covering what CUDA
      reports separately as LOCAL and STACK. This ISA does not separate them, so the whole
      figure is reported as local bytes and the stack figure is zero because it is not an
      observable quantity here -- not because no stack frame exists.
    """
    # A kernel's first metadata key carries a YAML list dash, so every field pattern
    # here tolerates one. Relying on the keys staying in an order that keeps `-` off
    # the fields being read would make this parser depend on an external format's
    # incidental ordering.
    names = re.findall(r"^[ \t]*(?:-[ \t]+)?\.name:\s*(\S+)\s*$", assembly, flags=re.MULTILINE)
    if names.count(entry_point) != 1 or len(names) != 1:
        raise ValueError("AMDGCN metadata does not name exactly one requested kernel")
    result: dict[str, int] = {}
    for label, name in (
        (".vgpr_count", "registers_per_thread"),
        (".group_segment_fixed_size", "static_shared_bytes"),
        (".private_segment_fixed_size", "local_bytes"),
    ):
        values = re.findall(
            rf"^[ \t]*(?:-[ \t]+)?{re.escape(label)}:\s*(\d+)\s*$", assembly, flags=re.MULTILINE
        )
        if len(values) != 1:
            raise ValueError(f"AMDGCN metadata has missing or ambiguous {label}")
        result[name] = int(values[0])
    result["stack_bytes"] = 0
    return result


def inspect_amdgcn_resources(compilation: TritonCompilation) -> CompiledResources:
    """Inspect AMDGPU compiler output, with no GPU runtime and no external utility.

    The CUDA peer of this shells out to `cuobjdump`. That utility has no counterpart in
    the Hygon DTK image, and it needs none: the allocation facts are already in the
    assembly Triton returned, so the inspector here is the compiler that produced them.
    """
    route = route_for_code_object(compilation.code_object)
    if route.text_role != "amdgcn":
        raise ValueError(f"target {compilation.target!r} does not produce AMDGCN assembly")
    return CompiledResources(
        source_sha256=sha256(compilation.source).hexdigest(),
        cubin_sha256=sha256(compilation.artifacts[route.binary_role]).hexdigest(),
        target=compilation.target, entry_point=compilation.entry_point,
        threads_per_cta=compilation.threads_per_cta,
        dynamic_shared_bytes=compilation.dynamic_shared_bytes,
        compiler_version=compilation.compiler_version,
        inspector_version=f"amdgpu-metadata via triton {compilation.compiler_version}",
        **_parse_amdgcn_resources(
            compilation.artifacts[route.text_role].decode("utf-8"), compilation.entry_point
        ),
    )


def _parse_cuobjdump_resources(report: str, entry_point: str) -> dict[str, int]:
    """Read one unambiguous per-function block from cuobjdump --dump-resource-usage.

    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html documents REG as a
    count and STACK/SHARED/LOCAL as bytes. Unknown fields are not interpreted.
    """
    blocks = re.findall(
        r"^\s*Function\s+([^\s:]+)\s*:\s*\n(.*?)(?=^\s*Function\s+|\Z)",
        report, flags=re.MULTILINE | re.DOTALL,
    )
    matches = [body for name, body in blocks if name == entry_point]
    if len(matches) != 1:
        raise ValueError("CUBIN resource report does not name exactly one requested kernel")
    result: dict[str, int] = {}
    for label, name in (("REG", "registers_per_thread"), ("STACK", "stack_bytes"),
                        ("SHARED", "static_shared_bytes"), ("LOCAL", "local_bytes")):
        values = re.findall(rf"(?<!\S){label}:(\d+)(?=\s|$)", matches[0])
        if len(values) != 1:
            raise ValueError(f"CUBIN resource report has missing or ambiguous {label}")
        result[name] = int(values[0])
    return result


def inspect_triton_resources(compilation: TritonCompilation, cuobjdump: str | Path) -> CompiledResources:
    """Inspect compiler output using NVIDIA's binary utility, with no GPU runtime."""
    # An AMDGCN compilation has no CUBIN for this utility to read. Say that, rather
    # than failing on a missing artifact key several frames further in.
    if route_for_code_object(compilation.code_object).text_role != "ptx":
        raise ValueError(
            f"target {compilation.target!r} produces no CUBIN; use inspect_amdgcn_resources"
        )
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
        **_parse_cuobjdump_resources(report, compilation.entry_point),
    )
