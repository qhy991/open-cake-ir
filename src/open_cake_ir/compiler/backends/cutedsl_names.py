"""Generated namespaces owned by the CuTe emitter family."""
from dataclasses import replace

from .common import PythonNamespace

PYTHON_NAMESPACE = PythonNamespace(
    reserved_names=frozenset({"torch", "cutlass", "cute"}),
    generated_prefixes=("N_", "D_", "BLOCK_", "NUM_WARPS", "_work"),
)
REGISTER_NAMESPACE = replace(
    PYTHON_NAMESPACE,
    reserved_names=PYTHON_NAMESPACE.reserved_names | {"warp", "open_cake_cute_launch"},
    forbidden_fragments=("__",),
    identifier_code="CUTE_REGISTER_IDENTIFIER",
    check_source_id=False,
)
