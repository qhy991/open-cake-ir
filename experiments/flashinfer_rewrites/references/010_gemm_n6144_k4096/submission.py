import importlib.util
from pathlib import Path

_IMPL = None

def _load_impl():
    global _IMPL
    if _IMPL is None:
        path = Path(__file__).with_name("n1a1-num_stages4.py")
        spec = importlib.util.spec_from_file_location("_kersor_entry", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IMPL = getattr(mod, "run")
    return _IMPL


def run(A, B):
    return _load_impl()(A, B)
