"""Run a bound worker module from this Executor's source tree in isolated Python.

Both local broker hops use this script's absolute path. Its owning Executor and
broker command bind the script bytes; ambient Python paths never select a checkout.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import runpy
import sys


def _module_name(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"open_cake_ir(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value) is None:
        raise ValueError("worker must be a bound open_cake_ir module")


def module_command(python: str | Path, module: str, *arguments: str) -> list[str]:
    """Preserve the exact bound Python invocation and this script's source root."""
    if not Path(python).is_absolute():
        raise ValueError("worker Python invocation must be absolute")
    _module_name(module)
    return [str(python), "-I", str(Path(__file__).resolve(strict=True)), module, *arguments]


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not sys.flags.isolated:
        raise ValueError("source worker requires isolated Python (-I)")
    if not arguments:
        raise ValueError("source worker requires a module")
    module = arguments[0]
    _module_name(module)
    script = Path(__file__).resolve(strict=True)
    source = script.parents[2]
    package = source / "open_cake_ir"
    if (script != package / "evaluation/source_bootstrap.py" or
            not (package / "__init__.py").is_file()):
        raise ValueError("source bootstrap is outside its bound package tree")
    module_path = source.joinpath(*module.split("."))
    target = module_path.with_suffix(".py")
    if not target.is_file():
        target = module_path / "__init__.py"
        entrypoint = module_path / "__main__.py"
        if not entrypoint.is_file() or not entrypoint.resolve(strict=True).is_relative_to(package):
            raise ValueError("worker module resolved outside the bound source tree")
    if not target.is_file() or not target.resolve(strict=True).is_relative_to(package):
        raise ValueError("worker module resolved outside the bound source tree")
    sys.path.insert(0, str(source))
    package_spec = importlib.util.find_spec("open_cake_ir")
    if (package_spec is None or package_spec.origin is None or
            Path(package_spec.origin).resolve(strict=True) != package / "__init__.py"):
        raise ValueError("open_cake_ir resolved outside the bound source tree")
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or Path(spec.origin).resolve(strict=True) != target.resolve(strict=True):
        raise ValueError("worker module resolved outside the bound source tree")
    sys.argv = [module, *arguments[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
