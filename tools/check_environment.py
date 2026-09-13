#!/usr/bin/env python3
"""Report this host against the preconditions the repository already declares.

This installs nothing and changes nothing. AGENTS.md refuses a repair run placed just
before the check it satisfies, so remediation is printed for a person to run as its own
deliberate act -- not executed here and not executed for you.

Expectations are read from the repository rather than restated: the current Executor
descriptor for a target owns the interpreter, package and helper closure, and
`compiler/targets/` owns which targets exist at all. A precondition this host cannot
observe is reported `unchecked`, never assumed to pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Report against this checkout, whichever interpreter is asking; a doctor that cannot
# import the project must still produce a report rather than a traceback.
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# A host declaring no `kind` is the explicitly named pre-`kind` CUDA form, not a
# fall-through; a kind this tool cannot examine must say so rather than be assumed CUDA.
HOST_KINDS = {"metal": "metal", "cuda": None}
AMD_TOKENS = ("rocm", "hip", "gfx", "amd")


@dataclass
class Check:
    name: str
    status: str
    observed: str
    expected: str
    remediation: str = ""


def _text(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def current_executor(target: str) -> tuple[str, dict] | None:
    """The Executor whose pinned host this checkout would have to match for `target`."""
    inventory = json.loads((ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text())
    entry = inventory.get("current_by_target", {}).get(target)
    if entry is None:
        return None
    return entry["executor_id"], json.loads((ROOT / entry["path"]).read_text())["host_environment"]


def check_interpreter(host: dict, executor_id: str) -> list[Check]:
    """The pinned interpreter is the one admission checks, not whichever runs this tool."""
    python = host["python"]
    expected, observed = Path(python["invocation_path"]), Path(sys.executable).absolute()
    if expected != observed:
        return [Check("executor interpreter", "failed", str(observed), str(expected),
                      f"run Executor work with {expected}; this tool ran under {observed}")]
    digest = sha256(observed.resolve(strict=True).read_bytes()).hexdigest()
    if digest != python["resolved_sha256"] or sys.version.split()[0] != python["version"]:
        return [Check("executor interpreter", "failed", f"{sys.version.split()[0]} {digest[:12]}",
                      f"{python['version']} {python['resolved_sha256'][:12]}",
                      f"{executor_id} pins this interpreter by digest; a toolchain upgrade "
                      "invalidates its admission and needs a successor Executor, not a repair")]
    return [Check("executor interpreter", "ok", f"{python['version']} {digest[:12]}", "pinned digest")]


def check_packages(host: dict) -> list[Check]:
    checks = []
    for distribution, expected in sorted(host.get("packages", {}).items()):
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = "absent"
        checks.append(Check(f"package {distribution}", "ok" if observed == expected else "failed",
                            observed, expected,
                            "" if observed == expected else
                            f"the Executor pins {distribution}=={expected}; installing it here does not "
                            "re-admit the host, it only lets the check pass"))
    return checks


def check_metal_toolchain() -> list[Check]:
    checks = []
    sdk = {name: _text(["/usr/bin/xcrun", "--sdk", "macosx", f"--show-sdk-{name}"])
           for name in ("path", "version", "build-version")}
    if any(value is None for value in sdk.values()):
        return [Check("selected macOS SDK", "failed", "xcrun did not report an SDK", "a selected macOS SDK",
                      "install the Xcode Command Line Tools: xcode-select --install")]
    checks.append(Check("selected macOS SDK", "ok", f"{sdk['version']} ({sdk['build-version']})", "any"))
    swift = _text(["/usr/bin/xcrun", "swiftc", "--version"])
    if swift is None:
        return checks + [Check("swiftc", "failed", "absent", "a working swiftc",
                               "install the Xcode Command Line Tools: xcode-select --install")]
    checks.append(Check("swiftc", "ok", swift.splitlines()[0], "any"))
    # F-2026-09-13-001: a selected SDK whose Swift module was built by a newer compiler than
    # the installed one passes both checks above and fails only when something compiles.
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "probe.swift"
        probe.write_text("import Metal\n")
        done = subprocess.run(["/usr/bin/xcrun", "swiftc", "-typecheck", "-sdk", sdk["path"],
                               "-target", "arm64-apple-macosx15.0", str(probe)],
                              capture_output=True, text=True, timeout=120)
    checks.append(Check("SDK/compiler coherence", "ok" if done.returncode == 0 else "failed",
                        "the selected SDK type-checks against the installed swiftc"
                        if done.returncode == 0 else done.stderr.strip().splitlines()[0][:120],
                        "one coherent toolchain", "" if done.returncode == 0 else
                        "the selected SDK and Swift compiler are different toolchains (F-2026-09-13-001); "
                        "install the matching Command Line Tools, then release a successor Executor"))
    return checks


def check_metal_device(target: str) -> list[Check]:
    try:
        import mlx.core as mx

        from tools.metal.adapter import EXACT_DEVICE_NAMES
    except ImportError as error:
        return [Check("Metal device", "unchecked", f"no in-process device probe: {error}",
                      f"an exact {target} device",
                      "pip install -e '.[metal-host]' to let this tool read the device name")]
    name = mx.device_info()["device_name"]
    admitted = EXACT_DEVICE_NAMES.get(target, ())
    return [Check("Metal device", "ok" if name in admitted else "failed", name, " or ".join(admitted) or "none",
                  "" if name in admitted else
                  "the Compiler requires an exact target match and never steps a Schedule down")]


def check_cuda_toolchain() -> list[Check]:
    checks = []
    for name, command, remediation in (
        ("NVIDIA driver", ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
         "no NVIDIA driver is reachable from this host"),
        ("nvcc", ["nvcc", "--version"], "CUDA Toolkit nvcc is not on PATH"),
    ):
        observed = _text(command)
        checks.append(Check(name, "ok" if observed else "failed", (observed or "absent").splitlines()[-1],
                            "present", "" if observed else remediation))
    return checks


def check_optional(module: str, gates: str, remediation: str) -> Check:
    """An absent optional dependency leaves its checks unrun; it is not a failure here."""
    try:
        version = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return Check(f"optional {module}", "unchecked", "absent", f"gates {gates}", remediation)
    return Check(f"optional {module}", "ok", version, f"gates {gates}")


def _named(paths, *, stems: bool) -> list[str]:
    """Whatever in the checkout already names this vendor; empty means it is unimplemented."""
    found = []
    for path in sorted(paths):
        haystack = path.stem.lower() if stems else path.read_text(encoding="utf-8").lower()
        if any(token in haystack for token in AMD_TOKENS):
            found.append(path.stem if stems else path.name)
    return found


def amd_checks() -> list[Check]:
    """AMD is a declared peer target with no implementation; report that, do not imply one."""
    missing = ("AGENTS.md names AMD an in-scope peer, but this checkout has no AMD Target document, "
               "backend or Executor host kind; there is no environment to configure until they exist")
    observed = (
        ("Target document", "compiler/targets",
         _named((ROOT / "compiler/targets").glob("*.json"), stems=True)),
        ("backend module", "src/open_cake_ir/compiler/backends",
         _named((ROOT / "src/open_cake_ir/compiler/backends").glob("*.py"), stems=True)),
        ("Executor host kind", "src/open_cake_ir/lab/executor.py",
         _named([ROOT / "src/open_cake_ir/lab/executor.py"], stems=False)),
    )
    return [Check(f"AMD {name}", "ok" if found else "unsupported", ", ".join(found) or "none",
                  f"at least one under {location}", "" if found else missing)
            for name, location, found in observed]


def run(kind: str, target: str | None) -> list[Check]:
    checks = [Check("platform", "ok", f"{platform.system()} {platform.machine()}", "any")]
    if kind == "amd":
        return checks + amd_checks()
    if target is not None:
        released = current_executor(target)
        if released is None:
            checks.append(Check("current Executor", "unchecked", f"none for {target} in this checkout",
                                "an entry in inventory/EXECUTOR_REVISIONS.json",
                                "this checkout models one host; the pinned closure cannot be checked here"))
        else:
            executor_id, host = released
            checks.append(Check("current Executor", "ok", executor_id, "released"))
            if host.get("kind") != HOST_KINDS[kind]:
                checks.append(Check("Executor host kind", "failed", str(host.get("kind")),
                                    str(HOST_KINDS[kind]), f"the current {target} Executor is not a {kind} host"))
            else:
                checks += check_interpreter(host, executor_id) + check_packages(host)
    if kind == "metal":
        checks += check_metal_toolchain()
        checks += check_metal_device(target) if target else []
        checks.append(check_optional("mlx", "check_correctness.py --host mlx and its live tests",
                                     "pip install -e '.[metal-host]'"))
    else:
        checks += check_cuda_toolchain()
        # torch is declared by no extra; the CUDA Executor closure pins the version that counts.
        checks.append(check_optional("torch", "tests/native_cuda/test_qualification.py",
                                     "install the torch version the current CUDA Executor pins; without it "
                                     "those tests error rather than skip"))
    checks.append(check_optional("numpy", "tests/contracts/test_mma_k_ranges.py", "pip install -e '.[test]'"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("metal", "cuda", "amd"),
                        default="metal" if sys.platform == "darwin" else "cuda")
    parser.add_argument("--target", help="exact target whose Executor closure to check; no device fallback")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    targets = sorted(path.stem for path in (ROOT / "compiler/targets").glob("*.json"))
    if arguments.target is not None and arguments.target not in targets:
        parser.error(f"unknown target {arguments.target!r}; this checkout declares {', '.join(targets)}")
    checks = run(arguments.kind, arguments.target)
    counts = {status: sum(check.status == status for check in checks)
              for status in ("ok", "failed", "unchecked", "unsupported")}
    if arguments.json:
        print(json.dumps({"kind": arguments.kind, "target": arguments.target,
                          "checks": [asdict(check) for check in checks], "counts": counts,
                          "repairs_performed": 0}, indent=2))
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            print(f"  {check.status:<11} {check.name:<{width}}  {check.observed}")
            if check.remediation:
                print(f"  {'':<11} {'':<{width}}  -> {check.remediation}")
        print(f"\n{counts['ok']} ok, {counts['failed']} failed, {counts['unchecked']} unchecked, "
              f"{counts['unsupported']} unsupported. Nothing was installed or changed.")
    return 1 if counts["failed"] else (3 if counts["unsupported"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
