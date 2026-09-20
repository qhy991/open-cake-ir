#!/usr/bin/env python3
"""Survey a MACA host without installing packages or launching GPU kernels.

Run on the host (including via ``ssh host python3 - < tools/probe_metax_host.py``).
JSON goes to stdout; keep it outside the checkout. A host C++ query is compiled in
a temporary directory against the installed SDK, so no foreign ABI layout or enum
numbers are assumed. Only device-count and device-attribute APIs are called.
This is discovery evidence, not an Executor host capture or platform admission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


# These are SDK attribute names, not hardware limits. Compile against the host's
# header rather than copying CUDA/HIP enum values into a ctypes adapter.
ATTRIBUTES = (
    "MaxThreadsPerBlock", "MaxBlockDimX", "MaxBlockDimY", "MaxBlockDimZ",
    "MaxGridDimX", "MaxGridDimY", "MaxGridDimZ", "MaxSharedMemoryPerBlock",
    "WarpSize", "WaveSize", "MultiProcessorCount", "MaxThreadsPerMultiProcessor",
    "MaxSharedMemoryPerMultiprocessor", "MaxRegistersPerBlock",
    "MaxRegistersPerMultiprocessor", "L2CacheSize",
    "ComputeCapabilityMajor", "ComputeCapabilityMinor",
)


def device_query_source() -> str:
    rows = ",\n".join(
        f'    {{"{name}", mcDeviceAttribute{name}}}' for name in ATTRIBUTES
    )
    return r'''
#include <mcr/mc_runtime_api.h>
#include <iostream>
struct Attribute { const char* name; mcDeviceAttribute_t attribute; };
const Attribute attributes[] = {
''' + rows + r'''
};
int main() {
    int count = 0;
    const auto status = mcGetDeviceCount(&count);
    std::cout << "{\"count_status\":" << static_cast<int>(status)
              << ",\"device_count\":";
    if (status == mcSuccess) std::cout << count;
    else std::cout << "null";
    std::cout << ",\"devices\":[";
    bool failed = status != mcSuccess || count == 0;
    if (status == mcSuccess) for (int device = 0; device < count; ++device) {
        if (device) std::cout << ',';
        std::cout << "{\"ordinal\":" << device << ",\"attributes\":{";
        bool first = true;
        for (const auto& row : attributes) {
            int value = 0;
            const auto result = mcDeviceGetAttribute(&value, row.attribute, device);
            if (!first) std::cout << ',';
            first = false;
            std::cout << '\"' << row.name << "\":{\"status\":"
                      << static_cast<int>(result) << ",\"value\":";
            if (result == mcSuccess) std::cout << value;
            else { std::cout << "null"; failed = true; }
            std::cout << '}';
        }
        std::cout << "}}";
    }
    std::cout << "]}\n";
    return failed ? 2 : 0;
}
'''


def command(argv: list[str], timeout: float) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except OSError as error:
        return {"argv": argv, "returncode": None, "error": str(error)}
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        return {"argv": argv, "returncode": None, "error": "timeout",
                "stdout": text(error.stdout), "stderr": text(error.stderr)}
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def survey(root: Path, cxx: str, timeout: float) -> dict:
    commands = {
        "mx_smi": command([shutil.which("mx-smi") or "mx-smi"], timeout),
        "macainfo": command([str(root / "bin/macainfo")], timeout),
        "compiler_arch": command([str(root / "bin/__metaxgpu_arch")], timeout),
        "mxcc_version": command([str(root / "mxgpu_llvm/bin/mxcc"), "--version"], timeout),
    }
    packages = {}
    for name in ("torch", "triton", "flagtree", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Module presence and distribution name are different facts (FlagTree supplies
    # triton). Do not import torch/triton or activate a runtime just to inspect them.
    modules = {name: importlib.util.find_spec(name) is not None
               for name in ("torch", "triton", "numpy")}
    with tempfile.TemporaryDirectory(prefix="cake-metax-host-") as temporary:
        directory = Path(temporary)
        source, binary = directory / "device_query.cpp", directory / "device_query"
        source.write_text(device_query_source(), encoding="utf-8")
        commands["device_query_build"] = command([
            cxx, "-std=c++14", str(source), "-I" + str(root / "include"),
            "-L" + str(root / "lib"), "-Wl,-rpath," + str(root / "lib"),
            "-lmcruntime", "-o", str(binary),
        ], timeout)
        if commands["device_query_build"]["returncode"] == 0:
            commands["device_query"] = command([str(binary)], timeout)
    limitations = [name + " failed; inspect its command record"
                   for name, result in commands.items() if result["returncode"] != 0]
    if "device_query" not in commands:
        limitations.append("device attributes were not queried")
    for name, present in modules.items():
        if not present:
            limitations.append(name + " is absent from this Python environment")
    limitations.append("No kernel compilation, GPU correctness, timing, profiler, or Executor admission was performed")
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "system": platform.platform(),
        "python": sys.executable, "python_version": platform.python_version(),
        "maca_root": str(root), "maca_root_resolved": str(root.resolve()),
        "packages": packages, "modules_present": modules,
        "commands": commands, "limitations": limitations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maca-root", type=Path, default=Path("/opt/maca"))
    parser.add_argument("--cxx", default=shutil.which("c++") or "c++")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = survey(args.maca_root.absolute(), args.cxx, args.timeout)
    print(json.dumps(result, indent=2))
    # Missing Python packages are reported, but do not invalidate a device survey.
    # A missing SDK/tool, nonzero API status or timeout does invalidate that survey.
    return 0 if all(row["returncode"] == 0 for row in result["commands"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
