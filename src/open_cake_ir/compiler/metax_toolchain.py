"""MACA binary inspection without a runtime import or GPU handle.

The observed mxcc -fatbin container is Clang's binary offload bundle, containing an
empty host image, device bitcode and a device ELF. The device identity is admitted at
runtime; this boundary checks the code-generation family declared by its Target.
See https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass


_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def pointer_parameters(ttgir: bytes) -> int:
    """The admitted 3.1 launcher passes only non-constexpr source parameters.

    This backend adds no scratch parameters to that list. Check the actual emitted
    public TTGIR signature before sealing a pointer-only tensor manifest.
    """
    text = ttgir.decode("utf-8")
    signatures = re.findall(r"tt\.func public @\w+\((.*?)\)\s*attributes", text, re.S)
    if len(signatures) != 1:
        raise ValueError("MACA TTGIR must declare exactly one public kernel")
    types = re.findall(r"%[\w.]+\s*:\s*([^\s,]+)", signatures[0])
    if not types or any(re.fullmatch(r"!tt\.ptr<\w+>", value) is None for value in types):
        raise ValueError("MACA kernel ABI is not a nonempty pointer-only signature")
    return len(types)


def device_image(payload: bytes, architecture: str) -> bytes:
    """Read the sole native MACA image for this exact declared codegen family.

    Validate the directory before slicing any member. Extra device architectures,
    duplicate identifiers, truncated/overlapping entries and missing native code are
    refused; a runtime must not select a different image or compile bitcode instead.
    The returned bytes are a view's contents, not a rewritten launch artifact.
    """
    if (not isinstance(payload, bytes) or not payload.startswith(_BUNDLE_MAGIC)
            or not isinstance(architecture, str)
            or re.fullmatch(r"xcore[0-9]+", architecture) is None):
        raise ValueError("MACA binary requires an offload bundle and explicit xcore architecture")
    position = len(_BUNDLE_MAGIC)
    if len(payload) < position + 8:
        raise ValueError("MACA offload bundle header is truncated")
    count = struct.unpack_from("<Q", payload, position)[0]
    position += 8
    if count == 0 or count > (len(payload) - position) // 24:
        raise ValueError("MACA offload bundle entry count differs")
    entries = {}
    for _ in range(count):
        if position + 24 > len(payload):
            raise ValueError("MACA offload bundle directory is truncated")
        offset, size, name_size = struct.unpack_from("<QQQ", payload, position)
        position += 24
        if name_size == 0 or name_size > len(payload) - position:
            raise ValueError("MACA offload bundle identifier is truncated")
        try:
            name = payload[position:position + name_size].decode("ascii")
        except UnicodeError as error:
            raise ValueError("MACA offload bundle identifier is not ASCII") from error
        position += name_size
        if name in entries:
            raise ValueError("MACA offload bundle has duplicate identifiers")
        entries[name] = (offset, size)
    device = "maca-mxc-metax-macahca--" + architecture
    if set(entries) != {"host-x86_64-unknown-linux-gnu", device, device + "-bc"}:
        raise ValueError(f"MACA offload bundle does not declare only {architecture!r}")
    occupied = []
    for name, (offset, size) in entries.items():
        if offset < position or offset > len(payload) or size > len(payload) - offset:
            raise ValueError("MACA offload bundle member bounds differ")
        if name.startswith("host-"):
            if size:
                raise ValueError("MACA kernel bundle contains a host executable")
        else:
            if not size:
                raise ValueError("MACA offload bundle is missing a device image")
            if any(offset < end and start < offset + size for start, end in occupied):
                raise ValueError("MACA offload bundle members overlap")
            occupied.append((offset, offset + size))
    offset, size = entries[device]
    image = payload[offset:offset + size]
    # e_machine=253 is what this mxcc emits for MXC; the bundle's label alone must
    # not allow an x86 or another vendor's ELF to reach mcModuleLoadData.
    if (len(image) < 64 or not image.startswith(b"\x7fELF\x02\x01")
            or struct.unpack_from("<H", image, 18)[0] != 253):
        raise ValueError("MACA native device image is not an MXC ELF64 little-endian object")
    return image


@dataclass(frozen=True)
class MetaxRoute:
    """The observed FlagTree 0.5.1+metax3.1 compilation interface.

    This package emits no source/LLVM/PTX assembly entries. The source role records
    the exact kernel module passed to compilation; TTIR and TTGIR are actual outputs.
    It declares no explicit scratch pointers or scratch-size metadata. The SDK fills
    its own hidden dispatch fields; these are not user parameters to mcModuleLaunchKernel.
    """

    gpu_backend: str = "maca"
    architecture_type: type = int
    artifact_roles: tuple[str, ...] = ("source", "ttir", "ttgir", "mcfatbin")
    binary_role: str = "mcfatbin"
    text_role: str = "ttgir"
    scratch_fields: tuple[str, ...] = ()

    def compiler_version(self, module=None) -> str:
        # FlagTree provides the triton module but has a different distribution name
        # and version. Host capture separately binds that distribution's identity.
        if module is None:
            import triton as module
        return module.__version__

    def target_pattern(self, target: str) -> None:
        # The native bundle names the codegen family. Its authority is not a PTX
        # .target directive or an AMDGCN assembly line.
        return None

    def read_artifacts(self, compiled, source: bytes) -> dict[str, bytes]:
        artifacts = {"source": source}
        for role in self.artifact_roles[1:]:
            payload = compiled.asm[role]
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"MACA compilation artifact {role!r} differs")
            artifacts[role] = payload
        return artifacts

    def validate_artifacts(self, artifacts, requirements, metadata) -> None:
        target = metadata.target
        if (target.backend != self.gpu_backend or target.arch != requirements["triton_arch"]
                or target.warp_size != requirements["warp_size"]):
            raise ValueError("MACA compiler metadata differs from the declared target")
        device_image(artifacts[self.binary_role], requirements.get("codegen_arch"))
