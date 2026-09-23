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
HIDDEN_POINTER_COUNTS = frozenset({0, 2})


def pointer_parameters(ttgir: bytes) -> int:
    """Count the non-constexpr pointer parameters in the public TTGIR signature."""
    text = ttgir.decode("utf-8")
    signatures = re.findall(r"tt\.func public @\w+\((.*?)\)\s*attributes", text, re.S)
    if len(signatures) != 1:
        raise ValueError("MACA TTGIR must declare exactly one public kernel")
    types = re.findall(r"%[\w.]+\s*:\s*([^\s,]+)", signatures[0])
    if not types or any(re.fullmatch(r"!tt\.ptr<\w+>", value) is None for value in types):
        raise ValueError("MACA kernel ABI is not a nonempty pointer-only signature")
    return len(types)


def native_pointer_parameters(payload: bytes, architecture: str, kernel_name: str) -> int:
    """Count explicit pointer arguments in the sole native MACA kernel note.

    The MetaX 3.8 Triton launcher appends global and profile scratch pointers to
    the source signature; FlagTree 3.1 did not. The native ELF note, rather than a
    package-version table, states which ABI this compilation actually emitted.
    """
    try:
        import msgpack
    except ImportError as error:
        raise ValueError("MACA binary metadata requires msgpack") from error
    image = device_image(payload, architecture)
    section_offset = struct.unpack_from("<Q", image, 40)[0]
    section_size, section_count = struct.unpack_from("<HH", image, 58)
    if (section_size < 64 or section_count == 0 or section_offset > len(image)
            or section_count > (len(image) - section_offset) // section_size):
        raise ValueError("MACA ELF section directory differs")
    notes = []
    for index in range(section_count):
        section = section_offset + index * section_size
        if struct.unpack_from("<I", image, section + 4)[0] != 7:  # SHT_NOTE
            continue
        offset, size = struct.unpack_from("<QQ", image, section + 24)
        if offset > len(image) or size > len(image) - offset:
            raise ValueError("MACA ELF note bounds differ")
        end = offset + size
        while offset < end:
            if end - offset < 12:
                raise ValueError("MACA ELF note is truncated")
            name_size, data_size, note_type = struct.unpack_from("<III", image, offset)
            offset += 12
            name_end = offset + ((name_size + 3) & ~3)
            data_end = name_end + ((data_size + 3) & ~3)
            if name_end > end or data_end > end:
                raise ValueError("MACA ELF note is truncated")
            if image[offset:offset + name_size].rstrip(b"\0") == b"MetaX" and note_type == 48:
                notes.append(image[name_end:name_end + data_size])
            offset = data_end
    if len(notes) != 1:
        raise ValueError("MACA ELF must declare one MetaX kernel note")
    try:
        document = msgpack.unpackb(notes[0], raw=False)
    except (ValueError, TypeError) as error:
        raise ValueError("MACA kernel note is not MessagePack") from error
    kernels = document.get("macahca.kernels") if isinstance(document, dict) else None
    if (not isinstance(kernels, list) or len(kernels) != 1
            or not isinstance(kernels[0], dict) or kernels[0].get(".name") != kernel_name):
        raise ValueError("MACA kernel note does not name the sealed entry point")
    args = kernels[0].get(".args")
    if not isinstance(args, list) or not args:
        raise ValueError("MACA kernel note declares no arguments")
    pointers = 0
    for index, row in enumerate(args):
        if not isinstance(row, dict):
            raise ValueError("MACA kernel argument record differs")
        kind = row.get(".arg_param_pass")
        if kind == "global_buffer":
            if (pointers * 8 != row.get(".arg_offset_bytes")
                    or row.get(".arg_size_bytes") != 8 or pointers != index):
                raise ValueError("MACA pointer arguments are not contiguous 64-bit values")
            pointers += 1
        elif not isinstance(kind, str) or not kind.startswith("hidden_"):
            raise ValueError("MACA kernel argument kind differs")
    if (not pointers or type(kernels[0].get(".kernarg_size_bytes")) is not int
            or kernels[0][".kernarg_size_bytes"] < 8 * pointers):
        raise ValueError("MACA kernel pointer segment differs")
    return pointers


def hidden_pointer_parameters(ttgir: bytes, payload: bytes, architecture: str, kernel_name: str) -> int:
    """Require the native ABI to add either no scratch or exactly two null scratch pointers."""
    tensors = pointer_parameters(ttgir)
    hidden = native_pointer_parameters(payload, architecture, kernel_name) - tensors
    if hidden not in HIDDEN_POINTER_COUNTS:
        raise ValueError(f"MACA kernel declares {hidden} unsupported hidden pointer parameters")
    return hidden


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
    """The observed FlagTree 3.1 and MetaX Triton 3.6 compilation interfaces.

    This package emits no source/LLVM/PTX assembly entries. The source role records
    the exact kernel module passed to compilation; TTIR and TTGIR are actual outputs.
    The native kernel note owns the launch pointer count. The SDK fills its own
    hidden dispatch fields; those are not user parameters to mcModuleLaunchKernel.
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
        hidden = hidden_pointer_parameters(artifacts["ttgir"], artifacts[self.binary_role],
            requirements.get("codegen_arch"), requirements["kernel_entry_point"])
        if hidden == 2 and (getattr(metadata, "global_scratch_size", None) != 0
                            or getattr(metadata, "profile_scratch_size", None) != 0):
            raise ValueError("MACA compilation requires nonzero scratch buffers")
