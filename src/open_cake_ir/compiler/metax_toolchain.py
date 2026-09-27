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


def _metadata_object(payload: bytes):
    """Decode the maps, arrays, strings and integers in a MACA MessagePack note.

    This is a bounded metadata subset, not a general serialization interface.
    Unsupported encodings, duplicate keys and trailing bytes are refusals.
    """
    position = 0

    def take(size):
        nonlocal position
        if size < 0 or position + size > len(payload):
            raise ValueError("MACA metadata is truncated")
        result = payload[position:position + size]
        position += size
        return result

    def number(size):
        return int.from_bytes(take(size), "big")

    def read(depth=0):
        if depth > 32:
            raise ValueError("MACA metadata nesting exceeds the supported domain")
        tag = number(1)
        if tag < 0x80:
            return tag
        if tag in (0xCC, 0xCD, 0xCE, 0xCF):
            return number(1 << (tag - 0xCC))
        if 0xA0 <= tag <= 0xBF or tag in (0xD9, 0xDA, 0xDB):
            size = tag & 31 if tag <= 0xBF else number(1 << (tag - 0xD9))
            try:
                return take(size).decode("utf-8")
            except UnicodeError as error:
                raise ValueError("MACA metadata string is not UTF-8") from error
        if 0x90 <= tag <= 0x9F or tag in (0xDC, 0xDD):
            count = tag & 15 if tag <= 0x9F else number(2 if tag == 0xDC else 4)
            if count > len(payload) - position:
                raise ValueError("MACA metadata array is truncated")
            return [read(depth + 1) for _ in range(count)]
        if 0x80 <= tag <= 0x8F or tag in (0xDE, 0xDF):
            count = tag & 15 if tag <= 0x8F else number(2 if tag == 0xDE else 4)
            if count > (len(payload) - position) // 2:
                raise ValueError("MACA metadata map is truncated")
            result = {}
            for _ in range(count):
                key = read(depth + 1)
                if not isinstance(key, str) or key in result:
                    raise ValueError("MACA metadata keys must be unique strings")
                result[key] = read(depth + 1)
            return result
        raise ValueError(f"MACA metadata encoding {tag:#x} is unsupported")

    result = read()
    if position != len(payload):
        raise ValueError("MACA metadata has trailing bytes")
    return result


def native_pointer_parameters(payload: bytes, architecture: str, kernel_name: str | None = None) -> int:
    """Read public and launcher scratch pointers from the native ELF, not TTGIR.

    The C550-2 Triton 3.6 ELF has two additional global_buffer slots even though
    its TTGIR contains only tensor arguments. Runtime-populated hidden dispatch
    entries are a different class and are never counted as launcher arguments.
    """
    image = device_image(payload, architecture)
    offset = struct.unpack_from("<Q", image, 40)[0]
    width, count = struct.unpack_from("<HH", image, 58)
    if width != 64 or count == 0 or offset < 64 or offset + width * count > len(image):
        raise ValueError("MACA ELF section directory differs")
    notes = []
    for index in range(count):
        row = offset + width * index
        if struct.unpack_from("<I", image, row + 4)[0] != 7:  # SHT_NOTE
            continue
        start, size = struct.unpack_from("<QQ", image, row + 24)
        end = start + size
        if start < 64 or end > len(image):
            raise ValueError("MACA ELF note bounds differ")
        while start < end:
            if start + 12 > end:
                raise ValueError("MACA ELF note is truncated")
            names, descriptions, kind = struct.unpack_from("<III", image, start)
            name_start = start + 12
            desc_start = name_start + ((names + 3) & ~3)
            next_note = desc_start + ((descriptions + 3) & ~3)
            if names == 0 or next_note > end:
                raise ValueError("MACA ELF note is truncated")
            if image[name_start:name_start + names].rstrip(b"\0") == b"MetaX" and kind == 0x30:
                notes.append(_metadata_object(image[desc_start:desc_start + descriptions]))
            start = next_note
    if len(notes) != 1 or not isinstance(notes[0], dict):
        raise ValueError("MACA ELF must carry one MetaX kernel metadata note")
    kernels = notes[0].get("macahca.kernels")
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
        raise ValueError("MACA ELF must declare exactly one kernel")
    kernel = kernels[0]
    if (not isinstance(kernel.get(".name"), str)
            or kernel_name is not None and kernel[".name"] != kernel_name):
        raise ValueError("MACA native kernel name differs")
    args = kernel.get(".args")
    if not isinstance(args, list) or not args:
        raise ValueError("MACA native kernel has no argument metadata")
    pointers = 0
    hidden = False
    for argument in args:
        if not isinstance(argument, dict):
            raise ValueError("MACA native argument metadata differs")
        kind = argument.get(".arg_param_pass")
        if kind == "global_buffer":
            if (hidden or argument.get(".arg_size_bytes") != 8
                    or argument.get(".arg_offset_bytes") != pointers * 8):
                raise ValueError("MACA launcher arguments must be contiguous eight-byte pointers")
            pointers += 1
        elif kind in {"hidden_global_offset_x", "hidden_global_offset_y", "hidden_global_offset_z",
                      "hidden_none", "hidden_multigrid_sync_arg"}:
            hidden = True
        else:
            raise ValueError("MACA native argument kind is unsupported")
    if pointers == 0:
        raise ValueError("MACA native kernel declares no launcher pointers")
    return pointers


@dataclass(frozen=True)
class MetaxRoute:
    """The observed MACA Triton compilation interface.

    This package emits no source/LLVM/PTX assembly entries. The source role records
    the exact kernel module passed to compilation; TTIR and TTGIR are actual outputs.
    Triton 3.1 declares no launcher scratch pointers; the captured 3.6 route adds two
    zero-sized scratch pointers. Native ELF argument metadata owns their count.
    SDK-populated hidden dispatch fields are not launcher pointer parameters.
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
        scratch = ("global_scratch_size", "profile_scratch_size")
        present = [hasattr(metadata, name) for name in scratch]
        if any(present) and (not all(present) or any(
                type(getattr(metadata, name)) is not int or getattr(metadata, name) != 0 for name in scratch)):
            raise ValueError("MACA compilation has unmodeled auxiliary scratch requirements")
        native = native_pointer_parameters(artifacts[self.binary_role], requirements.get("codegen_arch"), metadata.name)
        extra = native - pointer_parameters(artifacts[self.text_role])
        if extra not in (0, 2) or extra == 2 and not all(present):
            raise ValueError("MACA native scratch slots lack a qualified zero-sized compiler contract")
