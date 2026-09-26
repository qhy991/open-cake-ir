"""Offline MACA bundle refusal cases; fixtures establish no hardware capability."""

import struct
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler.metax_toolchain import (
    MetaxRoute, device_image, hidden_pointer_parameters,
)


TTGIR_ONE_POINTER = b"tt.func public @kernel(%x: !tt.ptr<f32>) attributes {test = true}"


def native_with_note(pointer_arguments, *, kernel_name="kernel"):
    import msgpack
    native = bytearray(64)
    native[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", native, 18, 253)
    metadata = {"macahca.version": [1, 0], "macahca.kernels": [{
        ".name": kernel_name,
        ".kernarg_size_bytes": pointer_arguments * 8,
        ".args": [{".arg_offset_bytes": index * 8, ".arg_size_bytes": 8,
                   ".arg_param_pass": "global_buffer"}
                  for index in range(pointer_arguments)],
    }]}
    description = msgpack.packb(metadata, use_bin_type=True)
    note = struct.pack("<III", 6, len(description), 48) + b"MetaX\0\0\0" + description
    note += b"\0" * (-len(note) % 4)
    native.extend(note)
    section_offset = (len(native) + 7) & ~7
    native.extend(b"\0" * (section_offset - len(native) + 128))
    struct.pack_into("<Q", native, 40, section_offset)
    struct.pack_into("<HH", native, 58, 64, 2)
    struct.pack_into("<IIQQQQIIQQ", native, section_offset + 64,
                     0, 7, 0, 0, 64, len(note), 0, 0, 4, 0)
    return bytes(native)


def bundle(*, architecture="xcore1000", native=None, extra=None,
           note_pointer_arguments=None, note_kernel_name="kernel"):
    if native is None:
        if note_pointer_arguments is not None:
            native = native_with_note(note_pointer_arguments, kernel_name=note_kernel_name)
        else:
            native = bytearray(64)
            native[:6] = b"\x7fELF\x02\x01"
            struct.pack_into("<H", native, 18, 253)
            native = bytes(native)
    base = "maca-mxc-metax-macahca--" + architecture
    rows = [("host-x86_64-unknown-linux-gnu", b""),
            (base + "-bc", b"synthetic-bitcode"), (base, native)]
    if extra is not None:
        rows.append(extra)
    header = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", len(rows))
    offset = len(header) + sum(24 + len(name) for name, _ in rows)
    body = b""
    for name, payload in rows:
        header += struct.pack("<QQQ", offset, len(payload), len(name)) + name.encode()
        body += payload
        offset += len(payload)
    return header + body, native


class MetaxBinaryTests(unittest.TestCase):
    def test_native_note_owns_old_and_new_hidden_pointer_counts(self):
        for pointers, expected in ((1, 0), (3, 2)):
            with self.subTest(pointers=pointers):
                payload, _ = bundle(note_pointer_arguments=pointers)
                self.assertEqual(hidden_pointer_parameters(
                    TTGIR_ONE_POINTER, payload, "xcore1000", "kernel"), expected)
        with self.assertRaisesRegex(ValueError, "unsupported hidden"):
            hidden_pointer_parameters(TTGIR_ONE_POINTER, bundle(note_pointer_arguments=2)[0],
                                      "xcore1000", "kernel")

    def test_new_launcher_requires_no_allocated_scratch(self):
        payload, _ = bundle(note_pointer_arguments=3)
        artifacts = {"ttgir": TTGIR_ONE_POINTER, "mcfatbin": payload}
        requirements = {"triton_arch": 80, "warp_size": 64,
                        "codegen_arch": "xcore1000", "kernel_entry_point": "kernel"}
        metadata = SimpleNamespace(target=SimpleNamespace(backend="maca", arch=80, warp_size=64),
                                   global_scratch_size=0, profile_scratch_size=0)
        MetaxRoute().validate_artifacts(artifacts, requirements, metadata)
        metadata.global_scratch_size = 64
        with self.assertRaisesRegex(ValueError, "nonzero scratch"):
            MetaxRoute().validate_artifacts(artifacts, requirements, metadata)
        metadata.global_scratch_size = 0
        metadata.profile_scratch_size = None
        with self.assertRaisesRegex(ValueError, "omits scratch"):
            MetaxRoute().validate_artifacts(artifacts, requirements, metadata)
        old_artifacts = {"ttgir": TTGIR_ONE_POINTER,
                         "mcfatbin": bundle(note_pointer_arguments=1)[0]}
        metadata.global_scratch_size = 64
        with self.assertRaisesRegex(ValueError, "nonzero scratch"):
            MetaxRoute().validate_artifacts(old_artifacts, requirements, metadata)

    def test_returns_only_the_native_member_of_the_requested_family(self):
        payload, native = bundle()
        self.assertEqual(device_image(payload, "xcore1000"), native)
        with self.assertRaisesRegex(ValueError, "does not declare only"):
            device_image(payload, "xcore1002")

    def test_every_truncated_directory_or_native_image_is_refused(self):
        payload, _ = bundle()
        for end in (0, 23, 24, 31, 32, 55, 100, len(payload) - 1):
            with self.subTest(end=end), self.assertRaises(ValueError):
                device_image(payload[:end], "xcore1000")

    def test_a_device_elf_cannot_be_relabelled_as_mxc(self):
        wrong = bytearray(bundle()[1])
        struct.pack_into("<H", wrong, 18, 62)  # x86-64
        for native in (bytes(wrong), b"\x7fELF\x02\x01", b""):
            with self.subTest(native=native), self.assertRaises(ValueError):
                device_image(bundle(native=native)[0], "xcore1000")

    def test_ambiguous_or_additional_architectures_are_refused(self):
        base = "maca-mxc-metax-macahca--"
        for extra in ((base + "xcore1002", bundle()[1]),
                      (base + "xcore1000", bundle()[1])):
            with self.subTest(extra=extra[0]), self.assertRaises(ValueError):
                device_image(bundle(extra=extra)[0], "xcore1000")

    def test_payload_must_not_point_into_the_directory_or_overlap(self):
        payload, _ = bundle()
        # Locate the two device records from the documented binary directory, then
        # corrupt their offsets without changing the payload or architecture names.
        first = 24 + 8
        second = first + 24 + len("host-x86_64-unknown-linux-gnu")
        third = second + 24 + len("maca-mxc-metax-macahca--xcore1000-bc")
        for offset in (0, struct.unpack_from("<Q", payload, second)[0]):
            corrupted = bytearray(payload)
            struct.pack_into("<Q", corrupted, third, offset)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                device_image(bytes(corrupted), "xcore1000")


if __name__ == "__main__":
    unittest.main()
