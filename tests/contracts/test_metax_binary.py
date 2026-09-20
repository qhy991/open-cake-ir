"""Offline MACA bundle refusal cases; fixtures establish no hardware capability."""

import struct
import unittest

from open_cake_ir.compiler.metax_toolchain import device_image


def bundle(*, architecture="xcore1000", native=None, extra=None):
    if native is None:
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
