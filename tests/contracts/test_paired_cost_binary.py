"""Synthetic ELF sections: executable changes must never pass baseline replay."""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from paired_cost_binary import equivalent_except_debug_lines


def make_cubin(*, line: bytes = b"path-A-1234", merc_line: bytes = b"merc-A-1234",
               code: bytes = b"SYNTHETIC-CODE", info: bytes = b"SYNTHETIC-INFO",
               line_flags: int = 0) -> bytes:
    names = (".shstrtab", ".debug_line", ".nv.merc.debug_line", ".text.kernel", ".nv.info")
    strings = b"\x00"
    name_offsets = {}
    for name in names:
        name_offsets[name] = len(strings)
        strings += name.encode() + b"\x00"
    payloads = (strings, line, merc_line, code, info)
    binary = bytearray(64)
    sections = [(0, 0, 0, 0, 0)]
    for name, payload in zip(names, payloads, strict=True):
        start = len(binary)
        binary.extend(payload)
        flags = line_flags if name == ".debug_line" else 0x6 if name == ".text.kernel" else 0
        sections.append((name_offsets[name], 3 if name == ".shstrtab" else 1,
                         flags, start, len(payload)))
    while len(binary) % 8:
        binary.append(0)
    table = len(binary)
    for name_offset, kind, flags, start, size in sections:
        binary.extend(struct.pack("<IIQQQQIIQQ", name_offset, kind, flags, 0,
                                  start, size, 0, 0, 1, 0))
    binary[:16] = b"\x7fELF\x02\x01\x01" + b"\x00" * 9
    struct.pack_into("<HHIQQQIHHHHHH", binary, 16,
                     2, 190, 1, 0, 0, table, 0, 64, 0, 0, 64,
                     len(sections), 1)
    return bytes(binary)


class PairedCostBinaryTest(unittest.TestCase):
    def test_only_two_nonallocated_line_tables_may_change(self):
        frozen = make_cubin()
        debug_only = make_cubin(line=b"path-B-1234", merc_line=b"merc-B-1234")
        self.assertTrue(equivalent_except_debug_lines(frozen, debug_only))
        self.assertFalse(equivalent_except_debug_lines(
            frozen, make_cubin(line=b"path-B-1234", code=b"DIFFERENT-CODE")))
        self.assertFalse(equivalent_except_debug_lines(
            frozen, make_cubin(merc_line=b"merc-B-1234", info=b"DIFFERENT-INFO")))

    def test_malformed_or_allocated_debug_sections_are_refused(self):
        frozen = make_cubin()
        self.assertFalse(equivalent_except_debug_lines(frozen, frozen[:-1] + b"x"))
        self.assertFalse(equivalent_except_debug_lines(
            frozen, make_cubin(line=b"path-B-1234", line_flags=0x2)))
        overlapping = bytearray(make_cubin(line=b"path-B-1234"))
        table = struct.unpack_from("<Q", overlapping, 40)[0]
        text_offset = struct.unpack_from("<Q", overlapping, table + 4 * 64 + 24)[0]
        struct.pack_into("<Q", overlapping, table + 2 * 64 + 24, text_offset)
        self.assertFalse(equivalent_except_debug_lines(frozen, bytes(overlapping)))
        self.assertFalse(equivalent_except_debug_lines(
            frozen, b"\x7fELF" + b"not a cubin"))


if __name__ == "__main__":
    unittest.main()
