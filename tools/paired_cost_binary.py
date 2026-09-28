"""Bounded CUBIN equality for a frozen baseline recompiled by the same toolchain.

Triton/ptxas embeds non-executing line tables that can change between builds.
Everything outside the two admitted debug sections must remain byte-identical.
This never rewrites or strips the CUBIN that Evaluation will launch.
"""
from __future__ import annotations

import struct


_DEBUG_SECTIONS = frozenset({".debug_line", ".nv.merc.debug_line"})
_ELF_HEADER_SIZE = 64
_SECTION_HEADER_SIZE = 64


def _sections(binary: bytes) -> dict[str, tuple[int, int, int, int]]:
    if (len(binary) < _ELF_HEADER_SIZE or binary[:4] != b"\x7fELF"
            or binary[4:7] != b"\x02\x01\x01"):
        raise ValueError("paired baseline CUBIN is not little-endian ELF64")
    table = struct.unpack_from("<Q", binary, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", binary, 58)
    if (entry_size != _SECTION_HEADER_SIZE or not count or names_index >= count
            or table < _ELF_HEADER_SIZE
            or table + count * entry_size > len(binary)):
        raise ValueError("paired baseline CUBIN section table differs")
    rows = [struct.unpack_from("<IIQQQQIIQQ", binary, table + index * entry_size)
            for index in range(count)]
    strings = rows[names_index]
    if strings[1] != 3 or strings[4] + strings[5] > len(binary):
        raise ValueError("paired baseline CUBIN section names differ")
    names = binary[strings[4]:strings[4] + strings[5]]
    result: dict[str, tuple[int, int, int, int]] = {}
    occupied = []
    for row in rows[1:]:
        name_offset, kind, flags, _, offset, size, *_ = row
        if name_offset >= len(names):
            raise ValueError("paired baseline CUBIN section name offset differs")
        end = names.find(b"\x00", name_offset)
        if end < 0:
            raise ValueError("paired baseline CUBIN section name is unterminated")
        try:
            name = names[name_offset:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("paired baseline CUBIN section name is not ASCII") from error
        if name in result:
            raise ValueError("paired baseline CUBIN repeats a section name")
        if kind != 8 and size:  # SHT_NOBITS occupies no file bytes.
            if offset < _ELF_HEADER_SIZE or offset + size > len(binary):
                raise ValueError("paired baseline CUBIN section bytes escape the file")
            occupied.append((name, offset, offset + size))
        result[name] = (kind, flags, offset, size)
    for name in _DEBUG_SECTIONS:
        item = result.get(name)
        if item is None or item[0] != 1 or item[1] & 0x6 or item[3] <= 0:
            raise ValueError("paired baseline CUBIN has no unallocated debug line section")
        _, _, start, size = item
        end = start + size
        if start < table + count * entry_size and end > table:
            raise ValueError("paired baseline CUBIN debug line overlaps section headers")
        if any(other != name and start < stop and end > begin
               for other, begin, stop in occupied):
            raise ValueError("paired baseline CUBIN debug line overlaps another section")
    return result


def equivalent_except_debug_lines(frozen: bytes, rebuilt: bytes) -> bool:
    """Refuse any change beyond the two known non-allocated line-table sections."""
    if frozen == rebuilt:
        return True
    if len(frozen) != len(rebuilt):
        return False
    try:
        old, new = _sections(frozen), _sections(rebuilt)
    except ValueError:
        return False
    if old != new:
        return False
    masked = bytearray(frozen)
    for name in _DEBUG_SECTIONS:
        _, _, offset, size = old[name]
        masked[offset:offset + size] = rebuilt[offset:offset + size]
    return bytes(masked) == rebuilt
