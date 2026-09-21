"""Native MCPTI activity records for the captured MACA 3.5.3 ABI.

Kernel8's prefix is read as declared by the SDK; no CUDA activity layouts or event
elapsed times are substituted. The SDK's CPU layout probe at qualification owns the
ABI evidence. Callbacks live for the process lifetime, while each collection is
exclusive and bounded. The caller must synchronize its device before finish().
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import threading


_API_VERSION = 18
_KERNEL = 10  # MCPTI_ACTIVITY_KIND_CONCURRENT_KERNEL; never the serializing kind 3.
_KINDS = (_KERNEL, 1, 2, 4, 5)  # kernel, memcpy, memset, driver, runtime
_MAX_BUFFER_BYTES = 64 * 1024 * 1024
_BUFFER_BYTES = 8 * 1024 * 1024
_MAX_RECORDS = 100000
_COLLECTOR = None
_CREATION = threading.RLock()


class _Kernel8Prefix(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32), ("cache", C.c_uint8), ("shared_config", C.c_uint8),
        ("registers", C.c_uint16), ("partition_requested", C.c_uint32),
        ("partition_executed", C.c_uint32), ("start", C.c_uint64), ("end", C.c_uint64),
        ("completed", C.c_uint64), ("device", C.c_uint32), ("context", C.c_uint32),
        ("stream", C.c_uint32), ("grid", C.c_int32 * 3), ("block", C.c_int32 * 3),
        ("static_shared", C.c_int32), ("dynamic_shared", C.c_int32),
        ("local_per_thread", C.c_uint32), ("local_total", C.c_uint32),
        ("correlation", C.c_uint32), ("grid_id", C.c_int64), ("name", C.c_void_p),
    ]


class _ApiActivity(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("cbid", C.c_uint32),
                ("start", C.c_uint64), ("end", C.c_uint64),
                ("process", C.c_uint32), ("thread", C.c_uint32),
                ("correlation", C.c_uint32), ("return_value", C.c_uint32)]


_Request = C.CFUNCTYPE(None, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.POINTER(C.c_size_t))
_Complete = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_void_p, C.c_size_t, C.c_size_t)


class McptiActivity:
    """One process-owned callback collector bound to an admitted absolute library."""

    def __init__(self, library: str):
        with _CREATION:
            if _COLLECTOR is not None:
                raise RuntimeError("MCPTI callbacks already have a process owner; use activity_collector")
            self._initialize(library)

    def _initialize(self, library: str):
        global _COLLECTOR
        self._ready = False
        path = Path(library)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("MCPTI requires the admitted resolved absolute library")
        self.library = str(path)
        self.api = C.CDLL(self.library)
        signatures = {
            "mcptiGetVersion": [C.POINTER(C.c_uint32)],
            "mcptiActivityEnable": [C.c_uint32], "mcptiActivityDisable": [C.c_uint32],
            "mcptiActivityFlushAll": [C.c_uint32],
            "mcptiActivityGetNumDroppedRecords": [C.c_void_p, C.c_uint32, C.POINTER(C.c_size_t)],
            "mcptiActivityGetNextRecord": [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p)],
            "mcptiActivityRegisterCallbacks": [_Request, _Complete],
        }
        for name, args in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, C.c_int
        version = C.c_uint32()
        self._call("mcptiGetVersion", C.byref(version))
        if version.value != _API_VERSION:
            raise ValueError(f"MCPTI activity ABI {version.value} is not the qualified API {_API_VERSION}")
        self.version = version.value
        self._session = threading.Lock()
        self._buffers = {}
        self._rows = []
        self._errors = []
        self._dropped = 0
        self._enabled = []
        self._active = False
        self._owner_thread = None
        self._requested_callback = _Request(self._requested)
        self._completed_callback = _Complete(self._completed)
        # Keep callbacks alive even if registration reports an ambiguous failure.
        _COLLECTOR = self
        self._call("mcptiActivityRegisterCallbacks", self._requested_callback, self._completed_callback)
        self._ready = True

    def _call(self, name, *args):
        status = getattr(self.api, name)(*args)
        if status != 0:
            raise RuntimeError(f"MCPTI {name} failed with status {status}")

    def _requested(self, pointer, size, count):
        # ctypes callback exceptions cannot propagate to the caller. Preserve them and
        # decline a buffer, then refuse the session rather than dropping evidence.
        try:
            if len(self._buffers) * _BUFFER_BYTES >= _MAX_BUFFER_BYTES:
                raise RuntimeError("MCPTI buffer allocation exceeds its bounded capacity")
            buffer = C.create_string_buffer(_BUFFER_BYTES)
            address = C.addressof(buffer)
            self._buffers[address] = buffer
            pointer[0], size[0], count[0] = address, _BUFFER_BYTES, 0
        except BaseException as error:
            self._errors.append(str(error))
            pointer[0], size[0], count[0] = None, 0, 0

    def _completed(self, context, stream, address, size, valid):
        try:
            if address not in self._buffers or size != _BUFFER_BYTES or valid > size:
                raise ValueError("MCPTI returned an unowned or invalid buffer")
            pointer = C.c_void_p()
            while valid:
                status = self.api.mcptiActivityGetNextRecord(address, valid, C.byref(pointer))
                if status == 12:  # MCPTI_ERROR_MAX_LIMIT_REACHED is normal end of buffer.
                    break
                if status != 0:
                    raise RuntimeError(f"MCPTI record iteration failed with status {status}")
                at = pointer.value
                if at is None or not address <= at <= address + valid - 4:
                    raise ValueError("MCPTI record is outside its returned buffer")
                kind = C.c_uint32.from_address(at).value
                row = {"kind": kind}
                if kind == _KERNEL:
                    if at + C.sizeof(_Kernel8Prefix) > address + valid:
                        raise ValueError("MCPTI kernel record is truncated")
                    item = _Kernel8Prefix.from_address(at)
                    if not item.name:
                        raise ValueError("MCPTI kernel has no name")
                    row.update(name=C.string_at(item.name).decode("utf-8"), start_ns=item.start,
                        end_ns=item.end, device=item.device, context=item.context, stream=item.stream,
                        correlation=item.correlation, grid=list(item.grid), block=list(item.block),
                        registers_per_thread=item.registers, static_shared_bytes=item.static_shared,
                        dynamic_shared_bytes=item.dynamic_shared, local_bytes_per_thread=item.local_per_thread)
                elif kind in (4, 5):
                    if at + C.sizeof(_ApiActivity) > address + valid:
                        raise ValueError("MCPTI API record is truncated")
                    item = _ApiActivity.from_address(at)
                    row.update(cbid=item.cbid, start_ns=item.start, end_ns=item.end,
                               correlation=item.correlation, return_value=item.return_value)
                if len(self._rows) >= _MAX_RECORDS:
                    raise ValueError("MCPTI record count exceeds the bounded session")
                self._rows.append(row)
            dropped = C.c_size_t()
            self._call("mcptiActivityGetNumDroppedRecords", context, stream, C.byref(dropped))
            self._dropped += dropped.value
        except BaseException as error:
            self._errors.append(str(error))
        finally:
            self._buffers.pop(address, None)

    def begin(self):
        if not self._ready:
            raise RuntimeError("MCPTI callback registration did not succeed")
        if not self._session.acquire(blocking=False):
            raise RuntimeError("MCPTI activity collection is already active")
        try:
            self._call("mcptiActivityFlushAll", 1)
            if self._buffers:
                self._errors.append("MCPTI retained activity buffers after forced drain")
            if self._errors or self._dropped:
                raise ValueError("MCPTI has unreconciled errors from the preceding activity session")
            self._rows, self._errors, self._dropped = [], [], 0
            self._active = True
            self._owner_thread = threading.get_ident()
            for kind in _KINDS:
                self._call("mcptiActivityEnable", kind)
                self._enabled.append(kind)
        except BaseException:
            self._disable()
            self._active = False
            self._owner_thread = None
            self._session.release()
            raise

    def _disable(self):
        errors = []
        for kind in reversed(self._enabled):
            try:
                self._call("mcptiActivityDisable", kind)
            except BaseException as error:
                errors.append(str(error))
        self._enabled = []
        self._errors.extend(errors)

    def finish(self):
        if not self._active or self._owner_thread != threading.get_ident():
            raise RuntimeError("MCPTI activity collection must finish on its owning thread")
        failure = None
        try:
            self._call("mcptiActivityFlushAll", 1)
            self._disable()
            self._call("mcptiActivityFlushAll", 1)
            if self._buffers:
                self._errors.append("MCPTI retained activity buffers after forced drain")
            if self._errors or self._dropped:
                raise ValueError(f"MCPTI incomplete activity: dropped={self._dropped}, errors={self._errors}")
        except BaseException as error:
            failure = error
        finally:
            self._disable()
            # Freeze this session while still holding ownership. A subsequent
            # begin() may replace the collector's rows immediately after release.
            snapshot = {"source": "mcpti_activity", "api_version": self.version,
                        "dropped_records": self._dropped, "pending_buffers": len(self._buffers),
                        "records": list(self._rows)}
            if failure is not None:
                snapshot['collection_errors'] = [*self._errors, str(failure)]
            self._active = False
            self._owner_thread = None
            self._session.release()
        if failure is not None:
            failure.activity_snapshot = snapshot
            raise failure
        return snapshot




def activity_collector(library: str) -> McptiActivity:
    global _COLLECTOR
    with _CREATION:
        if _COLLECTOR is None:
            _COLLECTOR = McptiActivity(library)
        if not _COLLECTOR._ready:
            raise RuntimeError("MCPTI callback registration did not succeed")
        if _COLLECTOR.library != str(Path(library).resolve(strict=True)):
            raise ValueError("MCPTI collector cannot change its admitted library in one process")
        return _COLLECTOR
