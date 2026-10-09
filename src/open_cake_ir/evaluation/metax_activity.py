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
import time


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
_Timestamp = C.CFUNCTYPE(C.c_uint64)
_Complete = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_void_p, C.c_size_t, C.c_size_t)


class McptiActivity:
    """One process-owned callback collector bound to an admitted absolute library."""

    def __init__(self, library: str, *, timestamp_callback=None, timestamp_source=None):
        with _CREATION:
            if _COLLECTOR is not None:
                raise RuntimeError("MCPTI callbacks already have a process owner; use activity_collector")
            self._initialize(library, timestamp_callback=timestamp_callback, timestamp_source=timestamp_source)

    def _initialize(self, library: str, *, timestamp_callback=None, timestamp_source=None):
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
        if timestamp_callback is not None:
            if (not isinstance(timestamp_callback, _Timestamp) or not bool(timestamp_callback)
                    or not isinstance(timestamp_source, str) or not timestamp_source.strip()):
                raise ValueError('MCPTI custom clock requires a nonnull uint64(void) callback and source')
            signatures['mcptiActivityRegisterTimestampCallback'] = [_Timestamp]
        elif timestamp_source is not None:
            raise ValueError('MCPTI clock source requires its explicit callback')
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
        self._flushes = []
        self._buffer_observations = []
        self._phase = 'initialization'
        self._active = False
        self._owner_thread = None
        # The SDK clock remains the default. Custom clocks are explicit diagnostics,
        # registered before any activity and kept alive for the process lifetime.
        self._timestamp_source = timestamp_source or 'sdk_default'
        self._timestamp_callback = timestamp_callback
        self._requested_callback = _Request(self._requested)
        self._completed_callback = _Complete(self._completed)
        # Keep callbacks alive even if registration reports an ambiguous failure.
        _COLLECTOR = self
        self._call("mcptiActivityRegisterCallbacks", self._requested_callback, self._completed_callback)
        if self._timestamp_callback is not None:
            self._call('mcptiActivityRegisterTimestampCallback', self._timestamp_callback)
        if self._errors:
            raise ValueError(f'MCPTI timestamp registration failed: {self._errors}')
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
        sequence = getattr(self, '_buffer_sequence', 0) + 1
        self._buffer_sequence = sequence
        phase = getattr(self, '_phase', 'unspecified')
        record_count = 0
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
                row = {"kind": kind, "capture_buffer": sequence}
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
                        dynamic_shared_bytes=item.dynamic_shared, local_bytes_per_thread=item.local_per_thread,
                        completed_ns=item.completed, grid_id=item.grid_id,
                        raw_prefix_hex=C.string_at(at, C.sizeof(_Kernel8Prefix)).hex())
                elif kind in (4, 5):
                    if at + C.sizeof(_ApiActivity) > address + valid:
                        raise ValueError("MCPTI API record is truncated")
                    item = _ApiActivity.from_address(at)
                    row.update(cbid=item.cbid, start_ns=item.start, end_ns=item.end,
                               correlation=item.correlation, return_value=item.return_value)
                if len(self._rows) >= _MAX_RECORDS:
                    raise ValueError("MCPTI record count exceeds the bounded session")
                self._rows.append(row)
                record_count += 1
            dropped = C.c_size_t()
            self._call("mcptiActivityGetNumDroppedRecords", context, stream, C.byref(dropped))
            self._dropped += dropped.value
        except BaseException as error:
            self._errors.append(str(error))
        finally:
            if not hasattr(self, '_buffer_observations'):
                self._buffer_observations = []
            if len(self._buffer_observations) < _MAX_RECORDS:
                self._buffer_observations.append({
                    'phase': phase, 'stream': stream, 'sequence': sequence,
                    'buffer_bytes': size, 'valid_bytes': valid, 'record_count': record_count})
            else:
                self._errors.append('MCPTI buffer observation count exceeds the bounded session')
            self._buffers.pop(address, None)

    def _flush(self, phase, flag=0):
        """Retain host collection order; these timestamps are not the device timer."""
        self._phase = phase
        if not hasattr(self, '_flushes'):
            self._flushes = []
        observation = {'phase': phase, 'flag': flag, 'host_start_ns': time.monotonic_ns(),
                       'pending_buffers_before': len(self._buffers)}
        self._flushes.append(observation)
        try:
            self._call('mcptiActivityFlushAll', flag)
        except BaseException as error:
            observation['error'] = str(error)
            raise
        finally:
            observation.update(host_end_ns=time.monotonic_ns(),
                               pending_buffers_after=len(self._buffers), records_after=len(self._rows))

    def _drain_rejected(self):
        # Forced flush is teardown evidence only. It can expose incomplete records,
        # so its rows remain in a rejected snapshot, never an accepted measurement.
        try:
            self._flush('rejected_session_teardown', 1)
        except BaseException as error:
            self._errors.append(str(error))

    def _snapshot(self, failure=None):
        snapshot = {'source': 'mcpti_activity', 'api_version': self.version,
                    'dropped_records': self._dropped, 'pending_buffers': len(self._buffers),
                    'records': list(self._rows),
                    'timestamp_source': getattr(self, '_timestamp_source', None),
                    'collection': {'flush_policy': 'completed_records_only',
                        'flushes': list(getattr(self, '_flushes', [])),
                        'buffers': list(getattr(self, '_buffer_observations', []))}}
        if failure is not None:
            snapshot['collection_errors'] = [*self._errors, str(failure)]
        return snapshot

    def begin(self):
        if not self._ready:
            raise RuntimeError("MCPTI callback registration did not succeed")
        if not self._session.acquire(blocking=False):
            raise RuntimeError("MCPTI activity collection is already active")
        try:
            self._flushes, self._buffer_observations = [], []
            self._flush('begin_drain')
            if self._buffers:
                self._errors.append("MCPTI retained activity buffers after completed-record drain")
            if self._errors or self._dropped:
                raise ValueError("MCPTI has unreconciled errors from the preceding activity session")
            self._rows, self._errors, self._dropped = [], [], 0
            self._buffer_observations, self._buffer_sequence = [], 0
            self._active = True
            self._phase = 'active_collection'
            self._owner_thread = threading.get_ident()
            for kind in _KINDS:
                self._call("mcptiActivityEnable", kind)
                self._enabled.append(kind)
        except BaseException as error:
            self._disable()
            self._drain_rejected()
            error.activity_snapshot = self._snapshot(error)
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
            self._flush('finish_before_disable')
            self._disable()
            self._flush('finish_after_disable')
            if self._buffers:
                self._errors.append("MCPTI retained activity buffers after completed-record drain")
            if self._errors or self._dropped:
                raise ValueError(f"MCPTI incomplete activity: dropped={self._dropped}, errors={self._errors}")
        except BaseException as error:
            failure = error
        finally:
            self._disable()
            if failure is not None:
                self._drain_rejected()
            # Freeze this session while still holding ownership. A subsequent
            # begin() may replace the collector's rows immediately after release.
            snapshot = self._snapshot(failure)
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
        if (_COLLECTOR._timestamp_callback is not None
                or _COLLECTOR._timestamp_source != 'sdk_default'):
            raise ValueError('MCPTI production collection cannot reuse a diagnostic clock')
        if _COLLECTOR.library != str(Path(library).resolve(strict=True)):
            raise ValueError("MCPTI collector cannot change its admitted library in one process")
        return _COLLECTOR
