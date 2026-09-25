"""Read a running development mailbox through nonblocking CUDA copy streams.

Copy beside one `adapter_run.py` bundle and invoke it inside that bundle's
four-GPU broker job. Snapshots are diagnostic observations, not a consistent
memory snapshot, an oracle, or a timing result. The probe never launches a
second kernel or changes a mailbox value.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import replace
import json
from threading import Event, Thread
import time

import adapter_run as adapter


def mailbox_type(tokens: int, width: int, routes: int, ranks: int):
    payload_cap = (ranks - 1) * tokens
    task_cap = ranks * tokens * routes

    class Mailbox(C.Structure):
        _fields_ = [
            ('payload_tail', C.c_int), ('task_tail', C.c_int),
            ('task_head', C.c_int), ('dispatch_sources_done', C.c_int),
            ('dispatch_cursor', C.c_int), ('dispatch_completed', C.c_int),
            ('comm_done', C.c_int), ('combine_cursor', C.c_int),
            ('steal_permits', C.c_int), ('stolen', C.c_int),
            ('compute_completed', C.c_int), ('status', C.c_int),
            ('payload_ready', C.c_int * payload_cap),
            ('payload', C.c_uint16 * (payload_cap * width)),
            ('task_ready', C.c_int * task_cap),
            ('source', C.c_int * task_cap), ('token', C.c_int * task_cap),
            ('route_slot', C.c_int * task_cap), ('expert', C.c_int * task_cap),
            ('payload_slot', C.c_int * task_cap),
            ('contributions', C.c_float * (tokens * routes * width)),
            ('contribution_ready', C.c_int * (tokens * routes)),
            ('chunk_completed', C.c_int * tokens),
            ('output', C.c_uint16 * (tokens * width)),
        ]

    return Mailbox


class ProbedRuntime(adapter.Runtime):
    def __init__(self):
        super().__init__()
        shape = self.case['shape']
        self.mailbox_type = mailbox_type(shape['T'], shape['H'], shape['K'], shape['R'])
        checks = {
            'mailbox_bytes': C.sizeof(self.mailbox_type),
            'status_offset': self.mailbox_type.status.offset,
            'output_offset': self.mailbox_type.output.offset,
            'payload_tail_offset': self.mailbox_type.payload_tail.offset,
            'task_tail_offset': self.mailbox_type.task_tail.offset,
            'steal_permits_offset': self.mailbox_type.steal_permits.offset,
            'stolen_offset': self.mailbox_type.stolen.offset,
            'compute_completed_offset': self.mailbox_type.compute_completed.offset,
            'dispatch_done_offset': self.mailbox_type.dispatch_sources_done.offset,
        }
        for key, expected in checks.items():
            if self.query[key]() != expected:
                raise RuntimeError(f'probe mailbox layout differs at {key}')
        cuda = self.cuda
        cuda.cudaHostAlloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint]
        cuda.cudaFreeHost.argtypes = [C.c_void_p]
        cuda.cudaStreamCreateWithFlags.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
        cuda.cudaStreamDestroy.argtypes = [C.c_void_p]
        cuda.cudaMemcpyAsync.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t,
                                         C.c_int, C.c_void_p]
        cuda.cudaStreamQuery.argtypes = [C.c_void_p]
        self.probe_stop = Event()
        self.probe_thread = None
        self.probe_buffers = {}

    def load_source(self, lowered):
        executable = super().load_source(lowered)
        original_bind = executable.bind

        def bind(inputs, mailboxes, plans, contexts):
            launch = original_bind(inputs, mailboxes, plans, contexts)

            def launch_with_probe(bound):
                launch(bound)
                self.start_probe(mailboxes)

            return launch_with_probe

        return replace(executable, bind=bind)

    def start_probe(self, mailboxes):
        # Leave CUDA stream and pinned-memory creation until after the kernel
        # has started. Prelaunch resource creation changed the failing run's
        # interleaving and hid the progress stall in the first probe revision.
        self.probe_mailboxes = mailboxes
        self.probe_thread = Thread(target=self.probe, daemon=True)
        self.probe_thread.start()

    def allocate_probe_buffers(self):
        size = C.sizeof(self.mailbox_type)
        for rank in range(adapter.R):
            adapter.checked(self.cuda.cudaSetDevice(rank), 'probe device owner')
            host, stream = C.c_void_p(), C.c_void_p()
            adapter.checked(self.cuda.cudaHostAlloc(C.byref(host), size, 0),
                            'probe pinned host allocation')
            adapter.checked(self.cuda.cudaStreamCreateWithFlags(C.byref(stream), 1),
                            'probe nonblocking stream')
            self.probe_buffers[rank] = (host, stream,
                                        self.probe_mailboxes[rank].pointer)

    def probe(self):
        try:
            for seconds in (5, 10, 20):
                if self.probe_stop.wait(seconds if seconds == 5 else seconds - previous):
                    return
                previous = seconds
                if not self.probe_buffers:
                    (adapter.HERE / 'probe_stage.json').write_text(
                        '{"stage":"allocating_after_launch"}\n')
                    self.allocate_probe_buffers()
                    (adapter.HERE / 'probe_stage.json').write_text(
                        '{"stage":"copying"}\n')
                if not self.snapshot(seconds):
                    return
        except Exception as error:
            (adapter.HERE / 'probe_error.json').write_text(json.dumps({
                'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')

    def snapshot(self, seconds: int) -> bool:
        size = C.sizeof(self.mailbox_type)
        for rank, (host, stream, device) in self.probe_buffers.items():
            adapter.checked(self.cuda.cudaSetDevice(rank), 'probe copy owner')
            adapter.checked(self.cuda.cudaMemcpyAsync(host, device, size, 2, stream),
                            'probe D2H enqueue')
        rows = {}
        pending = set(range(adapter.R))
        deadline = time.monotonic() + 3
        while pending and time.monotonic() < deadline:
            for rank in tuple(pending):
                host, stream, _ = self.probe_buffers[rank]
                adapter.checked(self.cuda.cudaSetDevice(rank), 'probe query owner')
                code = self.cuda.cudaStreamQuery(stream)
                if code == 0:
                    value = self.mailbox_type.from_buffer_copy(
                        C.string_at(host, size))
                    rows[rank] = {
                        name: getattr(value, name) for name in (
                            'payload_tail', 'task_tail', 'task_head',
                            'dispatch_sources_done', 'dispatch_cursor',
                            'dispatch_completed', 'comm_done', 'combine_cursor',
                            'steal_permits', 'stolen', 'compute_completed', 'status')
                    }
                    rows[rank]['payload_ready_count'] = sum(value.payload_ready)
                    rows[rank]['task_ready_count'] = sum(value.task_ready)
                    rows[rank]['contribution_ready'] = list(value.contribution_ready)
                    rows[rank]['chunk_completed'] = list(value.chunk_completed)
                    pending.remove(rank)
            if pending:
                time.sleep(.02)
        for rank in pending:
            rows[rank] = {'copy_pending_after_seconds': 3}
        (adapter.HERE / f'probe_{seconds:02d}s.json').write_text(json.dumps({
            'source_commit': adapter.SOURCE_COMMIT,
            'case_id': self.case['case_id'],
            'plans': self.case['plans'],
            'rows': rows,
            'scope': 'racy D2H diagnostic; no coherent snapshot or timing claim',
        }, indent=2) + '\n')
        return not pending

    def close(self):
        self.probe_stop.set()
        if self.probe_thread is not None:
            self.probe_thread.join(timeout=1)
        for rank, (host, stream, _) in self.probe_buffers.items():
            adapter.checked(self.cuda.cudaSetDevice(rank), 'probe cleanup owner')
            adapter.checked(self.cuda.cudaStreamDestroy(stream), 'probe stream free')
            adapter.checked(self.cuda.cudaFreeHost(host), 'probe host free')
        super().close()


def main():
    runtime = ProbedRuntime()
    try:
        runtime.run()
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
