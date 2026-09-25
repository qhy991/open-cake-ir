"""Diagnostic concurrent completion wait for a frozen ranked CUDA bundle.

Copy beside `adapter_run.py` and invoke only inside the four-GPU broker job.
The four wait threads are established before any kernel launch. After the
unchanged combined launch, each thread synchronizes its own logical device;
the main thread reads statuses only after all four waits have finished.
This tests the development adapter's sequential wait, not performance.
"""
from __future__ import annotations

import ctypes as C
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import adapter_run as adapter


class ConcurrentStatusRuntime(adapter.Runtime):
    def __init__(self):
        super().__init__()
        self.pool = ThreadPoolExecutor(max_workers=adapter.R)
        barrier = Barrier(adapter.R + 1)
        workers = [self.pool.submit(barrier.wait) for _ in range(adapter.R)]
        barrier.wait()
        for worker in workers:
            worker.result()

    def statuses(self, mailboxes, offset, contexts):
        def wait_rank(rank):
            adapter.checked(self.cuda.cudaSetDevice(rank),
                            'parallel rank completion owner')
            adapter.checked(self.cuda.cudaDeviceSynchronize(),
                            'parallel rank completion')

        waits = [self.pool.submit(wait_rank, rank) for rank in range(adapter.R)]
        for wait in waits:
            wait.result()
        values = []
        for rank in range(adapter.R):
            value = C.c_int32()
            pointer = C.c_void_p(mailboxes[rank].pointer.value + offset)
            adapter.checked(self.cuda.cudaSetDevice(rank), 'status owner')
            adapter.checked(self.cuda.cudaMemcpy(
                C.cast(C.byref(value), C.c_void_p), pointer, 4, 2),
                'status D2H')
            values.append(value.value)
        return tuple(values)

    def close(self):
        try:
            super().close()
        finally:
            self.pool.shutdown(wait=True)


def main():
    runtime = ConcurrentStatusRuntime()
    try:
        runtime.run()
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
