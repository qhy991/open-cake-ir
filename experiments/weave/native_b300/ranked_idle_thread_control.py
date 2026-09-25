"""Negative control for the ranked multi-device synchronization diagnosis.

Copy beside an unchanged `adapter_run.py` bundle and invoke through its
four-GPU broker lease. One idle Python thread is established before the
launch; it calls no CUDA API. The original adapter's sequential status wait
is preserved. A pass here would show that concurrent CUDA synchronization
was not necessary for the observed schedule to complete.
"""
from __future__ import annotations

from threading import Event, Thread

import adapter_run as adapter


class IdleThreadRuntime(adapter.Runtime):
    def __init__(self):
        super().__init__()
        self.stop_idle = Event()
        self.idle_ready = Event()

        def idle():
            self.idle_ready.set()
            self.stop_idle.wait()

        self.idle_thread = Thread(target=idle, daemon=True)
        self.idle_thread.start()
        if not self.idle_ready.wait(timeout=5):
            raise RuntimeError('negative-control idle thread did not start')

    def close(self):
        self.stop_idle.set()
        self.idle_thread.join(timeout=5)
        super().close()


def main():
    runtime = IdleThreadRuntime()
    try:
        runtime.run()
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
