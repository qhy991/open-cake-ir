"""Diagnostic host-delay control for the ranked four-device launch.

Copy beside a frozen `adapter_run.py` bundle and invoke inside its broker
lease. The CUDA source, four launch calls and status synchronization remain
the bundle's; only the host waits 10 ms after the combined launch returns.
This tests whether the unsynchronized launch-to-wait transition changes
progress. It is not a timing measurement or a production Executor policy.
"""
from __future__ import annotations

from dataclasses import replace
import time

import adapter_run as adapter


class PausedRuntime(adapter.Runtime):
    def load_source(self, lowered):
        executable = super().load_source(lowered)
        original_bind = executable.bind

        def bind(inputs, mailboxes, plans, contexts):
            launch = original_bind(inputs, mailboxes, plans, contexts)

            def launch_then_pause(bound):
                launch(bound)
                time.sleep(.01)

            return launch_then_pause

        return replace(executable, bind=bind)


def main():
    runtime = PausedRuntime()
    try:
        runtime.run()
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
