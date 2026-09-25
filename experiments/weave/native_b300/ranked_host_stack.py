"""Bounded host-stack capture for an unchanged ranked CUDA bundle.

Copy beside `adapter_run.py` and invoke only through a four-GPU broker job.
The target process runs the ordinary adapter without a background thread.
Its supervisor sends SIGUSR1 after eight seconds, retaining the Python stack,
then stops this exact diagnostic child after a further twenty seconds.
No GPU timing or candidate correctness is inferred from this probe.
"""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def target():
    log = HERE / 'host_stack.log'
    with log.open('x') as stream:
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True)
        import adapter_run
        if os.environ.get('CAKE_RANK_TRACE') == '1':
            original_statuses = adapter_run.Runtime.statuses

            def traced_statuses(runtime, mailboxes, offset, contexts):
                with (HERE / 'rank_wait_trace.jsonl').open('x', buffering=1) as trace:
                    for rank in range(adapter_run.R):
                        trace.write(json.dumps({'rank': rank,
                                                'phase': 'before_device_sync'}) + '\n')
                        adapter_run.checked(runtime.cuda.cudaSetDevice(rank),
                                            'traced completion owner')
                        adapter_run.checked(runtime.cuda.cudaDeviceSynchronize(),
                                            'traced rank completion')
                        trace.write(json.dumps({'rank': rank,
                                                'phase': 'after_device_sync'}) + '\n')
                return original_statuses(runtime, mailboxes, offset, contexts)

            adapter_run.Runtime.statuses = traced_statuses
        adapter_run.device_run()


def supervise():
    result = HERE / 'host_stack_supervisor.json'
    output = HERE / 'host_stack_target.log'
    if result.exists() or output.exists() or (HERE / 'host_stack.log').exists():
        raise RuntimeError('host-stack diagnostic output must be create-only')
    interrupted = False
    terminated = False
    with output.open('x') as stream:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'target'],
            cwd=HERE, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGUSR1)
                interrupted = True
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    child.terminate()
                    terminated = True
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
    result.write_text(json.dumps({
        'target_exit_code': child.returncode,
        'host_stack_signal_sent': interrupted,
        'diagnostic_child_terminated': terminated,
        'scope': 'one broker-contained host stack probe; no GPU result',
    }, indent=2) + '\n')
    print(f'host-stack probe child exit={child.returncode}; '
          f'signal={interrupted}; terminated={terminated}')
    return 0 if child.returncode == 0 else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['target']:
        target()
    elif not sys.argv[1:]:
        raise SystemExit(supervise())
    else:
        raise SystemExit('usage: python3 ranked_host_stack.py')
