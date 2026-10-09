"""Host phase observations for a dedicated, single-thread C550 diagnostic process.

This changes host timing. Its observations are not performance receipts and must
not replace a frozen A/A result. No device call is initiated by this module.
"""
from contextlib import contextmanager, ExitStack
import threading
import time
from unittest.mock import patch


class HostTimeline:
    def __init__(self, *, wall_clock=time.perf_counter_ns, cpu_clock=time.thread_time_ns):
        self.rows = []
        self.wall_clock, self.cpu_clock = wall_clock, cpu_clock
        self.thread = threading.get_ident()

    def call(self, phase, function, *args, **kwargs):
        if threading.get_ident() != self.thread:
            raise RuntimeError('host timing diagnostics require one caller thread')
        if len(self.rows) >= 256:
            raise RuntimeError('host timing diagnostic record limit reached')
        row = {'phase': phase, 'sequence': len(self.rows),
               'wall_start_ns': self.wall_clock(), 'cpu_start_ns': self.cpu_clock()}
        self.rows.append(row)
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            row['error_type'] = type(error).__name__
            raise
        else:
            row['error_type'] = None
            return result
        finally:
            row['cpu_end_ns'] = self.cpu_clock()
            row['wall_end_ns'] = self.wall_clock()


_ACTIVE = threading.Lock()


@contextmanager
def instrument_host_phases(loaded, torch_module, timeline):
    """Observe an existing timer in a dedicated process; restore all attributes.

    The caller owns source admission, GPU locking, sample count, original
    correctness and output retention. Use the ordinary timer and launch path.
    Do not run other Torch work concurrently while its Event factory is wrapped.
    """
    from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
    if type(loaded) is not LoadedMetaxCandidate or loaded.closed:
        raise ValueError('diagnostic requires an open single MACA dispatch')
    if threading.get_ident() != timeline.thread:
        raise RuntimeError('host timing diagnostics require one caller thread')
    if not _ACTIVE.acquire(blocking=False):
        raise RuntimeError('host timing diagnostics cannot overlap')
    try:
        event_factory = torch_module.cuda.Event
        prepare = loaded.prepare_arguments
        api = loaded._api
        event_count = 0

        class ObservedEvent:
            def __init__(self, *args, **kwargs):
                nonlocal event_count
                if event_count >= 2:
                    raise ValueError('diagnostic permits one event pair')
                self.role = ('begin', 'end')[event_count]
                event_count += 1
                self.event = event_factory(*args, **kwargs)

            def record(self, *args, **kwargs):
                return timeline.call(self.role + '_record', self.event.record, *args, **kwargs)

            def synchronize(self):
                return timeline.call(self.role + '_synchronize', self.event.synchronize)

            def elapsed_time(self, other):
                return self.event.elapsed_time(other.event)

        class ObservedAPI:
            def __getattr__(self, name):
                return getattr(api, name)

            def mcModuleLaunchKernel(self, *args):
                return timeline.call('native_submit', api.mcModuleLaunchKernel, *args)

        def prepare_arguments(*args, **kwargs):
            return timeline.call('prepare_arguments', prepare, *args, **kwargs)

        with ExitStack() as restore:
            restore.enter_context(patch.object(loaded, 'prepare_arguments', prepare_arguments))
            restore.enter_context(patch.object(loaded, '_api', ObservedAPI()))
            restore.enter_context(patch.object(torch_module.cuda, 'Event', ObservedEvent))
            yield timeline
    finally:
        _ACTIVE.release()
