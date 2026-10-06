"""Exercise the compiled submission loop with deterministic CPU runtime APIs."""
import ctypes as C
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import metax_native_events as native
from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.tasks.normalization.study import evaluation_policy
from tests.contracts import test_metax_paired as fixtures


class NativeSubmissionTests(unittest.TestCase):
    def run_cohort(self, fail_launch=None):
        calls, resets, closed, ordering = [], [], [], []
        Create = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_void_p), C.c_uint)
        Record = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p)
        Event = C.CFUNCTYPE(C.c_int, C.c_void_p)
        Elapsed = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.c_void_p, C.c_void_p)
        Reset = C.CFUNCTYPE(C.c_int, C.c_size_t, C.c_uint, C.c_size_t, C.c_void_p)
        Launch = C.CFUNCTYPE(C.c_int, C.c_void_p, *([C.c_uint] * 7), C.c_void_p,
                            C.POINTER(C.c_void_p), C.c_void_p)
        count = [0]
        def create(out, flags):
            count[0] += 1; out[0] = count[0]; return 0
        def record(event, stream):
            ordering.append(('event', event)); return 0
        def reset(address, value, words, stream):
            resets.append((address, value, words, stream)); ordering.append(('reset', words)); return 0
        def launch(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra):
            value = C.cast(args[0], C.POINTER(C.c_void_p)).contents.value
            calls.append(value); ordering.append(('target', value))
            return 32 if fail_launch == len(calls) else 0
        def elapsed(out, begin, end): out[0] = .001; return 0
        def destroy(event): closed.append(event); return 0
        callbacks = [Create(create), Record(record), Event(lambda event: 0),
                     Elapsed(elapsed), Event(destroy), Reset(reset), Launch(launch),
                     Event(lambda stream: 0)]
        api = (C.c_void_p * 8)(*(C.cast(fn, C.c_void_p).value for fn in callbacks))
        values = [C.c_void_p(i + 100) for i in range(16)]
        packed = [(C.c_void_p * 1)(C.addressof(value)) for value in values]
        pointers = (C.POINTER(C.c_void_p) * 16)(*packed)
        output, actual, phase = (C.c_float * 5)(), C.c_uint(), C.c_uint()
        helper = native.prepare_helper()
        status = helper.cake_maca_event_cohort(api, 9, (C.c_uint * 6)(1, 1, 1, 64, 1, 1),
            0, pointers, 11, 5, 1234, 8388608, output, C.byref(actual), C.byref(phase))
        return status, actual.value, phase.value, calls, resets, closed, ordering, list(output)

    def test_real_native_loop_uses_every_fresh_output_and_reset_before_each_sample(self):
        status, actual, phase, calls, resets, closed, order, samples = self.run_cohort()
        self.assertEqual((status, actual), (0, 16))
        self.assertEqual(calls, list(range(100, 116)))
        self.assertEqual(len(resets), 6)  # One reset warmup and five formal resets.
        self.assertTrue(all(row == (1234, 0x3f800000, 8388608, None) for row in resets))
        self.assertEqual(sorted(closed), [1, 2])
        for value in range(111, 116):
            at = order.index(('target', value))
            self.assertEqual(order[at - 2:at], [('reset', 8388608), ('event', 1)])
            self.assertEqual(order[at + 1], ('event', 2))
        self.assertTrue(all(value > 0 for value in samples))

    def test_failed_launch_does_not_return_partial_success_and_closes_both_events(self):
        status, actual, phase, calls, resets, closed, order, samples = self.run_cohort(fail_launch=13)
        self.assertEqual((status, actual, phase), (32, 12, 9))
        self.assertEqual(sorted(closed), [1, 2])

    def test_cpu_build_does_not_load_the_device_runtime(self):
        with patch.object(native, '_HELPER', None), patch.object(native, '_DIRECTORY', None):
            helper = native.prepare_helper()
            self.assertTrue(hasattr(helper, 'cake_maca_event_cohort'))

    def test_native_protocol_is_distinct_and_preserves_legacy_and_ten_sample_contract(self):
        f = fixtures.MacaPairedReceipts(); f.setUp(); self.addCleanup(f.doCleanups)
        policy = evaluation_policy(f.workload, metax_native_mean10=True)
        self.assertEqual(policy['paired_timing']['kind'], 'fixed_baseline_paired_maca_native_event_v1')
        protocol = paired_protocol(policy)
        self.assertEqual(len(protocol.pair_order) * protocol.samples_per_cohort, 10)
        self.assertEqual(protocol.statistic, 'mean')
        self.assertEqual(evaluation_policy(f.workload)['paired_timing']['kind'],
                         'fixed_baseline_paired_mcpti_dispatch_v1')
        with self.assertRaises(ValueError):
            evaluation_policy(f.workload, metax_mean10=True, metax_native_mean10=True)
