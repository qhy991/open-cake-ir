"""MCPTI timing must account for the complete observed device sequence."""
from copy import deepcopy
from pathlib import Path
import ctypes
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.metax_activity import McptiActivity, _ApiActivity, _Kernel8Prefix
from open_cake_ir.evaluation.metax_benchmark import dispatch_samples, kernel_records


def kernel(name, correlation, start, *, grid=(8, 1, 1), block=(64, 1, 1)):
    return dict(kind=10, name=name, start_ns=start, end_ns=start+2048, device=0, context=0,
                stream=0, correlation=correlation, grid=list(grid), block=list(block),
                registers_per_thread=16, static_shared_bytes=0, dynamic_shared_bytes=0,
                local_bytes_per_thread=0)


def capture(*kernels):
    return dict(source="mcpti_activity", api_version=18, dropped_records=0,
                records=[*kernels, *[dict(kind=5, correlation=k["correlation"], return_value=0) for k in kernels]])


class McptiMeasurements(unittest.TestCase):
    def setUp(self):
        self.reset = kernel("fill", 10, 1000, grid=(32768,1,1), block=(256,1,1))
        self.raw = capture(self.reset, kernel("cake", 11, 4000),
                           {**self.reset, "correlation": 12, "start_ns": 7000, "end_ns": 9048},
                           kernel("cake", 13, 10000))

    def samples(self, raw):
        return dispatch_samples(raw, kernel_name="cake", grid=(8,1,1), block=(64,1,1),
                                repeats=2, reset_record=self.reset)

    def test_complete_cold_sequence_uses_device_interval_only(self):
        self.assertEqual(self.samples(self.raw), [0.002048, 0.002048])
        # A host API's duration is not included or used to rank the device sample.
        self.raw["records"][-1].update(start_ns=1, end_ns=1000000000)
        self.assertEqual(self.samples(self.raw), [0.002048, 0.002048])

    def test_missing_extra_and_non_kernel_work_never_disappear(self):
        variants = []
        missing = deepcopy(self.raw); missing["records"].pop(1); variants.append(missing)
        extra = deepcopy(self.raw); extra["records"].append(kernel("extra", 99, 14000)); variants.append(extra)
        copy = deepcopy(self.raw); copy["records"].append({"kind":1}); variants.append(copy)
        no_api = deepcopy(self.raw); no_api["records"].pop(); variants.append(no_api)
        dropped = deepcopy(self.raw); dropped["dropped_records"] = 1; variants.append(dropped)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.samples(raw)

    def test_wrong_identity_zero_timestamps_and_repeated_ids_are_refused(self):
        for mutation in ({"name":"cake_extra"}, {"grid":[9,1,1]}, {"block":[128,1,1]},
                         {"start_ns":0}, {"end_ns":4000}, {"correlation":10},
                         {"device":1}, {"stream":1}):
            raw=deepcopy(self.raw);raw["records"][1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.samples(raw)

    def test_reset_identity_and_order_are_observed_not_subtracted(self):
        for mutation in ({"name":"some_other_kernel"}, {"stream":2}, {"end_ns":5000}):
            raw=deepcopy(self.raw);raw["records"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.samples(raw)

    def test_warm_capture_is_still_exact_name_and_complete_count(self):
        raw=capture(kernel("cake",1,1000),kernel("cake",2,5000))
        self.assertEqual(dispatch_samples(raw,kernel_name="cake",grid=(8,1,1),block=(64,1,1),
                                         repeats=2,reset_record=None),[0.002048]*2)
        with self.assertRaises(ValueError):
            dispatch_samples(raw,kernel_name="cak",grid=(8,1,1),block=(64,1,1),repeats=2,reset_record=None)

    def test_device_drop_and_failed_launch_api_invalidate_attribution_too(self):
        for change in ({"api_version":19},{"dropped_records":True},{"dropped_records":2}):
            with self.subTest(change=change),self.assertRaises(ValueError):kernel_records({**self.raw,**change})
        raw=deepcopy(self.raw);raw["records"][-1]["return_value"]=2
        with self.assertRaisesRegex(ValueError,"failed"):kernel_records(raw)

    def test_python_prefix_agrees_with_the_qualified_sdk_layout(self):
        # These are the installed SDK ABI offsets, not C550 hardware constants.
        observed={name:getattr(_Kernel8Prefix,name).offset for name in
                  ("registers","start","end","device","stream","grid","block","static_shared","local_per_thread","correlation","grid_id","name")}
        self.assertEqual(observed,dict(registers=6,start=16,end=24,device=40,stream=48,
            grid=52,block=64,static_shared=76,local_per_thread=84,correlation=92,grid_id=96,name=104))
        self.assertEqual(ctypes.sizeof(_ApiActivity),40)

    def test_callback_errors_are_retained_instead_of_becoming_silent_native_exceptions(self):
        collector=object.__new__(McptiActivity)
        collector._buffers={};collector._errors=[];collector._rows=[];collector._dropped=0
        collector._completed(None,0,123,8,99)
        self.assertIn("unowned or invalid buffer",collector._errors[0])
        with patch("open_cake_ir.evaluation.metax_activity.C.create_string_buffer",side_effect=MemoryError("capacity")):
            pointer=ctypes.c_void_p();size=ctypes.c_size_t();count=ctypes.c_size_t()
            collector._requested(ctypes.pointer(pointer),ctypes.pointer(size),ctypes.pointer(count))
        self.assertEqual((pointer.value,size.value,count.value),(None,0,0))
        self.assertIn("capacity",collector._errors[-1])
