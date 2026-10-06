"""Real assay adapters with CPU-only runtime boundaries; no substitute worker class."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from open_cake_ir.evaluation import metax_event_benchmark as event


class RealAdapterTests(unittest.TestCase):
    def test_native_and_torch_reset_adapters_select_the_real_submission_and_record_coverage(self):
        manifest=SimpleNamespace(target='xcore1002',aligned_variant=False,kernel_name='candidate',
                                 grid=(1,1,1),block=(64,1,1),dynamic_shared_memory_bytes=0)
        torch=SimpleNamespace(version=SimpleNamespace(maca='captured'),float32=object(),
            empty=lambda *args,**kwargs:object(),cuda=SimpleNamespace(
                default_stream=lambda device:SimpleNamespace(cuda_stream=0),
                get_device_properties=lambda device:SimpleNamespace(L2_cache_size=8388608)))
        for cls,torch_reset in [(event.MacaNativeEventBenchmark,False),(event.MacaTorchResetEventBenchmark,True),(event.MacaGatedEventBenchmark,True)]:
            assay=cls(manifest,l2_cache_bytes=8388608)
            loaded=SimpleNamespace(manifest=manifest)
            with patch.dict(sys.modules,torch=torch),patch(
                    'open_cake_ir.evaluation.metax_native_events.capture',return_value=[.001]*5) as capture:
                self.assertEqual(assay.capture_loaded_cohort(loaded,[object() for _ in range(16)],
                    dry_run_iters=11,repeat_iters=5),[.001]*5)
                self.assertEqual(capture.call_args.kwargs['torch_reset'],torch_reset)
            gated=cls is event.MacaGatedEventBenchmark
            self.assertEqual(assay.last_activity['stream'],'owned_nonblocking' if gated else 0)
            event.validate_cohort({'native_activity':assay.last_activity,'samples_ms':[.001]*5},
                                  manifest,sample_count=5,native=not torch_reset,torch_reset=torch_reset and not gated,gated=gated)
