"""The MACA loader retains native code and refuses ABI drift before dispatch."""

from hashlib import sha256
from types import SimpleNamespace
import unittest

from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
from tests.contracts.test_metax_binary import bundle


class API:
    def __init__(self):
        self.loads, self.launches, self.unloads = [], [], []
    def mcModuleLoadData(self, pointer, image):
        self.loads.append(image.raw)
        pointer._obj.value = 17
        return 0
    def mcModuleGetFunction(self, pointer, module, name):
        pointer._obj.value = 19
        return 0
    def mcFuncGetAttribute(self, pointer, attribute, function):
        pointer._obj.value = 16 if attribute == 4 else 0
        return 0
    def mcModuleLaunchKernel(self, *args):
        self.launches.append(args)
        return 0
    def mcModuleUnload(self, module):
        self.unloads.append(module.value)
        return 0


class MetaxDriverTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.native = bundle()
        self.api = API()
        self.candidate = SimpleNamespace(target="xcore1002", entry_point="kernel",
            launch_spec_sha256="c" * 64, artifact_payloads={"mcfatbin": self.payload},
            artifact_roles={"mcfatbin": sha256(self.payload).hexdigest()})
        self.manifest = SimpleNamespace(target="xcore1002", kernel_name="kernel", canonical_sha256="c" * 64,
            grid=(1, 1, 1), block=(64, 1, 1), dynamic_shared_memory_bytes=0,
            hidden_null_pointer_parameters=0, tensor_abi=(("x", (128,), "fp32", "input"),))
        self.admission = SimpleNamespace(target="xcore1002", device_name="MetaX C550",
                                         device_arch="xcore1002", warp_size=64)

    def load(self):
        return LoadedMetaxCandidate.load(self.candidate, self.manifest, self.admission, api=self.api)

    def argument(self, **overrides):
        fields = dict(shape=(128,), dtype="torch.float32", device=SimpleNamespace(type="cuda", index=0),
                      is_contiguous=lambda: True, data_ptr=lambda: 4096)
        return SimpleNamespace(**{**fields, **overrides})

    def test_the_runtime_receives_native_code_and_unloads_it_once(self):
        loaded = self.load()
        self.assertEqual(self.api.loads, [self.native + b"\0"])
        loaded.launch([self.argument()], tensor_contract=self.manifest)
        self.assertEqual(loaded.launch_calls, 1)
        loaded.close()
        loaded.close()
        self.assertEqual(self.api.unloads, [17])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            loaded.launch([self.argument()], tensor_contract=self.manifest)

    def test_shape_dtype_and_device_mismatch_are_rejected_before_launch(self):
        loaded = self.load()
        for argument in (self.argument(shape=(64,)), self.argument(dtype="torch.float16"),
                         self.argument(device=SimpleNamespace(type="cpu", index=None)),
                         self.argument(is_contiguous=lambda: False)):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                loaded.launch([argument], tensor_contract=self.manifest)
        self.assertEqual(self.api.launches, [])
        loaded.close()

    def test_narrow_and_integer_tensor_abis_keep_their_declared_runtime_dtype(self):
        for dtype, runtime_dtype in (("fp16", "torch.float16"), ("bf16", "torch.bfloat16"),
                                     ("int32", "torch.int32")):
            with self.subTest(dtype=dtype):
                self.manifest.tensor_abi = (("x", (128,), dtype, "input"),)
                loaded = self.load()
                before = len(self.api.launches)
                loaded.launch([self.argument(dtype=runtime_dtype)], tensor_contract=self.manifest)
                with self.assertRaises(ValueError):
                    loaded.launch([self.argument()], tensor_contract=self.manifest)
                self.assertEqual(len(self.api.launches), before + 1)
                loaded.close()

    def test_another_native_device_is_not_admitted_by_its_compatibility_capability(self):
        self.admission.device_arch = "xcore1000"
        with self.assertRaisesRegex(ValueError, "device admission differs"):
            self.load()
        self.assertEqual(self.api.loads, [])

    def test_an_inherited_cuda_hidden_pointer_count_is_rejected(self):
        self.manifest.hidden_null_pointer_parameters = 2
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(self.api.loads, [])
