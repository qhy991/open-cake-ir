from __future__ import annotations

import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import CudaDeviceAdmission, LaunchableCandidate, LoadedCudaCandidate, launch_candidate_once, launch_cubin_once
from open_cake_ir.tasks.flash_kmeans.cuda import CudaTensorContract
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import parse_cuda_launch_manifest

CUBIN = b"\x7fELFopen-cake-driver-fixture"


class FakeTensor:
    def __init__(self, shape, dtype: str, pointer: int, device: str = "cuda:0") -> None:
        self.shape = shape
        self.dtype = dtype
        self._pointer = pointer
        self.device = device
        self.is_cuda = True

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return self._pointer


class Attributes:
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13


class FakeDriver:
    CUfunction_attribute = Attributes

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.context = 41
        self.attributes = {0: 1024, 1: 0, 6: 100, 10: 0, 11: 0, 12: 0, 13: 0}

    def cuCtxGetCurrent(self):
        self.calls.append(("cuCtxGetCurrent",))
        return 0, self.context

    def cuModuleLoadData(self, cubin):
        self.calls.append(("cuModuleLoadData", cubin))
        return 0, 42

    def cuModuleGetFunction(self, module, name):
        self.calls.append(("cuModuleGetFunction", module, name))
        return 0, 43

    def cuFuncGetAttribute(self, attribute, function):
        self.calls.append(("cuFuncGetAttribute", attribute, function))
        return 0, self.attributes[attribute]

    def cuFuncSetAttribute(self, function, attribute, value):
        self.calls.append(("cuFuncSetAttribute", function, attribute, value))
        return (0,)

    def cuLaunchKernel(self, *arguments):
        self.calls.append(("cuLaunchKernel", *arguments))
        return (0,)

    def cuModuleUnload(self, module):
        self.calls.append(("cuModuleUnload", module))
        return (0,)


def tensors(batch: int = 32, tokens: int = 65_536) -> list[FakeTensor]:
    return [
        FakeTensor((batch, tokens, 128), "torch.bfloat16", 1_000),
        FakeTensor((batch, 1_024, 128), "torch.bfloat16", 2_000),
        FakeTensor((batch, 1_024), "torch.float32", 3_000),
        FakeTensor((batch, tokens), "torch.int32", 4_000),
    ]


class CudaDriverContractTests(unittest.TestCase):
    def test_persistent_candidate_rejects_a_changed_cuda_context_before_launch(self) -> None:
        manifest = parse_cuda_launch_manifest(
            (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        )
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point=manifest.kernel_name,
            artifact_roles={"cubin": sha256(CUBIN).hexdigest()},
            launch_spec_sha256=manifest.canonical_sha256,
        )
        driver = FakeDriver()
        loaded = LoadedCudaCandidate.load(
            candidate,
            CUBIN,
            manifest,
            CudaDeviceAdmission(
                "NVIDIA B200", (10, 0), "GPU-fixture", "gpuq-000000000001", "exclusive"
            ),
            driver=driver,
        )
        driver.context = 99

        with self.assertRaisesRegex(ValueError, "CUDA context changed"):
            loaded.launch(
                tensors(),
                tensor_contract=CudaTensorContract(32, 65_536, 1_024, 128),
                stream=99,
            )

        driver.context = 41
        loaded.close(synchronize=lambda: None)

    def test_persistent_shape_bound_module_reuses_one_load_and_unloads_once(self) -> None:
        manifest = parse_cuda_launch_manifest(
            (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        )
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point=manifest.kernel_name,
            artifact_roles={"cubin": sha256(CUBIN).hexdigest()},
            launch_spec_sha256=manifest.canonical_sha256,
        )
        driver = FakeDriver()
        loaded = LoadedCudaCandidate.load(
            candidate,
            CUBIN,
            manifest,
            CudaDeviceAdmission(
                "NVIDIA B200", (10, 0), "GPU-fixture", "gpuq-000000000001", "exclusive"
            ),
            driver=driver,
        )
        contract = CudaTensorContract(32, 65_536, 1_024, 128)

        loaded.launch(tensors(), tensor_contract=contract, stream=99)
        loaded.launch(tensors(), tensor_contract=contract, stream=99)
        with self.assertRaisesRegex(ValueError, "tokens shape"):
            loaded.launch(
                tensors(tokens=512),
                tensor_contract=contract,
                stream=99,
            )
        loaded.close(synchronize=lambda: None)

        names = [call[0] for call in driver.calls]
        self.assertEqual(names.count("cuModuleLoadData"), 1)
        self.assertEqual(names.count("cuLaunchKernel"), 2)
        self.assertEqual(names.count("cuModuleUnload"), 1)
        self.assertTrue(loaded.closed)
        self.assertEqual(loaded.launch_calls, 2)

    def test_triton_launch_passes_dynamic_shared_memory_and_two_hidden_nulls(self) -> None:
        from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest

        manifest = CudaLaunchManifest.from_dict(
            {
                "schema_version": 1,
                "abi": "flash_kmeans_assign_v1",
                "target": "sm_100a",
                "kernel_name": "_cake_flash_kmeans_assign_kernel",
                "grid": [2, 32, 1],
                "block": [256, 1, 1],
                "dynamic_shared_memory_bytes": 164880,
                "hidden_null_pointer_parameters": 2,
            }
        )
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point=manifest.kernel_name,
            artifact_roles={"cubin": sha256(CUBIN).hexdigest()},
            launch_spec_sha256=manifest.canonical_sha256,
        )
        driver = FakeDriver()
        loaded = LoadedCudaCandidate.load(
            candidate,
            CUBIN,
            manifest,
            CudaDeviceAdmission(
                "NVIDIA B200", (10, 0), "GPU-fixture", "gpuq-000000000001", "exclusive"
            ),
            driver=driver,
        )

        loaded.launch(
            tensors(tokens=512),
            tensor_contract=CudaTensorContract(32, 512, 1_024, 128),
            stream=99,
        )
        loaded.close(synchronize=lambda: None)

        launch = next(call for call in driver.calls if call[0] == "cuLaunchKernel")
        self.assertEqual(launch[8], 164880)
        self.assertEqual(len(launch[-2]), 6)
        self.assertIn("cuFuncSetAttribute", [call[0] for call in driver.calls])

    def test_host_owns_one_exact_launch_and_unloads_after_sync(self) -> None:
        manifest = parse_cuda_launch_manifest(
            (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        )
        driver = FakeDriver()
        synchronized: list[bool] = []

        receipt = launch_cubin_once(
            CUBIN,
            sha256(CUBIN).hexdigest(),
            manifest,
            tensors(),
            tensor_contract=CudaTensorContract(32, 65_536, 1_024, 128),
            stream=99,
            synchronize=lambda: synchronized.append(True),
            driver=driver,
        )

        names = [call[0] for call in driver.calls]
        self.assertEqual(names.count("cuLaunchKernel"), 1)
        self.assertEqual(names[-1], "cuModuleUnload")
        self.assertEqual(synchronized, [True])
        self.assertEqual(receipt.kernel_calls, 1)
        self.assertTrue(receipt.same_cubin)
        self.assertTrue(receipt.module_unloaded)
        self.assertEqual(receipt.fallback_calls, 0)

    def test_resource_failure_unloads_without_launching(self) -> None:
        manifest = parse_cuda_launch_manifest(
            (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        )
        driver = FakeDriver()
        driver.attributes[6] = 90

        with self.assertRaisesRegex(ValueError, "binary version"):
            launch_cubin_once(
                CUBIN,
                sha256(CUBIN).hexdigest(),
                manifest,
                tensors(),
                tensor_contract=CudaTensorContract(32, 65_536, 1_024, 128),
                stream=99,
                synchronize=lambda: None,
                driver=driver,
            )

        names = [call[0] for call in driver.calls]
        self.assertNotIn("cuLaunchKernel", names)
        self.assertEqual(names[-1], "cuModuleUnload")

    def test_shape_bound_contract_launches_the_heldout_b32_case(self) -> None:
        from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest

        manifest = CudaLaunchManifest.from_dict(
            {
                "schema_version": 1,
                "abi": "flash_kmeans_assign_v1",
                "target": "sm_100a",
                "kernel_name": "cake_flash_kmeans_assign",
                "grid": [2, 32, 1],
                "block": [64, 1, 1],
                "dynamic_shared_memory_bytes": 0,
            }
        )
        driver = FakeDriver()
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point=manifest.kernel_name,
            artifact_roles={"cubin": sha256(CUBIN).hexdigest()},
            launch_spec_sha256=manifest.canonical_sha256,
        )

        receipt = launch_candidate_once(
            candidate,
            CUBIN,
            manifest,
            tensors(tokens=512),
            tensor_contract=CudaTensorContract(32, 512, 1_024, 128),
            stream=99,
            synchronize=lambda: None,
            driver=driver,
        )

        self.assertEqual(receipt.tensor_contract["tokens"]["shape"], [32, 512, 128])
        self.assertEqual(
            [call[0] for call in driver.calls].count("cuLaunchKernel"),
            1,
        )

        wrong_entry = LaunchableCandidate(
            candidate_sha256="2" * 64,
            target="sm_100a",
            entry_point="wrong_kernel",
            artifact_roles={"cubin": sha256(CUBIN).hexdigest()},
            launch_spec_sha256=manifest.canonical_sha256,
        )
        calls_before = len(driver.calls)
        with self.assertRaisesRegex(ValueError, "authority"):
            launch_candidate_once(
                wrong_entry,
                CUBIN,
                manifest,
                tensors(tokens=512),
                tensor_contract=CudaTensorContract(32, 512, 1_024, 128),
                stream=99,
                synchronize=lambda: None,
                driver=driver,
            )
        self.assertEqual(len(driver.calls), calls_before)


if __name__ == "__main__":
    unittest.main()
