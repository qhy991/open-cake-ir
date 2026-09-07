from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.cuda_driver import CudaLifecycleError
from open_cake_ir.tasks.qsa.cuda import LoadedQsaProgram, QsaProgramArtifact


class _Tensor:
    def __init__(self, pointer: int) -> None:
        self._pointer = pointer
        self.is_cuda = True

    def data_ptr(self) -> int:
        return self._pointer

    def is_contiguous(self) -> bool:
        return True


class _Driver:
    def __init__(self) -> None:
        self.launches: list[object] = []
        self.context = 1
        self.loads: list[int] = []
        self.unloads: list[int] = []
        self.function_calls = 0
        self.load_error: tuple[int, BaseException] | None = None
        self.function_error: tuple[int, BaseException] | None = None
        self.unload_errors: dict[int, BaseException] = {}

    def cuCtxGetCurrent(self):
        return (0, self.context)

    def cuModuleLoadData(self, _payload):
        module = len(self.loads) + 1
        self.loads.append(module)
        if self.load_error is not None and self.load_error[0] == module:
            raise self.load_error[1]
        return (0, module)

    def cuModuleGetFunction(self, _module, name):
        self.function_calls += 1
        if self.function_error is not None and self.function_error[0] == self.function_calls:
            raise self.function_error[1]
        return (0, name)

    def cuLaunchKernel(self, function, *_arguments):
        self.launches.append(function)
        return (0,)

    def cuModuleUnload(self, module):
        self.unloads.append(module)
        if module in self.unload_errors:
            raise self.unload_errors[module]
        return (0,)


class QsaCudaProgramContractTests(unittest.TestCase):
    def _program(self, root: Path, *, distinct_modules: bool = False) -> Path:
        candidate = root / "candidate"
        candidate.mkdir()
        (candidate / "program.cubin").write_bytes(b"\x7fELFqsa")
        arguments = {
            "pool_layernorm": ["index_k", "k_norm_weight", "normalized_keys"],
            "score_topk": ["index_q", "normalized_keys", "block_indices"],
            "expand": ["block_indices", "token_indices"],
            "attention": ["q", "k", "v", "token_indices", "output"],
        }
        document = {
            "schema_version": 1,
            "abi": "qsa_prefill_task_geometry_v1",
            "arm": "direct_cuda",
            "kernels": [
                {
                    "id": kernel_id,
                    "cubin": "candidate/program.cubin",
                    "kernel_name": f"qsa_{kernel_id}",
                    "grid": [32768, 1, 1],
                    "block": [256, 1, 1],
                    "dynamic_shared_memory_bytes": 0,
                    "arguments": arguments[kernel_id],
                    "hidden_null_pointer_parameters": 0,
                }
                for kernel_id in (
                    "pool_layernorm",
                    "score_topk",
                    "expand",
                    "attention",
                )
            ],
        }
        if distinct_modules:
            for kernel in document["kernels"]:
                cubin = candidate / f"{kernel['id']}.cubin"
                cubin.write_bytes(b"\x7fELFqsa")
                kernel["cubin"] = cubin.relative_to(root).as_posix()
        source = candidate / "program.json"
        source.write_text(json.dumps(document), encoding="utf-8")
        return source

    def test_direct_program_loads_only_the_closed_qsa_launch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = QsaProgramArtifact.load(root, self._program(root))
            self.assertEqual(artifact.arm, "direct_cuda")
            self.assertEqual(
                tuple(kernel.kernel_id for kernel in artifact.kernels),
                ("pool_layernorm", "score_topk", "expand", "attention"),
            )
            self.assertEqual(len({kernel.cubin_path for kernel in artifact.kernels}), 1)

    def test_program_rejects_an_argument_binding_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._program(root)
            document = json.loads(source.read_text(encoding="utf-8"))
            document["kernels"][1]["arguments"] = [
                "normalized_keys",
                "index_q",
                "block_indices",
            ]
            source.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract differs"):
                QsaProgramArtifact.load(root, source)

    def test_loaded_program_marks_boundaries_without_changing_launch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = QsaProgramArtifact.load(root, self._program(root))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            names = {
                name
                for kernel in artifact.kernels
                for name in kernel.arguments
            }
            tensors = {
                name: _Tensor(index + 1)
                for index, name in enumerate(sorted(names))
            }
            boundaries: list[tuple[str, str]] = []

            loaded.launch(tensors, stream=0, boundary=lambda *row: boundaries.append(row))

            order = [kernel.kernel_id for kernel in artifact.kernels]
            self.assertEqual(
                boundaries,
                [
                    (kernel_id, phase)
                    for kernel_id in order
                    for phase in ("before", "after")
                ],
            )
            self.assertEqual(
                [function.decode().removeprefix("qsa_") for function in driver.launches],
                order,
            )
            self.assertEqual(loaded.launch_calls, len(order))
            loaded.close(synchronize=lambda: None)
            self.assertEqual(driver.loads, [1])
            self.assertEqual(driver.unloads, [1])
            self.assertTrue(loaded.closed)
            with self.assertRaisesRegex(ValueError, "already closed"):
                loaded.close(synchronize=lambda: None)


    def _artifact(self, root: Path) -> QsaProgramArtifact:
        return QsaProgramArtifact.load(root, self._program(root, distinct_modules=True))

    def _tensors(self, artifact: QsaProgramArtifact) -> dict[str, _Tensor]:
        names = sorted({name for kernel in artifact.kernels for name in kernel.arguments})
        return {name: _Tensor(index + 1) for index, name in enumerate(names)}

    def test_sync_failure_still_unloads_every_module_in_reverse_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            failure = RuntimeError("synchronization failed")

            def synchronize():
                raise failure

            with self.assertRaises(RuntimeError) as caught:
                loaded.close(synchronize=synchronize)
            self.assertIs(caught.exception, failure)
            self.assertEqual(driver.unloads, [4, 3, 2, 1])
            self.assertTrue(loaded.closed)

    def test_sync_and_multiple_unload_failures_are_all_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            primary = RuntimeError("synchronization failed")
            first, second = RuntimeError("unload 4 failed"), RuntimeError("unload 2 failed")
            driver.unload_errors = {4: first, 2: second}

            def synchronize():
                raise primary

            with self.assertRaises(CudaLifecycleError) as caught:
                loaded.close(synchronize=synchronize)
            self.assertIs(caught.exception.primary, primary)
            self.assertIs(caught.exception.__cause__, primary)
            self.assertIs(caught.exception.teardown, first)
            self.assertEqual(caught.exception.teardown_errors, (first, second))
            self.assertEqual(driver.unloads, [4, 3, 2, 1])
            self.assertFalse(loaded.closed)
            with self.assertRaisesRegex(ValueError, "teardown has started"):
                loaded.launch(self._tensors(artifact), stream=0)
            self.assertEqual(driver.launches, [])

            # A caller may retry the failed handles; successful unloads are never repeated.
            driver.unload_errors.clear()
            loaded.close(synchronize=lambda: None)
            self.assertEqual(driver.unloads, [4, 3, 2, 1, 4, 2])
            self.assertTrue(loaded.closed)

    def test_multiple_unload_failures_without_sync_failure_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            first, second = RuntimeError("unload 4 failed"), RuntimeError("unload 2 failed")
            driver.unload_errors = {4: first, 2: second}
            with self.assertRaises(CudaLifecycleError) as caught:
                loaded.close(synchronize=lambda: None)
            self.assertIs(caught.exception.primary, first)
            self.assertEqual(caught.exception.teardown_errors, (second,))
            self.assertEqual(driver.unloads, [4, 3, 2, 1])
            self.assertFalse(loaded.closed)

    def test_second_module_load_failure_unloads_the_first_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            primary = RuntimeError("second load failed")
            driver.load_error = (2, primary)
            with self.assertRaises(RuntimeError) as caught:
                LoadedQsaProgram(artifact, driver=driver)
            self.assertIs(caught.exception, primary)
            self.assertEqual(driver.loads, [1, 2])
            self.assertEqual(driver.unloads, [1])
            self.assertEqual(driver.launches, [])

    def test_constructor_keeps_primary_and_every_unload_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            primary = RuntimeError("fourth function lookup failed")
            first, second = RuntimeError("unload 4 failed"), RuntimeError("unload 2 failed")
            driver.function_error = (4, primary)
            driver.unload_errors = {4: first, 2: second}
            with self.assertRaises(CudaLifecycleError) as caught:
                LoadedQsaProgram(artifact, driver=driver)
            self.assertIs(caught.exception.primary, primary)
            self.assertEqual(caught.exception.teardown_errors, (first, second))
            self.assertEqual(driver.unloads, [4, 3, 2, 1])
            self.assertEqual(driver.launches, [])

    def test_context_change_blocks_launch_and_cleanup_in_the_wrong_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            driver.context = 9
            synchronizations = []
            with self.assertRaisesRegex(ValueError, "CUDA context changed"):
                loaded.launch(self._tensors(artifact), stream=0)
            with self.assertRaisesRegex(ValueError, "CUDA context changed"):
                loaded.close(synchronize=lambda: synchronizations.append(True))
            self.assertEqual(driver.launches, [])
            self.assertEqual(driver.unloads, [])
            self.assertEqual(synchronizations, [])
            self.assertFalse(loaded.closed)
            driver.context = 1
            loaded.close(synchronize=lambda: None)
            self.assertEqual(driver.unloads, [4, 3, 2, 1])

    def test_sync_failure_and_context_change_keep_handles_in_original_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            primary = RuntimeError("sync changed context and failed")

            def synchronize():
                driver.context = 9
                raise primary

            with self.assertRaises(CudaLifecycleError) as caught:
                loaded.close(synchronize=synchronize)
            self.assertIs(caught.exception.primary, primary)
            self.assertRegex(str(caught.exception.teardown), "CUDA context changed")
            self.assertEqual(driver.unloads, [])
            self.assertFalse(loaded.closed)
            driver.context = 1
            loaded.close(synchronize=lambda: None)
            self.assertEqual(driver.unloads, [4, 3, 2, 1])

    def test_boundary_context_change_refuses_the_next_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(Path(directory))
            driver = _Driver()
            loaded = LoadedQsaProgram(artifact, driver=driver)
            boundaries = []

            def boundary(kernel, phase):
                boundaries.append((kernel, phase))
                driver.context = 9

            with self.assertRaisesRegex(ValueError, "CUDA context changed"):
                loaded.launch(self._tensors(artifact), stream=0, boundary=boundary)
            self.assertEqual(boundaries, [(artifact.kernels[0].kernel_id, "before")])
            self.assertEqual(driver.launches, [])
            driver.context = 1
            loaded.close(synchronize=lambda: None)


if __name__ == "__main__":
    unittest.main()
