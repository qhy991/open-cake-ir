import csv
import io
from types import SimpleNamespace
import unittest

from tools.compare_flashinfer_reference import LoadedCallable
from tools.compare_rewrite_artifacts import reference_arguments
from tools.profile_rewrite_artifacts import METRICS, parse_metrics
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import create_task


class ExternalDtypeTests(unittest.TestCase):
    def test_fp16_gemm_and_bf16_norm_keep_their_abi(self):
        class Tensor:
            counter = 0
            def __init__(self, dtype):
                Tensor.counter += 1
                self.pointer = Tensor.counter
                self.dtype = dtype
                self.device = SimpleNamespace(type='cuda', index=0)
            def reshape(self, shape):
                self.shape = shape
                return self
            def clone(self):
                return Tensor(self.dtype).reshape(self.shape)
            def data_ptr(self):
                return self.pointer
            def is_contiguous(self):
                return True
        torch = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', float16='fp16',
            tensor=lambda values, dtype, device: Tensor(dtype),
            full=lambda shape, value, dtype, device: Tensor(dtype).reshape(shape))
        for task, rows, columns, dtype, names in [
            ('fib_gemm_n128_k2048', 1, 128, 'fp16', ['a', 'b']),
            ('fib_rmsnorm_h2048', 79, 2048, 'bf16', ['x', 'weight'])]:
            with self.subTest(task=task):
                doc, _ = create_task(task, backend='triton-b300', rows=rows, columns=columns)
                workload = WorkloadContract(doc)
                self.assertEqual(reference_arguments(workload), names)
                loaded = LoadedCallable(workload, {n: [1] for n in names}, lambda v, out: out, torch)
                arguments = loaded.fresh_argument_sets(1)[0]
                self.assertTrue(all(v.dtype == dtype for v in arguments['inputs'].values()))
                loaded.launch(arguments)
                self.assertEqual(arguments['result'].dtype, dtype)
                loaded.launch_function = lambda v, out: Tensor('bf16' if dtype == 'fp16' else 'fp16').reshape(out.shape)
                with self.assertRaisesRegex(ValueError, 'output ABI'):
                    loaded.launch(arguments)
        doc, _ = create_task('rmsnorm', backend='triton-b300', rows=1, columns=128)
        with self.assertRaisesRegex(ValueError, 'ABI differs'):
            reference_arguments(WorkloadContract(doc))


class ProfileCsvTests(unittest.TestCase):
    def csv(self, rows):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
        writer.writerows(rows)
        return '==PROF== capture\n' + stream.getvalue()

    def test_multiple_launches_remain_separate_and_incomplete_metrics_refuse(self):
        rows = [[str(i), 'kernel_' + str(i), metric, 'count', '1']
                for i in range(2) for metric in METRICS]
        self.assertEqual(len(parse_metrics(self.csv(rows))), 2)
        for broken in [rows[:-1], rows + [rows[0]],
                       [r[:-1] + ['nan'] if j == 0 else r for j, r in enumerate(rows)],
                       [r[:-2] + ['', '1'] if j == 0 else r for j, r in enumerate(rows)]]:
            with self.subTest(broken=broken[-1]), self.assertRaises(ValueError):
                parse_metrics(self.csv(broken))


if __name__ == '__main__':
    unittest.main()
