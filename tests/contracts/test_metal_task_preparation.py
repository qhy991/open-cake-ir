"""The Metal development projection keeps refusals; it never qualifies a device."""
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from tools.metal.prepare_tasks import inspect_task, prepare

ROOT = Path(__file__).resolve().parents[2]


class MetalTaskPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_ready_starter_is_not_device_qualification(self):
        row, files = inspect_task(self.compiler, 'rmsnorm')
        self.assertEqual(row['status'], 'msl_generated')
        self.assertFalse(row['device_qualified'])
        self.assertIn('starter.metal', files)

    def test_original_dtype_is_not_silently_replaced(self):
        row, files = inspect_task(self.compiler, 'add_rmsnorm_bf16')
        self.assertEqual(row['status'], 'workload_refused')
        self.assertIn('bf16', row['reason'])
        self.assertNotIn('starter.metal', files)

    def test_explicit_smaller_development_case_does_not_hide_default_refusal(self):
        default, _ = inspect_task(self.compiler, 'gemm_silu')
        bounded, _ = inspect_task(self.compiler, 'gemm_silu', rows=128, columns=32, depth=256)
        self.assertEqual(default['status'], 'assessment_refused')
        self.assertEqual(bounded['status'], 'msl_generated')
        self.assertNotEqual(default['shape'], bounded['shape'])

    def test_no_materials_written_inside_source(self):
        with self.assertRaisesRegex(ValueError, 'outside Git'):
            prepare(ROOT, ROOT / 'never-create-this-directory', tasks=['rmsnorm'])

    def test_retained_task_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / 'tasks'
            prepare(ROOT, output, tasks=['rmsnorm'])
            with self.assertRaises(FileExistsError):
                prepare(ROOT, output, tasks=['rmsnorm'])


if __name__ == '__main__':
    unittest.main()
