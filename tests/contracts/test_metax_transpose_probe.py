"""Qualification probes keep production admission closed and preserve movement."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transpose_probe', ROOT / 'tools/qualify_metax_transpose.py')
probe_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_tool)


class MetaxTransposeProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        cls.probe = replace(cls.target, operation_kinds=cls.target.operation_kinds | {OperationKind.TRANSPOSE})

    def test_actual_emission_preserves_coordinates_tails_and_each_output(self):
        for dtype in probe_tool.DTYPES:
            for rows, columns in probe_tool.SHAPES:
                with self.subTest(dtype=dtype, shape=(rows, columns)):
                    raw = frontend.parse(probe_tool.source(rows, columns, dtype)).document
                    self.assertIn('TARGET_OPERATION_UNSUPPORTED',
                                  {f.code for f in self.compiler.assess(raw).findings})
                    schedule = Schedule.from_dict(raw)
                    findings = (*verify(schedule, self.probe), *preflight(schedule, self.probe))
                    self.assertFalse([f for f in findings if f.blocks_lowering])
                    values = ([bool(i % 2) for i in range(rows * columns)] if dtype == 'bool' else
                              [i + 2**54 for i in range(rows * columns)] if dtype == 'int64' else
                              [i % 47 - 23 for i in range(rows * columns)])
                    memory = {'x': values[:], 'out': [None] * len(values)}
                    trace = _execute(emit(schedule, self.probe), memory)
                    self.assertEqual(memory['out'], [values[r * columns + c] for c in range(columns) for r in range(rows)])
                    self.assertEqual(memory['x'], values)
                    self.assertEqual(set(trace.stores.values()), {1})

    def test_wrong_transpose_shape_and_fp8_refuse_at_their_owners(self):
        raw = frontend.parse(probe_tool.source(16, 32, 'fp32')).document
        next(b for b in raw['buffers'] if b['name'] == 'transposed')['shape'] = [16, 32]
        self.assertIn('VALUE_OPERATION_TYPE', {f.code for f in verify(Schedule.from_dict(raw), self.probe)})
        fp8 = Schedule.from_dict(frontend.parse(probe_tool.source(16, 32, 'fp8_e4m3')).document)
        self.assertIn('VALUE_OPERATION_TYPE', {f.code for f in verify(fp8, self.probe)})
        self.assertIn('MACA_FP8_OPERATION_UNQUALIFIED', {f.code for f in preflight(fp8, self.probe)})


if __name__ == '__main__':
    unittest.main()
