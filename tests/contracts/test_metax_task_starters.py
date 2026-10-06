"""Whole admitted GEMM tasks with bounded ownership at small and original widths."""
from pathlib import Path
import unittest
import operator
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.gemm.workload import workload_document
from open_cake_ir.tasks.gemm.authoring import starter_source
from open_cake_ir.tasks.metax_authoring import native_event_starter
from tests.contracts.test_triton_loop_scopes import _execute, _Tile

ROOT = Path(__file__).resolve().parents[2]


class MetaXTaskStarters(unittest.TestCase):
    def test_full_abi_and_reduction_with_one_write_per_original_output(self):
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        for columns in (16, 1024):
            w = WorkloadContract(workload_document('gemm_bias', rows=2, depth=8,
                                                  columns=columns, backend='triton-metax'))
            original = starter_source(w)
            adapted = native_event_starter(w, original)
            schedule = frontend.parse(adapted).document
            self.assertEqual([b for b in schedule['buffers'] if b['space']=='global'],
                             [b for b in frontend.parse(original).document['buffers'] if b['space']=='global'])
            self.assertTrue(compiler.assess(schedule).lowering_eligible)
            emission = emit(Schedule.from_dict(schedule), compiler._revision.targets['xcore1002'])
            self.assertEqual(emission.toolchain['grid'], [2, columns, 1])
            a = [float(i%3-1) for i in range(16)]
            b = [float(i%5-2) for i in range(8*columns)]
            bias = [float(i%4) for i in range(columns)]
            memory={'a':a[:], 'b':b[:], 'bias':bias[:], 'out':[None]*(2*columns)}
            with patch.object(_Tile, '__ge__', lambda self, other: self.binary(other, operator.ge), create=True):
                observed = _execute(emission,memory)
            self.assertEqual(memory['out'], [sum(a[m*8+k]*b[k*columns+n] for k in range(8))+bias[n]
                                             for m in range(2) for n in range(columns)])
            self.assertEqual(set(observed.stores.values()), {1})
            self.assertEqual(memory['a'],a); self.assertEqual(memory['b'],b); self.assertEqual(memory['bias'],bias)

    def test_rejects_a_foreign_target_and_an_unrecognized_starting_program(self):
        w = WorkloadContract(workload_document('gemm_bias',backend='triton-metax'))
        with self.assertRaises(ValueError):native_event_starter(w,starter_source(w)+'\n')
        other = WorkloadContract(workload_document('gemm_bias',backend='triton-b200'))
        with self.assertRaises(ValueError):native_event_starter(other,starter_source(other))
