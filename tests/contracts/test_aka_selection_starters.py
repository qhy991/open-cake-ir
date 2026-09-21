"""Exact target successors and boundary-preserving complete selection starters."""
from pathlib import Path
from contextlib import ExitStack
import operator
import struct
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.aka_v3 import workload as aka
from open_cake_ir.tasks.workloads import create_task
from tools.launch_task import _default_shape

ROOT = Path(__file__).resolve().parents[2]


class SelectionStarters(unittest.TestCase):
    def test_emitted_memory_operations_match_original_oracles_and_center_rounding(self):
        from tests.contracts.test_epilogue_fusion import execute, _Tile
        with ExitStack() as stack:
            for method, operation in [('__ge__', operator.ge), ('__le__', operator.le),
                                      ('__gt__', operator.gt), ('__ne__', operator.ne)]:
                stack.enter_context(patch.object(_Tile, method, lambda a, b, op=operation: a.binary(b, op)))
            for task, rows, columns in [('histogram', 1024, 16), ('max_pool1d', 2, 8)]:
                document, source = create_task('aka_' + task, backend='triton-metax', rows=rows, columns=columns)
                workload = WorkloadContract(document)
                schedule = frontend.parse(source).document
                for case in workload.case_ids:
                    inputs = aka.materialize_case(workload, case)
                    observed, _ = execute(schedule, inputs)
                    self.assertEqual(observed, aka.reference_outputs(workload, case, inputs))
            # Reproduce the exact oracle discrepancy found at zero, including
            # negative subnormal input. Exercise emitted source, not a copied bin formula.
            document, source = create_task('aka_histogram', backend='triton-metax', rows=1, columns=16)
            workload = WorkloadContract(document)
            for value in [-2**-149, -2**-54, -2**-52, -2**-51, 0.0, 2**-149]:
                inputs = {'values': [struct.unpack('<f', struct.pack('<f', value))[0]]}
                observed, _ = execute(frontend.parse(source).document, inputs)
                self.assertEqual(observed, aka.reference_outputs(workload, 'primary', inputs))

    def test_c550_successors_preserve_all_original_cases_and_oracles(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        for name in ('histogram', 'max_pool1d'):
            old = WorkloadContract(aka.workload_document(name))
            new = WorkloadContract(aka.workload_document(name, backend='triton-metax'))
            aka.validate_aka_v3_contract(new.document)
            for field in ('tensors', 'cases', 'oracle'):
                self.assertEqual(new.document[field], old.document[field])
            for case in old.case_ids:
                inputs = aka.materialize_case(old, case)
                self.assertEqual(inputs, aka.materialize_case(new, case))
                self.assertEqual(aka.reference_outputs(old, case, inputs), aka.reference_outputs(new, case, inputs))
            rows, columns = _default_shape('aka_' + name, None, None)
            document, source = create_task('aka_' + name, backend='triton-metax', rows=rows, columns=columns)
            self.assertEqual(document, new.document)
            schedule = frontend.parse(source).document
            assessment = compiler.assess(schedule)
            self.assertTrue(assessment.lowering_eligible, assessment.findings)
            compiler.lower(assessment)
            self.assertEqual(len([op for op in schedule['operations'] if op['kind'] == 'store']), 1)

    def test_histogram_boundaries_and_pool_negative_edges_keep_original_meaning(self):
        histogram = WorkloadContract(aka.workload_document('histogram', backend='triton-metax'))
        for case, index in [('lower_edge', 0), ('upper_edge', 15)]:
            values = aka.reference_outputs(histogram, case, aka.materialize_case(histogram, case))['counts']
            self.assertEqual(values[index], 1024)
            self.assertEqual(sum(values), 1024)
        self.assertEqual(sum(aka.reference_outputs(histogram, 'out_of_range',
                         aka.materialize_case(histogram, 'out_of_range'))['counts']), 0)
        pool = WorkloadContract(aka.workload_document('max_pool1d', backend='triton-metax'))
        self.assertTrue(all(v < 0 for v in aka.reference_outputs(pool, 'negative_only',
                            aka.materialize_case(pool, 'negative_only'))['output']))
        for kwargs in ({'bins': 3}, {'elements': 2**25}):
            with self.assertRaises(ValueError): aka.workload_document('histogram', backend='triton-metax', **kwargs)
        with self.assertRaises(ValueError):
            aka.workload_document('max_pool1d', backend='triton-metax', kernel_size=1, pad=1, output_length=9)
        with self.assertRaisesRegex(ValueError, 'signed int32 linear'):
            aka.workload_document('max_pool1d', backend='triton-metax', batch=2**25)
        # Historical mathematical contracts remain loadable, but their new starter
        # must reject the same unrepresentable address domain.
        from open_cake_ir.tasks.aka_v3.authoring import starter_source
        old = WorkloadContract(aka.workload_document('max_pool1d', batch=2**25))
        with self.assertRaisesRegex(ValueError, 'signed int32 linear'):
            starter_source(old)
