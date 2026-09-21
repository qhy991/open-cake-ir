"""Exact target successors and boundary-preserving complete selection starters."""
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.aka_v3 import workload as aka
from open_cake_ir.tasks.workloads import create_task
from tools.launch_task import _default_shape

ROOT = Path(__file__).resolve().parents[2]


class SelectionStarters(unittest.TestCase):
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
