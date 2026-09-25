"""A four-rank activity view requires complete device and oracle evidence."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from experiments.weave.native_b300.ranked_activity import analyze, read_kernel_rows


def evidence():
    commit = 'a' * 40
    revision = 'open-cake-ir@' + commit
    case = {'shape': {'R': 4}, 'source_commit': commit,
            'compiler_revision_id': revision,
            'target': 'sm_103a', 'case_id': 'skew_to_rank0'}
    requirements = {'compiler_commit': commit, 'target': 'sm_103a',
                    'world_size': 4, 'grid_per_rank': [[148, 1, 1]] * 4,
                    'block': [32, 1, 1]}
    receipt = {'job_id': 'gpuq-test', 'mode': 'exclusive', 'gpu_count': 4,
               'gpu_ids': [2, 3, 4, 5]}
    device = {'broker_job': 'gpuq-test', 'source_commit': commit,
              'compiler_revision_id': revision, 'launch_calls': 1}
    report = {'broker_job': 'gpuq-test', 'source_commit': commit,
              'compiler_revision_id': revision, 'case_id': 'skew_to_rank0',
              'passed': True,
              'ranks': [{'rank': rank, 'passed': True} for rank in range(4)]}
    rows = [dict(device_id=rank, start_ns=start, end_ns=end,
                 registers=44, grid_x=148, block_x=32,
                 name='<unnamed>::cake_ranked_ep4_kernel(Params *)')
            for rank, start, end in ((0, 100, 300), (1, 120, 250),
                                     (2, 130, 270), (3, 140, 310))]
    return case, requirements, receipt, device, report, rows


class RankedActivity(unittest.TestCase):
    def test_joint_window_requires_all_four_correct_ranked_kernels(self):
        original = evidence()
        result = analyze(*original)
        self.assertEqual(result['kernel_activity_union_ns'], 210)
        self.assertEqual(result['all_rank_active_intersection_ns'], 110)
        self.assertEqual([row['physical_gpu'] for row in result['ranks']],
                         [2, 3, 4, 5])
        self.assertIn('not an admitted MoE-layer timer', result['scope'])

        for name, change in (
            ('missing kernel', lambda rows: rows.pop()),
            ('duplicate device', lambda rows: rows[-1].update(device_id=0)),
            ('invalid timestamp', lambda rows: rows[0].update(start_ns=0)),
            ('unrelated kernel', lambda rows: rows[0].update(name='other')),
        ):
            with self.subTest(name=name):
                case, requirements, receipt, device, report, rows = deepcopy(original)
                change(rows)
                with self.assertRaisesRegex(ValueError, 'Nsight'):
                    analyze(case, requirements, receipt, device, report, rows)
        case, requirements, receipt, device, report, rows = deepcopy(original)
        report['ranks'][2]['passed'] = False
        with self.assertRaisesRegex(ValueError, 'oracle'):
            analyze(case, requirements, receipt, device, report, rows)
        case, requirements, receipt, device, report, rows = deepcopy(original)
        receipt['mode'] = 'shared'
        with self.assertRaisesRegex(ValueError, 'broker'):
            analyze(case, requirements, receipt, device, report, rows)

    def test_reads_nsys_kernel_identity_from_string_table(self):
        with tempfile.TemporaryDirectory(prefix='cake-ranked-activity-') as directory:
            database = Path(directory) / 'trace.sqlite'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
                connection.execute('''CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    deviceId INTEGER, start INTEGER, end INTEGER,
                    registersPerThread INTEGER, gridX INTEGER, blockX INTEGER,
                    demangledName INTEGER)''')
                connection.execute('INSERT INTO StringIds VALUES (?, ?)',
                                   (7, 'cake_ranked_ep4_kernel'))
                connection.execute('''INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                    VALUES (?, ?, ?, ?, ?, ?, ?)''', (0, 100, 200, 44, 148, 32, 7))
                connection.commit()
            rows = read_kernel_rows(database)
            self.assertEqual(rows[0]['name'], 'cake_ranked_ep4_kernel')
            self.assertEqual(rows[0]['device_id'], 0)


if __name__ == '__main__':
    unittest.main()
