from array import array
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import create_task
from tools.check_alignment_candidate import prepare_cases, verify_capture, phase_contract


class AlignmentReplayPhases(unittest.TestCase):
    def test_cpu_verification_retains_all_cases_and_detects_output_and_input_changes(self):
        document, _ = create_task('rmsnorm',backend='triton-b300',rows=1,columns=8)
        workload = WorkloadContract(document)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, captured = root / 'prepare', root / 'guard'
            prepared.mkdir(); captured.mkdir()
            prepare_cases(workload,prepared)
            checks = []
            for case_index, case in enumerate(workload.case_ids):
                abi = workload.tensor_abi(case)
                variants = [(None,0)] + [(i,b) for i in range(len(abi)) for b in (2,4,8)]
                for index, offset in variants:
                    target = captured / str(len(checks)); target.mkdir()
                    for number,_ in enumerate(abi):
                        name = f'{case_index}-{number}.f64'
                        shutil.copyfile(prepared / name,target / name)
                    checks.append({'case':case,'case_index':case_index,'offset_index':index,
                                   'offset_bytes':offset,'selected':'generic' if offset else 'aligned'})
            report = {'workload_sha256':workload.canonical_sha256,'capture_complete':True,'checks':checks}
            path = captured / 'capture.json'
            path.write_text(json.dumps(report))
            result = verify_capture(workload,prepared,captured)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['checks']),len(checks))
            self.assertEqual(result['verification_phase'],'CPU after GPU worker exit')
            for mode in ('input','output'):
                number = next(i for i,arg in enumerate(workload.tensor_abi('primary')) if arg.mode == mode)
                tensor = captured / '0' / f'0-{number}.f64'
                original = tensor.read_bytes()
                values = array('d');values.frombytes(original);values[0] += 1.0
                tensor.write_bytes(values.tobytes())
                self.assertFalse(verify_capture(workload,prepared,captured)['passed'])
                tensor.write_bytes(original)
            report['checks'] = checks[:-1]
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'every required case'):
                verify_capture(workload,prepared,captured)

    def test_cpu_work_cannot_be_assigned_a_broker_lease_by_the_replay_contract(self):
        commit = 'a' * 40
        task = {'stages':[{'id':name,'execution':execution,'judge':{'identity':'open-cake-ir@'+commit}}
                          for name,execution in [('prepare','local'),('guard','broker'),('verify','local')]]}
        phase_contract(commit,task)
        task['stages'][0]['execution'] = 'broker'
        with self.assertRaisesRegex(ValueError,'resource phase'):
            phase_contract(commit,task)


if __name__ == '__main__':
    unittest.main()
