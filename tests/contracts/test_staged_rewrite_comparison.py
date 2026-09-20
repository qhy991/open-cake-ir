from array import array
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.tasks.normalization.study import evaluation_policy
from tools.check_alignment_candidate import prepare_cases, case_data, phase_contract
from tools import staged_rewrite_comparison as staged


class StagedComparisonTests(unittest.TestCase):
    def test_physical_dtype_decoding_keeps_values_and_signed_zero(self):
        for dtype,payload,expected in [
            ('bf16',array('H',[0x3fc0,0x8000]).tobytes(),[1.5,-0.0]),
            ('fp16',struct.pack('=ee',1.5,-0.0),[1.5,-0.0]),
            ('fp32',array('f',[1.5,-0.0]).tobytes(),[1.5,-0.0]),
            ('int32',array('i',[1,-2]).tobytes(),[1,-2]),
        ]:
            values = staged.decode_values(payload,dtype)
            self.assertEqual(list(values),expected)
            if dtype != 'int32':self.assertEqual(math.copysign(1,values[-1]),-1)

    def test_native_executable_handoff_refuses_changed_bytes_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve();path = root/'reference.so'
            path.write_bytes(b'compiled reference fixture')
            metadata = {'native_library':{'path':path.name,'sha256':sha256(path.read_bytes()).hexdigest()}}
            self.assertEqual(staged.prepared_library(root,metadata),path)
            path.write_bytes(b'changed executable')
            with self.assertRaisesRegex(ValueError,'changed before GPU loading'):
                staged.prepared_library(root,metadata)

    def test_three_phases_are_ordered_and_only_capture_holds_a_lease(self):
        task = {'stages':[{'id':name,'execution':execution,'judge':{'identity':'open-cake-ir@'+'a'*40}}
            for name,execution in [('prepare','local'),('capture','broker'),('verify','local')]]}
        phase_contract('a'*40,task,gpu_stage='capture')
        task['stages'][0],task['stages'][1] = task['stages'][1],task['stages'][0]
        with self.assertRaisesRegex(ValueError,'ordered'):
            phase_contract('a'*40,task,gpu_stage='capture')

    def test_full_capture_replay_checks_all_calls_and_never_times_bad_numerics(self):
        document,_ = create_task('rmsnorm',backend='triton-b300',rows=1,columns=2)
        workload = WorkloadContract(document)
        policy = evaluation_policy(workload);protocol = paired_protocol(policy)
        plan = staged.observation_plan(workload,protocol)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepared,captured = root/'prepare',root/'capture'
            prepared.mkdir();captured.mkdir();prepare_cases(workload,prepared)
            raw = {'capture_complete':True,'workload_sha256':workload.canonical_sha256,
                'byteorder':sys.byteorder,'evaluation_protocol':policy,
                'participants':{'optimized':{'id':'optimized'},'starter':{'id':'starter'}},
                'snapshot_count':sum(row['count'] for row in plan),'groups':[]}
            stream = io.BytesIO()
            for observation in plan:
                case = observation['case'];index = workload.case_ids.index(case)
                data = {**case_data(prepared,workload,index,'input'),**case_data(prepared,workload,index,'output')}
                for _ in range(observation['count']):
                    for arg in workload.tensor_abi(case):stream.write(array('f',data[arg.name]).tobytes())
                row = {'observation':observation}
                if observation['phase']=='timed':row['samples_ms'] = [1.0]*protocol.samples_per_cohort
                raw['groups'].append(row)
            header = captured/'capture.json';header.write_text(json.dumps(raw))
            snapshots = captured/'snapshots.bin';snapshots.write_bytes(stream.getvalue())
            serial = 0
            def verify():
                nonlocal serial
                output = root/str(serial);output.mkdir();serial += 1
                return staged.verify(root,output,prepared,captured,workload)
            with patch.object(staged,'load_participants',return_value=({'optimized':'optimized','starter':'starter'},{})), \
                 patch.object(staged,'candidate_identity',side_effect=lambda value:{'id':value}):
                result = verify()
                self.assertTrue(result['correctness_passed'])
                self.assertTrue(result['measurement_quality_passed'])
                self.assertEqual(len(result['cases']),2*len(workload.case_ids)*3)
                self.assertEqual(result['snapshot_count'],2550)
                for mode in ('input','output'):
                    changed = bytearray(stream.getvalue());offset = 0
                    for arg in workload.tensor_abi('primary'):
                        if arg.mode == mode:break
                        offset += math.prod(arg.shape)*staged.WIDTHS[arg.dtype]
                    value = struct.unpack_from('=f',changed,offset)[0]
                    struct.pack_into('=f',changed,offset,value+1)
                    snapshots.write_bytes(changed)
                    failed = verify()
                    self.assertFalse(failed['correctness_passed'])
                    self.assertTrue(all('timing' not in row for row in failed['comparisons'].values()))
                snapshots.write_bytes(stream.getvalue()+b'extra')
                with self.assertRaisesRegex(ValueError,'trailing'):
                    verify()
                snapshots.write_bytes(stream.getvalue())
                raw['groups'].pop();header.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError,'every prescribed call'):
                    verify()
