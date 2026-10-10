"""Failed observations remain versioned diagnostic evidence, never receipts."""
from copy import deepcopy
import unittest
from open_cake_ir.evaluation.failures import failure_artifacts
from open_cake_ir.tasks.evaluate import _base_result

class FailedObservationProtocol(unittest.TestCase):
    def test_legacy_and_failed_v2_are_distinct(self):
        legacy=_base_result('maca-0123456789ab')
        self.assertEqual(failure_artifacts(legacy),{})
        failed={**legacy,'schema_version':2,'error':'evaluator_failed','failure_class':'ValueError',
                'failure_artifacts':{'paired_activity':'failure-paired_activity.bin'}}
        self.assertEqual(failure_artifacts(failed),failed['failure_artifacts'])
        for field,value in [('receipt',{}),('error',None),('failure_class',None),('schema_version',True),
                             ('failure_artifacts',{}),('failure_artifacts',{'raw':'../escape'}),
                             ('failure_artifacts',{'raw':'/outside'}),('failure_artifacts',{'raw':['not-a-path']}),
                             ('failure_artifacts',{'raw':'same','other':'same'}),
                             ('failure_artifacts',{'bad-role':'diagnostic'})]:
            mutated={**failed,field:value}
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):failure_artifacts(mutated)

from pathlib import Path
import tempfile
import os
import pwd
import grp
import sys
import json
from hashlib import sha256
from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.lab.runtime import CommandBrokerSubmitter, BoundedBrokerEvaluator
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.archive import _archive_logical_attempt, _logical_attempt_document
from open_cake_ir.lab.replay.attempts import _replay_broker_attempt_ledger
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from tests.contracts import test_lab as lab_fixture
from tests.contracts._executor_fixture import compiler_reference
from open_cake_ir.serialization import canonical_json_bytes

ROOT=Path(__file__).resolve().parents[2]

class FailedNativeArtifactHandoff(lab_fixture.SemanticLabTestCase):
    def test_actual_broker_handoff_and_replay_retain_failed_bytes_and_reject_omission(self):
        workload_path=ROOT/'contracts/workloads/flash-kmeans-assign-v2.json'
        workload=json.loads(workload_path.read_bytes())
        protocol={'case_id':'headline_b32','kind':'fixture'}
        protocol_sha=sha256(canonical_json_bytes(protocol)).hexdigest()
        payloads={'cubin':b'never executed CPU fixture','launch_manifest':b'CPU fixture manifest'}
        candidate=LaunchableCandidate('a'*64,'sm_100a','kernel',
            {k:sha256(v).hexdigest() for k,v in payloads.items()},sha256(payloads['launch_manifest']).hexdigest(),payloads)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            worker=root/'worker.py'
            worker.write_text('''import argparse,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=pathlib.Path(a.request).parent
(r/'failed.bin').write_bytes(b'{"records":[{"start_ns":256,"end_ns":128}]}')
o={'schema_version':2,'job_id':'gpuq-000000000000','mode':'exclusive','admitted':True,
'error':'evaluator_failed','failure_class':'ValueError','receipt':None,
'failure_artifacts':{'paired_activity':'failed.bin'},'counters':{'compiler_invocations':0,
'module_loads':1,'preflight_calls':1,'kernel_calls':7,'timing_samples':0,'fallback_calls':0}}
print('[gpu-run] accepted job gpuq-000000000001 label=fixture mode=exclusive gpus=1',file=sys.stderr)
pathlib.Path(a.output).write_text(json.dumps(o))
''')
            submitter=CommandBrokerSubmitter(command=(sys.executable,str(worker)),workload_path=workload_path,
                workload_sha256=sha256(canonical_json_bytes(workload)).hexdigest(),protocol_sha256=protocol_sha,
                cwd=root,executor=self.executor_fixture.revision(ROOT),compiler_reference=compiler_reference(ROOT),
                service_user=pwd.getpwuid(os.geteuid()).pw_name,service_group=grp.getgrgid(os.getegid()).gr_name)
            result=BoundedBrokerEvaluator(protocol,submitter).evaluate(candidate,case_id='headline_b32',purpose='search')
            self.assertIsNone(result.final_receipt)
            self.assertEqual(len(result.attempts),1)
            self.assertIn('failure_paired_activity',result.attempts[0].artifact_payloads)
            # Actual TemporaryDirectory transport has closed before this assertion.
            self.assertEqual(json.loads(result.attempts[0].artifact_payloads['failure_paired_activity'])['records'][0]['end_ns'],128)
            evidence=EvidenceStore.create(root/'evidence')
            refs=_archive_logical_attempt(evidence,result)
            arguments=dict(evidence=evidence,references=refs,document=_logical_attempt_document(result),
                candidate=candidate,protocol_sha256=protocol_sha,compiler_reference=compiler_reference(ROOT),
                final_receipt=None,purpose='search',case_id='headline_b32')
            _replay_broker_attempt_ledger(**arguments)
            arguments['references']=[r for r in refs if r['role']!='attempt_1_failure_paired_activity']
            with self.assertRaises(ReplayRefusal):_replay_broker_attempt_ledger(**arguments)

class TerminalWorkerArtifactHandoff(lab_fixture.SemanticLabTestCase):
    def test_exit_74_retains_only_custody_checked_diagnostics_and_never_a_receipt(self):
        from open_cake_ir.lab.faults import RunProtocolFault
        workload_path=ROOT/'contracts/workloads/flash-kmeans-assign-v2.json'
        workload=json.loads(workload_path.read_bytes())
        payloads={'cubin':b'CPU fixture','launch_manifest':b'CPU fixture'}
        candidate=LaunchableCandidate('a'*64,'sm_100a','kernel',
            {k:sha256(v).hexdigest() for k,v in payloads.items()},sha256(payloads['launch_manifest']).hexdigest(),payloads)
        for bad_path in (False,True):
            with self.subTest(bad_path=bad_path),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();script=root/'worker.py'
                script.write_text('''import argparse,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=pathlib.Path(a.request).parent
(r/'failed.bin').write_bytes(b'{"drained":false}')
o={'schema_version':2,'job_id':'gpuq-000000000000','mode':'exclusive','admitted':True,
'error':'evaluator_failed','failure_class':'UndrainedDeviceWork','receipt':None,
'failure_artifacts':{'native_observation':PATH},'counters':{'compiler_invocations':0,
'module_loads':1,'preflight_calls':1,'kernel_calls':7,'timing_samples':0,'fallback_calls':0}}
pathlib.Path(a.output).write_text(json.dumps(o))
sys.exit(74)
'''.replace('PATH',repr('../outside' if bad_path else 'failed.bin')))
                submitter=CommandBrokerSubmitter(command=(sys.executable,str(script)),workload_path=workload_path,
                    workload_sha256=sha256(canonical_json_bytes(workload)).hexdigest(),protocol_sha256='f'*64,
                    cwd=root,executor=self.executor_fixture.revision(ROOT),compiler_reference=compiler_reference(ROOT),
                    service_user=pwd.getpwuid(os.geteuid()).pw_name,service_group=grp.getgrgid(os.getegid()).gr_name)
                with self.assertRaises(RunProtocolFault) as caught:
                    submitter.submit(candidate,case_id='headline_b32',purpose='search',attempt=1)
                fault=caught.exception
                self.assertIn('exited 74',str(fault))
                self.assertIsNone(json.loads(fault.artifact_payloads['broker_result'])['receipt'])
                if bad_path:self.assertNotIn('failure_native_observation',fault.artifact_payloads)
                else:self.assertEqual(fault.artifact_payloads['failure_native_observation'],b'{"drained":false}')
