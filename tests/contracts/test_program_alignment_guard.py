"""Real Program seals reach the guard planner and complete CPU replay."""
from array import array
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.compiler.toolchain import TritonCompilation, triton_route
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.tasks.workloads import create_task
from tools.check_alignment_candidate import prepare_cases, case_data
from tools import program_alignment_guard as guard
from tools.staged_rewrite_comparison import SnapshotEncoder, INPUT_REFERENCES

ROOT=Path(__file__).resolve().parents[2]


class ProgramGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document,source=create_task('rmsnorm',backend='triton-b300',rows=1,columns=8)
        cls.workload=WorkloadContract(document);schedule=frontend.parse(source).document
        program=Program.from_schedule(schedule).document
        program['tensors']['partial']=deepcopy(program['tensors']['out'])
        program['stages'][0]['bindings']['out']='partial'
        copy='''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy",target="sm_103a",backend="triton",entry_point="copy")
def candidate(lm,partial:cake.Tensor((1,8),"fp32"),out:cake.Tensor((1,8),"fp32",mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(out,axis=0,dimension=0,tile=1)
    with compute:
        value=lm.load(partial[row,:])
        lm.store(out[row,:],value,coalesced=False)
'''
        program['stages'].append({'name':'final','schedule':frontend.parse(copy).document,'bindings':{'partial':'partial','out':'out'}})
        cls.program=Program.from_dict(program)
        class CPUCompiler:
            def compile(self,source,requirements):
                route=triton_route(requirements)
                payloads={role:b'CPU fixture not devicecode' for role in route.artifact_roles}
                payloads['source']=b'expanded CPUfixture\n'+source
                payloads['cubin']=b'\x7fELF CPUfixture'
                return TritonCompilation(source,requirements['target'],requirements['kernel_entry_point'],payloads,
                    requirements['compile_options']['num_warps']*32,0,'CPUfixture','cubin')
        builder=TritonToolchainBuilder(workload=cls.workload,case_id='primary',isolated_compiler=CPUCompiler(),pointer_alignment=16)
        env=OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),builder,
            workload=cls.workload,case_id='primary',authority_document={'input_format':'schedule_or_python_v1','lowering_route':schedule['lowering']})
        built=env.build(CandidateSubmission.seal(env.media_type,cls.program.document_bytes))
        if built.launchable is None:raise AssertionError(built.feedback)
        cls.candidate=built.launchable;cls.manifest=program_components(cls.candidate)[0]

    def test_plan_includes_private_storage_and_only_element_aligned_offsets(self):
        plan=guard.guard_plan(self.candidate,self.workload)
        self.assertEqual(len(plan),45)
        for name in self.program.tensors:
            rows=[r for r in plan if r['case']=='primary' and r['offset_tensor']==name]
            self.assertEqual([r['offset_bytes'] for r in rows],[4,8])
        self.assertEqual(len([r for r in plan if r['offset_tensor'] is None]),5)
        first,second=[s.name for s in self.program.stages]
        selected,rejected=guard.expected_stage_checks(self.manifest,'partial')
        self.assertEqual(selected,{first:'generic',second:'generic'})
        self.assertEqual(rejected,[first,second])
        selected,rejected=guard.expected_stage_checks(self.manifest,'out')
        self.assertEqual(selected,{first:'aligned',second:'generic'})
        self.assertEqual(rejected,[second])

    def test_compact_replay_checks_all_outputs_and_changed_input_literals(self):
        import sys
        plan=guard.guard_plan(self.candidate,self.workload);budget=guard.storage_budget(self.candidate,self.workload)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();prepared=root/'prepare';captured=root/'guard';prepared.mkdir();captured.mkdir()
            prepare_cases(self.workload,prepared)
            rows=[]
            for row in plan:
                selected,rejected=guard.expected_stage_checks(self.manifest,row['offset_tensor'])
                rows.append({**row,'selected':selected,'rejected_leaves':rejected,'kernel_calls':2})
            raw={'kind':'program_alignment_capture','judge_commit':'frozen','capture_complete':True,
                 'candidate':candidate_identity(self.candidate),'workload_sha256':self.workload.canonical_sha256,
                 'byteorder':sys.byteorder,'snapshot_encoding':INPUT_REFERENCES,'storage_budget':budget,
                 'snapshot_count':len(rows),'checks':rows}
            header=captured/'capture.json';header.write_text(json.dumps(raw));stream_path=captured/'snapshots.bin'
            def encode(change=None):
                with stream_path.open('wb') as stream:
                    encoder=SnapshotEncoder(stream,self.workload,budget['byte_limit'])
                    for index,row in enumerate(plan):
                        values={**case_data(prepared,self.workload,row['case_index'],'input'),**case_data(prepared,self.workload,row['case_index'],'output')}
                        if index==len(plan)-1 and change is not None:
                            mode=change;arg=next(a for a in self.workload.tensor_abi(row['case']) if a.mode==mode)
                            values[arg.name][0]+=1
                        encoder.append(row['case'],[array('f',values[a.name]).tobytes() for a in self.workload.tensor_abi(row['case'])])
            def verify():return guard.verify(self.candidate,self.workload,prepared,captured,'frozen')
            encode();correct=verify();self.assertTrue(correct['passed']);self.assertEqual(len(correct['checks']),45)
            self.assertEqual(correct['input_check_counts'],{'literal':10,'reference':80})
            for mode in ('input','output'):
                encode(mode);self.assertFalse(verify()['passed'])
            encode();valid=stream_path.read_bytes()
            for payload,message in ((valid[:-1],'truncated'),(valid+b'x','trailing')):
                stream_path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError,message):verify()
            stream_path.write_bytes(valid)
            for field,value in [('snapshot_count',44),('checks',rows[:-1]),('judge_commit','different')]:
                header.write_text(json.dumps({**raw,field:value}))
                with self.assertRaisesRegex(ValueError,'bound candidate and complete plan'):verify()
            changed=deepcopy(raw);changed['checks'][-1]['rejected_leaves']=[];header.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'leaf refusal'):verify()
            changed=deepcopy(raw);changed['checks'][-1]['kernel_calls']=1;header.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'stage count'):verify()

    def test_private_allocator_instrumentation_restores_owner_and_checks_coverage(self):
        seen=[]
        def original(spec):seen.append(spec);return 'allocated'
        loaded=SimpleNamespace(_allocate=original)
        private=[spec for name,spec in self.program.tensors.items() if name not in self.program.inputs+self.program.outputs]
        with patch.object(guard,'offset_arguments',return_value=['offset']) as shift:
            with guard.offset_private_allocator(loaded,self.program,'partial',4,None):
                self.assertEqual(loaded._allocate(private[0]),'offset')
            self.assertIs(loaded._allocate,original);shift.assert_called_once()
        with self.assertRaisesRegex(ValueError,'every declared private'):
            with guard.offset_private_allocator(loaded,self.program,None,0,None):pass
        self.assertIs(loaded._allocate,original)
        with self.assertRaisesRegex(RuntimeError,'execution failure'):
            with guard.offset_private_allocator(loaded,self.program,None,0,None):raise RuntimeError('execution failure')
        self.assertIs(loaded._allocate,original)
