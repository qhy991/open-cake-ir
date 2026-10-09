"""Program task inputs traverse real package/preflight/build boundaries with CPU doubles."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.lab import admission, bindings
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.python_reference import (read_skeleton, read_skeleton_reference,
    bind_python_reference, skeleton_route, lower_skeleton)
from open_cake_ir.lab.pairing import bind_baseline
from open_cake_ir.lab.provider_documents import PYTHON_CANDIDATE_BUNDLE_V1
from open_cake_ir.lab.environments import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.lab.task_package import build_run_reference_documents
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.serialization import canonical_json_bytes as canonical
from open_cake_ir.tasks.authoring import prepare_schedule
from open_cake_ir.tasks.launch import parse_launch_manifest
from open_cake_ir.tasks.normalization.study import task_run_inputs
from open_cake_ir.tasks.preparation import prepare_task_run
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import load_workload, create_task
from tests.contracts import test_metal_preflight
from tests.contracts.test_native_triton_pairing import CompilationFixture
from tests.contracts.test_runtime_config import runtime_document
from tests.contracts._executor_fixture import compiler_reference

ROOT=Path(__file__).resolve().parents[2]


def program_source(single):
    return '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy-input", target="sm_100a", backend="triton", entry_point="copy_input")
def copy_in(lm, x: cake.Tensor((2,8), "fp32"), tmp: cake.Tensor((2,8), "fp32", mode="output")):
    row = lm.program(x, axis=0, dimension=0, tile=1)
    compute = lm.role(execution_groups=[0])
    with compute:
        v = lm.load(x[row,:])
        lm.store(tmp[row,:],v,coalesced=False)
''' + single.split('\n',1)[1] + '''
cake.program(program_id="copy-then-silu", inputs=["x"], outputs=["out"], stages=[
    cake.stage(name="copy", schedule=copy_in, bindings={"x":"x", "tmp":"tmp"}),
    cake.stage(name="activate", schedule=candidate, bindings={"x":"tmp", "out":"out"})])
'''


class ProgramTaskStarters(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name).resolve()
        self.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
        _, self.study, self.executor, self.receipt, _ = test_metal_preflight.MetalPreflightTests.fixture(
            self,self.directory,'claude-code',backend='triton-b200')
        self.workload=load_workload(self.directory/'workload.json')
        self.source=program_source((self.directory/'starter.py').read_text())
        self.path=self.directory/'program.py';self.path.write_text(self.source)
        self.raw=read_skeleton(self.path)
        self.route=skeleton_route(self.raw)
        self.authority={'lowering_route':self.route,'input_format':'python_source_v1',
            'provider':{'submission_contract':PYTHON_CANDIDATE_BUNDLE_V1}}
        self.prepared=prepare_schedule(self.raw,self.workload,'primary',self.authority)
        self.builder=TritonToolchainBuilder(workload=self.workload,case_id='primary',isolated_compiler=CompilationFixture())
        environment=OpenCakeEnvironment(self.compiler,self.builder,workload=self.workload,
                                        case_id='primary',authority_document=self.authority)
        result=environment.build(CandidateSubmission.seal(environment.media_type,canonical(
            {'python_program_source':self.source,'program_id':self.raw['program_id']})))
        self.assertEqual(result.disposition,'launchable',result.feedback)
        self.candidate=result.launchable

    def test_reader_binding_and_task_package_preserve_complete_python_program(self):
        reference={'path':str(self.path),'canonical_sha256':sha256(canonical(self.raw)).hexdigest()}
        payload,document=read_skeleton_reference(ROOT,reference)
        self.assertEqual(document,self.raw)
        self.assertEqual(payload,self.source.encode())
        self.assertEqual(bind_python_reference(self.source,self.prepared,filename=str(self.path)),payload)
        self.assertTrue(all(s['schedule']['metadata']['workload_contract_sha256']==self.workload.canonical_sha256
                            for s in self.prepared['stages']))
        inputs=task_run_inputs(ROOT,self.workload,self.directory/'workload.json',self.path,
            harness='claude-code',model='exact-test-model',effort='high',turns=2)
        docs=build_run_reference_documents(ROOT,SimpleNamespace(document=inputs),inputs['authoring'],
            workload_contract=self.workload,prepare_schedule=prepare_schedule)
        self.assertEqual(docs['schedule-starter.py'],payload)
        self.assertIn(b'not a Program device function',docs['program-starter.md'])
        with self.assertRaisesRegex(ValueError,'candidate-bundle'):
            task_run_inputs(ROOT,self.workload,self.directory/'workload.json',self.path,
                harness='claude-code',model='exact-test-model',effort='high',maximum_candidates=1,
                searches_per_turn=1,source_file=True)

    def test_no_parser_fallback_or_partial_program_binding(self):
        broken=self.source.replace('outputs=["out"]','outputs=dynamic_outputs')
        self.path.write_text(broken)
        with self.assertRaisesRegex(ValueError,'static literals'):
            read_skeleton(self.path)
        for mutation in ('stage_body','missing','reordered','backend','target'):
            document=deepcopy(self.prepared)
            if mutation=='stage_body':document['stages'][1]['schedule']['operations'][-1]['parameters']['coalesced']=True
            if mutation=='missing':document['stages'].pop()
            if mutation=='reordered':document['stages'].reverse()
            if mutation=='backend':document['stages'][1]['schedule']['lowering']['backend']='native_cuda'
            if mutation=='target':document['stages'][1]['schedule']['target']='sm_103a'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_python_reference(self.source,document,filename='program.py')
        mixed=deepcopy(self.raw);mixed['stages'][1]['schedule']['lowering']['backend']='native_cuda'
        with self.assertRaisesRegex(ValueError,'one authoring backend'):
            skeleton_route(mixed)
        with self.assertRaisesRegex(ValueError,'native comparison'):
            bind_baseline(self.raw,self.workload,'primary')

    def validate_baseline(self, candidate, lowered):
        execution={'fixed_baseline':{'candidate':candidate_identity(candidate),'bundle_path':'fixture'}}
        with patch.object(admission,'admit_paired_baseline_artifact',return_value=(candidate,False)):
            admission.validate_paired_baseline(project_root=ROOT,workload=self.workload,
                evaluation={'case_id':'primary'},execution=execution,route=self.route,
                baseline_lowering=lowered,manifest_parser=parse_launch_manifest)

    def test_sealed_baseline_checks_every_stage_not_only_the_first(self):
        lowered=lower_skeleton(self.compiler,self.prepared)
        self.validate_baseline(self.candidate,lowered)
        manifest,children,_=program_components(self.candidate)
        self.assertEqual(set(children),{'copy','activate'})
        for change in ('later_body','missing','order'):
            document=deepcopy(self.prepared)
            if change=='later_body':document['stages'][1]['schedule']['operations'][-1]['parameters']['coalesced']=True
            if change=='missing':document['stages'].pop()
            if change=='order':document['stages'].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                changed=lower_skeleton(self.compiler,document)
                self.validate_baseline(self.candidate,changed)

    def test_later_stage_source_and_launch_drift_refuse_even_with_coherent_bundle(self):
        from open_cake_ir.evaluation import LaunchableCandidate
        from open_cake_ir.evaluation.kernel_bundle import pack_candidates
        manifest, original_children, _ = program_components(self.candidate)
        lowered = lower_skeleton(self.compiler,self.prepared)
        for change in ('source', 'grid'):
            children = dict(original_children)
            child = children['activate']
            payloads = dict(child.artifact_payloads)
            outer = manifest.as_dict()
            report = json.loads(payloads['stage_compilation'])
            launch = json.loads(payloads['launch_manifest'])
            if change == 'source':
                replaced = payloads['lowered_source'].replace(b'tl.exp(', b'tl.exp2(', 1)
                self.assertNotEqual(replaced,payloads['lowered_source'])
                payloads['lowered_source'] = replaced
                report['source_sha256'] = sha256(replaced).hexdigest()
                outer['lowered_sources']['activate'] = report['source_sha256']
            else:
                launch['grid'][0] += 1
                report['grid'] = launch['grid']
            payloads['stage_compilation'] = canonical(report)
            payloads['launch_manifest'] = canonical(launch)
            children['activate'] = LaunchableCandidate(child.candidate_sha256,child.target,child.entry_point,
                {role:sha256(value).hexdigest() for role,value in payloads.items()},
                sha256(payloads['launch_manifest']).hexdigest(),payloads)
            payloads = {'launch_manifest':canonical(outer),'program_bundle':pack_candidates(children)}
            changed = LaunchableCandidate(self.candidate.candidate_sha256,self.candidate.target,self.candidate.entry_point,
                {role:sha256(value).hexdigest() for role,value in payloads.items()},
                sha256(payloads['launch_manifest']).hexdigest(),payloads)
            # The graph and internal metadata are coherent; only comparison to the
            # frozen Compiler's second stage exposes this substitution.
            program_components(changed)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,"stage 'activate'"):
                self.validate_baseline(changed,lowered)

    def test_prepare_task_run_and_tasklab_preflight_use_the_complete_starter(self):
        inputs=task_run_inputs(ROOT,self.workload,self.directory/'workload.json',self.path,
            harness='claude-code',model='exact-test-model',effort='high',turns=2)
        executable=self.directory/'provider';executable.write_bytes(b'CPU provider fixture; never executed')
        self.receipt.executable_sha256=sha256(executable.read_bytes()).hexdigest()
        runtime=runtime_document('triton')
        runtime['provider']={'executable':str(executable),'workspace_root':str(self.directory/'actors')}
        runtime['broker']['cwd']=str(ROOT)
        runtime_path=self.directory/'runtime.json';runtime_path.write_bytes(canonical(runtime))
        baseline=self.directory/'baseline.json';baseline.write_text('{}')
        selection={'schema_version':1,'policy':'starter_reference','source':'starter_reference',
                   'incumbent_key':None,'promotion_run_id':None,'registry_root':None}
        toolchain=SimpleNamespace(canonical_sha256='1'*64,check_executor=lambda *args,**kwargs:None)
        with patch.object(admission.ProviderQualificationReceipt,'load',return_value=self.receipt), \
             patch.object(ExecutorRevision,'load_reference',return_value=self.executor), \
             patch('open_cake_ir.lab.triton_build.IsolatedTritonCompiler',return_value=toolchain), \
             patch.object(bindings,'broker_execution_sha256',return_value='2'*64), \
             patch.object(bindings,'load_baseline_bundle',return_value=self.candidate), \
             patch.object(admission,'load_baseline_bundle',return_value=self.candidate):
            run=prepare_task_run(ROOT,inputs,compiler_reference=compiler_reference(ROOT),executor=self.executor,
                qualification_path=self.directory/'receipt-double.json',qualification_anchor_path=self.directory/'anchor-double.json',
                runtime_config_path=runtime_path,baseline_path=baseline,baseline_selection=selection)
            self.assertEqual(TaskLab(ROOT).preflight_run(run).document,run.document)
            package=TaskLab(ROOT).task_package(run,run.run_id)
            self.assertIsNotNone(package)
            changed=deepcopy(inputs)
            changed['authoring']['lowering_route']['entry_point']='not_the_starter_default'
            with self.assertRaisesRegex(ValueError,'lowering route differs'):
                prepare_task_run(ROOT,changed,compiler_reference=compiler_reference(ROOT),executor=self.executor,
                    qualification_path=self.directory/'receipt-double.json',qualification_anchor_path=self.directory/'anchor-double.json',
                    runtime_config_path=runtime_path,baseline_path=baseline,baseline_selection=selection)

    def test_launcher_baseline_build_submits_a_complete_program(self):
        from tools import launch_task
        workspace=self.directory/'launch';workspace.mkdir()
        strategy=SimpleNamespace(baseline_builder=lambda *args:self.builder)
        with patch.object(launch_task,'_launch_toolchain',return_value=strategy):
            bundle=launch_task._prepare_baseline(ROOT,workspace,self.compiler,None,None,self.workload,
                self.authority,self.source,compiler_reference(ROOT),route='triton')
        candidate=bindings.load_baseline_bundle(ROOT,str(bundle))
        manifest,children,_=program_components(candidate)
        self.assertEqual([s.name for s in manifest.program.stages],['copy','activate'])
        self.assertEqual(len(children),2)
