"""Program stage alignment retains unrestricted ABI, guards and physical counts."""
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission, LoadedCudaCandidate
from open_cake_ir.evaluation.kernel_bundle import alignment_component, pack_candidates
from open_cake_ir.evaluation.paired import candidate_identity, participant_work, validate_pair_candidates
from open_cake_ir.evaluation.program import LoadedProgram, ProgramLaunchManifest, program_components
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts.test_program_evaluation import workload_for, replay_program_candidate
from tests.contracts.test_program_rewrites import epilogue_program
from tests.contracts.test_native_triton_pairing import CompilationFixture
from test_cuda_driver import FakeDriver, FakeTensor

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {'fp32': 'torch.float32', 'bf16': 'torch.bfloat16', 'fp16': 'torch.float16'}


class ProgramAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def build(self, alignment):
        program = Program.from_dict(epilogue_program())
        workload = workload_for(program)
        fixture = CompilationFixture()
        builder = TritonToolchainBuilder(workload=workload,case_id='primary',
            isolated_compiler=fixture,pointer_alignment=alignment)
        environment = OpenCakeEnvironment(self.compiler,builder,workload=workload,case_id='primary',
            authority_document={'input_format':'schedule_or_python_v1',
                                'lowering_route':program.document['stages'][0]['schedule']['lowering']})
        result = environment.build(CandidateSubmission.seal(environment.media_type,program.document_bytes))
        self.assertIsNotNone(result.launchable,result.feedback)
        return program, workload, result.launchable, fixture

    def test_stage_variants_replay_and_count_modules_separately_from_kernel_calls(self):
        program,workload,candidate,fixture = self.build(16)
        old_program,_,baseline,old_fixture = self.build(None)
        old,_,_ = program_components(baseline)
        self.assertEqual(old.as_dict()['schema_version'],1)
        self.assertNotIn('aligned_stages',old.as_dict())
        manifest,children,_ = program_components(candidate)
        self.assertEqual(manifest.aligned_stages,tuple(s.name for s in program.stages))
        self.assertEqual(manifest.as_dict()['schema_version'],2)
        self.assertEqual(len(fixture.requests),4)
        self.assertEqual(len(old_fixture.requests),2)
        self.assertEqual(len(children),2)
        self.assertEqual(candidate.kernels_per_call,2)
        parsed = validate_pair_candidates(candidate,baseline,workload,'primary')
        work = participant_work({'participants':{'candidate':candidate_identity(candidate),'baseline':candidate_identity(baseline)},
                                 'launch_manifests':{role:value.as_dict() for role,value in parsed.items()}})
        self.assertEqual(work,{'candidate':{'modules':4,'kernels':2},'baseline':{'modules':2,'kernels':2}})
        replayed = replay_program_candidate(self.compiler,program,candidate,candidate.artifact_payloads)
        self.assertEqual(replayed.canonical_sha256,candidate.canonical_sha256)
        self.assertEqual(program.document,old_program.document)

    def test_every_public_and_private_address_selects_the_stage_guard(self):
        program,_,candidate,_ = self.build(16)
        manifest,_,_ = program_components(candidate)
        admission = CudaDeviceAdmission('NVIDIA B200',(10,0),'test-gpu','gpuq-123456789abc','exclusive')
        public = program.inputs + program.outputs
        scenarios = [(None,0)] + [(name,offset) for name in program.tensors for offset in (2,4,8)]
        for changed,offset in scenarios:
            with self.subTest(changed=changed,offset=offset):
                drivers=[];next_address=1<<20
                def tensor(name,spec):
                    nonlocal next_address
                    next_address+=1<<20
                    return FakeTensor(spec.shape,DTYPES[spec.dtype.value],next_address+(offset if name==changed else 0))
                private = {name:tensor(name,spec) for name,spec in program.tensors.items() if name not in public}
                arguments = [tensor(name,program.tensors[name]) for name in public]
                def loader(child,spec,observed):
                    driver=FakeDriver();drivers.append(driver)
                    return LoadedCudaCandidate.load(child,child.artifact_payloads['cubin'],spec,observed,driver=driver)
                def check_tensor(value,spec):
                    if tuple(value.shape)!=spec.shape or value.dtype!=DTYPES[spec.dtype.value]:raise ValueError('fixture ABI differs')
                def span(value):
                    width=4 if value.dtype=='torch.float32' else 2
                    return ('cuda:0',value.data_ptr(),value.data_ptr()+math.prod(value.shape)*width)
                # This fixture has exactly one private intermediate.
                self.assertEqual(len(private),1)
                loaded = LoadedProgram(candidate,manifest,admission,loader,
                    allocate=lambda spec:next(iter(private.values())),view=lambda t,shape:t,
                    check_tensor=check_tensor,storage_span=span,stream=0)
                try:
                    loaded.prepare_arguments(arguments)
                    loaded.launch(arguments,tensor_contract=manifest,stream=0)
                    self.assertEqual(loaded.launch_calls,2)
                    for stage in program.stages:
                        affected=changed in {binding.tensor for binding in stage.bindings.values()}
                        stage_loaded=loaded._children[stage.name]
                        self.assertEqual(stage_loaded.last_variant,'generic' if affected else 'aligned')
                        if affected:
                            args=loaded._prepared[id(arguments)][3][stage.name]
                            before=stage_loaded.aligned.launch_calls
                            with self.assertRaisesRegex(ValueError,'alignment contract'):
                                stage_loaded.aligned.launch(args,tensor_contract=stage_loaded.aligned_manifest,stream=0)
                            self.assertEqual(stage_loaded.aligned.launch_calls,before)
                    self.assertEqual(len(drivers),4)
                    self.assertEqual(sum(sum(c[0]=='cuLaunchKernel' for c in d.calls) for d in drivers),2)
                finally:
                    loaded.close(synchronize=lambda:None)
                self.assertTrue(loaded.closed)
                self.assertEqual(sum(sum(c[0]=='cuModuleUnload' for c in d.calls) for d in drivers),4)

    def test_seal_refuses_hidden_dispatch_and_restricted_generic_or_corrupt_leaf_compile(self):
        _,_,candidate,_=self.build(16)
        manifest,children,_=program_components(candidate)
        def rebuild(payloads):
            return replace(candidate,artifact_payloads=payloads,
                artifact_roles={k:sha256(v).hexdigest() for k,v in payloads.items()},
                launch_spec_sha256=sha256(payloads['launch_manifest']).hexdigest())
        hidden=manifest.as_dict();hidden.pop('aligned_stages');hidden['schema_version']=1
        with self.assertRaisesRegex(ValueError,'artifact or ABI binding'):
            rebuild({**candidate.artifact_payloads,'launch_manifest':canonical_json_bytes(hidden)})
        for names in ([],['missing'],list(reversed(manifest.aligned_stages)),[manifest.aligned_stages[0]]*2):
            with self.subTest(names=names),self.assertRaisesRegex(ValueError,'ordered subset'):
                ProgramLaunchManifest.from_dict({**manifest.as_dict(),'aligned_stages':names})
        stage=manifest.aligned_stages[0];child=children[stage]
        parent_manifest=TensorLaunchManifest.from_dict(json.loads(child.artifact_payloads['launch_manifest']))
        aligned,leaf=alignment_component(child,parent_manifest)
        restricted_children={**children,stage:aligned}
        with self.assertRaisesRegex(ValueError,'artifact or ABI binding'):
            rebuild({**candidate.artifact_payloads,'program_bundle':pack_candidates(restricted_children)})
        report=json.loads(aligned.artifact_payloads['stage_compilation']);report['threads_per_cta']+=32
        altered={**aligned.artifact_payloads,'stage_compilation':canonical_json_bytes(report)}
        aligned=replace(aligned,artifact_payloads=altered,artifact_roles={k:sha256(v).hexdigest() for k,v in altered.items()})
        altered={**child.artifact_payloads,'kernel_bundle':pack_candidates({'aligned':aligned})}
        child=replace(child,artifact_payloads=altered,artifact_roles={k:sha256(v).hexdigest() for k,v in altered.items()})
        with self.assertRaisesRegex(ValueError,'compiler metadata'):
            rebuild({**candidate.artifact_payloads,'program_bundle':pack_candidates({**children,stage:child})})

    def test_failed_variant_load_cleans_up_all_earlier_stage_modules(self):
        _,_,candidate,_=self.build(16)
        manifest,_,_=program_components(candidate)
        modules=[]
        class Module:
            closed=False
            def close(self,*,synchronize):synchronize();self.closed=True
        def loader(*args):
            if len(modules)==3:raise RuntimeError('last aligned module failed')
            item=Module();modules.append(item);return item
        with self.assertRaisesRegex(RuntimeError,'last aligned module failed'):
            LoadedProgram(candidate,manifest,None,loader,allocate=lambda _:None,view=lambda *args:None,
                          check_tensor=lambda *args:None,storage_span=lambda _:None,stream=0)
        self.assertEqual(len(modules),3)
        self.assertTrue(all(module.closed for module in modules))
