"""Native source admission uses the emission's declared execution-group width."""
import copy
from pathlib import Path
import unittest
from unittest import mock

from open_cake_ir.compiler import Compiler,frontend
from open_cake_ir.compiler.toolchain import project_triton_kernel
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.environments import CandidateSubmission,NativeTritonEnvironment
from open_cake_ir.lab.faults import CandidateCompileRejected
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.contraction.workload import workload_document
from open_cake_ir.tasks.contraction.authoring import starter_source

ROOT=Path(__file__).resolve().parents[2]


class RejectingCompiler:
    def __init__(self):self.requests=[]
    def build(self,request):
        self.requests.append(request)
        raise CandidateCompileRejected('CPU fixture: source reached the builder; no compilation or device work', artifact_payloads={})


class NativeTritonLaunchWidth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def fixture(self,backend='triton-metax'):
        w=WorkloadContract(workload_document('pairwise_sqdist',rows=128,depth=256,columns=32,backend=backend))
        d=frontend.parse(starter_source(w)).document
        low=self.compiler.lower(self.compiler.assess(d));req=copy.deepcopy(dict(low.toolchain_requirements))
        source=project_triton_kernel(low.source.encode(),req).decode()
        payload={'kernel_source':source,'compile_constants':req['compile_constants'],
                 'compile_options':req['compile_options'],'grid':req['grid']}
        builder=RejectingCompiler()
        env=NativeTritonEnvironment(builder,toolchain_requirements=req,
             authority_document={'environment_kind':'native_triton','reference_access':'known_kernel_reproduction'},
             workload=w,case_id='primary')
        return w,req,payload,builder,env

    def test_wave64_oversized_launch_is_refused_before_a_build(self):
        for backend in ('triton-metax','triton-dcu'):
            with self.subTest(backend=backend):
                w,req,d,builder,env=self.fixture(backend)
                self.assertEqual(req['warp_size'],64)
                d['compile_options']['num_warps']=32
                result=env.build(CandidateSubmission.seal(env.media_type,canonical_json_bytes(d)))
                self.assertEqual(result.feedback['stage'],'source_admission')
                self.assertEqual(builder.requests,[])
                self.assertNotIn('CUDA',result.feedback['error'])

    def test_group_width_is_taken_from_route_facts_for_every_target(self):
        from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec
        original=CudaKernelSpec.from_dict
        for backend in ('triton-metax','triton-dcu','triton-b300'):
            with self.subTest(backend=backend):
                w,req,d,builder,env=self.fixture(backend)
                d['compile_options']['num_warps']=4
                with mock.patch.object(CudaKernelSpec,'from_dict',wraps=original) as check:
                    result=env.build(CandidateSubmission.seal(env.media_type,canonical_json_bytes(d)))
                self.assertEqual(result.feedback['stage'],'compile')
                self.assertEqual(len(builder.requests),1)
                self.assertEqual(check.call_args.args[0]['block'],[4*declared_target(w.target).warp_size,1,1])

    def test_native_cuda_width32_can_admit_the_same_group_count(self):
        _,_,d,builder,env=self.fixture('triton-b300')
        d['compile_options']['num_warps']=32
        result=env.build(CandidateSubmission.seal(env.media_type,canonical_json_bytes(d)))
        self.assertEqual(result.feedback['stage'],'compile')
        self.assertEqual(len(builder.requests),1)

    def test_a_missing_width_is_not_substituted_with32(self):
        w,req,_,builder,_=self.fixture();req.pop('warp_size')
        with self.assertRaisesRegex(ValueError,'Triton compile contract differs'):
            NativeTritonEnvironment(builder,toolchain_requirements=req,
                authority_document={'environment_kind':'native_triton'},workload=w,case_id='primary')
        self.assertEqual(builder.requests,[])
