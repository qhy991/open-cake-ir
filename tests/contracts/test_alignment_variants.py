import json
from hashlib import sha256
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.toolchain import TritonCompilation, pointer_alignment_attributes, triton_route
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission, LoadedCudaCandidate
from open_cake_ir.evaluation.kernel_bundle import LoadedAlignmentCandidate, alignment_component
from open_cake_ir.evaluation.paired import validate_pair_candidates, participant_work, candidate_identity
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
from open_cake_ir.tasks.workloads import create_task
from test_cuda_driver import FakeDriver, FakeTensor

ROOT = Path(__file__).resolve().parents[2]


class AlignmentVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document, source = create_task('fib_rmsnorm_h2048', backend='triton-b300', rows=1, columns=2048)
        cls.workload = WorkloadContract(document)
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.lowering = compiler.lower(compiler.assess(frontend.parse(source).document))

    def build(self, alignment):
        calls = []
        class Isolated:
            def compile(self, source, requirements):
                calls.append(dict(requirements))
                label = b'aligned' if requirements.get('pointer_alignments') else b'generic'
                route = triton_route(requirements)
                artifacts = {role: b'fixture-' + role.encode() + label for role in route.artifact_roles}
                artifacts['source'] = source
                return TritonCompilation(source, requirements['target'], requirements['kernel_entry_point'],
                    artifacts, requirements['compile_options']['num_warps'] * 32,
                    16 if label == b'aligned' else 0, 'fixture', 'cubin')
        lower = self.lowering
        payload = lower.source.encode()
        request = BuildRequest('a' * 64, payload, 'lowered_source', sha256(payload).hexdigest(),
                               'sm_103a', lower.route.entry_point, lower.toolchain_requirements)
        result = TritonToolchainBuilder(workload=self.workload, case_id='primary',
            isolated_compiler=Isolated(), pointer_alignment=alignment).build(request)
        return result, TensorLaunchManifest.from_dict(json.loads(result.artifact_payloads['launch_manifest'])), calls

    def test_attributes_are_explicit_and_invalid_contracts_refuse(self):
        base = {'signature': {'x':'*bf16','n':'i32'}}
        self.assertEqual(pointer_alignment_attributes(base), {})
        self.assertEqual(pointer_alignment_attributes({**base,'pointer_alignments':{'x':16}}),
                         {(0,):[('tt.divisibility',16)]})
        for value in [{'n':16},{'unknown':16},{'x':0},{'x':3},{'x':True}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                pointer_alignment_attributes({**base,'pointer_alignments':value})

    def test_generic_baseline_stays_v1_and_bundle_binds_the_same_program(self):
        baseline, old, calls = self.build(None)
        self.assertEqual(len(calls),1)
        self.assertEqual(old.as_dict()['schema_version'],1)
        self.assertNotIn('kernel_bundle',baseline.artifact_roles)
        candidate, manifest, calls = self.build(16)
        self.assertEqual(len(calls),2)
        self.assertNotIn('pointer_alignments',calls[0])
        self.assertEqual(set(calls[1]['pointer_alignments']),{r[0] for r in manifest.tensor_abi})
        child, leaf = alignment_component(candidate,manifest)
        self.assertEqual(leaf.dynamic_shared_memory_bytes,16)
        self.assertEqual(manifest.dynamic_shared_memory_bytes,0)
        self.assertEqual(leaf.tensor_abi,old.tensor_abi)
        validate_pair_candidates(candidate,baseline,self.workload,'primary')
        with self.assertRaisesRegex(ValueError,'restricted leaf'):
            validate_pair_candidates(child,baseline,self.workload,'primary')
        work = participant_work({'participants':{'candidate':candidate_identity(candidate),'baseline':candidate_identity(baseline)},
             'launch_manifests':{'candidate':manifest.as_dict(),'baseline':old.as_dict()}})
        self.assertEqual(work,{'candidate':{'modules':2,'kernels':1},'baseline':{'modules':1,'kernels':1}})

    def test_each_launch_checks_inputs_and_output_and_leaf_cannot_bypass_guard(self):
        candidate, manifest, _ = self.build(16)
        admission = CudaDeviceAdmission('NVIDIA B300',(10,3),'test-gpu','gpuq-123456789abc','exclusive')
        drivers = []
        def loader(item, spec, observed):
            driver = FakeDriver();driver.attributes[6]=103;drivers.append(driver)
            return LoadedCudaCandidate.load(item,item.artifact_payloads['cubin'],spec,observed,driver=driver)
        loaded = LoadedAlignmentCandidate(candidate,manifest,admission,loader,lambda:None)
        args = [FakeTensor(shape,dtype,(i+1)*1048576) for i,(_,shape,dtype) in enumerate(manifest.tensors)]
        try:
            loaded.launch(args,tensor_contract=manifest,stream=0)
            self.assertEqual(loaded.last_variant,'aligned')
            for index in range(len(args)):
                for offset in (2,4,8):
                    args[index]._pointer += offset
                    loaded.launch(args,tensor_contract=manifest,stream=0)
                    self.assertEqual(loaded.last_variant,'generic')
                    with self.assertRaisesRegex(ValueError,'alignment contract'):
                        loaded.aligned.launch(args,tensor_contract=loaded.aligned_manifest,stream=0)
                    args[index]._pointer -= offset
            loaded.launch(args,tensor_contract=manifest,stream=0)
            self.assertEqual(loaded.last_variant,'aligned')
            args[0].dtype='torch.float16'
            with self.assertRaisesRegex(ValueError,'dtype differs'):
                loaded.launch(args,tensor_contract=manifest,stream=0)
            self.assertEqual(loaded.dispatch_counts,{'generic':9,'aligned':2})
            self.assertEqual(sum(sum(c[0]=='cuLaunchKernel' for c in d.calls) for d in drivers),11)
        finally:
            loaded.close(synchronize=lambda:None)
        self.assertTrue(loaded.closed)
