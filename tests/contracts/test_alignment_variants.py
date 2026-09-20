import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.toolchain import TritonCompilation, pointer_alignment_attributes, triton_route
from open_cake_ir.evaluation.core import LaunchableCandidate, TensorLaunchManifest
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

    def build(self, alignment, *, compilation=None):
        calls = []
        class Isolated:
            def compile(self, source, requirements):
                calls.append(dict(requirements))
                label = b'aligned' if requirements.get('pointer_alignments') else b'generic'
                route = triton_route(requirements)
                artifacts = {role: b'fixture-' + role.encode() + label for role in route.artifact_roles}
                artifacts['cubin'] = b'\x7fELF' + artifacts['cubin']
                # Real Triton exposes expanded compiler IR here, including the
                # alignment attributes and per-compilation source locations.
                artifacts['source'] = b'expanded-ir-' + label
                return TritonCompilation(source, requirements['target'], requirements['kernel_entry_point'],
                    artifacts, requirements['compile_options']['num_warps'] * 32,
                    16 if label == b'aligned' else 0, 'fixture', 'cubin')
        lower = self.lowering
        payload = lower.source.encode()
        request = BuildRequest('a' * 64, payload, 'lowered_source', sha256(payload).hexdigest(),
                               'sm_103a', lower.route.entry_point, lower.toolchain_requirements, compilation=compilation)
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
        self.assertNotEqual(child.artifact_roles['compiler_expanded_source'],
                            candidate.artifact_roles['compiler_expanded_source'])
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
            with self.assertRaisesRegex(ValueError, 'tensor contract differs'):
                loaded.aligned.launch(args,tensor_contract=manifest,stream=0)
            changed = replace(loaded.aligned_manifest,
                tensor_abi=tuple(reversed(loaded.aligned_manifest.tensor_abi)))
            with self.assertRaisesRegex(ValueError, 'tensor contract differs'):
                loaded.aligned.launch(args,tensor_contract=changed,stream=0)
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

    def test_incomplete_bundle_refuses_at_construction_before_any_evaluation(self):
        candidate, _, _ = self.build(16)
        payloads = dict(candidate.artifact_payloads)
        payloads.pop('kernel_bundle')
        with self.assertRaisesRegex(ValueError, 'complete sealed kernel bundle'):
            replace(candidate, artifact_payloads=payloads,
                    artifact_roles={k: sha256(v).hexdigest() for k,v in payloads.items()})
        # Old identity-only records do not pretend to contain launchable bytes.
        LaunchableCandidate(candidate.candidate_sha256, candidate.target, candidate.entry_point,
                            candidate.artifact_roles, candidate.launch_spec_sha256)

    def test_generic_contract_and_source_cannot_change_in_the_aligned_component(self):
        from open_cake_ir.evaluation.kernel_bundle import pack_candidates
        from open_cake_ir.serialization import canonical_json_bytes
        candidate, manifest, _ = self.build(16)
        child, leaf = alignment_component(candidate,manifest)
        for field, value in [('grid',[2,1,1]),
                             ('hidden_null_pointer_parameters',(leaf.hidden_null_pointer_parameters + 1) % 3)]:
            raw = {**leaf.as_dict(),field:value}
            spec = canonical_json_bytes(raw)
            child_payloads = {**child.artifact_payloads, 'launch_manifest':spec}
            changed = replace(child, artifact_payloads=child_payloads,
                artifact_roles={k:sha256(v).hexdigest() for k,v in child_payloads.items()},
                launch_spec_sha256=sha256(spec).hexdigest())
            payloads = {**candidate.artifact_payloads,'kernel_bundle':pack_candidates({'aligned':changed})}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'preserve source'):
                replace(candidate,artifact_payloads=payloads,
                        artifact_roles={k:sha256(v).hexdigest() for k,v in payloads.items()})
        child_payloads = {**child.artifact_payloads,'lowered_source':b'different authored program'}
        changed = replace(child,artifact_payloads=child_payloads,
                          artifact_roles={k:sha256(v).hexdigest() for k,v in child_payloads.items()})
        payloads = {**candidate.artifact_payloads,'kernel_bundle':pack_candidates({'aligned':changed})}
        with self.assertRaisesRegex(ValueError,'changes its source program'):
            replace(candidate,artifact_payloads=payloads,
                    artifact_roles={k:sha256(v).hexdigest() for k,v in payloads.items()})

    def test_partial_teardown_can_resume_and_all_failures_are_preserved(self):
        from open_cake_ir.evaluation.loaders import LifecycleError
        from unittest.mock import Mock
        candidate, manifest, _ = self.build(16)
        generic, aligned = Mock(closed=False), Mock(closed=False)
        primary, cleanup = RuntimeError('load aligned'), RuntimeError('unload generic')
        generic.close.side_effect = cleanup
        loader = Mock(side_effect=[generic, primary])
        with self.assertRaises(LifecycleError) as caught:
            LoadedAlignmentCandidate(candidate,manifest,None,loader,lambda:None)
        self.assertIs(caught.exception.primary,primary)
        self.assertIs(caught.exception.teardown,cleanup)
        loaded = LoadedAlignmentCandidate(candidate,manifest,None,
                                          Mock(side_effect=[generic,aligned]),lambda:None)
        def closed_aligned(**_): aligned.closed = True
        aligned.close.side_effect = closed_aligned
        with self.assertRaisesRegex(RuntimeError,'unload generic'):
            loaded.close(synchronize=lambda:None)
        def closed_generic(**_): generic.closed = True
        generic.close.side_effect = closed_generic
        loaded.close(synchronize=lambda:None)
        self.assertTrue(loaded.closed)
        self.assertEqual(aligned.close.call_count,1)
        generic.closed = aligned.closed = False
        generic.close.side_effect = cleanup
        aligned.close.side_effect = primary
        with self.assertRaises(LifecycleError) as caught:
            loaded.close(synchronize=lambda:None)
        self.assertIs(caught.exception.primary,primary)
        self.assertIs(caught.exception.teardown,cleanup)
