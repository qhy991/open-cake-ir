"""Input closure for sealed rewrite versus external implementation comparisons."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_rewrite_artifacts import regular, reference_spec, comparison_roles


class RewriteArtifactComparisonTests(unittest.TestCase):
    def test_new_schedule_and_fixed_control_are_not_labeled_as_confirmed_promotion(self):
        for control,label in [('optimized','unchanged old optimized binary'),
                              ('starter','unchanged original Cake starter')]:
            roles = comparison_roles({'kind':'authored_schedule_comparison','control_role':control})
            self.assertEqual(roles['starter'],label)
            self.assertEqual(roles['optimized'],'new authored complete Cake Schedule')
        with self.assertRaisesRegex(ValueError,'control role'):
            comparison_roles({'kind':'authored_schedule_comparison','control_role':'unknown'})
        self.assertEqual(comparison_roles({'kind':'explicit_alignment_ablation'})['starter'],
                         'unchanged pre-specialization optimized binary')

    def test_native_reference_preserves_declared_entry_and_source_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = root / 'reference'
            ref.mkdir()
            (ref / 'kernel.cu').write_text('// unchanged external source\n')
            document = {'kind': 'cuda_cpp', 'files': ['kernel.cu'],
                        'entry_point': 'kernel.cu::run', 'compile_options': {'cuda_cflags': ['-O3']}}
            path = ref / 'reference.json'
            path.write_text(json.dumps(document))
            spec, files, entry, function = reference_spec(root)
            self.assertEqual(spec, document)
            self.assertEqual(files, [ref / 'kernel.cu'])
            self.assertEqual((entry, function), ('kernel.cu', 'run'))
            for changes in ({'entry_point': 'missing.cu::run'}, {'files': ['kernel.cu', 'kernel.cu']},
                            {'files': ['../kernel.cu']}, {'kind': 'unknown'},
                            {'entry_point': 'kernel.cu::not-a-symbol'}):
                path.write_text(json.dumps({**document, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    reference_spec(root)

    def test_reference_files_cannot_be_redirected_outside_input_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'source.py').write_text('pass\n')
            (root / 'alias.py').symlink_to(root / 'source.py')
            for value in ('../source.py', str(root / 'source.py'), 'alias.py', 'a\\b.py'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    regular(root, value)
            self.assertEqual(regular(root, 'source.py'), root / 'source.py')


class CachedOutputObservationTests(unittest.TestCase):
    def pool(self, outputs, functions=None):
        from types import SimpleNamespace
        from tools.compare_rewrite_artifacts import RetainedExternal
        loaded = object.__new__(RetainedExternal)
        loaded.cached_output = True
        loaded.seen = set()
        loaded.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        loaded.launches = functions or [(lambda values, out, tensor=tensor: tensor) for tensor in outputs]
        loaded.launch_function = loaded.launches[0]
        return loaded

    def parent_patches(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from tools.compare_rewrite_artifacts import LoadedCallable
        stack = ExitStack()
        stack.enter_context(patch.object(LoadedCallable, 'fresh_argument_sets',
            lambda self, count: [{'inputs': {}, 'out': None, 'result': None} for _ in range(count)]))
        def launch(self, arguments):
            arguments['result'] = self.launch_function(arguments['inputs'], arguments['out'])
        stack.enter_context(patch.object(LoadedCallable, 'launch', launch))
        return stack

    def test_shared_buffers_and_changed_timed_buffer_are_refused(self):
        one, two = FakeOutput(1), FakeOutput(2)
        with self.parent_patches():
            with self.assertRaisesRegex(ValueError, 'share an output'):
                self.pool([one, one]).fresh_argument_sets(2)
            loaded = self.pool([one])
            arguments = loaded.fresh_argument_sets(1)
            loaded.launches = [lambda values, out: two]
            with self.assertRaisesRegex(ValueError, 'differs from poisoned'):
                loaded.launch(arguments[0])

    def test_no_write_cannot_reuse_correct_warm_output(self):
        import math
        from open_cake_ir.evaluation.core import compare_tile_outputs
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
        document, _ = create_task('rmsnorm', backend='triton-b300', rows=1, columns=1)
        workload = WorkloadContract(document)
        inputs = materialize_case(workload, 'primary')
        expected = reference_outputs(workload, 'primary', inputs)
        tensor = FakeOutput(1)
        tensor.value = expected['out'][0]
        with self.parent_patches():
            loaded = self.pool([tensor])
            arguments = loaded.fresh_argument_sets(1)
            self.assertTrue(math.isnan(tensor.value))
            loaded.launch(arguments[0])  # Reference returns its buffer but writes nothing.
            passed, _ = compare_tile_outputs(workload, inputs, expected,
                                             {'out': [arguments[0]['result'].value]}, inputs)
            self.assertFalse(passed)


class FakeOutput:
    def __init__(self, pointer):
        self.pointer = pointer
        self.value = 1.0

    def data_ptr(self):
        return self.pointer

    def fill_(self, value):
        self.value = value


class ProgramComparisonPreparationTests(unittest.TestCase):
    def test_program_preparation_seals_all_stages_and_keeps_the_selected_control(self):
        from copy import deepcopy
        from unittest.mock import patch
        from open_cake_ir.compiler import Compiler, Program, frontend
        from open_cake_ir.compiler.toolchain import TritonCompilation, triton_route
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.evaluation.paired import candidate_identity
        from open_cake_ir.evaluation.program import program_components
        from open_cake_ir.lab.build import TritonToolchainBuilder
        from open_cake_ir.lab.environments import OpenCakeEnvironment, CandidateSubmission
        from open_cake_ir.lab.bindings import load_baseline_bundle
        from open_cake_ir.tasks.workloads import create_task
        from open_cake_ir.serialization import canonical_json_bytes
        from tools import prepare_schedule_comparison as preparation
        class CPUCompiler:
            def check_executor(self,*args,**kwargs): pass
            def compile(self,source,requirements):
                route=triton_route(requirements)
                payloads={role:b'CPU fixture; not executable' for role in route.artifact_roles}
                payloads['source']=b'CPU compiler expansion\n'+source
                payloads['cubin']=b'\x7fELF CPU fixture'
                return TritonCompilation(source,requirements['target'],requirements['kernel_entry_point'],
                    payloads,requirements['compile_options']['num_warps']*32,0,'CPU fixture','cubin')
        document,source=create_task('rmsnorm',backend='triton-b300',rows=1,columns=8)
        workload=WorkloadContract(document);schedule=frontend.parse(source).document
        compiler=Compiler.load(preparation.ROOT,preparation.ROOT/'compiler/revision.json')
        builder=TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=CPUCompiler())
        environment=OpenCakeEnvironment(compiler,builder,workload=workload,case_id='primary',
            authority_document={'input_format':'schedule_or_python_v1','lowering_route':schedule['lowering']})
        control=environment.build(CandidateSubmission.seal(environment.media_type,canonical_json_bytes(schedule))).launchable
        self.assertIsNotNone(control)
        program=Program.from_schedule(schedule).document
        program['tensors']['partial']=deepcopy(program['tensors']['out'])
        program['stages'][0]['bindings']['out']='partial'
        copy_source='''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy-final",target="sm_103a",backend="triton",entry_point="copy_final")
def candidate(lm,partial:cake.Tensor((1,8),"fp32"),out:cake.Tensor((1,8),"fp32",mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(out,axis=0,dimension=0,tile=1)
    with compute:
        value=lm.load(partial[row,:])
        lm.store(out[row,:],value,coalesced=False)
'''
        program['stages'].append({'name':'finalize','schedule':frontend.parse(copy_source).document,
                                  'bindings':{'partial':'partial','out':'out'}})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();input_root=root/'input';input_root.mkdir()
            (input_root/'workload.json').write_text(json.dumps(document));ref=input_root/'reference';ref.mkdir()
            (ref/'ref.py').write_text('def run(*args): pass\n')
            (ref/'reference.json').write_text(json.dumps({'kind':'python','files':['ref.py'],'entry_point':'ref.py::run'}))
            sealed=input_root/'optimized';sealed.mkdir();paths={}
            for role,payload in control.artifact_payloads.items():paths[role]=role+'.bin';(sealed/paths[role]).write_bytes(payload)
            (sealed/'candidate.json').write_text(json.dumps({'candidate':candidate_identity(control),'artifact_paths':paths}))
            runtime=root/'runtime.json';runtime.write_text(json.dumps({'toolchain':{}}))
            authored=root/'program.json';authored.write_text(json.dumps(program))
            out=root/'comparison'
            with patch.object(preparation,'IsolatedTritonCompiler',return_value=CPUCompiler()):
                metadata=preparation.prepare(authored,input_root,out,runtime,'optimized','CPU composition fixture')
                candidate=load_baseline_bundle(preparation.ROOT,out/'optimized/candidate.json')
                manifest,children,_=program_components(candidate)
                self.assertEqual(candidate.kernels_per_call,2)
                self.assertEqual(len(children),2)
                manifest.check_workload(workload,'primary')
                self.assertEqual((out/'starter/candidate.json').read_bytes(),(sealed/'candidate.json').read_bytes())
                self.assertEqual(json.loads((out/'authored-program.json').read_text()),program)
                self.assertEqual(metadata['kind'],'authored_program_comparison')
                roles=comparison_roles(metadata)
                self.assertEqual(roles['optimized'],'new authored complete Cake Program')
                self.assertEqual(roles['starter'],'unchanged old optimized binary')
                incomplete=deepcopy(program);incomplete['stages'].pop();authored.write_text(json.dumps(incomplete))
                with self.assertRaises(ValueError):
                    preparation.prepare(authored,input_root,root/'incomplete',runtime,'optimized','missing writer')
                self.assertFalse((root/'incomplete').exists())
