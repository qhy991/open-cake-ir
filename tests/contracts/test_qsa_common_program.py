"""CPU-only proof of QSA adapter ownership; no QSA device or timing qualification."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import project_triton_kernel
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.tasks.qsa import evaluate, cake, launch
from tests.contracts.test_program_rewrites import epilogue_program
from tests.contracts.test_program_evaluation import workload_for
from tests.contracts.test_native_triton_pairing import CompilationFixture
from tests.contracts.test_epilogue_fusion import execute, rounded

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Tensor:
    shape: tuple
    dtype: str
    data: list
    address: int
    device = 'cuda:0'
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 2 if self.dtype in {'bf16','fp16'} else 4
    def data_ptr(self): return self.address
    def is_contiguous(self): return True
    def view(self,shape): return Tensor(tuple(shape),self.dtype,self.data,self.address)


class Torch:
    float32='fp32';bfloat16='bf16';int32='int32'
    cuda = SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=7),synchronize=lambda:None)
    def __init__(self): self.position=1024;self.allocations=[]
    def full(self,shape,value,*,dtype,device):
        tensor = Tensor(tuple(shape),dtype,[value]*math.prod(shape),self.position)
        self.position += tensor.numel()*tensor.element_size()+1024
        self.allocations.append(tensor)
        return tensor


class QsaCommonProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT,ROOT/'compiler/revision.json')
        document = epilogue_program()
        document['outputs'] = ['output']
        document['tensors']['output'] = document['tensors'].pop('out')
        for stage in document['stages']:
            stage['bindings'] = {name:('output' if value=='out' else value) for name,value in stage['bindings'].items()}
        cls.program = Program.from_dict(document)
        cls.reference = SimpleNamespace(implementation=cls.program,workload=workload_for(cls.program))

    def compile(self,root,program,name):
        candidate = {'schema_version':2,'arm':'open_cake','program':program.document}
        fixture = CompilationFixture()
        def build(request):
            return fixture.compile(project_triton_kernel(request.source,request.toolchain_requirements),request.toolchain_requirements)
        output = root/name
        with patch.object(evaluate.ProgramContract,'load',return_value=self.reference), \
             patch.object(evaluate,'_compile_node',side_effect=build):
            profile = evaluate._compile_open_cake(ROOT,root,candidate,output,compiler=self.compiler,
                target=Target.load(ROOT/'compiler/targets/sm_100a.json'),candidate_sha256='a'*64)
        built,manifest,inspection = cake.read_cake_artifact(ROOT,output/'program.json')
        self.assertEqual(tuple(profile['nodes']),tuple(stage.name for stage in program.stages))
        self.assertEqual(len(fixture.requests),len(program.stages))
        self.assertEqual([kernel.kernel_id for kernel in inspection.kernels],[stage.name for stage in program.stages])
        return output,built,manifest

    def test_variable_stage_programs_use_common_build_storage_launch_and_oracle(self):
        fused = self.compiler.rewrite_program(self.program,'fuse_pointwise_epilogue',
            {'producer':'producer','epilogue':'epilogue','schedule_id':'fused','entry_point':'fused'}).program
        for index,program in enumerate((self.program,fused)):
            with self.subTest(stages=len(program.stages)),tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                _,candidate,manifest = self.compile(root,program,'candidate')
                _,children,_ = program_components(candidate)
                stages = {children[stage.name].entry_point:stage for stage in manifest.program.stages}
                torch = Torch()
                inputs = {name:torch.full(spec.shape,1. if name=='bias' else 0.,dtype=spec.dtype.value,device='cuda:0')
                          for name,spec in program.tensors.items() if name in program.inputs}
                output = torch.full(program.tensors['output'].shape,float('nan'),dtype='bf16',device='cuda:0')
                before = len(torch.allocations);calls=[];kernels=[]
                class Kernel:
                    launch_calls=0;closed=False;resources={}
                    def __init__(self,stage): self.stage=stage
                    def launch(self,args,*,tensor_contract,stream):
                        self.launch_calls+=1;calls.append(self.stage.name)
                        values = {name:arg.data for (name,_,_,mode),arg in zip(tensor_contract.tensor_abi,args,strict=True) if mode=='input'}
                        results,_ = execute(json.loads(self.stage.schedule_bytes),values)
                        for (name,_,_,mode),arg in zip(tensor_contract.tensor_abi,args,strict=True):
                            if mode=='output': arg.data[:]=results[name]
                    def close(self,*,synchronize): synchronize();self.closed=True
                def loader(child,spec,admission):
                    result = Kernel(stages[child.entry_point]);kernels.append(result);return result
                with patch.dict('sys.modules',{'torch':torch}),patch.object(cake,'_load_cubin',side_effect=loader), \
                     patch.object(evaluate.ProgramContract,'load',return_value=self.reference):
                    loaded = evaluate._load_program(root,'candidate',root=ROOT,compiler=self.compiler,
                        inputs=inputs,output=output,admission=object())
                    expected_private = set(program.tensors)-set(program.inputs)-set(program.outputs)
                    self.assertEqual(len(torch.allocations)-before,len(expected_private))
                    self.assertEqual(set(loaded.tensors),set(program.tensors))
                    boundaries=[]
                    loaded.launch(loaded.tensors,stream=7,boundary=lambda name,phase:boundaries.append((name,phase)))
                    self.assertEqual(calls,[stage.name for stage in program.stages])
                    self.assertEqual(boundaries,[(stage.name,phase) for stage in program.stages for phase in ('before','after')])
                    self.assertEqual(output.data,[rounded(1/(1+math.exp(-1)),'bf16')]*16)
                    self.assertEqual(len(torch.allocations)-before,len(expected_private))
                    loaded.close(synchronize=torch.cuda.synchronize)
                    self.assertTrue(all(kernel.closed for kernel in kernels))

    def test_v2_seed_carries_one_complete_program_and_legacy_nodes_are_only_an_input_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cake_root,_ = launch._materialize_candidates(root,self.reference)
            document = json.loads((cake_root/'candidate.json').read_bytes())
            self.assertEqual(document,{'schema_version':2,'arm':'open_cake','program':self.program.document})
            nodes=[]
            for index,stage in enumerate(self.program.stages):
                name=f'stage-{index}.json';(root/name).write_bytes(stage.schedule_bytes)
                nodes.append({'id':stage.name,'schedule':name})
            projected = cake.candidate_program({'schema_version':1,'arm':'open_cake','nodes':nodes},root,
                self.reference,evaluate._owned_file)
            self.assertEqual(projected.document,self.program.document)
            nodes.reverse()
            with self.assertRaisesRegex(ValueError,'frozen reference'):
                cake.candidate_program({'schema_version':1,'arm':'open_cake','nodes':nodes},root,self.reference,evaluate._owned_file)

    def test_stage_labels_are_data_and_cannot_become_output_paths(self):
        document = self.program.document;document['stages'][0]['name']='../outside'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            self.compile(root,Program.from_dict(document),'candidate')
            self.assertFalse((root/'outside').exists())
            self.assertEqual({path.name for path in (root/'candidate').iterdir()},
                             {'program.json','launch_manifest.bin','program_bundle.bin','static-profile.json'})

    def test_fusion_does_not_invent_reference_intermediate_observations(self):
        result = evaluate._qsa_failure_diagnostics(None,{}, {})
        self.assertEqual(result['first_divergence'],'unknown')
        self.assertEqual(result['boundaries'],{})
        self.assertEqual(set(result['unobserved_boundaries']),{'pool','layernorm','score_topk','expand'})
