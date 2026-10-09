"""Existing sealed-baseline and original Bench callback contracts, CPU only."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from benchmarks.c550 import native_bridge as bridge
import qualify_c550_bench_native as command
from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab import OpenCakeEnvironment,CandidateSubmission,TritonToolchainBuilder
from open_cake_ir.lab.provider_documents import PYTHON_CANDIDATE_BUNDLE_V1
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from tests.contracts.test_portable_program_evaluation import NativeCompiler

POLICY={'float32_matmul_precision':'highest','allow_tf32':False,
        'allow_fp16_reduced_precision_reduction':True,'allow_bf16_reduced_precision_reduction':True,
        'initialization':{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}}


def source(program=False):
    def stage(name,read,write):
        return f'''@cake.schedule(name="{name}", target="xcore1002", backend="triton", entry_point="{name}")
def {name}(lm, {read}: cake.Tensor((2,4), "fp32"), {write}: cake.Tensor((2,4), "fp32", mode="output")):
    row=lm.program({read},axis=0,dimension=0,tile=1)
    compute=lm.role(execution_groups=[0])
    with compute:
        value=lm.load({read}[row,:])
        result=value+1.0
        lm.store({write}[row,:],result)
'''
    text='from open_cake_ir.compiler import frontend as cake\n'+stage('first','x','tmp' if program else 'out')
    if program:
        text+=stage('second','tmp','out')+'\ncake.program(program_id="bridge_program",inputs=("x",),outputs=("out",),stages=(cake.stage(name="first",schedule=first,bindings={"x":"x","tmp":"tmp"}),cake.stage(name="second",schedule=second,bindings={"tmp":"tmp","out":"out"})))\n'
    return text


class Problem:
    task_id='L1/bridge_fixture'
    workloads=tuple(SimpleNamespace(uuid=f'case-{i:02d}') for i in range(16))
    def workload_document(self,uuid,*,input_views=None,oracle_numerics=POLICY):
        return {'schema_version':1,'workload_id':uuid,'revision':'1','state':'frozen','operator':'bridge_fixture',
            'provenance':[],'cases':[{'case_id':'primary','shape':{'R':2,'C':4},'seed':0,'mode':'fixture'}],
            'tensors':{n:{'shape':['R','C'],'dtype':'fp32','layout':'contiguous_row_major'} for n in ('x','out')},
            'semantics':{'target':'xcore1002','candidate_abi':{'inputs':['x'],'outputs':['out']},
                'ordered_original_inputs':['x','alpha'],'ordered_original_outputs':['out'],
                'original_tensor_shapes':{'x':[2,4],'out':[2,4]},'input_views':{'x':[0,1]} if input_views is None else input_views,
                'fixed_scalar_inputs':{'alpha':{'dtype':'float32','value':2.,'binding':'literal_input'}},
                'oracle_numerics':deepcopy(oracle_numerics)},
            'oracle':{'kind':'CPU fixture'},'validation':{'primary_case':'primary','all_cases_required':True,
                'comparison':'elementwise_atol_rtol','atol':0.,'rtol':0.}}


def prepare(root,compiler,program):
    problem=Problem();index={'bench_commit':BENCH_COMMIT,'task':problem.task_id,'cases':[]}
    text=source(program)
    for item in problem.workloads:
        folder=root/item.uuid;folder.mkdir();(folder/'baseline').mkdir()
        document=problem.workload_document(item.uuid);workload=BenchWorkload(document)
        envelope={'python_program_source':text,'program_id':'bridge_program'} if program else {'python_source':text}
        submission=CandidateSubmission.seal(OpenCakeEnvironment.media_type,canonical_json_bytes(envelope))
        builder=TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=NativeCompiler())
        environment=OpenCakeEnvironment(compiler,builder,workload=workload,case_id='primary',authority_document={
            'input_format':'python_source_v1','provider':{'submission_contract':PYTHON_CANDIDATE_BUNDLE_V1},
            'lowering_route':{'backend':'triton','entry_point':'first'}})
        result=environment.build(submission)
        if result.launchable is None:raise AssertionError(result.feedback)
        candidate=result.launchable;paths={}
        for role,payload in candidate.artifact_payloads.items():
            filename=role+'.bin';(folder/'baseline'/filename).write_bytes(payload);paths[role]=filename
        identity=candidate_identity(candidate)
        (folder/'baseline/candidate.json').write_bytes(canonical_json_bytes({'candidate':identity,'artifact_paths':paths}))
        selection={'schema_version':1,'policy':'starter_reference','source':'starter_reference',
                   'incumbent_key':None,'promotion_run_id':None,'registry_root':None}
        (folder/'prepared-baseline.json').write_bytes(canonical_json_bytes({'schema_version':1,
            'fixed_baseline_bundle_path':str(folder/'baseline/candidate.json'),'fixed_baseline_candidate':identity,
            'fixed_baseline_selection':selection}))
        (folder/'workload.json').write_bytes(canonical_json_bytes(document));(folder/'starter.py').write_text(text)
        (folder/'compiler-gate.json').write_bytes(canonical_json_bytes({'passed':True,'compiler_revision_id':compiler._revision.revision_id}))
        index['cases'].append({'uuid':item.uuid,'prepared_baseline':str(folder/'prepared-baseline.json')})
    path=root/'locators.json';path.write_bytes(canonical_json_bytes(index));return problem,path


class Bytes:
    def __init__(self,value):self.value=value
    def payload(self):
        return self.value if isinstance(self.value,bytes) else struct.pack('<'+'f'*len(self.value.data),*self.value.data)
    def cpu(self):return self
    def clone(self):return Bytes(self.payload())
    def tolist(self):raise AssertionError('bridge must not convert tensors to lists')


class Tensor:
    device='cuda:0'
    def __init__(self,shape,dtype,data,address,strides=None):
        self.shape=tuple(shape);self.dtype=dtype;self.data=data;self.address=address
        step=1;dense=[]
        for n in reversed(shape):dense.insert(0,step);step*=n
        self.strides=tuple(dense if strides is None else strides)
    @property
    def ndim(self):return len(self.shape)
    def numel(self):return math.prod(self.shape)
    def element_size(self):return 4
    def data_ptr(self):return self.address
    def stride(self):return self.strides
    def storage_offset(self):return 0
    def untyped_storage(self):return SimpleNamespace(nbytes=lambda:len(self.data)*4)
    def is_contiguous(self):
        step=1
        for n,stride in zip(reversed(self.shape),reversed(self.strides)):
            if n!=1 and stride!=step:return False
            step*=n
        return True
    def permute(self,order):return Tensor([self.shape[i] for i in order],self.dtype,self.data,self.address,[self.strides[i] for i in order])
    def view(self,value):
        if value=='uint8':return Bytes(self)
        return Tensor(value,self.dtype,self.data,self.address)
    def tolist(self):raise AssertionError('bridge must not convert tensors to lists')


class Torch:
    float32='fp32';bfloat16='bf16';float16='fp16';int32='int32';int64='int64';bool='bool';uint8='uint8'
    cuda=SimpleNamespace(current_device=lambda:0,current_stream=lambda:SimpleNamespace(cuda_stream=7),synchronize=lambda:None)
    def __init__(self):self.position=4096;self.allocations=[]
    def full(self,shape,value,*,dtype,device):
        result=Tensor(shape,dtype,[value]*math.prod(shape),self.position)
        self.position+=result.numel()*4+1024;self.allocations.append(result);return result
    @staticmethod
    def equal(a,b):return a.payload()==b.payload()


class Kernels:
    def __init__(self,mode='normal'):self.mode=mode;self.children=[]
    def load(self,candidate,manifest,admission):
        if self.mode=='partial_load' and self.children:raise ValueError('second load failed')
        owner=self
        class Kernel:
            launch_calls=0;closed=False;resources={}
            def launch(self,args,*,tensor_contract,stream):
                if owner.mode=='omit':return
                self.launch_calls+=1
                if owner.mode=='launch_failure':raise ValueError('native launch failed')
                args[-1].data[:]=[v+1 for v in args[0].data]
                if owner.mode=='mutate':args[0].data[0]+=1
                if owner.mode=='wrong_output':args[-1].shape=(8,)
            def close(self,*,synchronize):
                synchronize();self.closed=True
                if owner.mode=='close_failure':raise ValueError('native close failed')
        result=Kernel();self.children.append(result);return result


class NativeBridgeContracts(unittest.TestCase):
    def setUp(self):
        observer=patch('open_cake_ir.tasks.c550_bench.binding.observe_oracle_numerics',return_value=deepcopy(POLICY))
        self.observer=observer.start();self.addCleanup(observer.stop)
    @classmethod
    def setUpClass(cls):
        import open_cake_ir.compiler
        assert Path(open_cake_ir.compiler.__file__).resolve().is_relative_to((ROOT/'src').resolve())
        cls.compiler=Compiler.load(ROOT)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.fixtures={}
        for program in (False,True):
            folder=cls.root/str(program);folder.mkdir();problem,path=prepare(folder,cls.compiler,program)
            cls.fixtures[program]=(problem,path,bridge.bind_all(problem,path,compiler=cls.compiler))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_both_existing_baseline_formats_bind_all_original_duplicate_shapes(self):
        for program in (False,True):
            _,_,cases=self.fixtures[program]
            self.assertEqual(len(cases),16)
            self.assertTrue(all(case.candidate.is_program==program for case in cases))
            self.assertEqual({case.manifest.kernels_per_call for case in cases},{2 if program else 1})

    def test_all_160_callbacks_keep_order_original_tensors_and_exact_stage_calls(self):
        for program in (False,True):
            _,_,cases=self.fixtures[program];torch=Torch();kernels=Kernels();trace=[]
            adapter=bridge.NativeBench(cases,10,object(),trace.append,torch_module=torch,loader=kernels.load)
            with patch.dict(sys.modules,torch=torch):
                for _ in range(160):
                    value=torch.full((2,4),1.,dtype='fp32',device='cuda:0')
                    result=adapter.run(value,2.)
                    self.assertEqual(result.data,[3. if program else 2.]*8)
                    self.assertEqual(value.data,[1.]*8)
            self.assertEqual([(r['uuid'],r['round']) for r in trace],[(c.uuid,j) for c in cases for j in range(10)])
            self.assertTrue(all(r['kernel_calls']==(2 if program else 1) and r['input_unchanged'] and r['module_closed'] and r['error'] is None for r in trace))
            with self.assertRaisesRegex(ValueError,'extra callback'):adapter.run(value,2.)

    def test_wrong_scalar_shape_dtype_or_argument_count_never_loads_a_module(self):
        cases=self.fixtures[False][2]
        for mutation in ('scalar','shape','dtype','count'):
            torch=Torch();kernels=Kernels();trace=[]
            value=torch.full((2,4),1.,dtype='fp32',device='cuda:0');args=[value,2.]
            if mutation=='scalar':args[1]=3.
            elif mutation=='shape':value.shape=(1,8)
            elif mutation=='dtype':value.dtype='bf16'
            else:args.pop()
            adapter=bridge.NativeBench(cases,10,object(),trace.append,torch_module=torch,loader=kernels.load)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):adapter.run(*args)
            self.assertFalse(kernels.children);self.assertEqual(len(torch.allocations),1)
            self.assertIsNotNone(trace[0]['error'])

    def test_numeric_drift_refuses_before_candidate_storage_or_module_allocation(self):
        torch=Torch();kernels=Kernels();trace=[]
        value=torch.full((2,4),1.,dtype='fp32',device='cuda:0')
        self.observer.return_value={**POLICY,'float32_matmul_precision':'high'}
        adapter=bridge.NativeBench(self.fixtures[False][2],10,object(),trace.append,torch_module=torch,loader=kernels.load)
        with self.assertRaisesRegex(ValueError,'numeric.*callback'):
            adapter.run(value,2.)
        self.assertEqual(len(torch.allocations),1);self.assertFalse(kernels.children)
        self.assertIsNotNone(trace[0]['error'])

    def test_qualification_delegates_original_rng_loop_and_binds_its_report(self):
        import importlib.util
        cases=self.fixtures[False][2]
        for mode in ('normal','wrong_order','failed_verdict','numeric_drift'):
            torch=Torch();kernels=Kernels();calls=[]
            def check_problem(task,**kwargs):
                self.assertEqual(kwargs['workload_scope'],'all');self.assertEqual(kwargs['rounds'],10)
                self.assertEqual(kwargs['seed'],401);self.assertFalse(kwargs['reference_selfcheck'])
                spec=importlib.util.spec_from_file_location('original_bench_candidate',kwargs['candidate_path'])
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                records=[]
                for case in cases:
                    for round_index in range(10):
                        # Only this fake original owner advances its input sequence.
                        value=torch.full((2,4),float(len(calls)),dtype='fp32',device='cuda:0')
                        out=module.run(value,2.)
                        self.assertEqual(out.data,[float(len(calls)+1)]*8)
                        calls.append((case.uuid,round_index))
                        records.append({'workload_uuid':case.uuid,'round':round_index,'passed':True})
                if mode=='wrong_order':records[0],records[1]=records[1],records[0]
                if mode=='failed_verdict':records[-1]['passed']=False
                if mode=='numeric_drift':self.observer.return_value={**POLICY,'float32_matmul_precision':'high'}
                return {'status':'passed','full_device_correctness':True,'cases':records}
            problem=SimpleNamespace(task_id='L1/bridge_fixture',api=SimpleNamespace(
                document=lambda name:{'correctness_rounds':10,'seed':401},
                task_record=lambda task:{'id':task},check_problem=check_problem))
            native_class=bridge.NativeBench
            def construct(*args):return native_class(*args,torch_module=torch,loader=kernels.load)
            self.observer.reset_mock();self.observer.return_value=deepcopy(POLICY)
            with tempfile.TemporaryDirectory() as temporary,patch.dict(sys.modules,torch=torch), \
                 patch('open_cake_ir.lab.executor.ExecutorRevision.for_target',return_value=SimpleNamespace(admit_host=lambda:{'runtime_library':'CPU fixture'})), \
                 patch('open_cake_ir.evaluation.local_broker.admit_local_job',return_value='maca-123456789abc') as allocate, \
                 patch('open_cake_ir.evaluation.triton_metax.observe_local_metax',return_value=object()), \
                 patch.object(bridge,'NativeBench',side_effect=construct):
                if mode=='numeric_drift':
                    with self.assertRaisesRegex(ValueError,'after original check_problem'):
                        command.evaluate_original(problem,cases,Path(temporary),physical_device=7,runtime_device=0,expected_pci='0000:34:00')
                else:
                    result=command.evaluate_original(problem,cases,Path(temporary),physical_device=7,runtime_device=0,expected_pci='0000:34:00')
                    self.assertEqual(result['passed'],mode=='normal')
                    self.assertEqual(result['native_kernel_calls'],160)
                allocate.assert_called_once_with('maca',device=7,runtime_device=0,expected_pci='0000:34:00',lock_scope='device',queue_seconds=300)
            self.assertIsNone(bridge._ACTIVE)
            self.assertEqual(len(calls),160)
            self.assertEqual(self.observer.call_count,163)
            self.assertTrue(all(kernel.closed for kernel in kernels.children))

    def test_original_noncontiguous_input_uses_the_checked_view_without_copy(self):
        case=self.fixtures[False][2][0];document=case.workload.document
        document['semantics']['input_views']['x']=[1,0]
        document['semantics']['original_tensor_shapes']['x']=[4,2]
        changed=replace(case,workload=BenchWorkload(document));cases=(changed,)+self.fixtures[False][2][1:]
        torch=Torch();kernels=Kernels();trace=[]
        value=Tensor((4,2),'fp32',[1.]*8,64,(1,4))
        result=bridge.NativeBench(cases,10,object(),trace.append,torch_module=torch,loader=kernels.load).run(value,2.)
        self.assertEqual(result.data,[2.]*8);self.assertEqual(value.data,[1.]*8)
        self.assertTrue(trace[0]['input_unchanged'])

    def test_dispatch_output_input_and_cleanup_failures_never_report_native_success(self):
        for program in (False,True):
            for mode in ('omit','mutate','wrong_output','launch_failure','close_failure'):
                cases=self.fixtures[program][2];torch=Torch();kernels=Kernels(mode);trace=[]
                value=torch.full((2,4),1.,dtype='fp32',device='cuda:0')
                adapter=bridge.NativeBench(cases,10,object(),trace.append,torch_module=torch,loader=kernels.load)
                with self.subTest(program=program,mode=mode),patch.dict(sys.modules,torch=torch),self.assertRaises(Exception):
                    adapter.run(value,2.)
                self.assertTrue(all(child.closed for child in kernels.children))
                self.assertIsNotNone(trace[0]['error'])
        torch=Torch();kernels=Kernels('partial_load');trace=[]
        value=torch.full((2,4),1.,dtype='fp32',device='cuda:0')
        with patch.dict(sys.modules,torch=torch),self.assertRaisesRegex(ValueError,'second load'):
            bridge.NativeBench(self.fixtures[True][2],10,object(),trace.append,torch_module=torch,loader=kernels.load).run(value,2.)
        self.assertTrue(all(child.closed for child in kernels.children))

    def test_missing_case_changed_source_gate_view_and_payload_refuse_during_cpu_binding(self):
        problem,path,_=self.fixtures[False];index=json.loads(path.read_text())
        for mutation in ('missing','source','gate','view','payload','escape','old_numerics','manifest'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                import shutil
                folder=Path(temporary).resolve()/ 'copy';shutil.copytree(path.parent,folder)
                copied=deepcopy(index)
                for row in copied['cases']:
                    case=folder/row['uuid'];row['prepared_baseline']=str(case/'prepared-baseline.json')
                    handoff=json.loads((case/'prepared-baseline.json').read_text())
                    handoff['fixed_baseline_bundle_path']=str(case/'baseline/candidate.json')
                    (case/'prepared-baseline.json').write_bytes(canonical_json_bytes(handoff))
                last=folder/copied['cases'][-1]['uuid']
                if mutation=='missing':copied['cases'].pop()
                elif mutation=='source':(last/'starter.py').write_text(source()+'\n# changed\n')
                elif mutation=='gate':(last/'compiler-gate.json').write_text(json.dumps({'passed':True,'compiler_revision_id':'open-cake-ir@'+'9'*40}))
                elif mutation=='view':
                    document=json.loads((last/'workload.json').read_text());document['semantics']['input_views']['x']=[1,0]
                    (last/'workload.json').write_bytes(canonical_json_bytes(document))
                elif mutation=='payload':
                    p=last/'baseline/mcfatbin.bin';p.write_bytes(p.read_bytes()+b'changed')
                elif mutation=='old_numerics':
                    document=json.loads((last/'workload.json').read_text());document['semantics'].pop('oracle_numerics')
                    (last/'workload.json').write_bytes(canonical_json_bytes(document))
                elif mutation=='manifest':
                    from hashlib import sha256
                    from open_cake_ir.evaluation.core import LaunchableCandidate
                    from open_cake_ir.lab.bindings import load_baseline_bundle
                    candidate=load_baseline_bundle(ROOT,last/'baseline/candidate.json')
                    payloads=dict(candidate.artifact_payloads)
                    manifest=json.loads(payloads['launch_manifest']);manifest['tensor_abi'][-1]['shape']=[1,8]
                    payloads['launch_manifest']=canonical_json_bytes(manifest)
                    changed=LaunchableCandidate(candidate.candidate_sha256,candidate.target,candidate.entry_point,
                        {role:sha256(value).hexdigest() for role,value in payloads.items()},sha256(payloads['launch_manifest']).hexdigest(),payloads)
                    bundle=json.loads((last/'baseline/candidate.json').read_text());bundle['candidate']=candidate_identity(changed)
                    (last/'baseline/candidate.json').write_bytes(canonical_json_bytes(bundle))
                    (last/'baseline/launch_manifest.bin').write_bytes(payloads['launch_manifest'])
                    handoff=json.loads((last/'prepared-baseline.json').read_text());handoff['fixed_baseline_candidate']=candidate_identity(changed)
                    (last/'prepared-baseline.json').write_bytes(canonical_json_bytes(handoff))
                else:
                    p=last/'baseline/mcfatbin.bin';p.unlink();p.symlink_to(path.parent/index['cases'][0]['uuid']/'baseline/mcfatbin.bin')
                (folder/'locators.json').write_bytes(canonical_json_bytes(copied))
                with self.assertRaises(ValueError):bridge.bind_all(problem,folder/'locators.json',compiler=self.compiler)


if __name__=='__main__':unittest.main()
