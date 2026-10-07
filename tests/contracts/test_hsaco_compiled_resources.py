"""HSACO allocation reaches the existing portable profile and feedback consumer."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, CompiledResources
from open_cake_ir.compiler.toolchain import TritonCompilation, inspect_amdgcn_resources
from open_cake_ir.compiler.performance.compiled_resources import load_compiled_resources

ROOT = Path(__file__).resolve().parents[2]


class HsacoCompiledResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.schedule = ROOT / 'corpus/schedules/gfx938-rmsnorm-b8-smoke.json'
        cls.assessment = cls.compiler.assess_file(cls.schedule)
        cls.lowered = cls.compiler.lower(cls.assessment)
        cls.entry = cls.lowered.toolchain_requirements['kernel_entry_point']
        cls.binary = b'\x7fELFfixture-hsaco'
        cls.asm = (f'.amdgpu_metadata\n---\namdhsa.kernels:\n  - .name: {cls.entry}\n'
                   '    .vgpr_count: 48\n    .group_segment_fixed_size: 512\n'
                   '    .private_segment_fixed_size: 132\n...\n.end_amdgpu_metadata\n').encode()
        cls.compilation = TritonCompilation(cls.lowered.source.encode(), 'gfx938', cls.entry,
            {'hsaco':cls.binary, 'amdgcn':cls.asm},
            cls.lowered.toolchain_requirements['compile_options']['num_warps'] * 64,
            512, 'fixture', 'hsaco')

    def test_static_and_launch_lds_remain_distinct_and_stack_stays_unknown(self):
        resource = inspect_amdgcn_resources(self.compilation)
        self.assertEqual(resource.shared_bytes, 1024)
        self.assertEqual(resource.dynamic_shared_bytes, 512)
        self.assertEqual(resource.local_bytes, 132)
        self.assertIsNone(resource.stack_bytes)
        self.assertEqual(resource.registers_per_thread, 48)
        self.assertEqual(resource.binary_sha256, sha256(self.binary).hexdigest())
        self.assertEqual(resource.as_dict()['code_object'], 'hsaco')
        self.assertNotIn('cubin_sha256', resource.as_dict())
        with self.assertRaisesRegex(ValueError, 'no CUBIN'):
            _ = resource.cubin_sha256
        self.assertEqual(CompiledResources.from_dict(resource.as_dict()), resource)

    def test_zero_static_lds_retains_the_actual_dynamic_launch_request(self):
        compilation = replace(self.compilation, artifacts={**self.compilation.artifacts,
            'amdgcn':self.asm.replace(b'.group_segment_fixed_size: 512', b'.group_segment_fixed_size: 0')},
            dynamic_shared_bytes=8192)
        resource = inspect_amdgcn_resources(compilation)
        self.assertEqual(resource.static_shared_bytes, 0)
        self.assertEqual(resource.shared_bytes, 8192)
        with self.assertRaises(ValueError):
            replace(resource, stack_bytes=0)

    def test_native_facts_do_not_create_ncu_metrics_or_latency_claims(self):
        resource = inspect_amdgcn_resources(self.compilation)
        report = self.compiler.profile(self.assessment, compiled_resources=resource).as_dict()
        self.assertEqual(report['compiled_resources'], resource.as_dict())
        self.assertEqual(report['ncu_metrics'], [])
        shared = next(r for r in report['residency']['bounds'] if r['resource']=='shared_memory')
        self.assertEqual(shared['per_cta'], 1024)
        self.assertTrue(any('not dynamic spill traffic' in x for x in report['abstentions']))
        for changes in ({'source_sha256':'0'*64}, {'entry_point':'other'}, {'threads_per_cta':1024}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compiler.profile(self.assessment, compiled_resources=replace(resource, **changes))

    def test_exported_cuda_constructor_keeps_its_legacy_keyword(self):
        args = dict(source_sha256=self.lowered.source_sha256,
                    target='sm_100a',entry_point='kernel',threads_per_cta=128,
                    registers_per_thread=32,static_shared_bytes=0,dynamic_shared_bytes=0,
                    local_bytes=0,stack_bytes=0,compiler_version='fixture',inspector_version='fixture')
        identity=sha256(self.binary).hexdigest()
        old=CompiledResources(cubin_sha256=identity,**args)
        new=CompiledResources(binary_sha256=identity,**args)
        self.assertEqual(old,new)
        self.assertEqual(old.as_dict()['schema_version'],1)
        self.assertEqual(CompiledResources.from_dict(old.as_dict()),old)
        with self.assertRaises(ValueError):
            CompiledResources(cubin_sha256=identity,binary_sha256=identity,**args)

    def test_cubin_record_cannot_describe_an_hsaco_target(self):
        resource = inspect_amdgcn_resources(self.compilation)
        with self.assertRaisesRegex(ValueError, 'code object differs'):
            self.compiler.profile(self.assessment, compiled_resources=replace(
                resource, code_object='cubin', stack_bytes=0))

    def write_report(self, folder):
        resource = inspect_amdgcn_resources(self.compilation)
        artifacts = folder/'0000';artifacts.mkdir()
        (artifacts/'kernel.hsaco').write_bytes(self.binary)
        (artifacts/'lowered.py').write_text(self.lowered.source)
        report = folder/'report.json'
        report.write_text(json.dumps({'schema_version':1,'rows':[{'profile':{
            'compiled_resources':resource.as_dict()}}]}))
        return report, resource

    def test_existing_report_and_author_feedback_replay_without_gpu_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, resource = self.write_report(Path(tmp))
            self.assertEqual(load_compiled_resources(report)[self.lowered.source_sha256], resource)
            for command in ([sys.executable,'tools/report_schedule_profile.py','--json'],
                            [sys.executable,'src/open_cake_ir/tasks/qsa/project_feedback.py','compiler']):
                with self.subTest(command=command):
                    result = subprocess.run(command+['--revision',str(ROOT/'compiler/revision.json'),'--compiled-report',str(report),str(self.schedule)],
                        cwd=ROOT,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    data=json.loads(result.stdout)
                    profile=data['rows'][0]['profile'] if 'rows' in data else data['static_profile']
                    self.assertEqual(profile['compiled_resources'],resource.as_dict())
                    self.assertEqual(profile['ncu_metrics'],[])
            (Path(tmp)/'0000/kernel.hsaco').write_bytes(b'\x7fELFchanged')
            with self.assertRaisesRegex(ValueError,'binary differs'):
                load_compiled_resources(report)


if __name__=='__main__':
    unittest.main()
