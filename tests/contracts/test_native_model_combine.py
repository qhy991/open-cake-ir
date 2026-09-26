"""The model top-8 combine remains one explicit Cake reduction and BF16 cast."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_model_combine import preflight


ROOT=Path(__file__).resolve().parents[2]
DOCUMENT=ROOT/'examples/schedules/native/weave-model-weighted-combine-b300.json'
RANK512=ROOT/'examples/schedules/native/weave-model-weighted-combine-rank512-b300.json'


class NativeModelCombine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT)
        cls.document=json.loads(DOCUMENT.read_text())

    def test_complete_top8_h2048_combine_lowers_with_one_bf16_round(self):
        assessment=self.compiler.assess(self.document)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code,f.path) for f in assessment.findings
                         if f.blocks_lowering])
        lowered=self.compiler.lower(assessment)
        req=lowered.toolchain_requirements
        self.assertEqual(req['target'],'sm_103a')
        self.assertEqual(req['grid'],[2048,8,1])
        self.assertEqual(req['block'],[256,1,1])
        self.assertIn('--fmad=false',req['nvcc_flags'])
        self.assertIn('for (int route=0; route<8; ++route)',lowered.source)
        self.assertIn('__float2bfloat16_rn(cake_sum)',lowered.source)
        self.assertEqual(lowered.source.count('cake_sum += cake_product'),1)
        for operation in self.document['operations']:
            self.assertIn(operation['id'],lowered.source_map)

    def test_wrong_reduction_or_feature_mapping_is_refused(self):
        target=Target.load(ROOT/'compiler/targets/sm_103a.json')
        changed=deepcopy(self.document)
        next(op for op in changed['operations']
             if op['kind']=='reduce')['parameters']['axis']=1
        self.assertIn('NATIVE_MODEL_COMBINE_OPERATIONS',
                      [f.code for f in preflight(Schedule.from_dict(changed),target)])
        changed=deepcopy(self.document)
        next(access for access in changed['access_maps']
             if access['buffer']=='output')['indices'][-1]={
                 'source':'dimension','dimension':1}
        self.assertIn('NATIVE_MODEL_COMBINE_ACCESS',
                      [f.code for f in preflight(Schedule.from_dict(changed),target)])

    def test_rank_local_512_tokens_admits_the_same_explicit_reduction(self):
        short=json.loads(RANK512.read_text())
        assessment=self.compiler.assess(short)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code,f.path) for f in assessment.findings
                         if f.blocks_lowering])
        if self.compiler.commit is not None:
            lowered=self.compiler.lower(assessment)
            self.assertEqual(lowered.toolchain_requirements['grid'],[512,8,1])
            self.assertIn('cake_token < 512',lowered.source)
            self.assertIn('__float2bfloat16_rn(cake_sum)',lowered.source)
        wrong=deepcopy(short)
        for buffer in wrong['buffers']:
            if buffer['space']=='global' and buffer['shape'][0]==512:
                buffer['shape'][0]=513
        target=Target.load(ROOT/'compiler/targets/sm_103a.json')
        self.assertIn('NATIVE_MODEL_COMBINE_SHAPE',
                      [f.code for f in preflight(Schedule.from_dict(wrong),target)])

    def test_other_target_and_fp32_output_are_refused(self):
        schedule=Schedule.from_dict(self.document)
        other=Target.load(ROOT/'compiler/targets/sm_100a.json')
        self.assertIn('NATIVE_MODEL_COMBINE_TARGET',
                      [f.code for f in preflight(schedule,other)])
        changed=deepcopy(self.document)
        next(buffer for buffer in changed['buffers']
             if buffer['name']=='output')['dtype']='fp32'
        self.assertIn('NATIVE_MODEL_COMBINE_GLOBALS',
                      [f.code for f in preflight(Schedule.from_dict(changed),
                                               Target.load(ROOT/'compiler/targets/sm_103a.json'))])


if __name__=='__main__':
    unittest.main()
