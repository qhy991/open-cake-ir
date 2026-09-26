"""Exact model-scale EP4 semantic and independent-oracle boundaries."""
from __future__ import annotations

import json
from math import exp
from pathlib import Path
import unittest

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.weave_ep import model_workload
from open_cake_ir.tasks.workloads import load_workload,validate_workload_document


ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'contracts/workloads/weave-model-ep4-bf16-moe-b300-v1.json'


class WeaveModelWorkloadTest(unittest.TestCase):
    def test_exact_model_contract_and_rank_abi(self):
        workload=load_workload(CONTRACT)
        self.assertEqual(workload.document,model_workload.workload_document())
        self.assertTrue(workload.requires_distributed_execution)
        self.assertEqual(workload.case_ids,tuple(name for name,_ in
                                                  model_workload.CASES))
        self.assertEqual([row.shape for row in
                          workload.tensor_abi('mixed_full_early_terminal')],
                         [(4,512,2048),(4,512,8),(4,512,8),
                          (128,1536,2048),(128,2048,768),(4,512,2048)])
        changed=workload.document
        changed['semantics']['top_k']=7
        with self.assertRaisesRegex(ValueError,'semantic or placement'):
            validate_workload_document(changed)
        changed=workload.document
        changed['cases'][1]['shape']['E']=64
        with self.assertRaisesRegex(ValueError,'semantic or placement'):
            model_workload.validate_contract(changed)

    def test_small_fp64_formula_and_route_override_patterns(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest('NumPy is needed for the independent CPU oracle')
        shape={'R':2,'T':1,'K':1,'E':2,'H':2,'I':1}
        hidden=np.zeros((2,1,2),dtype=np.float32)
        hidden[0,0,0]=1
        ids=np.array([[[0]],[[1]]],dtype=np.int32)
        weights=np.ones((2,1,1),dtype=np.float32)
        upgate=np.zeros((2,2,2),dtype=np.float32)
        upgate[0,0,0]=upgate[0,1,0]=1
        down=np.zeros((2,2,1),dtype=np.float32)
        down[0,0,0]=1
        actual=model_workload.reference_values(
            shape,hidden,ids,weights,upgate,down)
        self.assertAlmostEqual(float(actual[0,0,0]),
                               1/(1+exp(-1)),delta=0.005)
        self.assertTrue(np.all(actual[1]==0))
        initial=np.zeros((4,512,8),dtype=np.int32)
        hot=model_workload._route_ids('override_hot8_ids',initial)
        mixed=model_workload._route_ids('override_mixed_ids',initial)
        dense=model_workload._route_ids('override_dense_ids',initial)
        self.assertEqual(hot[3,511].tolist(),list(range(8)))
        self.assertEqual(mixed[0,128].tolist(),list(range(8,16)))
        self.assertEqual(dense[0,0].tolist(),[0,1,16,17,18,19,20,21])


if __name__=='__main__':
    unittest.main()
