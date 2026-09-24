"""Rank ownership and CPU mathematics for the EP4 development Workload."""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.build import build_program_candidate
from open_cake_ir.tasks.weave_ep.workload import (
    RankVolume, materialize_tensors, reference_tensors, reference_values,
    routed_volume_values, routed_volumes, validate_contract, workload_document,
)
from open_cake_ir.tasks.workloads import load_workload, validate_workload_document


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'


class WeaveEP4Workload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document = json.loads(CONTRACT.read_text())
        cls.workload = WorkloadContract.from_document(
            document, CONTRACT, validate=validate_contract)

    def test_exact_contract_and_single_device_refusal(self):
        workload = self.workload
        self.assertEqual(load_workload(CONTRACT).document, workload.document)
        validate_workload_document(workload.document)
        self.assertEqual(workload.document, workload_document())
        self.assertTrue(workload.requires_distributed_execution)
        self.assertEqual(workload.case_ids,
                         ('balanced', 'skew_to_rank0', 'local_only',
                          'remote_only', 'tail_tokens'))
        self.assertEqual([row.shape for row in workload.tensor_abi('tail_tokens')][:2],
                         [(4, 7, 16), (4, 7, 2)])
        launch = {'target': 'sm_103a', 'kernel_name': 'incorrect_single_device',
                  'grid': [1, 1, 1], 'block': [32, 1, 1],
                  'dynamic_shared_memory_bytes': 0,
                  'hidden_null_pointer_parameters': 0}
        with self.assertRaisesRegex(ValueError, 'single-device launch manifest'):
            TensorLaunchManifest.for_workload(workload, 'balanced', **launch)
        with self.assertRaisesRegex(ValueError, 'distributed candidate builder'):
            build_program_candidate(None, None, candidate_sha256='0' * 64,
                                    workload=workload, case_id='balanced')
        changed = workload.document
        changed['semantics']['execution_topology']['world_size'] = 1
        with self.assertRaisesRegex(ValueError, 'execution topology differs'):
            WorkloadContract(changed)
        changed = workload.document
        changed['semantics']['expert_placement'] = 'replicated'
        with self.assertRaisesRegex(ValueError, 'semantic or placement contract differs'):
            validate_contract(changed)

    def test_independent_small_route_formula(self):
        shape = self.workload.case('balanced')['shape']
        ranks, tokens, experts, hidden, intermediate = (
            shape[name] for name in ('R', 'T', 'E', 'H', 'I'))
        activation = [[[0.0] * hidden for _ in range(tokens)] for _ in range(ranks)]
        activation[0][0][0] = 1.0
        ids = [[[0, 1] for _ in range(tokens)] for _ in range(ranks)]
        weights = [[[1.0, 0.0] for _ in range(tokens)] for _ in range(ranks)]
        up = [[[0.0] * hidden for _ in range(2 * intermediate)]
              for _ in range(experts)]
        down = [[[0.0] * intermediate for _ in range(hidden)]
                for _ in range(experts)]
        up[0][0][0] = up[0][intermediate][0] = down[0][0][0] = 1.0
        result = reference_values(shape, activation, ids, weights, up, down)
        self.assertAlmostEqual(result[0][0][0], 1 / (1 + math.exp(-1)))
        self.assertTrue(all(result[rank][token][feature] == 0.0
                            for rank in range(ranks) for token in range(tokens)
                            for feature in range(hidden)
                            if (rank, token, feature) != (0, 0, 0)))
        self.assertEqual(result[0][1], [0.0] * hidden)

    def test_routing_volume_counts_deduplicated_remote_tokens(self):
        shape = self.workload.case('skew_to_rank0')['shape']
        ids = [[[0, 1] for _ in range(shape['T'])] for _ in range(shape['R'])]
        volumes = routed_volume_values(shape, ids)
        self.assertEqual(volumes[0], RankVolume(16, 48, 24, 0))
        self.assertEqual(volumes[1], RankVolume(0, 0, 0, 8))
        local = [[[rank * 2, rank * 2 + 1] for _ in range(shape['T'])]
                 for rank in range(shape['R'])]
        self.assertEqual(routed_volume_values(shape, local),
                         (RankVolume(16, 0, 0, 0),) * 4)
        bad = [[[0, 0] for _ in range(shape['T'])] for _ in range(shape['R'])]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            routed_volume_values(shape, bad)

    def test_torch_materialization_and_oracle_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional Torch materializer is unavailable')
        for case_id in self.workload.case_ids:
            with self.subTest(case_id=case_id):
                tensors = materialize_tensors(self.workload, case_id)
                output = reference_tensors(self.workload, case_id, tensors)['output']
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(tuple(output.shape),
                                 tuple(self.workload.tensor_abi(case_id)[-1].shape))
                self.assertTrue(bool(torch.isfinite(output.float()).all()))
                volumes = routed_volumes(self.workload, case_id,
                                         tensors['expert_ids'])
                shape = self.workload.case(case_id)['shape']
                self.assertEqual(sum(row.local_routes + row.remote_in_routes
                                     for row in volumes),
                                 shape['R'] * shape['T'] * shape['K'])
                self.assertTrue(all(row.unique_remote_in_tokens <= row.remote_in_routes
                                    for row in volumes))
                if case_id == 'local_only':
                    self.assertTrue(all(row.remote_in_routes == 0 for row in volumes))
                if case_id == 'skew_to_rank0':
                    self.assertEqual(volumes[0].remote_in_routes, 48)
        invalid = materialize_tensors(self.workload, 'balanced')
        invalid['expert_ids'][0, 0, 1] = invalid['expert_ids'][0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            reference_tensors(self.workload, 'balanced', invalid)


if __name__ == '__main__':
    unittest.main()
