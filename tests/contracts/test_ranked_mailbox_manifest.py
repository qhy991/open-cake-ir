"""Frozen EP Workload shards bind one complete ranked launch plan."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.evaluation.ranked_manifest import RankedMailboxLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.weave_ep.workload import validate_contract
from tests.contracts.test_ranked_mailbox_launch import fixture


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'
COMBINE = ROOT / 'examples/schedules/triton/weave-weighted-combine-t7-h16.json'
BINDINGS = {'hidden_states': 'hidden', 'expert_ids': 'expert_ids',
            'route_weights': 'route_weights', 'w_up_gate': 'w_up_gate',
            'w_down': 'w_down'}


def material():
    document = json.loads(CONTRACT.read_text())
    workload = WorkloadContract.from_document(document, CONTRACT,
                                              validate=validate_contract)
    lowered, _, controls = fixture()
    requirements = dict(lowered.toolchain_requirements)
    requirements['compiler_commit'] = '1' * 40
    lowered = replace(lowered, toolchain_requirements=requirements)
    plans = [{'rank': rank, **controls[rank]} for rank in range(4)]
    return workload, lowered, plans


class RankedMailboxManifest(unittest.TestCase):
    def test_exact_tail_workload_math_shards_and_plan_bind(self):
        workload, lowered, plans = material()
        combine_document = json.loads(COMBINE.read_text())
        manifest = RankedMailboxLaunchManifest.from_lowered(
            lowered, combine_document=combine_document, workload=workload,
            case_id='tail_tokens', tensor_bindings=BINDINGS, plans=plans)
        self.assertEqual(manifest.as_dict()['abi'], 'ranked_mailbox_v1')
        self.assertEqual(manifest.as_dict()['case_id'], 'tail_tokens')
        self.assertEqual(manifest.as_dict()['plans'][0]['chunks'], 2)
        self.assertEqual(RankedMailboxLaunchManifest.from_dict(
            manifest.as_dict()), manifest)
        manifest.check_workload(workload, 'tail_tokens')
        manifest.check_lowered(lowered)

    def test_shape_plan_source_and_binding_drift_refuse(self):
        workload, lowered, plans = material()
        combine_document = json.loads(COMBINE.read_text())
        with self.assertRaisesRegex(ValueError, 'public output or input ABI'):
            RankedMailboxLaunchManifest.from_lowered(
                lowered, combine_document=combine_document,
                workload=workload, case_id='skew_to_rank0',
                tensor_bindings=BINDINGS, plans=plans)
        changed = dict(BINDINGS, hidden_states='expert_ids', expert_ids='hidden')
        with self.assertRaisesRegex(ValueError, 'shard ABI'):
            RankedMailboxLaunchManifest.from_lowered(
                lowered, combine_document=combine_document,
                workload=workload, case_id='tail_tokens',
                tensor_bindings=changed, plans=plans)
        changed_combine = deepcopy(combine_document)
        next(op for op in changed_combine['operations']
             if op['id'] == 'weight')['parameters']['op'] = 'add'
        self.assertNotEqual(Schedule.from_dict(changed_combine),
                            lowered.combine_schedule)
        with self.assertRaisesRegex(ValueError, 'combine document differs'):
            RankedMailboxLaunchManifest.from_lowered(
                lowered, combine_document=changed_combine,
                workload=workload, case_id='tail_tokens',
                tensor_bindings=BINDINGS, plans=plans)
        manifest = RankedMailboxLaunchManifest.from_lowered(
            lowered, combine_document=combine_document, workload=workload,
            case_id='tail_tokens', tensor_bindings=BINDINGS, plans=plans)
        changed = manifest.as_dict()
        changed['plans'][0]['chunks'] = 8
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            RankedMailboxLaunchManifest.from_dict(changed)
        changed = manifest.as_dict()
        changed['grid_per_rank'][2] = [147, 1, 1]
        with self.assertRaisesRegex(ValueError, 'grid or runtime gate'):
            RankedMailboxLaunchManifest.from_dict(changed)
        changed = manifest.as_dict()
        changed['compiler_commit'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            RankedMailboxLaunchManifest.from_dict(changed)
        changed = manifest.as_dict()
        changed['schema_version'] = True
        with self.assertRaisesRegex(ValueError, 'fields or ABI'):
            RankedMailboxLaunchManifest.from_dict(changed)
        changed = manifest.as_dict()
        changed['rank_inputs'][0]['shape'][0] = 6
        with self.assertRaisesRegex(ValueError, 'exact Compiler lowering'):
            RankedMailboxLaunchManifest.from_dict(changed).check_lowered(lowered)


if __name__ == '__main__':
    unittest.main()
