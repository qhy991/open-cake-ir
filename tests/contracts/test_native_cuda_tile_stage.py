"""Schedule-driven tensor-core stage bodies used inside a persistent worker."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
    emit_source_event_device, emit_source_event_library,
)
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda_tile_stage import (
    compose_model_ranked_tile_stages, emit_model_activation_stage,
    emit_tensor_tile_stage,
)


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300.json'
EFFECTS = ROOT / 'examples/programs/weave-model-ranked-tile-effects-b300.json'
COMBINE = ROOT / 'examples/schedules/native/weave-model-weighted-combine-rank512-b300.json'


class NativeCudaTileStageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = Program.from_dict(json.loads(PROGRAM.read_text()))
        cls.target = Compiler.load(ROOT)._revision.targets['sm_103a']

    def test_both_model_tensor_stages_use_schedule_operations(self):
        for index, function in ((0, 'cake_upgate_stage_work'),
                                (2, 'cake_down_stage_work')):
            with self.subTest(stage=index):
                stage = self.program.stages[index]
                result = emit_tensor_tile_stage(stage.schedule, self.target,
                                                function_name=function)
                self.assertEqual(result.mapped_operations,
                                 tuple(op.op_id for op in stage.schedule.operations))
                self.assertEqual(result.dynamic_shared_bytes, 49200)
                self.assertEqual(result.tmem_columns, 64)
                self.assertIn('cp.async.bulk.tensor.2d', result.instruction_helpers)
                self.assertIn('tcgen05.mma', result.instruction_helpers)
                self.assertIn('maps_a[0]', result.source)
                self.assertIn('map_b[0]', result.source)
                self.assertIn('n_tile * 64', result.source)
                self.assertIn('asm volatile("trap;")', result.source)
                self.assertNotIn('if (((0 * 128', result.source)
                self.assertNotIn('tcgen05.alloc', result.source)
                self.assertNotIn('tcgen05.dealloc', result.source)
                self.assertEqual(result.source.count('CAKE_OP:'), 5)

    def test_refuses_wrong_route_and_tile_geometry(self):
        stage = self.program.stages[0]
        with self.assertRaisesRegex(EmitError, 'identifier'):
            emit_tensor_tile_stage(stage.schedule, self.target,
                                   function_name='bad-name')
        changed = deepcopy(json.loads(PROGRAM.read_text())['stages'][0]['schedule'])
        changed['program_map']['axes'][1]['tile'] = 32
        with self.assertRaisesRegex(EmitError, 'ACCESS_TILE_MISMATCH'):
            emit_tensor_tile_stage(Schedule.from_dict(changed), self.target,
                                   function_name='tile_stage')
        other = Compiler.load(ROOT)._revision.targets['sm_100a']
        with self.assertRaisesRegex(EmitError, 'exact B300'):
            emit_tensor_tile_stage(stage.schedule, other,
                                   function_name='tile_stage')

    def test_activation_stage_comes_from_its_cast_explicit_schedule(self):
        schedule = self.program.stages[1].schedule
        result = emit_model_activation_stage(schedule, self.target,
                                             function_name='cake_activation_stage_work')
        self.assertEqual(result.mapped_operations,
                         tuple(op.op_id for op in schedule.operations))
        self.assertIn('expf(', result.source)
        self.assertIn('__float2bfloat16_rn', result.source)
        self.assertIn('fence.proxy.async.global', result.source)
        self.assertEqual(result.dynamic_shared_bytes, 0)
        self.assertNotIn('tcgen05', result.source)

    def test_ranked_tile_composition_reports_safe_capacity_and_actual_shared_bytes(self):
        effects = RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        combine = Schedule.from_dict(json.loads(COMBINE.read_text()))
        result = compose_model_ranked_tile_stages(
            effects, self.program, combine, self.target)
        self.assertEqual(result.stage_work_units,
                         (('up_gate',24),('activation',128),('down',32)))
        self.assertEqual((result.safe_logical_tile_slots_per_rank,
                          result.safe_stage_task_slots_per_rank),(255,46920))
        self.assertEqual((result.declared_shared_bytes,
                          result.emitted_shared_bytes),(49152,49200))
        self.assertEqual((result.tensor_bytes,result.required_execution_groups,
                          result.target_multiprocessors),(32768,6,148))
        self.assertEqual(len(result.stages),3)
        device=emit_source_event_device(
            result,combine_source='#include "combine/kernel.cu"')
        self.assertIn('__global__ void dispatch_source_wave',device)
        self.assertIn('atom.relaxed.sys.global.add.s32',device)
        self.assertNotIn('int main(',device)
        with self.assertRaisesRegex(EmitError,'exact checked B300 composition'):
            emit_source_event_device(result,combine_source='')
        library=emit_source_event_library(
            result,combine_source=native_cuda.emit(combine,self.target).source,
            entry=effects.lowering.entry_point)
        self.assertIn('cake_ranked_tile_b300_create(',library)
        self.assertIn('cake_ranked_tile_b300_launch(',library)
        self.assertIn('cake_ranked_tile_b300_destroy(',library)
        self.assertNotIn('int main(',library)
        self.assertNotIn('fopen(',library)
        bad=effects.document
        bad['partial_threshold_rows']=128
        with self.assertRaisesRegex(EmitError,'exact model EP4 domain'):
            compose_model_ranked_tile_stages(
                RankedTileEffects.from_dict(bad),self.program,combine,self.target)
        other=Compiler.load(ROOT)._revision.targets['sm_100a']
        with self.assertRaisesRegex(EmitError,'exact observed B300'):
            compose_model_ranked_tile_stages(effects,self.program,combine,other)

    def test_clean_compiler_composes_tensor_stages_into_worker(self):
        commit = Compiler.load(ROOT).commit
        if commit is None:
            self.skipTest('generated worker needs a fixed clean Compiler commit')
        script = ROOT / 'experiments/weave/native_b300/generate_tile_ready_from_cake.py'
        for generation in ('cake_full_ffn_stages', 'cake_two_expert_stages',
                           'cake_ep4_owner_stages',
                           'cake_ep4_live_chain_stages',
                           'cake_ep4_gpu_plan_stages',
                           'cake_ep4_temporal_stages',
                           'cake_ep4_capacity_stages',
                           'cake_ep4_source_event_stages'):
            with self.subTest(generation=generation), tempfile.TemporaryDirectory(
                    prefix='cake-tile-stage-') as directory:
                root = Path(directory)
                (root / 'manifest.json').write_text(json.dumps({
                    'target': 'sm_103a', 'source_commit': commit,
                    'generation': generation,
                }))
                environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
                subprocess.run([sys.executable, str(script), '--evidence-root',
                                str(root)], env=environment, check=True,
                               capture_output=True, text=True)
                source = (root / 'model_tile_ready_ffn_capped.cu').read_text()
                lowering = json.loads((root / 'stage_lowering.json').read_text())
                self.assertNotIn('@CAKE_', source)
                self.assertNotIn('#include "down/kernel.cu"', source)
                self.assertEqual([stage['mapped_operations']
                                  for stage in lowering['stages']],
                                 [[op.op_id for op in self.program.stages[index].schedule.operations]
                                  for index in (0, 1, 2)])
                self.assertEqual(source.count('CAKE_OP: up_gate.'), 0)
                self.assertEqual(source.count('CAKE_OP: load_a'), 2)
                self.assertEqual(source.count('CAKE_OP: exp_gate'), 1)
                if generation == 'cake_two_expert_stages':
                    self.assertIn('tile_expert[tile]', source)
                    self.assertIn('up_map_b+expert', source)
                    self.assertIn('down_map_b+expert', source)
                if generation == 'cake_ep4_owner_stages':
                    self.assertIn('constexpr int kExperts = 32;', source)
                    self.assertIn('up_map_b+expert', source)
                if generation == 'cake_ep4_live_chain_stages':
                    self.assertIn('dispatch_routes<<<', source)
                    self.assertIn('gather_tiles<<<', source)
                    self.assertIn('scatter_returns<<<', source)
                    self.assertIn('cake_weave_rank512_combine_kernel<<<', source)
                if generation == 'cake_ep4_gpu_plan_stages':
                    self.assertIn('derive_expert_order<<<', source)
                    self.assertIn('assign_tile_plan<<<', source)
                    self.assertIn('expand_stage_tasks<<<', source)
                if generation == 'cake_ep4_temporal_stages':
                    self.assertIn('derive_early_wave<<<', source)
                    self.assertIn('assign_terminal_tiles<<<', source)
                    self.assertIn('publish_wave_ready<<<', source)
                    composition=lowering['ranked_tile_stage_composition']
                    self.assertEqual(composition['safe_logical_tile_slots_per_rank'],255)
                    self.assertEqual(composition['safe_stage_task_slots_per_rank'],46920)
                    self.assertEqual(composition['emitted_shared_bytes'],49200)
                    self.assertIs(composition['complete_ranked_tile_lowering'],False)
                if generation == 'cake_ep4_capacity_stages':
                    self.assertIn('constexpr int kLogicalTiles = 255;', source)
                    self.assertIn('derive_wave_order<<<', source)
                    self.assertIn('assign_wave_tiles<<<', source)
                    self.assertIn('plan_device_report.json', source)
                    self.assertEqual(
                        lowering['ranked_tile_stage_composition'][
                            'safe_logical_tile_slots_per_rank'],255)
                if generation == 'cake_ep4_source_event_stages':
                    self.assertIn('constexpr int kEvents = kWaves * (kSourceRanks+1);',
                                  source)
                    self.assertIn('derive_wave_order<<<', source)
                    self.assertIn('tile_events_by_owner', source)


if __name__ == '__main__':
    unittest.main()
