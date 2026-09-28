"""Schedule-driven tensor-core stage bodies used inside a persistent worker."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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
    emit_source_event_device, emit_source_event_library, source_event_map,
)
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda_tile_stage import (
    compose_model_ranked_tile_stages, emit_model_activation_stage,
    emit_tensor_tile_stage,
)


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300.json'
PROGRAM_N128 = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300-n128.json'
EFFECTS = ROOT / 'examples/programs/weave-model-ranked-tile-effects-b300.json'
COMBINE = ROOT / 'examples/schedules/native/weave-model-weighted-combine-rank512-b300.json'


class NativeCudaTileStageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = Program.from_dict(json.loads(PROGRAM.read_text()))
        cls.program_n128 = Program.from_dict(json.loads(PROGRAM_N128.read_text()))
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
                self.assertEqual(result.tensor_address_offset, 49192)
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

    def test_n128_tensor_stages_and_ranked_composition(self):
        for index, function in ((0, 'cake_upgate_stage_work'),
                                (2, 'cake_down_stage_work')):
            with self.subTest(stage=index):
                stage = self.program_n128.stages[index]
                result = emit_tensor_tile_stage(
                    stage.schedule, self.target, function_name=function)
                self.assertEqual(result.dynamic_shared_bytes, 65584)
                self.assertEqual(result.tmem_columns, 128)
                self.assertEqual(result.tensor_address_offset, 65576)
                self.assertIn('n_tile * 128', result.source)
                self.assertEqual(result.mapped_operations,
                                 tuple(op.op_id for op in stage.schedule.operations))
        effects = RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        combine = Schedule.from_dict(json.loads(COMBINE.read_text()))
        result = compose_model_ranked_tile_stages(
            effects, self.program_n128, combine, self.target)
        self.assertEqual(result.stage_work_units,
                         (('up_gate', 12), ('activation', 22), ('down', 16)))
        self.assertEqual(result.safe_stage_task_slots_per_rank, 12750)
        self.assertEqual(result.tensor_bytes, 65536)
        self.assertEqual(result.emitted_shared_bytes, 65584)
        device = emit_source_event_device(
            result, combine_source='#include "combine/kernel.cu"')
        self.assertIn('constexpr int kTensorColumns = 128;', device)
        self.assertIn('shared + 65576', device)
        self.assertIn('"n"(kTensorColumns)', device)
        library = emit_source_event_library(
            result, combine_source=native_cuda.emit(combine, self.target).source,
            entry=effects.lowering.entry_point)
        self.assertIn('box_b[2]={64,kTensorColumns}', library)

    def test_activation_stage_comes_from_its_cast_explicit_schedule(self):
        schedule = self.program.stages[1].schedule
        result = emit_model_activation_stage(schedule, self.target,
                                             function_name='cake_activation_stage_work')
        self.assertEqual(result.mapped_operations,
                         tuple(op.op_id for op in schedule.operations))
        self.assertIn('expf(', result.source)
        self.assertIn('__float2bfloat16_rn', result.source)
        self.assertIn('fence.proxy.async.global', result.source)
        self.assertIn('row_tile * 6 + cake_warp_row', result.source)
        self.assertIn('row_tile * 6 + cake_warp_row < 128', result.source)
        self.assertEqual(result.dynamic_shared_bytes, 0)
        self.assertNotIn('tcgen05', result.source)

    def test_ranked_tile_composition_reports_safe_capacity_and_actual_shared_bytes(self):
        effects = RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        combine = Schedule.from_dict(json.loads(COMBINE.read_text()))
        result = compose_model_ranked_tile_stages(
            effects, self.program, combine, self.target)
        self.assertEqual(result.stage_work_units,
                         (('up_gate',24),('activation',22),('down',32)))
        self.assertEqual((result.safe_logical_tile_slots_per_rank,
                          result.safe_stage_task_slots_per_rank),(255,19890))
        self.assertEqual((result.declared_shared_bytes,
                          result.emitted_shared_bytes),(49152,49200))
        self.assertEqual((result.tensor_bytes,result.required_execution_groups,
                          result.target_multiprocessors),(32768,6,148))
        self.assertEqual(len(result.stages),3)
        device=emit_source_event_device(
            result,combine_source='#include "combine/kernel.cu"')
        self.assertIn('__device__ void dispatch_source_chunk',device)
        self.assertIn('dispatch_source_chunk(source_params,block',device)
        self.assertNotIn('__global__ void dispatch_source_wave',device)
        self.assertIn('atom.relaxed.sys.global.add.s32',device)
        self.assertNotIn('int main(',device)
        with self.assertRaisesRegex(EmitError,'exact checked B300 composition'):
            emit_source_event_device(result,combine_source='')
        combine_source=native_cuda.emit(combine,self.target).source
        library=emit_source_event_library(
            result,combine_source=combine_source,
            entry=effects.lowering.entry_point)
        self.assertIn('cake_ranked_tile_b300_create(',library)
        self.assertIn('cake_ranked_tile_b300_launch(',library)
        self.assertIn('cake_ranked_tile_b300_abi_version() { return 4; }',library)
        self.assertIn('const int* communication_ctas',library)
        self.assertIn('cake_ranked_tile_b300_destroy(',library)
        self.assertIn('cake_ranked_tile_b300_stolen(',library)
        self.assertIn('cake_ranked_tile_b300_payloads(',library)
        self.assertIn('cake_ranked_tile_b300_bin_bytes(',library)
        self.assertNotIn('cudaStreamWaitEvent(s.compute,s.fence[event-1])',library)
        self.assertNotIn('dispatch_source_wave<<<',library)
        self.assertNotIn('derive_wave_order<<<',library)
        self.assertNotIn('s.communication',library)
        self.assertIn('derive_expert_snapshot(source_params,expert,source_wave',library)
        gather=library.index('gather_event_rows(source_params,selected_event,block')
        proxy=library.index('asm volatile("fence.proxy.async.global;',gather)
        publish=library.index('publish_event_snapshot(source_params,selected_event)',proxy)
        acquire=library.index('// CAKE_EFFECT: tile.acquire',publish)
        consumer_proxy=library.index('asm volatile("fence.proxy.async.global;',acquire)
        upgate=library.index('cake_upgate_stage_work(shared,tensor_address,warp')
        self.assertLess(gather,proxy)
        self.assertLess(proxy,publish)
        self.assertLess(publish,acquire)
        self.assertLess(acquire,consumer_proxy)
        self.assertLess(consumer_proxy,upgate)
        self.assertNotIn('int main(',library)
        self.assertNotIn('fopen(',library)
        mapped=source_event_map(library,result,combine_source=combine_source)
        self.assertEqual(len([name for name in mapped
                              if name.startswith('effect.')]),27)
        self.assertEqual({name for name in mapped if name.startswith('up_gate.')},
                         {'up_gate.'+op.op_id for op in
                          self.program.stages[0].schedule.operations})
        self.assertEqual({name for name in mapped if name.startswith('combine.')},
                         {'combine.'+op.op_id for op in combine.operations})
        with self.assertRaisesRegex(EmitError,'omits'):
            source_event_map(library.replace('// CAKE_EFFECT: payload.publish',''),
                             result,combine_source=combine_source)
        observed=replace(self.target,
                         occupancy=replace(self.target.occupancy,
                                           multiprocessor_count=149),
                         device_names=('Synthetic B300 audit',))
        changed=compose_model_ranked_tile_stages(
            effects,self.program,combine,observed)
        routed=emit_source_event_library(
            changed,combine_source=native_cuda.emit(combine,observed).source,
            entry=effects.lowering.entry_point)
        self.assertIn('prop.multiProcessorCount!=149',routed)
        self.assertIn('Synthetic B300 audit',routed)
        self.assertNotIn('NVIDIA B300 SXM6 AC',routed)
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
                    self.assertEqual(composition['safe_stage_task_slots_per_rank'],19890)
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

    def test_clean_compiler_emits_one_pointer_abi_with_cake_combine(self):
        commit=Compiler.load(ROOT).commit
        if commit is None:
            self.skipTest('ranked-tile ABI generation needs a clean commit')
        script=ROOT/'experiments/weave/native_b300/generate_ranked_tile_library.py'
        with tempfile.TemporaryDirectory(prefix='cake-ranked-tile-abi-') as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps({
                'target':'sm_103a','source_commit':commit,
                'generation':'cake_ranked_tile_pointer_abi'}))
            environment=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
            subprocess.run([sys.executable,str(script),'--evidence-root',str(root)],
                           env=environment,check=True,capture_output=True,text=True)
            source=(root/'ranked_tile.cu').read_text()
            report=json.loads((root/'lowering_report.json').read_text())
            self.assertEqual(Program.from_dict(json.loads(
                (root/'local_program.json').read_text())),self.program)
            self.assertEqual(RankedTileEffects.from_dict(json.loads(
                (root/'effects.json').read_text())).lowering.entry_point,
                             report['entry_point'])
            self.assertIn('cake_ranked_tile_b300_create(',source)
            self.assertIn('cake_ranked_tile_b300_launch(',source)
            self.assertIn('cake_weave_rank512_combine_kernel',source)
            self.assertIn('CAKE_OP: exp_gate',source)
            self.assertNotIn('int main(',source)
            self.assertNotIn('fopen(',source)
            self.assertEqual(report['logical_tile_capacity'],255)
            self.assertEqual(report['stage_task_capacity'],19890)
            from experiments.weave.native_b300.model_ranked_tile_pointer_run import contract
            self.assertEqual(contract(root)['source_commit'],commit)
            self.assertEqual(report['toolchain_requirements']['rank_local_controls'],
                             True)
            self.assertEqual(len([name for name in report['source_map']
                                  if name.startswith('effect.')]),27)

    def test_clean_compiler_emits_n128_pointer_abi(self):
        commit = Compiler.load(ROOT).commit
        if commit is None:
            self.skipTest('N128 ranked-tile ABI generation needs a clean commit')
        script = ROOT / 'experiments/weave/native_b300/generate_ranked_tile_library.py'
        with tempfile.TemporaryDirectory(prefix='cake-ranked-tile-n128-') as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({
                'target': 'sm_103a', 'source_commit': commit,
                'generation': 'cake_ranked_tile_pointer_abi',
                'tensor_n_tile': 128,
            }))
            environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
            subprocess.run([sys.executable, str(script), '--evidence-root',
                            str(root)], env=environment, check=True,
                           capture_output=True, text=True)
            source = (root / 'ranked_tile.cu').read_text()
            report = json.loads((root / 'lowering_report.json').read_text())
            self.assertEqual(Program.from_dict(json.loads(
                (root / 'local_program.json').read_text())), self.program_n128)
            self.assertEqual(report['tensor_n_tile'], 128)
            self.assertEqual(report['stage_work_units'],
                             [['up_gate', 12], ['activation', 22], ['down', 16]])
            self.assertEqual(report['stage_task_capacity'], 12750)
            self.assertEqual(report['emitted_shared_bytes'], 65584)
            self.assertIn('constexpr int kTensorColumns = 128;', source)
            self.assertIn('shared + 65576', source)
            from experiments.weave.native_b300.model_ranked_tile_pointer_run import contract
            self.assertEqual(contract(root)['tensor_n_tile'], 128)

    def test_exact_model_program_has_complete_ranked_tile_lowering(self):
        compiler=Compiler.load(ROOT)
        if compiler.commit is None:
            self.skipTest('public ranked-tile lowering needs a clean Compiler')
        effects=RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        combine=Schedule.from_dict(json.loads(COMBINE.read_text()))
        lowered=compiler.lower_ranked_tiles(effects,self.program,combine)
        lowered.validate_binding()
        req=lowered.toolchain_requirements
        self.assertEqual((req['world_size'],req['logical_tile_capacity'],
                          req['stage_task_capacity'],req['source_events']),
                         (4,255,19890,20))
        self.assertEqual(req['supported_chunks'],[1,2,4])
        self.assertEqual(req['source_chunk_tokens_by_chunks'],
                         {'1':512,'2':256,'4':128})
        self.assertEqual(req['host_abi']['abi_version'],
                         'cake_ranked_tile_b300_abi_version')
        self.assertEqual(req['host_abi']['tile_counts'],
                         'cake_ranked_tile_b300_tile_counts')
        self.assertEqual(req['rank_inputs'][0],
                         {'name':'hidden','shape':[512,2048],'dtype':'bf16'})
        self.assertEqual(len([name for name in lowered.source_map
                              if name.startswith('effect.')]),27)
        self.assertIn('cake_ranked_tile_b300_launch(',lowered.source)


if __name__ == '__main__':
    unittest.main()
