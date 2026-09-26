"""K128 state geometry around a fixed C32 forward solve, two carried phases."""
from __future__ import annotations

import copy
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_two_phase_carried import document as small_document, target


def document() -> dict:
    value=small_document()
    value['schedule_id']='native-two-phase-carried-k128-c32'
    value['lowering']['entry_point']='cake_two_phase_carried_k128_c32'
    buffers={b['name']:b for b in value['buffers']}
    for name in ('initial_state','final_state','initial_reg','state_tmem',
                 'correction_acc','correction','next_state'):
        buffers[name]['shape']=[128,128]
    for name in ('b_base','b_correction'):
        buffers[name]['shape']=[2,128,32]
    buffers['p']['shape']=[2,32,32]
    buffers['rhs']['shape']=[2,128,32]
    for name in ('base_stage','correction_stage'):
        buffers[name]['shape']=[128,32]
    buffers['updates_tmem']['byte_offset']=32768
    buffers['base_acc']['byte_offset']=40960
    buffers['correction_acc']['byte_offset']=57344
    allocations={a['name']:a for a in value['allocations']}
    allocations['base_smem']['size_bytes']=8192
    allocations['correction_smem']['size_bytes']=8192
    allocations['tensor']['size_bytes']=131072
    allocations['tensor']['tensor_columns']=256
    operations={op['id']:op for op in value['operations']}
    for name in ('load_base','load_correction'):
        operations[name]['parameters']['descriptor_box']=[128,32]
    base=operations['mma_base']['parameters']
    base['tile_shape']=[128,32,128]
    base['instruction']['operand_major']=['k','mn']
    correction=operations['mma_correction']['parameters']
    correction['tile_shape']=[128,128,32]
    correction['instruction']['shape']=[128,128,16]
    value['tile_loops'][0]['tile']=1
    def axis(n):return {'source':'dimension','dimension':n}
    chunk={'source':'loop','name':'chunk_index'}
    access={a['operation']:a for a in value['access_maps']}
    for name in ('load_base','load_rhs','load_p','load_correction'):
        access[name]['indices']=[copy.deepcopy(chunk),axis(1),axis(2)]
    return value


class NativeTwoPhaseK128(unittest.TestCase):
    def test_exact_k128_mapping_has_typed_stages(self):
        schedule=Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule,target()) if f.blocks_lowering],[])
        self.assertEqual(native_cuda.preflight(schedule,target()),())
        source=native_cuda.emit(schedule,target()).source
        self.assertIn('cake_tma3(',source)
        self.assertLess(source.index('// CAKE_OP: mma_base'),source.index('// CAKE_OP: solve'))
        self.assertLess(source.index('// CAKE_OP: solve'),source.index('// CAKE_OP: mma_correction'))
        self.assertIn('atom*1024, 512, 4), 134808720u',source)

    def test_chunk_index_and_operand_domain_are_owned(self):
        value=document()
        access=next(a for a in value['access_maps'] if a['operation']=='load_base')
        access['indices'][0]={'source':'dimension','dimension':0}
        self.assertIn('NATIVE_CARRIED_MMA_DOMAIN',
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value),target())})


if __name__=='__main__':
    unittest.main()
