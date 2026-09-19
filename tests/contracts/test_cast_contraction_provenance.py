"""Cast-preserved contraction coordinates; CPU source execution is not GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.backends.triton import emit
from tests.contracts.test_triton_loop_scopes import _gemm, _execute, TARGET

ROOT = Path(__file__).resolve().parents[2]


def cast_gemm(*, nested=False, chained=False):
    document = _gemm(tail=True)
    if not nested:
        document['tile_loops'] = document['tile_loops'][:1]
        document['program_map']['axes'] = [
            dict(name='m_block', axis=0, buffer='a', dimension=0, tile=16),
            dict(name='n_block', axis=1, buffer='b', dimension=0, tile=16)]
        for access in document['access_maps']:
            for index in access['indices']:
                if index.get('name') == 'm':
                    index.update(source='program_tile', name='m_block')
    dot = next(op for op in document['operations'] if op['id'] == 'dot')
    dot['parameters']['instruction'] = {'contract':'triton.dot.fp32_ieee'}
    for name in ('a_tile', 'b_tile'):
        source = name
        for index in range(2 if chained else 1):
            destination = f'{name}_cast_{index}'
            buffer = deepcopy(next(b for b in document['buffers'] if b['name'] == name))
            buffer.update(name=destination, dtype='fp32')
            document['buffers'].append(buffer)
            operation = {'id':destination, 'kind':'cast', 'role':dot['role'],
                'reads':[source], 'writes':[destination], 'parameters':{'to':'fp32'}}
            document['operations'].insert(document['operations'].index(dot), operation)
            body = document['tile_loops'][0]['body']
            body.insert(body.index('dot'), destination)
            source = destination
        dot['reads'][dot['reads'].index(name)] = source
    return document


class CastContractionProvenanceTests(unittest.TestCase):
    def test_casted_k_tiles_are_carried_and_emitted_source_sums_every_k_tile(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        for chained in (False, True):
            with self.subTest(chained=chained):
                document = cast_gemm(chained=chained)
                schedule = Schedule.from_dict(document)
                self.assertTrue(schedule.mma_accumulates_over(schedule.operation('dot'), schedule.tile_loop('k_loop')))
                assessment = compiler.assess(document)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                emission = emit(schedule, TARGET)
                shapes = {b['name']:b['shape'] for b in document['buffers']}
                m, k = shapes['a']; n = shapes['b'][0]
                a = [(i*3+t)%7-3 for i in range(m) for t in range(k)]
                b = [(j+t*2)%5-2 for j in range(n) for t in range(k)]
                bias = [j-2 for j in range(n)]
                memories = {'a':a, 'b':b, 'bias':bias, 'c':[None]*(m*n)}
                execution = _execute(emission, memories)
                expected = [sum(a[i*k+t]*b[j*k+t] for t in range(k))+bias[j]
                            for i in range(m) for j in range(n)]
                self.assertEqual(memories['c'], expected)
                self.assertEqual(set(execution.stores.values()), {1})

    def test_casts_do_not_turn_output_axis_loops_into_contractions(self):
        schedule = Schedule.from_dict(cast_gemm(nested=True, chained=True))
        dot = schedule.operation('dot')
        self.assertTrue(schedule.mma_accumulates_over(dot, schedule.tile_loop('k_loop')))
        self.assertFalse(schedule.mma_accumulates_over(dot, schedule.tile_loop('m_loop')))

    def test_invalid_cast_graphs_do_not_supply_axis_proof(self):
        for mutation in ('shape', 'cycle', 'ambiguous_writer'):
            document = cast_gemm()
            for name in ('a_tile_cast_0', 'b_tile_cast_0'):
                operation = next(op for op in document['operations'] if op['id'] == name)
                if mutation == 'shape':
                    next(b for b in document['buffers'] if b['name'] == name)['shape'] = [8,32]
                elif mutation == 'cycle':
                    operation['reads'] = [name]
                else:
                    extra = deepcopy(operation); extra['id'] += '_second_writer'
                    document['operations'].append(extra)
            schedule = Schedule.from_dict(document)
            self.assertFalse(schedule.mma_accumulates_over(schedule.operation('dot'), schedule.tile_loop('k_loop')))


if __name__ == '__main__':
    unittest.main()
