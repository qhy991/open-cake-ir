"""Indexed MMA source arithmetic on CPU; no native precision or device claim."""
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]


def indexed_source(target='xcore1002', *, batches=2, sequence=65, rows=17, columns=35,
                   selected=(0, 1), cast=False, indices='scalar', outer_columns=False):
    heads = batches if indices == 'repeated' else 2
    prefix = (batches, heads) if indices in {'multiple', 'repeated'} else (batches,)
    left_shape, right_shape = (*prefix, sequence, rows), (*prefix, sequence, columns)
    source = f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="indexed_regions", target="{target}", backend="triton", entry_point="indexed_regions")
def candidate(lm, left: cake.Tensor({left_shape!r}, "bf16"), right: cake.Tensor({right_shape!r}, "bf16"), output: cake.Tensor(({rows}, {columns}), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    m = lm.program(output, axis=0, dimension=0, tile=16)
    n = lm.program(output, axis=1, dimension=1, tile=32)
    with compute:
        coordinate = lm.coordinate(source="program", name="m")
        zero = coordinate * 0
'''
    vector = indices in {'vector', 'multiple', 'repeated'}
    if vector:
        source += '''        m_features = lm.coordinate(source="program_tile", name="m")
        n_features = lm.coordinate(source="program_tile", name="n")
'''
    for region, batch in enumerate(selected):
        source += f'    with compute:\n        batch_{region} = zero + {batch}\n'
        address = [f'lm.scalar_index(batch_{region})']
        if indices == 'multiple':
            source += f'        head_{region} = zero + {region % heads}\n'
            address.append(f'lm.scalar_index(head_{region})')
        elif indices == 'repeated':
            address.append(f'lm.scalar_index(batch_{region})')
        source += f'''    for sequence_{region} in lm.range(left, name="sequence_{region}", dimension={len(prefix)}, tile=32, num_stages=1):
        with compute:
            left_{region} = lm.load(left[{', '.join((*address, f'sequence_{region}', 'm_features' if vector else 'm'))}], id="left_{region}")
            right_{region} = lm.load(right[{', '.join((*address, f'sequence_{region}', 'n_features' if vector else 'n'))}], id="right_{region}")
'''
        left, right = f'left_{region}', f'right_{region}'
        if cast:
            source += f'''            left_cast_{region} = lm.cast({left}, to="fp32", id="left_cast_{region}")
            right_cast_{region} = lm.cast({right}, to="fp32", id="right_cast_{region}")
'''
            left, right = f'left_cast_{region}', f'right_cast_{region}'
        contract = 'triton.dot.fp32_ieee' if cast else 'triton.dot.bf16_fp32'
        source += f'''            a_{region} = lm.transpose({left}, id="a_{region}")
            b_{region} = lm.transpose({right}, id="b_{region}")
            product_{region} = lm.mma(a_{region}, b_{region}, instruction={{"contract": "{contract}"}}, tile_shape=(16, 32, 32), id="dot_{region}")
'''
    source += ('    with compute:\n        total = ' + ' + '.join(f'product_{i}' for i in range(len(selected)))
               + '\n        lm.store(output[m, n], total, id="store")\n')
    if outer_columns:
        source = source.replace('    n = lm.program(output, axis=1, dimension=1, tile=32)\n', '')
        source = source.replace('        n_features = lm.coordinate(source="program_tile", name="n")\n', '')
        boundary = source.index('    with compute:\n        batch_0')
        body = source[boundary:]
        source = source[:boundary] + (
            f'    for n in lm.range(right, name="columns", dimension={len(prefix) + 1}, tile=32):\n'
            '        with compute:\n'
            '            n_features = lm.coordinate(source="loop_tile", name="n")\n'
            + ''.join('    ' + line + '\n' for line in body.splitlines()))
    return source


def nested_source(target):
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="nested_non_k", target="{target}", backend="triton", entry_point="nested_non_k")
def candidate(lm, left: cake.Tensor((2, 65, 32), "bf16"), right: cake.Tensor((2, 65, 32), "bf16"), output: cake.Tensor((32, 32), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    m = lm.program(output, axis=0, dimension=0, tile=16)
    n = lm.program(output, axis=1, dimension=1, tile=32)
    for sequence in lm.range(left, name="sequence", dimension=1, tile=32, num_stages=1):
        for batch in lm.range(left, name="batch", dimension=0, tile=1, num_stages=1):
            with compute:
                batch_index = lm.coordinate(source="loop_tile", name="batch")
                left_stored = lm.load(left[lm.scalar_index(batch_index), sequence, m], id="left")
                right_stored = lm.load(right[lm.scalar_index(batch_index), sequence, n], id="right")
                a = lm.transpose(left_stored)
                b = lm.transpose(right_stored)
                product = lm.mma(a, b, instruction={{"contract": "triton.dot.bf16_fp32"}}, tile_shape=(16, 32, 32), id="dot")
    with compute:
        lm.store(output[m, n], product, id="store")
'''


class IndexedMMAProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def execute(self, target, **options):
        document = frontend.parse(indexed_source(target, **options)).document
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        schedule = assessment.typed_schedule
        selected = options.get('selected', (0, 1))
        for region in range(len(selected)):
            self.assertTrue(schedule.mma_accumulates_over(
                schedule.operation(f'dot_{region}'), schedule.tile_loop(f'sequence_{region}')))
        emission = emit(schedule, self.compiler._revision.targets[target])
        batches, sequence, rows, columns = (options.get(k, v) for k, v in
            [('batches', 2), ('sequence', 65), ('rows', 17), ('columns', 35)])
        indices = options.get('indices', 'scalar')
        heads = batches if indices == 'repeated' else 2 if indices == 'multiple' else 1
        left = [float((batch + 1) * (head + 1) * ((step + row) % 5 - 2))
                for batch in range(batches) for head in range(heads)
                for step in range(sequence) for row in range(rows)]
        right = [float((batch + 2) * (head + 2) * ((2 * step + column) % 7 - 3))
                 for batch in range(batches) for head in range(heads)
                 for step in range(sequence) for column in range(columns)]
        pairs = [(batch, region % heads if indices == 'multiple' else batch if indices == 'repeated' else 0)
                 for region, batch in enumerate(selected)]
        expected = [sum(left[((batch * heads + head) * sequence + step) * rows + row]
                        * right[((batch * heads + head) * sequence + step) * columns + column]
                        for batch, head in pairs for step in range(sequence))
                    for row in range(rows) for column in range(columns)]
        memories = {'left': left[:], 'right': right[:], 'output': [None] * (rows * columns)}
        trace = _execute(emission, memories)
        self.assertEqual(memories['output'], expected)
        self.assertEqual(memories['left'], left)
        self.assertEqual(memories['right'], right)
        self.assertEqual(len(trace.stores), rows * columns)
        self.assertEqual(set(trace.stores.values()), {1})
        return memories['output'], schedule

    def test_indexed_cast_transpose_matrices_keep_batch_sequence_and_output_tails(self):
        for target in ('xcore1002', 'gfx938'):
            for cast in (False, True):
                with self.subTest(target=target, cast=cast):
                    self.execute(target, cast=cast)

    def test_scalar_selection_changes_the_batch_in_actual_emitted_arithmetic(self):
        first, _ = self.execute('xcore1002', batches=3, sequence=33, rows=3, columns=7, selected=(0, 0))
        other, _ = self.execute('xcore1002', batches=3, sequence=33, rows=3, columns=7, selected=(1, 2))
        self.assertNotEqual(first, other)

    def test_vector_and_multiple_scalar_dependencies_preserve_first_use_and_deduplication(self):
        for target in ('xcore1002', 'gfx938'):
            for indices, dependencies in (
                ('vector', ('left', 'batch_0', 'm_features')),
                ('multiple', ('left', 'batch_0', 'head_0', 'm_features')),
                ('repeated', ('left', 'batch_0', 'm_features')),
            ):
                with self.subTest(target=target, indices=indices):
                    _, schedule = self.execute(target, sequence=33, rows=3, columns=7, indices=indices)
                    self.assertEqual(schedule.operation('left_0').reads, dependencies)

    def test_outer_feature_index_is_available_to_each_inner_k_region(self):
        for target in ('xcore1002', 'gfx938'):
            with self.subTest(target=target):
                self.execute(target, sequence=33, rows=3, columns=35,
                             indices='multiple', outer_columns=True)

    def test_missing_reordered_or_unwritten_index_refuses_at_its_existing_owner(self):
        for mutation in ('missing', 'reordered', 'unwritten'):
            document = frontend.parse(indexed_source(indices='multiple')).document
            load = next(op for op in document['operations'] if op['id'] == 'left_0')
            if mutation == 'missing':
                load['reads'].remove('head_0')
            elif mutation == 'reordered':
                load['reads'][1:3] = reversed(load['reads'][1:3])
            else:
                writer = next(op for op in document['operations'] if op['writes'] == ['head_0'])
                document['operations'].remove(writer)
                document['operations'].insert(document['operations'].index(load) + 1, writer)
            with self.subTest(mutation=mutation):
                assessment = self.compiler.assess(document)
                self.assertFalse(assessment.lowering_eligible)
                wanted = 'OP_READ_BEFORE_WRITE' if mutation == 'unwritten' else 'ACCESS_INDEX_BUFFER_READS'
                self.assertIn(wanted, [f.code for f in assessment.findings])

    def test_ambiguous_access_and_invalid_cast_chain_have_no_backend_operand_proof(self):
        for mutation in ('access', 'cast'):
            document = frontend.parse(indexed_source(cast=True)).document
            if mutation == 'access':
                access = next(a for a in document['access_maps'] if a['operation'] == 'left_0')
                document['access_maps'].append(deepcopy(access))
            else:
                cast = next(op for op in document['operations'] if op['id'] == 'left_cast_0')
                cast['reads'].append('batch_0')
            with self.subTest(mutation=mutation):
                findings = preflight(Schedule.from_dict(document), self.compiler._revision.targets['xcore1002'])
                self.assertIn('TRITON_NESTED_MMA_OPERAND', [f.code for f in findings])

    def test_nested_non_k_value_still_cannot_escape_to_the_output(self):
        for target in ('xcore1002', 'gfx938'):
            with self.subTest(target=target):
                assessment = self.compiler.assess(frontend.parse(nested_source(target)).document)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn('BUFFER_ESCAPES_LOOP', [f.code for f in assessment.findings])
