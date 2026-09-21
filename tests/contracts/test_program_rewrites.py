"""Complete Program proofs and execution boundaries; no device performance claim."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.program import LoweredProgram
from open_cake_ir.evaluation.launch_plan import prepare_program
from tests.contracts.test_epilogue_fusion import stage, execute
from tests.contracts.test_ordered_launch_plan import document as scalar_program

ROOT = Path(__file__).resolve().parents[2]


def epilogue_program(dtype='bf16'):
    p, e = stage('producer', dtype), stage('consumer', dtype)
    tensors = {}
    for schedule in (p, e):
        for b in schedule['buffers']:
            if b['space'] == 'global':
                tensors[b['name']] = {'shape': b['shape'], 'dtype': b['dtype']}
    return {'schema_version': 1, 'program_id': 'rounded-epilogue', 'target': p['target'],
            'tensors': tensors, 'inputs': ['a', 'b', 'bias'], 'outputs': list(e['outputs']),
            'stages': [{'name': name, 'schedule': schedule,
                        'bindings': {b['name']: b['name'] for b in schedule['buffers'] if b['space'] == 'global'}}
                       for name, schedule in [('producer', p), ('epilogue', e)]]}


class CompleteProgramRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def fuse(self, document, **parameters):
        return self.compiler.rewrite_program(Program.from_dict(document), 'fuse_pointwise_epilogue',
                dict(producer='producer', epilogue='epilogue', schedule_id='fused', entry_point='fused') | parameters)

    def test_fusion_preserves_public_abi_and_rounding_with_local_output_rename(self):
        for dtype in ('bf16', 'fp16'):
            document = epilogue_program(dtype)
            before = deepcopy(document)
            result = self.fuse(document)
            self.assertTrue(result.applied, result.message)
            program = result.program
            self.assertEqual(document, before)
            self.assertEqual(program.inputs, tuple(document['inputs']))
            self.assertEqual(program.outputs, tuple(document['outputs']))
            self.assertEqual(len(program.stages), 1)
            self.assertNotIn('mid', program.tensors)
            fused = program.document['stages'][0]
            output = fused['schedule']['outputs'][0]
            self.assertEqual(fused['bindings'][output], document['outputs'][0])
            self.assertNotEqual(output, document['outputs'][0])
            inputs = {'a': [0.0]*16, 'b': [0.0]*64,
                      'bias': [1.0 + 2**(-8 if dtype == 'bf16' else -11)]*8}
            middle, _ = execute(document['stages'][0]['schedule'], inputs)
            expected, _ = execute(document['stages'][1]['schedule'], middle)
            actual, _ = execute(fused['schedule'], inputs)
            self.assertEqual({fused['bindings'][k]: v for k, v in actual.items()}, expected)
            self.assertEqual(program.tensors[program.outputs[0]], Program.from_dict(document).tensors[program.outputs[0]])

    def test_public_intermediate_and_extra_consumer_have_their_own_refusals(self):
        document = epilogue_program()
        document['outputs'].append('mid')
        self.assertEqual(self.fuse(document).reason, 'public_intermediate')
        document = epilogue_program()
        third = deepcopy(document['stages'][1]); third['name'] = 'other_consumer'
        output = document['outputs'][0]
        third['bindings'][output] = 'other_output'
        document['tensors']['other_output'] = deepcopy(document['tensors'][output])
        document['outputs'].append('other_output')
        document['stages'].append(third)
        self.assertEqual(self.fuse(document).reason, 'intermediate_consumers')

    def test_original_pass_guards_and_explicit_selection_remain_authoritative(self):
        self.assertEqual(self.fuse(epilogue_program('fp32')).reason, 'rounding_boundary')
        self.assertEqual(self.fuse(epilogue_program(), producer='missing').reason, 'stage_selection')
        self.assertEqual(self.fuse(epilogue_program(), producer='epilogue', epilogue='producer').reason, 'stage_order')
        self.assertEqual(self.fuse(epilogue_program(), schedule_id='producer').reason, 'result_identity')
        program = Program.from_dict(epilogue_program())
        self.assertEqual(self.compiler.rewrite_program(program, 'nonexistent', {}).reason, 'unknown_transform')
        self.assertEqual(self.compiler.rewrite_program(program, 'specialize_triton_warps', {}).reason, 'transform_parameters')

    def test_specialization_preserves_composition_and_other_stage(self):
        document = epilogue_program()
        result = self.compiler.rewrite_program(Program.from_dict(document), 'specialize_triton_warps',
            {'stage': 'epilogue', 'num_warps': 8, 'schedule_id': 'epilogue_wide', 'entry_point': 'epilogue_wide'})
        self.assertTrue(result.applied, result.message)
        observed = result.program.document
        self.assertEqual(observed['stages'][0], document['stages'][0])
        self.assertEqual(observed['stages'][1]['bindings'], document['stages'][1]['bindings'])
        self.assertEqual(observed['stages'][1]['schedule']['roles'][0]['execution_groups'], list(range(8)))
        for name in ('inputs', 'outputs', 'tensors', 'target'):
            self.assertEqual(observed[name], document[name])

    def test_output_column_rescue_is_reachable_through_the_complete_program(self):
        from tests.contracts.test_output_column_specialization import wide_document
        program = Program.from_schedule(wide_document())
        stage_name = program.stages[0].name
        parameters = {'stage': stage_name, 'schedule_id': 'rescued_columns', 'entry_point': 'rescued_columns'}
        result = self.compiler.rewrite_program(program, 'specialize_output_columns', parameters)
        self.assertTrue(result.applied, result.message)
        self.assertEqual(result.program.inputs, program.inputs)
        self.assertEqual(result.program.outputs, program.outputs)
        self.assertTrue(self.compiler.assess(result.program.stages[0].schedule).lowering_eligible)
        self.compiler.lower_program(result.program)

        wrong = wide_document()
        wrong['lowering']['entry_point'] = 'float4'
        refused = self.compiler.rewrite_program(Program.from_schedule(wrong), 'specialize_output_columns', parameters)
        self.assertFalse(refused.applied)
        self.assertEqual(refused.reason, 'input_refused')
        self.assertIn('METAL_ENTRY_POINT_UNSUPPORTED', refused.message)

        original = program.document
        other = deepcopy(original['stages'][0]); other['name'] = 'unselected_wide'
        output = program.outputs[0]
        other['bindings'][output] = 'other_output'
        original['tensors']['other_output'] = deepcopy(original['tensors'][output])
        original['outputs'].append('other_output'); original['stages'].append(other)
        refused = self.compiler.rewrite_program(Program.from_dict(original), 'specialize_output_columns', parameters)
        self.assertFalse(refused.applied)
        self.assertEqual(refused.region, ('unselected_wide',))
        self.assertIn('METAL_PRIVATE_STORAGE_LIMIT', refused.message)

    def test_public_typed_input_cannot_bypass_program_legality(self):
        program = Program.from_dict(scalar_program())
        changed = replace(program, outputs=('x',))
        self.assertEqual(changed.document['outputs'], ['x'])
        with self.assertRaisesRegex(ValueError, 'inputs are immutable'):
            self.compiler.lower_program(changed)
        result = self.compiler.rewrite_program(changed, 'specialize_triton_warps',
            {'stage': 'first', 'num_warps': 2, 'schedule_id': 'wide', 'entry_point': 'wide'})
        self.assertFalse(result.applied)

    def test_stage_lowering_substitution_is_rejected_before_device_side_effects(self):
        compiled = self.compiler.lower_program(Program.from_dict(scalar_program()))
        for altered in (replace(compiled, lowerings=compiled.lowerings[::-1]),
                        replace(compiled, compiler_revision_id='unrelated'),
                        replace(compiled, lowerings=compiled.lowerings[:1])):
            def forbidden(*args, **kwargs):
                self.fail('an invalid code binding reached device preparation')
            with self.assertRaisesRegex(ValueError, 'binding differs|stage count'):
                prepare_program(altered, {}, allocate=forbidden, load_kernel=forbidden,
                    check_tensor=forbidden, storage_span=forbidden, execution_context=forbidden)

    def test_program_projection_is_immutable_and_single_schedule_is_ergonomic(self):
        schedule = stage('producer')
        program = Program.from_schedule(schedule)
        self.assertEqual(program.outputs, ('mid',))
        self.assertEqual(len(program.stages), 1)
        projection = program.document
        projection['stages'].clear()
        schedule['outputs'].clear()
        self.assertEqual(len(program.stages), 1)
        self.assertEqual(program.stages[0].schedule.outputs, ('mid',))
        with self.assertRaises(TypeError):
            program.stages[0].bindings['new'] = 'a'


@dataclass
class Tensor:
    shape: tuple
    dtype: object
    start: int
    nbytes: int


class ProgramViewTests(unittest.TestCase):
    def document(self):
        document = scalar_program()
        document['tensors']['x']['shape'] = [1, 4, 1]
        document['stages'][0]['bindings']['x'] = {'tensor': 'x', 'view': 'singleton_axes'}
        return document

    def test_singleton_binding_requires_explicit_shape_preserving_view(self):
        self.assertEqual(Program.from_dict(self.document()).tensors['x'].shape, (1, 4, 1))
        document = self.document();document['stages'][0]['bindings']['x'] = 'x'
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            Program.from_dict(document)
        document = self.document();document['tensors']['x']['shape'] = [2, 2]
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            Program.from_dict(document)
        document = self.document();document['tensors']['x']['shape'] = [4]
        with self.assertRaisesRegex(ValueError, 'unnecessary singleton'):
            Program.from_dict(document)

    def test_runtime_view_must_preserve_storage_not_make_a_copy(self):
        program = Program.from_dict(self.document())
        lowered = Compiler.load(ROOT, ROOT/'compiler/revision.json').lower_program(program)
        dtype = program.tensors['x'].dtype
        inputs = {'x': Tensor((1, 4, 1), dtype, 100, 16)}
        next_address = iter((200, 300))
        def allocate(name, spec):
            return Tensor(spec.shape, spec.dtype, next(next_address), spec.nbytes)
        def check(tensor, spec):
            if tensor.shape != spec.shape or tensor.dtype != spec.dtype:
                raise ValueError('tensor does not match')
        def prepare(view):
            nonlocal next_address
            next_address = iter((200, 300))
            return prepare_program(lowered, inputs, allocate=allocate, check_tensor=check,
                storage_span=lambda t: ('device', t.start, t.start + t.nbytes),
                execution_context=lambda: ('device', 'stream'),
                load_kernel=lambda name, lowering: lambda **kw: None, view_tensor=view)
        with self.assertRaisesRegex(ValueError, 'view adapter'):
            prepare(None)
        with self.assertRaisesRegex(ValueError, 'exact storage'):
            prepare(lambda t, shape: Tensor(shape, t.dtype, 400, t.nbytes))
        self.assertEqual(len(prepare(lambda t, shape: Tensor(shape, t.dtype, t.start, t.nbytes)).calls), 2)
