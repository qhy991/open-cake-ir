"""Public CPU regression boundaries for the September Compiler contract audit."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import operator
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, CompilerError, frontend
from open_cake_ir.compiler.backends import triton, cutedsl, cutedsl_register
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from tests.contracts import test_triton_loop_scopes as cpu
from tests.contracts.test_cutedsl_register import register_schedule
from tests.contracts.test_top_k_half_selection import _bitonic_merge_descending

ROOT = Path(__file__).resolve().parents[2]


def document(name):
    return json.loads((ROOT / 'corpus/schedules' / (name + '.json')).read_text())


def rename(value, old, new):
    if isinstance(value, dict): return {k: rename(v, old, new) for k, v in value.items()}
    if isinstance(value, list): return [rename(v, old, new) for v in value]
    return new if value == old else value


class CompilerIssueContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.target = Target.load(ROOT / 'compiler/targets/sm_100a.json')

    def refuses(self, doc, code, *, accepted):
        result = self.compiler.assess(doc)
        self.assertEqual(result.accepted, accepted, result.findings)
        self.assertFalse(result.lowering_eligible, result.findings)
        self.assertIn(code, {f.code for f in result.findings})
        with self.assertRaises(CompilerError): self.compiler.lower(result)
        return result

    def test_unsafe_python_identifiers_refuse_before_emission_in_all_routes(self):
        for backend, base in ((triton, document('relu-b8-smoke')),
                              (cutedsl, document('flash-kmeans-assignment-full')),
                              (cutedsl_register, register_schedule())):
            first = next(b['name'] for b in base['buffers'] if b['space'] == 'global' and b['mode'] == 'input')
            for name in ('in', 'out', 'tl', 'torch', 'cutlass', 'cute', 'N_FAKE', 'D_X_0', 'BLOCK_X', '_work_id', '__debug__'):
                with self.subTest(route=backend.__name__, name=name):
                    changed = rename(base, first, name)
                    code = 'CUTE_REGISTER_IDENTIFIER' if backend is cutedsl_register else 'BACKEND_IDENTIFIER_UNSAFE'
                    self.refuses(changed, code, accepted=True)
                    with self.assertRaises(ValueError): backend.emit(Schedule.from_dict(changed), Target.load(ROOT/f"compiler/targets/{changed['target']}.json"))
        for name in ('tensor', 'shape', 'dtype'):
            self.refuses(rename(document('relu-b8-smoke'), 'x', name), 'BACKEND_IDENTIFIER_UNSAFE', accepted=True)

    def test_output_out_is_valid_but_generated_namespace_aliases_are_refused(self):
        d = rename(document('relu-b8-smoke'), 'y', 'out')
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        compile(self.compiler.lower(assessment).source, '<safe-output>', 'exec')
        d = document('fma-b8-smoke')
        self.refuses(rename(d, 'b', 'A'), 'BACKEND_IDENTIFIER_COLLISION', accepted=True)
        d = document('relu-b8-smoke')
        self.refuses(rename(d, 'batch', 'x'), 'BACKEND_IDENTIFIER_COLLISION', accepted=True)

    def test_argmin_loop_owner_must_cover_the_actual_loaded_candidate_extent(self):
        d = document('triton-argmin-runtime-domain-drift')
        d['buffers'][0].pop('valid_extent')
        d['buffers'][0]['shape'][2] = 32
        d['buffers'].append({'name':'domain_owner', 'space':'global', 'dtype':'fp32',
                             'shape':[64], 'mode':'input'})
        d['tile_loops'][0].update(buffer='domain_owner', dimension=0)
        assessment = self.refuses(d, 'TRITON_ARGMIN_DOMAIN', accepted=True)
        self.assertEqual([f.path for f in assessment.findings if f.code == 'TRITON_ARGMIN_DOMAIN'],
                         ['operations[1].reads'])
        d['buffers'][0]['shape'][2] = 64
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        schedule = Schedule.from_dict(d)
        self.assertEqual(schedule.argmin_domain(schedule.operation('select')), 64)
        compile(self.compiler.lower(assessment).source, '<different-owner-same-domain>', 'exec')

    def test_actual_generated_binding_namespace_cannot_overwrite_authored_values(self):
        base = document('relu-b8-smoke')
        schedule = Schedule.from_dict(base)
        source = self.compiler.lower(self.compiler.assess(base)).source
        tree = ast.parse(source)
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_' + schedule.lowering.entry_point + '_kernel')
        authored = {buffer.name for buffer in schedule.buffers}
        authored.update(axis.name for axis in schedule.program_map.axes)
        generated = {node.id for node in ast.walk(kernel)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)} - authored
        self.assertIn('x_d1_offsets', generated)
        # Exercise every actual generated kernel binding, not a duplicate list of
        # temporary spellings. Both pointer and scratch collisions must refuse.
        scratch = next(buffer.name for buffer in schedule.buffers if buffer.space.value == 'register')
        for name in sorted(generated):
            for original in ('y', scratch):
                with self.subTest(generated=name, original=original):
                    changed = rename(base, original, name)
                    if original == 'y':
                        emitted = triton._TritonEmitter(Schedule.from_dict(changed), self.target, _namespace=False).emit()
                        current_kernel = next(node for node in ast.parse(emitted.source).body
                                              if isinstance(node, ast.FunctionDef) and node.name.startswith('_'))
                        if not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                                   and node.id == name for node in ast.walk(current_kernel)):
                            # A self-derived vector changes with its renamed owner;
                            # it is no longer a collision and must stay accepted.
                            self.assertTrue(self.compiler.assess(changed).lowering_eligible)
                            continue
                    self.refuses(changed, 'BACKEND_IDENTIFIER_COLLISION', accepted=True)
                    with self.assertRaises(ValueError):
                        triton.emit(Schedule.from_dict(changed), self.target)
        changed = rename(base, 'y', 'x_d1_offsets')
        # Inspect the emitter's private unchecked text to establish the actual
        # collision, while all public assess/lower/emit boundaries refuse it.
        raw = triton._TritonEmitter(Schedule.from_dict(changed), self.target, _namespace=False).emit().source
        kernel = next(node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
                      and node.name.startswith('_'))
        self.assertIn('x_d1_offsets', {arg.arg for arg in kernel.args.args})
        self.assertTrue(any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                            and node.id == 'x_d1_offsets' for node in ast.walk(kernel)))

    def test_actual_wrapper_builtin_namespace_and_entry_point_are_checked(self):
        from types import SimpleNamespace
        base = document('relu-b8-smoke')
        fake = SimpleNamespace(shape=(8, 128), dtype='fp32', is_cuda=True,
                               is_contiguous=lambda: True, device='cpu-fixture')
        for name in ('any', 'tuple', 'ValueError'):
            changed = rename(base, 'x', name)
            self.refuses(changed, 'BACKEND_IDENTIFIER_COLLISION', accepted=True)
        changed = rename(base, 'x', 'any')
        raw = triton._TritonEmitter(Schedule.from_dict(changed), self.target, _namespace=False).emit().source
        wrapper = next(node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
                       and node.name == changed['lowering']['entry_point'])
        env = {'torch': SimpleNamespace(float32='fp32')}
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual-wrapper>', 'exec'), env)
        with self.assertRaises(TypeError):
            env[wrapper.name](fake)
        changed = deepcopy(base)
        changed['lowering']['entry_point'] = 'any'
        self.refuses(changed, 'BACKEND_IDENTIFIER_COLLISION', accepted=True)
        with self.assertRaises(ValueError):
            triton.emit(Schedule.from_dict(base), self.target, entry_point='any')
        # A valid output named out keeps the public wrapper ABI and executes all
        # validation plus its launch callback with CPU objects only.
        changed = rename(base, 'y', 'out')
        lowered = self.compiler.lower(self.compiler.assess(changed))
        wrapper = next(node for node in ast.parse(lowered.source).body if isinstance(node, ast.FunctionDef)
                       and node.name == changed['lowering']['entry_point'])
        calls = []
        class Launch:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append((grid, args, kwargs))
        env = {'torch': SimpleNamespace(float32='fp32'), '_' + changed['lowering']['entry_point'] + '_kernel': Launch()}
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<valid-wrapper>', 'exec'), env)
        self.assertIs(env[wrapper.name](fake, out=fake), fake)
        self.assertEqual(len(calls), 1)

    def test_cute_generated_host_names_and_register_allocator_preserve_ownership(self):
        base = document('flash-kmeans-assignment-full')
        for name in ('tiled_mma', 'compiled', 'from_dlpack', 'tuple'):
            self.refuses(rename(base, 'tokens', name), 'BACKEND_IDENTIFIER_COLLISION', accepted=True)
        # The register route already owns a disjoint generated prefix allocator;
        # it can keep an authored pointer with the default internal prefix.
        base = register_schedule()
        first = next(buffer['name'] for buffer in base['buffers'] if buffer['space'] == 'global')
        changed = rename(base, first, '_cake_tid')
        assessment = self.compiler.assess(changed)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        kernel = next(node for node in ast.parse(self.compiler.lower(assessment).source).body
                      if isinstance(node, ast.FunctionDef))
        self.assertFalse(any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                             and node.id == '_cake_tid' for node in ast.walk(kernel)))

    def test_operation_control_characters_are_not_python_source(self):
        d = document('triton-operation-id-control-drift')
        self.refuses(d, 'BACKEND_IDENTIFIER_UNSAFE', accepted=True)
        with self.assertRaises(ValueError): triton.emit(Schedule.from_dict(d), self.target)

    def test_dimension_walk_definitions_do_not_depend_on_suffix_or_declaration_order(self):
        d = document('block-scaled-gemm-b1-smoke')
        for order in (list(reversed(d['access_maps'])), sorted(d['access_maps'], key=lambda a:len(a['buffer']), reverse=True)):
            changed = {**d, 'access_maps':order}
            assessment = self.compiler.assess(changed)
            self.assertTrue(assessment.lowering_eligible, assessment.findings)
            tree = ast.parse(self.compiler.lower(assessment).source)
            loads = {node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id.endswith('_offsets')}
            stores = [node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id.endswith('_offsets')]
            self.assertFalse(loads - set(stores))
            self.assertEqual(len(stores), len(set(stores)))

    def test_each_loop_carried_cat_preserves_order_through_the_public_api(self):
        seen = 0
        for path in (ROOT/'corpus/schedules').glob('*.json'):
            d=json.loads(path.read_text())
            if d.get('lowering',{}).get('backend') != 'triton' or not any(o['kind']=='top_k' and o['parameters'].get('across_loop') for o in d['operations']):continue
            try: emitted=triton.emit(Schedule.from_dict(d), Target.load(ROOT/f"compiler/targets/{d['target']}.json"))
            except ValueError:continue
            tree=ast.parse(emitted.source)
            for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl' and n.func.attr=='cat'):
                self.assertIs(next(k.value.value for k in call.keywords if k.arg=='can_reorder'), False)
                seen += 1
            assignments={n.targets[0].id:n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
            for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bitonic_merge'):
                expression=assignments[call.args[0].id]
                self.assertIsInstance(expression, ast.Call)
                self.assertEqual(ast.unparse(expression.func), 'tl.cat')
                self.assertIs(next(k.value.value for k in expression.keywords if k.arg=='can_reorder'), False)
                k=next(operation['parameters']['k'] for operation in d['operations']
                       if operation['kind']=='top_k' and operation['parameters'].get('across_loop'))
                state=list(range(k,0,-1)); source=list(range(k))
                class TL:
                    @staticmethod
                    def cat(a,b,*,can_reorder):
                        assert can_reorder is False
                        return a+b
                ordered=eval(compile(ast.Expression(expression),'<generated-bitonic-input>','eval'),
                    {'tl':TL,expression.args[0].id:state,expression.args[1].id:source})
                self.assertEqual(ordered,state+source)
                self.assertEqual(_bitonic_merge_descending(ordered),sorted(state+source,reverse=True))
        self.assertGreater(seen,0)

    def test_store_value_shape_matches_its_vector_address_domain(self):
        d=document('store-access-rank-drift')
        result=self.refuses(d,'STORE_ACCESS_SHAPE_MISMATCH',accepted=False)
        self.assertIn('operations[2].reads[0]',{f.path for f in result.findings if f.code=='STORE_ACCESS_SHAPE_MISMATCH'})
        # Restore the one-dimensional output access as the positive dual.
        d['buffers'][1]['shape']=[8,16]
        d['access_maps'][-1]['indices'].pop()
        self.assertTrue(self.compiler.assess(d).lowering_eligible)

    def test_dynamic_stop_refuses_unmasked_carry_side_effects_and_escaping_values(self):
        self.refuses(document('triton-loop-stop-mma-drift'),'TRITON_LOOP_STOP_UNSUPPORTED',accepted=True)
        base=document('top-k-streaming-merge2-exact-half-k256-b8-smoke')
        for add in (-1,0,1,514):
            d=deepcopy(base);d['tile_loops'][0]['stop']['add']=add
            a=self.compiler.assess(d);self.assertTrue(a.lowering_eligible,a.findings)
            self.assertIn('source_valid =',self.compiler.lower(a).source)
        d=deepcopy(base)
        # An extra global store is a side effect even if top-k still supplies the outputs.
        store=next(o for o in d['operations'] if o['kind']=='store')
        d['tile_loops'][0]['body'].append(store['id'])
        findings=triton.preflight(Schedule.from_dict(d),self.target)
        self.assertIn('TRITON_LOOP_STOP_UNSUPPORTED',{f.code for f in findings})
        d=deepcopy(base);load=next(o for o in d['operations'] if o['kind']=='load')
        d['operations'].append({'id':'escape','kind':'elementwise','role':'compute','reads':load['writes'],
            'writes':['escaped'],'parameters':{'op':'square'}})
        source=next(b for b in d['buffers'] if b['name']==load['writes'][0]);d['buffers'].append({**source,'name':'escaped'})
        findings=triton.preflight(Schedule.from_dict(d),self.target)
        self.assertIn('TRITON_LOOP_STOP_UNSUPPORTED',{f.code for f in findings})

    def test_runtime_and_static_tail_argmin_domains_are_explicit_refusals(self):
        for name in ('triton-argmin-runtime-domain-drift','triton-argmin-static-tail-drift'):
            d=document(name);self.refuses(d,'TRITON_ARGMIN_DOMAIN',accepted=True)
            with self.assertRaises(ValueError):triton.emit(Schedule.from_dict(d),Target.load(ROOT/'compiler/targets/sm_103a.json'))

    def test_argmin_runtime_domain_propagates_through_cast_arithmetic_and_mma(self):
        for kind, parameters in (("cast", {"to": "fp32"}), ("elementwise", {"op": "square"})):
            d=document('triton-argmin-runtime-domain-drift')
            if kind == 'cast':
                next(b for b in d['buffers'] if b['name']=='scores')['dtype']='fp16'
                next(b for b in d['buffers'] if b['name']=='tile')['dtype']='fp16'
            d['buffers'].append({'name':'transformed','space':'register','dtype':'fp32','shape':[2,32],'mode':'scratch'})
            d['operations'].insert(1,{'id':'transform','kind':kind,'role':'compute','reads':['tile'],
                'writes':['transformed'],'depends_on':['read'],'parameters':parameters})
            d['operations'][2]['reads']=['transformed']; d['operations'][2]['depends_on']=['transform']
            d['tile_loops'][0]['body'].insert(1,'transform')
            self.refuses(d,'TRITON_ARGMIN_DOMAIN',accepted=True)
            d['buffers'][0]['valid_extent']['dimension']=1
            self.assertTrue(self.compiler.assess(d).lowering_eligible)
        d=document('flash-kmeans-b32-smoke-v2')
        centroids=next(b for b in d['buffers'] if b['name']=='centroids')
        d['buffers'].append({'name':'lengths','space':'global','dtype':'int32','shape':[32],'mode':'input'})
        centroids['valid_extent']={'dimension':1,'buffer':'lengths','indexed_by':[0]}
        self.refuses(d,'TRITON_ARGMIN_DOMAIN',accepted=True)
        centroids['valid_extent']['dimension']=2
        self.assertTrue(self.compiler.assess(d).lowering_eligible)
        d=document('triton-argmin-runtime-domain-drift')
        d['buffers'][0]['valid_extent'].update(dimension=1,indexed_by=[2])
        d['buffers'][1]['shape']=[64]
        schedule=Schedule.from_dict(d)
        self.assertIsNone(schedule.argmin_domain(schedule.operation('select')))

    def test_argmin_orthogonal_prefix_and_positive_ties_use_full_candidate_tiles(self):
        d=document('triton-argmin-runtime-domain-drift');d['buffers'][0]['valid_extent']['dimension']=1
        assessment=self.compiler.assess(d);self.assertTrue(assessment.lowering_eligible,assessment.findings)
        emitted=triton.emit(Schedule.from_dict(d),Target.load(ROOT/'compiler/targets/sm_103a.json'))
        def argmin(tile,axis,tie_break_left):
            def select(values):
                values=list(values);return min(range(len(values)),key=lambda i:(values[i],i))
            return cpu._TL.reduce(tile,axis,select)
        def where(mask,left,right):return mask.binary(left.binary(right,lambda a,b:(a,b)),lambda m,p:p[0] if m else p[1])
        with patch.object(cpu._TL,'int32',object(),create=True),patch.object(cpu._TL,'argmin',staticmethod(argmin),create=True),patch.object(cpu._TL,'min',staticmethod(lambda a,axis:cpu._TL.reduce(a,axis,min)),create=True),patch.object(cpu._TL,'where',staticmethod(where),create=True),patch.object(cpu._Tile,'__eq__',lambda a,b:a.binary(b,operator.eq)),patch.object(cpu._Tile,'__or__',lambda a,b:a.binary(b,operator.or_),create=True):
            memories={'scores':[9,5,5,8]+[100]*60+[9,8,4,4]+[100]*60+[2,3,4,5]+[100]*60,'lengths':[3],'winner':[-1]*3}
            observed=cpu._execute(emitted,memories)
            self.assertEqual(memories['winner'],[1,2,0])

    def test_tensor_columns_and_mma_n_have_owned_hardware_refusals(self):
        self.refuses(document('tensor-columns-illegal-drift'),'ALLOCATION_TENSOR_COLUMNS_ILLEGAL',accepted=False)
        self.refuses(document('mma-tile-n-drift'),'MMA_TILE_INSTRUCTION_MISMATCH',accepted=False)
        for columns in (16,33,384,513):
            d=document('flash-kmeans-assignment-full');allocation=next(a for a in d['allocations'] if a['space']=='tensor')
            allocation.update(size_bytes=columns*512,tensor_columns=columns)
            self.refuses(d,'ALLOCATION_TENSOR_COLUMNS_ILLEGAL',accepted=False)
        for columns in (256,512):
            d=document('flash-kmeans-assignment-full');allocation=next(a for a in d['allocations'] if a['space']=='tensor')
            allocation.update(size_bytes=columns*512,tensor_columns=columns)
            self.assertNotIn('ALLOCATION_TENSOR_COLUMNS_ILLEGAL',{f.code for f in self.compiler.assess(d).findings})

    def test_residency_cap_uses_target_capacity_not_setmaxnreg_encoding(self):
        self.refuses(document('residency-register-cap-drift'),'TARGET_REGISTER_CAP_LIMIT',accepted=False)
        for registers in (16,31,255):
            d=document('cast-b8-smoke');d['residency']={'registers_per_thread':registers}
            a=self.compiler.assess(d);self.assertTrue(a.lowering_eligible,a.findings)
        self.assertEqual(self.target.resource_limits.maximum_registers_per_thread,255)

    def test_controlled_emit_refusal_has_a_stable_public_type_without_hiding_internal_errors(self):
        from open_cake_ir.compiler import LoweringRefusedError
        from open_cake_ir.compiler.backends.common import EmitError
        assessment=self.compiler.assess(document('relu-b8-smoke'))
        with patch.object(triton,'emit',side_effect=EmitError('unrepresented source commitment')):
            with self.assertRaises(LoweringRefusedError) as caught:self.compiler.lower(assessment)
        self.assertEqual(caught.exception.code,'LOWERING_UNDETERMINED')
        self.assertIsInstance(caught.exception.__cause__,EmitError)
        failure=CompilerError('internal compiler fault')
        with patch.object(triton,'emit',side_effect=failure):
            with self.assertRaises(CompilerError) as caught:self.compiler.lower(assessment)
        self.assertIs(caught.exception,failure)
        self.assertNotIsInstance(caught.exception,LoweringRefusedError)

    def test_cute_mutable_state_diagnostic_is_not_borrowed_from_unknown_vocabulary(self):
        d=document('atomic-reservation-b8-smoke')
        codes={f.code for f in cutedsl.preflight(Schedule.from_dict(d),self.target)}
        self.assertIn('CUTE_STATE_UNSUPPORTED',codes)
        self.assertIn('BACKEND_OPERATION_UNEMITTABLE',codes)


class FrontendIssueContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def test_cast_frontend_is_canonical_and_dtype_alias_is_not_a_second_spelling(self):
        source=(ROOT/'examples/python/cast.py').read_text()
        parsed=frontend.parse(source)
        self.assertEqual(Schedule.from_dict(parsed.document),Schedule.load(ROOT/'corpus/schedules/cast-b8-smoke.json'))
        self.assertTrue(self.compiler.assess(parsed.document).lowering_eligible)
        with self.assertRaises(frontend.FrontendError) as caught:
            frontend.parse(source.replace('to="fp32"','dtype="fp32"'))
        self.assertIn('to=',str(caught.exception))
        self.assertIsNotNone(caught.exception.canonical_path)

    def test_mixed_dtype_promotion_is_operand_order_independent_and_ambiguous_types_refuse(self):
        source=(ROOT/'examples/python/mixed_dtype.py').read_text()
        for expr in ('x_tile + bias_tile','bias_tile + x_tile'):
            parsed=frontend.parse(source.replace('x_tile + bias_tile',expr))
            self.assertEqual(next(b['dtype'] for b in parsed.document['buffers'] if b['name']=='result'),'fp32')
            self.assertTrue(self.compiler.assess(parsed.document).lowering_eligible)
        source=source.replace('bias: cake.Tensor((128,), "fp32")','bias: cake.Tensor((128,), "fp16")')
        with self.assertRaises(frontend.FrontendError) as caught:frontend.parse(source,filename='mixed.py')
        self.assertEqual(caught.exception.code,'ELEMENTWISE_DTYPE_UNSUPPORTED')
        self.assertEqual(caught.exception.location.line,source.splitlines().index('        result = x_tile + bias_tile')+1)
        self.assertIn('lm.cast',str(caught.exception))

    def test_diagnostics_point_to_decorator_output_argument_and_latest_program_declaration(self):
        source=(ROOT/'examples/python/mixed_dtype.py').read_text()
        unknown=frontend.parse(source.replace('target="sm_100a"','target="not_available"'))
        target=self.compiler.assess(unknown.document)
        finding=next(f for f in target.findings if f.code=='TARGET_UNSUPPORTED')
        self.assertEqual(unknown.location_for(finding.path).line,4)
        no_store=frontend.parse(source.replace('        lm.store(y[batch, :], result)',''))
        finding=next(f for f in self.compiler.assess(no_store.document).findings if f.code=='OUTPUT_UNWRITTEN')
        self.assertEqual(no_store.location_for(finding.path).line,7)
        repeated=source.replace('    with compute:', '    again = lm.program(x, axis=0, dimension=1, tile=1)\n    with compute:')
        parsed=frontend.parse(repeated)
        finding=next(f for f in self.compiler.assess(parsed.document).findings if f.code=='PROGRAM_AXIS_DUPLICATE_NUMBER')
        self.assertEqual(parsed.location_for(finding.path).line,10)
        with self.assertRaises(frontend.FrontendError) as caught:
            frontend.parse(source.replace('entry_point="cake_mixed_dtype"','entry_point="bad-name"'))
        self.assertEqual(caught.exception.location.line,5)
        self.assertEqual(caught.exception.canonical_path,'schedule.lowering.entry_point')
        self.assertIn('@cake.schedule(entry_point=',str(caught.exception))

    def test_single_reduction_scope_default_is_the_explicit_cta_form(self):
        source=(ROOT/'examples/python/softmax.py').read_text()
        explicit=frontend.parse(source)
        implicit=frontend.parse(source.replace(', scope="cta"','').replace('scope="cta", ',''))
        self.assertEqual(implicit.document,explicit.document)
