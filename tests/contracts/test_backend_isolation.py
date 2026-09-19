"""Changing one backend's policy must not alter another backend's admission."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler.backends import BACKENDS, triton, cutedsl
from open_cake_ir.compiler.ir import LoweringBackend

ROOT = Path(__file__).resolve().parents[2]


def document(name):
    return json.loads((ROOT / 'corpus/schedules' / f'{name}.json').read_text())


def rename(value, old, new):
    if isinstance(value, dict):
        return {key: rename(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [rename(item, old, new) for item in value]
    return new if value == old else value


class BackendPolicyIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_foreign_backend_names_are_usable_in_generated_python(self):
        for base, old, names in (
            ('fma-b8-smoke', 'a', ('cute', 'cutlass')),
            ('flash-kmeans-assignment-full', 'tokens', ('tl', 'triton')),
        ):
            for name in names:
                with self.subTest(base=base, name=name):
                    assessment = self.compiler.assess(rename(document(base), old, name))
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    source = self.compiler.lower(assessment).source
                    compile(source, '<isolated namespace>', 'exec')
                    self.assertTrue(any(isinstance(node, ast.arg) and node.arg == name
                                        for node in ast.walk(ast.parse(source))))

    def test_changing_cute_namespace_does_not_change_triton_admission(self):
        triton_doc = rename(document('fma-b8-smoke'), 'a', 'adapter_value')
        cute_doc = rename(document('flash-kmeans-assignment-full'), 'tokens', 'adapter_value')
        self.assertTrue(self.compiler.assess(cute_doc).lowering_eligible)
        changed = replace(cutedsl.PYTHON_NAMESPACE,
                          reserved_names=cutedsl.PYTHON_NAMESPACE.reserved_names | {'adapter_value'})
        with patch.object(cutedsl, 'PYTHON_NAMESPACE', changed):
            rejected = self.compiler.assess(cute_doc)
            self.assertTrue(rejected.accepted)
            self.assertFalse(rejected.lowering_eligible)
            self.assertIn('BACKEND_IDENTIFIER_UNSAFE', [f.code for f in rejected.findings])
            admitted = self.compiler.assess(triton_doc)
            self.assertTrue(admitted.lowering_eligible, admitted.findings)
            self.compiler.lower(admitted)

    def test_backend_input_contract_is_selected_by_registration(self):
        triton_doc = document('fma-b8-smoke')
        metal_doc = document('metal-elementwise-odd')
        def refuses_input(document):
            from open_cake_ir.compiler.backends.common import EmitError
            raise EmitError('synthetic backend input restriction')
        registry = dict(BACKENDS)
        registry[LoweringBackend.TRITON] = replace(registry[LoweringBackend.TRITON], validate_input=refuses_input)
        with patch('open_cake_ir.compiler.core.BACKENDS', registry):
            with self.assertRaisesRegex(CompilerError, 'synthetic backend input restriction'):
                self.compiler.assess(triton_doc)
            self.assertTrue(self.compiler.assess(metal_doc).lowering_eligible)
        # The list-only rule is Triton's, not the typed IR's or the facade's.
        looping = document('flash-kmeans-b32-smoke-v2')
        looping['tile_loops'][0]['body'] = tuple(looping['tile_loops'][0]['body'])
        with self.assertRaisesRegex(CompilerError, r'tile_loops\[0\].body must be a list'):
            self.compiler.assess(looping)
        registry[LoweringBackend.TRITON] = replace(BACKENDS[LoweringBackend.TRITON], validate_input=None)
        with patch('open_cake_ir.compiler.core.BACKENDS', registry):
            self.assertTrue(self.compiler.assess(looping).lowering_eligible)
