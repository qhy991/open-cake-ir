"""A third native adapter must not enter either incumbent implementation."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.lab import pairing


class NativeAdapterDispatchTests(unittest.TestCase):
    def test_registered_third_adapter_owns_all_factories_and_payload_projection(self):
        adapter = Mock(spec=pairing.NativeAdapter)
        isolated, builder, environment = object(), object(), object()
        adapter.isolated_compiler.return_value = isolated
        adapter.builder.return_value = builder
        adapter.environment.return_value = environment
        adapter.source.return_value = b'admitted fixture source'
        adapter.block.return_value = [7, 1, 1]
        adapter.baseline.return_value = {'fixture_launch': 7}
        row = replace(pairing._NATIVE_BACKENDS[0], backend='fixture', arm='native_fixture',
                      label='Fixture', adapter=adapter)
        requirements = {'compiler': 'fixture', 'fixture_launch': 7}
        poison = Mock(spec=pairing.NativeAdapter)
        for method in ('source', 'block', 'baseline', 'isolated_compiler', 'builder', 'environment'):
            getattr(poison, method).side_effect = AssertionError('entered an incumbent adapter')
        incumbents = tuple(replace(item, adapter=poison) for item in pairing._NATIVE_BACKENDS)
        with patch.object(pairing, '_NATIVE_BACKENDS', (*incumbents, row)):
            self.assertIs(pairing.backend_policy('fixture'), row)
            self.assertIs(pairing.native_backend('native_fixture'), row)
            self.assertEqual(pairing.comparison_arm({'open_cake': {}, 'native_fixture': {}}), 'native_fixture')
            self.assertIs(row.isolated_compiler({'configured': True}), isolated)
            self.assertIs(row.builder(workload='workload', case_id='case', isolated_compiler=isolated), builder)
            self.assertIs(row.environment(builder, authority_document={'fixture': True}), environment)
            self.assertEqual(pairing.native_block(requirements, warp_size=64), [7, 1, 1])
            projected = pairing.native_baseline(SimpleNamespace(source='raw', toolchain_requirements=requirements))
            self.assertEqual(projected, {'fixture_launch': 7})
            adapter.source.assert_called_once_with(b'raw', requirements)
            adapter.baseline.assert_called_once_with('admitted fixture source', requirements)
            adapter.block.assert_called_once_with(requirements, warp_size=64)
            adapter.isolated_compiler.assert_called_once_with({'configured': True})
            adapter.builder.assert_called_once_with(workload='workload', case_id='case', isolated_compiler=isolated)
            adapter.environment.assert_called_once_with(builder, authority_document={'fixture': True})
            with self.assertRaisesRegex(ValueError, 'unsupported paired lowering backend'):
                pairing.native_source(b'raw', {'compiler': 'unregistered'})

    def test_factory_symbols_remain_late_bound_in_their_own_modules(self):
        for backend, module, symbol in (
            ('triton', 'triton_build', 'IsolatedTritonCompiler'),
            ('cutlass_cute_dsl', 'cute_build', 'IsolatedCuTeCompiler'),
        ):
            with self.subTest(backend=backend), patch(f'open_cake_ir.lab.{module}.{symbol}') as factory:
                result = pairing.backend_policy(backend).isolated_compiler({'fixture': True})
                factory.assert_called_once_with(fixture=True)
                self.assertIs(result, factory.return_value)

    def test_source_refusal_precedes_baseline_projection(self):
        adapter = Mock(spec=pairing.NativeAdapter)
        adapter.source.side_effect = ValueError('source refused')
        row = replace(pairing._NATIVE_BACKENDS[0], adapter=adapter)
        with patch.object(pairing, '_NATIVE_BACKENDS', (row,)):
            with self.assertRaisesRegex(ValueError, 'source refused'):
                pairing.native_baseline(SimpleNamespace(source='raw', toolchain_requirements={'compiler': row.backend}))
        adapter.baseline.assert_not_called()
