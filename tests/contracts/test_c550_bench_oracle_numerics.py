"""The external oracle keeps its observed math policy and its original comparator."""
import copy
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.tasks.c550_bench import binding
from open_cake_ir.lab.task_package import TaskPackage, _document_sections, render_task_request
from tests.contracts.test_c550_bench_binding import problem


def policy():
    return {'float32_matmul_precision': 'high', 'allow_tf32': True,
        'allow_fp16_reduced_precision_reduction': True,
        'allow_bf16_reduced_precision_reduction': True,
        'initialization': {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}}


def torch_fixture(state):
    matmul = SimpleNamespace(**{key: state[key] for key in state if key.startswith('allow_')})
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
        get_float32_matmul_precision=lambda: state['float32_matmul_precision'])
    return torch


class OracleNumerics(unittest.TestCase):
    def test_workload_freezes_explicit_policy_without_observing_the_preparation_host(self):
        observed = policy()
        with patch.dict(sys.modules, {'torch': None}):
            doc = problem().workload_document('original-7', oracle_numerics=observed)
        self.assertEqual(doc['semantics']['oracle_numerics'], observed)
        observed['initialization']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = None
        self.assertEqual(doc['semantics']['oracle_numerics']['initialization']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'], '1')
        self.assertIn('FP32 storage', doc['semantics']['oracle_numerics_contract'])
        with self.assertRaises(TypeError):
            problem().workload_document('original-7')

    def test_original_defect_changed_global_precision_refuses_without_resetting_it(self):
        state = policy()
        torch = torch_fixture(state)
        with patch.dict(sys.modules, {'torch': torch}), patch.dict(os.environ, {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}):
            binding.require_oracle_numerics(policy(), phase='before_reference')
            state['float32_matmul_precision'] = 'highest'
            with self.assertRaisesRegex(ValueError, 'before_reference.*float32_matmul_precision'):
                binding.require_oracle_numerics(policy(), phase='before_reference')
            self.assertEqual(state['float32_matmul_precision'], 'highest')

    def test_each_effective_setting_and_initialization_source_is_checked(self):
        for field in ('allow_tf32', 'allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction', 'initialization'):
            with self.subTest(field=field):
                state = policy(); torch = torch_fixture(state)
                env = {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}
                if field == 'initialization':
                    env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
                else:
                    setattr(torch.backends.cuda.matmul, field, False)
                with patch.dict(sys.modules, {'torch': torch}), patch.dict(os.environ, env):
                    with self.assertRaisesRegex(ValueError, 'before_input_factory.*' + field):
                        binding.require_oracle_numerics(policy(), phase='before_input_factory')

    def test_policy_is_closed_typed_data_not_an_ambient_default(self):
        for mutate in (
            lambda x: x.pop('allow_tf32'), lambda x: x.update(extra='ignored'),
            lambda x: x.update(allow_tf32=1), lambda x: x.update(float32_matmul_precision='default'),
            lambda x: x['initialization'].update(other='ignored'),
            lambda x: x['initialization'].update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=True),
        ):
            invalid = policy(); mutate(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                binding.validate_oracle_numerics(invalid)

    def test_frozen_policy_reaches_the_actual_task_request_and_evidence(self):
        doc = problem().workload_document('original-7', oracle_numerics=policy())
        markdown = _document_sections({'workload.json': json.dumps(doc).encode()}, access='known_kernel_reproduction')
        package = TaskPackage('run', 'author', markdown, 'Use the frozen Workload.')
        prompt, retained = render_task_request(package, {})
        delivered = json.loads(retained)['task_markdown']
        self.assertEqual(delivered, markdown)
        self.assertIn(json.dumps(markdown), prompt)
        for key, value in policy().items():
            self.assertIn(json.dumps(key), markdown)
            self.assertIn(json.dumps(value), markdown)
        self.assertIn('FP32 storage', delivered)

    def test_document_validation_preserves_policy_and_refuses_missing_policy(self):
        original = problem()
        doc = original.workload_document('original-7', oracle_numerics=policy())
        with patch.object(binding.BenchProblem, 'open', return_value=original):
            binding.validate_document(doc)
            del doc['semantics']['oracle_numerics']
            with self.assertRaisesRegex(ValueError, 'oracle numerics'):
                binding.validate_document(doc)

    def _prepare(self, *, drift=None):
        original = problem(); state = policy(); torch = torch_fixture(state); events = []
        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return self
            def is_contiguous(self): return True
        torch.Tensor = Tensor
        torch.cuda = SimpleNamespace(synchronize=lambda device: events.append('synchronize'))
        arguments = (Tensor(), Tensor(), .5)
        def phase(name):
            events.append(name)
            if drift == name:
                state['float32_matmul_precision'] = 'highest'
        def generate(*args, **kwargs):
            phase('factory'); return arguments
        def reference(*args):
            phase('reference'); return arguments[0]
        original.api.load_module = lambda *args: SimpleNamespace(run=reference)
        original.api.cloned_inputs = lambda inputs: inputs
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                'torch': torch,
                'sol_execbench.core.bench.correctness': SimpleNamespace(set_seed=lambda seed: None),
                'sol_execbench.core.bench.io': SimpleNamespace(gen_inputs=generate,
                    normalize_outputs=lambda output, **kwargs: {'out': output}),
                'sol_execbench.core.data.dtypes': SimpleNamespace(dtype_str_to_torch_dtype=lambda dtype: dtype),
            }))
            stack.enter_context(patch.dict(os.environ, {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}))
            stack.enter_context(patch('open_cake_ir.evaluation.triton_metax.observe_local_metax', return_value='lease'))
            def view(value, order):
                if drift == 'before_reference': state['float32_matmul_precision'] = 'highest'
                return value
            stack.enter_context(patch.object(binding, 'physical_input_view', side_effect=view))
            if drift == 'entry': state['float32_matmul_precision'] = 'highest'
            if drift:
                expected_phase = {'entry': 'before_input_factory', 'factory': 'after_input_factory', 'reference': 'after_reference', 'before_reference': 'before_reference'}[drift]
                with self.assertRaisesRegex(ValueError, expected_phase):
                    original.prepare_on_target('original-7', runtime_library='/not-loaded', oracle_numerics=policy())
            else:
                inputs, outputs, admission = original.prepare_on_target('original-7', runtime_library='/not-loaded', oracle_numerics=policy())
                self.assertEqual(admission, 'lease')
                self.assertEqual(set(inputs), {'positions', 'mask'})
                self.assertEqual(set(outputs), {'out'})
        return events

    def test_common_preparation_checks_real_factory_and_reference_boundaries(self):
        self.assertEqual(self._prepare(), ['factory', 'reference', 'synchronize'])
        self.assertEqual(self._prepare(drift='entry'), [])
        self.assertEqual(self._prepare(drift='factory'), ['factory'])
        self.assertEqual(self._prepare(drift='reference'), ['factory', 'reference'])
        self.assertEqual(self._prepare(drift='before_reference'), ['factory'])


    def test_preparation_cli_uses_retained_observation_for_all_original_workloads(self):
        from tools import prepare_c550_bench as entry
        original = problem()
        original.api.document = lambda name: {'seed': 200, 'tasks': [{'id': 'fixture'}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            observation = root / 'oracle-numerics.json'
            observation.write_text(json.dumps(policy()))
            output = root / 'prepared'
            with patch.object(entry.BenchProblem, 'open', return_value=original), patch.dict(sys.modules, {'torch': None}):
                entry.main(['--bench-root', str(root), '--output', str(output),
                    '--task', 'fixture', '--oracle-numerics', str(observation)])
            prepared = json.loads((output / 'prepared.json').read_text())
            self.assertEqual(prepared['oracle_numerics_observation'], str(observation.resolve()))
            cases = sorted((output / 'fixture').glob('case-*/workload.json'))
            self.assertEqual(len(cases), 16)
            for path in cases:
                self.assertEqual(json.loads(path.read_text())['semantics']['oracle_numerics'], policy())


    def test_observation_path_refuses_source_trees_links_and_relative_paths_before_preparation(self):
        from tools import prepare_c550_bench as entry
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            observation = root / 'observation.json'
            observation.write_text(json.dumps(policy()))
            link = root / 'link.json'; link.symlink_to(observation)
            checkout = root / 'another-checkout'; checkout.mkdir(); (checkout / '.git').mkdir()
            tracked = checkout / 'observation.json'; tracked.write_text(observation.read_text())
            for path in (link, tracked, Path('relative.json')):
                with self.subTest(path=path), patch.object(entry.BenchProblem, 'open') as opened:
                    with self.assertRaisesRegex(ValueError, 'external|source worktrees'):
                        entry.main(['--bench-root', str(root), '--output', str(root / 'unused'),
                            '--oracle-numerics', str(path)])
                    opened.assert_not_called()
            self.assertFalse((root / 'unused').exists())

    def test_actual_torch_precision_drift_is_read_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is absent; the accepted Linux CPU gate exercises this contract')
        saved = torch.get_float32_matmul_precision()
        expected = binding.observe_oracle_numerics()
        other = 'highest' if saved != 'highest' else 'high'
        try:
            torch.set_float32_matmul_precision(other)
            with self.assertRaisesRegex(ValueError, 'float32_matmul_precision'):
                binding.require_oracle_numerics(expected, phase='before_reference')
            self.assertEqual(torch.get_float32_matmul_precision(), other)
        finally:
            torch.set_float32_matmul_precision(saved)


if __name__ == '__main__':
    unittest.main()
