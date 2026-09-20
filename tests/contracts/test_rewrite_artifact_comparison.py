"""Input closure for sealed rewrite versus external implementation comparisons."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_rewrite_artifacts import regular, reference_spec, comparison_roles


class RewriteArtifactComparisonTests(unittest.TestCase):
    def test_new_schedule_and_fixed_control_are_not_labeled_as_confirmed_promotion(self):
        for control,label in [('optimized','unchanged old optimized binary'),
                              ('starter','unchanged original Cake starter')]:
            roles = comparison_roles({'kind':'authored_schedule_comparison','control_role':control})
            self.assertEqual(roles['starter'],label)
            self.assertEqual(roles['optimized'],'new authored complete Cake Schedule')
        with self.assertRaisesRegex(ValueError,'control role'):
            comparison_roles({'kind':'authored_schedule_comparison','control_role':'unknown'})
        self.assertEqual(comparison_roles({'kind':'explicit_alignment_ablation'})['starter'],
                         'unchanged pre-specialization optimized binary')

    def test_native_reference_preserves_declared_entry_and_source_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = root / 'reference'
            ref.mkdir()
            (ref / 'kernel.cu').write_text('// unchanged external source\n')
            document = {'kind': 'cuda_cpp', 'files': ['kernel.cu'],
                        'entry_point': 'kernel.cu::run', 'compile_options': {'cuda_cflags': ['-O3']}}
            path = ref / 'reference.json'
            path.write_text(json.dumps(document))
            spec, files, entry, function = reference_spec(root)
            self.assertEqual(spec, document)
            self.assertEqual(files, [ref / 'kernel.cu'])
            self.assertEqual((entry, function), ('kernel.cu', 'run'))
            for changes in ({'entry_point': 'missing.cu::run'}, {'files': ['kernel.cu', 'kernel.cu']},
                            {'files': ['../kernel.cu']}, {'kind': 'unknown'},
                            {'entry_point': 'kernel.cu::not-a-symbol'}):
                path.write_text(json.dumps({**document, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    reference_spec(root)

    def test_reference_files_cannot_be_redirected_outside_input_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'source.py').write_text('pass\n')
            (root / 'alias.py').symlink_to(root / 'source.py')
            for value in ('../source.py', str(root / 'source.py'), 'alias.py', 'a\\b.py'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    regular(root, value)
            self.assertEqual(regular(root, 'source.py'), root / 'source.py')


class CachedOutputObservationTests(unittest.TestCase):
    def pool(self, outputs, functions=None):
        from types import SimpleNamespace
        from tools.compare_rewrite_artifacts import RetainedExternal
        loaded = object.__new__(RetainedExternal)
        loaded.cached_output = True
        loaded.seen = set()
        loaded.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        loaded.launches = functions or [(lambda values, out, tensor=tensor: tensor) for tensor in outputs]
        loaded.launch_function = loaded.launches[0]
        return loaded

    def parent_patches(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from tools.compare_rewrite_artifacts import LoadedCallable
        stack = ExitStack()
        stack.enter_context(patch.object(LoadedCallable, 'fresh_argument_sets',
            lambda self, count: [{'inputs': {}, 'out': None, 'result': None} for _ in range(count)]))
        def launch(self, arguments):
            arguments['result'] = self.launch_function(arguments['inputs'], arguments['out'])
        stack.enter_context(patch.object(LoadedCallable, 'launch', launch))
        return stack

    def test_shared_buffers_and_changed_timed_buffer_are_refused(self):
        one, two = FakeOutput(1), FakeOutput(2)
        with self.parent_patches():
            with self.assertRaisesRegex(ValueError, 'share an output'):
                self.pool([one, one]).fresh_argument_sets(2)
            loaded = self.pool([one])
            arguments = loaded.fresh_argument_sets(1)
            loaded.launches = [lambda values, out: two]
            with self.assertRaisesRegex(ValueError, 'differs from poisoned'):
                loaded.launch(arguments[0])

    def test_no_write_cannot_reuse_correct_warm_output(self):
        import math
        from open_cake_ir.evaluation.core import compare_tile_outputs
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
        document, _ = create_task('rmsnorm', backend='triton-b300', rows=1, columns=1)
        workload = WorkloadContract(document)
        inputs = materialize_case(workload, 'primary')
        expected = reference_outputs(workload, 'primary', inputs)
        tensor = FakeOutput(1)
        tensor.value = expected['out'][0]
        with self.parent_patches():
            loaded = self.pool([tensor])
            arguments = loaded.fresh_argument_sets(1)
            self.assertTrue(math.isnan(tensor.value))
            loaded.launch(arguments[0])  # Reference returns its buffer but writes nothing.
            passed, _ = compare_tile_outputs(workload, inputs, expected,
                                             {'out': [arguments[0]['result'].value]}, inputs)
            self.assertFalse(passed)


class FakeOutput:
    def __init__(self, pointer):
        self.pointer = pointer
        self.value = 1.0

    def data_ptr(self):
        return self.pointer

    def fill_(self, value):
        self.value = value
