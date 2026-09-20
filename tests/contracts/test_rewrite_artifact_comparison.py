"""Input closure for sealed rewrite versus external implementation comparisons."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_rewrite_artifacts import regular, reference_spec


class RewriteArtifactComparisonTests(unittest.TestCase):
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
