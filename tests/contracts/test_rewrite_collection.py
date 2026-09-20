"""Packaged references reach actual authoring inputs without local collection paths."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import rewrite_collection as pack


class RewriteCollectionTests(unittest.TestCase):
    def test_catalog_accounts_for_every_task_and_admits_only_ready_sources(self):
        rows = pack.catalog()
        self.assertEqual(len(rows), 26)
        self.assertEqual(len({row['id'] for row in rows}), 26)
        ready = pack.select_tasks(None)
        self.assertEqual(len(ready), 17)
        self.assertEqual(sum(r['reference_kind'] == 'library_dispatch' for r in ready), 1)
        for row in rows:
            self.assertEqual(row['reference_status'], 'available')
            self.assertTrue(row['source'])
            for ref in row['references']:
                self.assertTrue(pack.reference_path(ref).is_file())
        for row in rows:
            if row['status'] != 'ready':
                with self.subTest(task=row['id']), self.assertRaises(ValueError):
                    pack.select_tasks([row['id']])
        with self.assertRaises(ValueError):
            pack.reference_path('../README.md')
        self.assertEqual(next(r for r in ready if r['id'].startswith('010'))['references'][-1],
                         'references/010_gemm_n6144_k4096/cudallm_iter_1_sample_1.cu')

    def test_all_ready_tasks_prepare_with_frozen_source_and_delivered_instructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / 'inputs'
            profile = pack.PACK / 'profiles/b300-m2.example.json'
            manifest = pack.prepare_batch(profile, workspace, root / 'runs')
            self.assertEqual(len(manifest['tasks']), 17)
            for row in pack.select_tasks(None):
                task = workspace / 'tasks' / row['id']
                config = json.loads((task / 'experiment.json').read_text())
                self.assertEqual(config['source_commit'], manifest['source_commit'])
                self.assertEqual(config['cells'][0]['node']['project_root'], str(pack.ROOT))
                scaffold = (task / 'scaffold.md').read_text()
                self.assertIn(row['objective'], scaffold)
                self.assertIn(row['source'], scaffold)
                self.assertIn((pack.PACK / 'AUTHORING.md').read_text(), scaffold)
                self.assertIn('structurally distinct', scaffold)
                for ref in row['references']:
                    # References are JSON-quoted by the canonical preparation owner.
                    self.assertIn(json.dumps(pack.reference_path(ref).read_text(), ensure_ascii=False), scaffold)
            first = manifest['tasks'][0]['id']
            with patch.object(pack.kernel_experiment, 'run_cell', return_value=255) as run:
                self.assertEqual(pack.run_batch(workspace), 255)
                run.assert_called_once_with(workspace / 'tasks' / first, first.replace('_', '-'))
            with self.assertRaises(FileExistsError):
                pack.prepare_batch(profile, workspace, root / 'runs')
            with patch.object(pack, 'checkout_commit', return_value='different commit'):
                with self.assertRaisesRegex(ValueError, 'clean source commit'):
                    pack.read_batch(workspace)

    def test_skipped_task_fails_before_creating_any_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, 'authoring_route_not_integrated'):
                pack.prepare_batch(pack.PACK / 'profiles/b300-m2.example.json', root / 'inputs',
                                   root / 'runs', ['020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048'])
            self.assertFalse((root / 'inputs').exists())
