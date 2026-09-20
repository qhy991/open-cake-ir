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
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({row['id'] for row in rows}), 30)
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

    def test_paper_tasks_retain_full_domains_and_cannot_enter_default_launches(self):
        from tools import rewrite_assessment
        rows = rewrite_assessment.select(pack.catalog(), None)
        self.assertEqual([r['id'] for r in rows], [
            '027_cake_kda_prefill', '028_cake_kda_decode',
            '029_cake_tinygemm2', '030_cake_alphamoe'])
        self.assertEqual([len(r['assessment']['benchmark_rows']) for r in rows], [6, 30, 3, 4])
        self.assertEqual({r['tokens'] for r in rows[1]['assessment']['benchmark_rows']}, set(range(1, 7)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for row in rows:
                with self.subTest(task=row['id']), self.assertRaisesRegex(ValueError, 'not_integrated'):
                    pack.prepare_batch(pack.PACK / 'profiles/b300-m2.example.json', root / 'batch',
                                       root / 'runs', [row['id']])
                self.assertFalse((root / 'batch').exists())

    def test_assessment_packages_observed_witnesses_without_claiming_whole_task_pass(self):
        from tools import rewrite_assessment
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'assessment'
            with patch.object(pack.kernel_experiment, 'run_cell', side_effect=AssertionError('must not launch')):
                report = rewrite_assessment.prepare(pack.ROOT, pack.catalog(), output, None, pack.reference_path)
            self.assertEqual(report['task_count'], 4)
            self.assertEqual(report['scope'], 'component_source_lowering_only')
            # A working state primitive on one route must not mask another route's refusal.
            self.assertEqual(report['probes']['sm_100a/state_triton']['status'], 'source_lowered')
            cute_state = report['probes']['sm_100a/state_cute']
            self.assertEqual(cute_state['status'], 'refused')
            self.assertIn('CUTE_STATE_UNSUPPORTED', [f['code'] for f in cute_state['findings']])
            for row in report['tasks']:
                self.assertEqual(row['whole_task'], 'not_evaluated')
                self.assertEqual(row['gpu_correctness'], 'not_run')
                task = output / 'tasks' / row['id']
                spec = json.loads((task / 'specification.json').read_text())
                self.assertIn('known_kernel_reproduction', (task / 'TASK.md').read_text())
                self.assertEqual((task / 'AGENTS.md').read_bytes(),
                                 (pack.ROOT / 'contracts/scaffolds/kernel-reproduction/AGENTS.md').read_bytes())
                for ref in spec['references']:
                    self.assertEqual((task / ref).read_bytes(), pack.reference_path(ref).read_bytes())
                for key in row['probe_keys']:
                    probe = report['probes'][key]
                    self.assertTrue((output / probe['input']).is_file())
                    if probe['status'] == 'source_lowered':
                        self.assertTrue((output / probe['emitted_source']).read_text())
            with self.assertRaisesRegex(ValueError, 'new canonical directory'):
                rewrite_assessment.prepare(pack.ROOT, pack.catalog(), output, None, pack.reference_path)
            with self.assertRaisesRegex(ValueError, 'no capability-assessment'):
                rewrite_assessment.select(pack.catalog(), ['001_fused_add_rmsnorm_h2048'])
