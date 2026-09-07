"""Existing indexed-load semantics must survive Python elaboration."""
from pathlib import Path
import unittest

from open_cake_ir.compiler.frontend import parse, FrontendError

ROOT = Path(__file__).resolve().parents[2]


class PythonGatherTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'examples/python/b300_indexed_gather.py').read_text()

    def gather(self, source):
        document = parse(source).document
        operation = next(op for op in document['operations'] if op['id'] == 'load_selected_rows')
        result = next(buffer for buffer in document['buffers'] if buffer['name'] == 'gathered_tile')
        return document, operation, result

    def explicit_destination(self, source):
        return source.replace('gathered_tile = lm.load(',
            'gathered_tile = lm.buffer(dtype="bf16", shape=(8, 16))\n        lm.load(').replace(
            'reuse="streamed", id="load_selected_rows")',
            'reuse="streamed", out=gathered_tile, id="load_selected_rows")')

    def test_zipped_domain_reads_and_dependency_edges_match_json(self):
        import json
        actual, operation, result = self.gather(self.source)
        expected = json.loads((ROOT / 'corpus/schedules/indexed-gather-b8-smoke-b300.json').read_text())
        self.assertEqual(operation, next(op for op in expected['operations'] if op['id'] == operation['id']))
        self.assertEqual(actual['access_maps'], expected['access_maps'])
        self.assertEqual(result['shape'], [8, 16])
        self.assertEqual(result['dtype'], 'bf16')

    def test_repeated_index_is_one_read_and_one_domain(self):
        _, operation, result = self.gather(self.source.replace(
            'expert_rows[expert_id_tile, row_id_tile, :]', 'expert_rows[expert_id_tile, expert_id_tile, :]'))
        self.assertEqual(operation['reads'], ['expert_rows', 'expert_id_tile'])
        self.assertEqual(operation['depends_on'], ['load_expert_ids'])
        self.assertEqual(result['shape'], [8, 16])

    def test_domain_position_and_single_element_are_preserved(self):
        _, _, result = self.gather(self.source.replace(
            'expert_rows[expert_id_tile, row_id_tile, :]', 'expert_rows[:, expert_id_tile, row_id_tile]'))
        self.assertEqual(result['shape'], [4, 8])
        _, _, result = self.gather(self.source.replace('((8, 8), "int32")', '((8, 1), "int32")'))
        self.assertEqual(result['shape'], [1, 16])

    def test_explicit_destination_keeps_implicit_reads_and_effect_order(self):
        source = self.explicit_destination(self.source).replace(
            '        lm.store(gathered_rows',
            '        lm.load(expert_ids[token, :], out=expert_id_tile, id="overwrite_index")\n'
            '        lm.store(gathered_rows')
        document, operation, _ = self.gather(source)
        self.assertEqual(operation['reads'], ['expert_rows', 'expert_id_tile', 'row_id_tile'])
        overwrite = next(op for op in document['operations'] if op['id'] == 'overwrite_index')
        self.assertEqual(overwrite['depends_on'], ['load_selected_rows'])

    def test_index_type_storage_rank_and_domain_fail_before_ir_emission(self):
        malformed = (
            self.source.replace('expert_ids: cake.Tensor((8, 8), "int32")',
                                'expert_ids: cake.Tensor((8, 8), "fp32")'),
            self.source.replace('expert_rows[expert_id_tile, row_id_tile, :]',
                                'expert_rows[expert_ids, row_id_tile, :]'),
            self.source.replace('expert_id_tile = lm.load(expert_ids[token, :], reuse="streamed", id="load_expert_ids")',
                                'expert_id_tile = lm.buffer(dtype="int32", shape=(2, 4))'),
            self.source.replace('row_ids: cake.Tensor((8, 8), "int32")',
                                'row_ids: cake.Tensor((8, 4), "int32")'),
        )
        for index, source in enumerate(malformed):
            for explicit in (False, True):
                with self.subTest(case=index, explicit=explicit), self.assertRaisesRegex(FrontendError, 'gather'):
                    parse(self.explicit_destination(source) if explicit else source)

    def test_buffer_indices_cannot_silently_enable_indexed_stores(self):
        source = self.source.replace('lm.store(gathered_rows[token, :, :]',
                                    'lm.store(expert_rows[expert_id_tile, row_id_tile, :]')
        with self.assertRaisesRegex(FrontendError, 'buffer-indexed destinations'):
            parse(source)


if __name__ == '__main__':
    unittest.main()
