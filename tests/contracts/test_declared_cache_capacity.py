"""A measurement cache capacity is a Target fact, never a shared default."""
import json
from pathlib import Path
import unittest
from open_cake_ir.compiler.target import Target,TargetParseError
ROOT=Path(__file__).resolve().parents[2]
class DeclaredCacheCapacityTests(unittest.TestCase):
    def test_absence_remains_unknown_and_each_vendor_can_declare_its_own_value(self):
        for path in (ROOT/'compiler/targets').glob('*.json'):
            document=json.loads(path.read_text());document.pop('l2_cache_bytes',None)
            with self.subTest(target=path.stem):
                self.assertIsNone(Target.from_dict(document).l2_cache_bytes)
                self.assertEqual(Target.from_dict({**document,'l2_cache_bytes':8388608}).l2_cache_bytes,8388608)
    def test_invalid_or_null_capacities_are_refused_at_their_own_boundary(self):
        document=json.loads((ROOT/'compiler/targets/xcore1002.json').read_text())
        for value in (None,0,-1,True,'8388608',8388608.0):
            with self.subTest(value=value),self.assertRaisesRegex(TargetParseError,'l2_cache_bytes'):
                Target.from_dict({**document,'l2_cache_bytes':value})
