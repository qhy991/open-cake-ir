"""Measured empty-owner steal coverage regression."""
import unittest

from model_ranked_tile_pointer_run import validate_stolen


class PointerStealCoverageTest(unittest.TestCase):
    def test_empty_owners_cannot_steal_and_need_no_coverage(self):
        validate_stolen([5888,0,0,0],5888,[16384,0,0,0])
        with self.assertRaisesRegex(ValueError,'nonempty-owner'):
            validate_stolen([0,0,0,0],5888,[16384,0,0,0])
        with self.assertRaisesRegex(ValueError,'nonempty-owner'):
            validate_stolen([5888,1,0,0],5888,[16384,0,0,0])

    def test_per_rank_cap_and_no_steal_control(self):
        validate_stolen([0,0,0,0],0,[4096]*4)
        validate_stolen([5888,0,5888,0],[5888,0,5888,0],[4096]*4)
        with self.assertRaisesRegex(ValueError,'cap'):
            validate_stolen([5889,0,0,0],5888,[16384,0,0,0])
        with self.assertRaisesRegex(ValueError,'coverage'):
            validate_stolen([5888,1,5888,0],[5888,0,5888,0],[4096]*4)


if __name__ == '__main__':
    unittest.main()
