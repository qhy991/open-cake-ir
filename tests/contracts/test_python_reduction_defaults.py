"""Familiar Python spellings retain one canonical reduction operation."""
from pathlib import Path
import unittest
from open_cake_ir.compiler.frontend import FrontendError, parse
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError

ROOT=Path(__file__).resolve().parents[2]

class PythonReductionDefaults(unittest.TestCase):
    def test_explicit_default_and_negative_axis_normalize_without_changing_ir(self):
        source=(ROOT/'examples/python/softmax.py').read_text()
        expected=parse(source).document
        for spelled in (source.replace('scope="cta",','scope="cta", across_loop=True,'),
                        source.replace('axis=1, scope="cta"','axis=-1, scope="cta"'),
                        source.replace('axis=1, scope="cta"','axis=-1, scope="cta", across_loop=True')):
            self.assertEqual(parse(spelled).document,expected)
        canonical=dict(expected)
        canonical['operations'][1]['parameters']['across_loop']=True
        with self.assertRaises(ScheduleParseError):Schedule.from_dict(canonical)

    def test_false_retains_local_fold_and_invalid_axes_are_refused(self):
        source=(ROOT/'examples/python/softmax.py').read_text()
        observed=parse(source.replace('scope="cta",','scope="cta", across_loop=False,')).document
        self.assertFalse(observed['operations'][1]['parameters']['across_loop'])
        for axis in ('-3','2','True','1.0'):
            with self.assertRaises(FrontendError):parse(source.replace('axis=1, scope="cta"',f'axis={axis}, scope="cta"'))

    def test_explicit_output_uses_the_same_normalization_before_analysis(self):
        source=(ROOT/'examples/python/softmax.py').read_text()
        # Introduce a named destination without changing its buffer contract.
        line='        rowmax = lm.reduce(x_tile, op="max", axis=1, scope="cta", id="row_max")'
        replacement='        rowmax = lm.buffer(dtype="fp32", shape=(64,))\n        lm.reduce(x_tile, op="max", axis=1, scope="cta", out=rowmax, id="row_max")'
        explicit=source.replace(line,replacement)
        expected=parse(explicit).document
        equivalent=explicit.replace('axis=1, scope="cta", out=rowmax','axis=-1, scope="cta", across_loop=True, out=rowmax')
        self.assertEqual(parse(equivalent).document,expected)
