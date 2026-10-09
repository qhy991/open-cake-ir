"""The diagnostic never turns a failed check into more measurement work."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import diagnose_c550_event_interval as diagnostic


class DiagnosticEntry(unittest.TestCase):
    def fixture(self):
        native=NS(launch_calls=0)
        def launch():native.launch_calls+=1
        loaded=NS(loaded=native,launch=launch,snapshot=lambda:({},{}),validation_inputs={})
        timer=NS(last_activity={'scope':'CPU double'})
        result={}
        def cohort(*args,**kwargs):
            self.assertEqual(kwargs,{'samples_per_cohort':5,'route_calls_per_cohort':16})
            native.launch_calls+=16
            return [1.0]*5, {'passed':True}
        return loaded,timer,result,cohort

    def test_one_cohort_and_original_checks_surround_exact_eighteen_calls(self):
        loaded,timer,result,cohort=self.fixture()
        with patch.object(diagnostic,'instrument_host_phases',return_value=nullcontext()), \
             patch('open_cake_ir.tasks.evaluate._fresh_tile_cohort',side_effect=cohort) as capture, \
             patch('open_cake_ir.evaluation.core.compare_tile_outputs',return_value=(True,{})) as compare:
            diagnostic.capture_once(loaded,object(),{},timer,object(),result)
        self.assertTrue(result['completed']);self.assertEqual(result['native_calls'],18)
        self.assertEqual(capture.call_count,1);self.assertEqual(compare.call_count,2)
        self.assertEqual(set(result['checks']),{'preflight','cohort','postflight'})

    def test_preflight_failure_stops_before_timed_work(self):
        loaded,timer,result,_=self.fixture()
        with patch('open_cake_ir.tasks.evaluate._fresh_tile_cohort') as capture, \
             patch('open_cake_ir.evaluation.core.compare_tile_outputs',return_value=(False,{'passed':False})), \
             self.assertRaisesRegex(ValueError,'correctness failed'):
            diagnostic.capture_once(loaded,object(),{},timer,object(),result)
        capture.assert_not_called();self.assertEqual(result['native_calls'],1)
        self.assertNotIn('completed',result)

    def test_capture_failure_keeps_host_trace_and_event_observation(self):
        loaded,timer,result,_=self.fixture();error=RuntimeError('device fault')
        def fail(*args,**kwargs):
            result['host_timeline'].append({'phase':'partial'})
            raise error
        with patch.object(diagnostic,'instrument_host_phases',return_value=nullcontext()), \
             patch('open_cake_ir.tasks.evaluate._fresh_tile_cohort',side_effect=fail) as capture, \
             patch('open_cake_ir.evaluation.core.compare_tile_outputs',return_value=(True,{})) as compare, \
             self.assertRaises(RuntimeError) as caught:
            diagnostic.capture_once(loaded,object(),{},timer,object(),result)
        self.assertIs(caught.exception,error)
        self.assertEqual(compare.call_count,1);self.assertEqual(capture.call_count,1)
        self.assertEqual(result['host_timeline'],[{'phase':'partial'}])
        self.assertIs(result['event_observation'],timer.last_activity)

    def test_cohort_wrong_answer_stops_before_postflight(self):
        loaded,timer,result,_=self.fixture()
        def wrong(*args,**kwargs):
            loaded.loaded.launch_calls+=16
            return [1.0]*5, {'passed':False}
        with patch.object(diagnostic,'instrument_host_phases',return_value=nullcontext()), \
             patch('open_cake_ir.tasks.evaluate._fresh_tile_cohort',side_effect=wrong), \
             patch('open_cake_ir.evaluation.core.compare_tile_outputs',return_value=(True,{})) as compare, \
             self.assertRaisesRegex(ValueError,'cohort correctness'):
            diagnostic.capture_once(loaded,object(),{},timer,object(),result)
        self.assertEqual(compare.call_count,1);self.assertEqual(result['native_calls'],17)
        self.assertFalse(result['checks']['cohort']['passed'])
