"""Failed observations remain versioned diagnostic evidence, never receipts."""
from copy import deepcopy
import unittest
from open_cake_ir.evaluation.failures import failure_artifacts
from open_cake_ir.tasks.evaluate import _base_result

class FailedObservationProtocol(unittest.TestCase):
    def test_legacy_and_failed_v2_are_distinct(self):
        legacy=_base_result('maca-0123456789ab')
        self.assertEqual(failure_artifacts(legacy),{})
        failed={**legacy,'schema_version':2,'error':'evaluator_failed','failure_class':'ValueError',
                'failure_artifacts':{'paired_activity':'failure-paired_activity.bin'}}
        self.assertEqual(failure_artifacts(failed),failed['failure_artifacts'])
        for field,value in [('receipt',{}),('error',None),('failure_class',None),('schema_version',True),
                             ('failure_artifacts',{}),('failure_artifacts',{'raw':'../escape'}),
                             ('failure_artifacts',{'raw':'/outside'}),('failure_artifacts',{'raw':['not-a-path']}),
                             ('failure_artifacts',{'raw':'same','other':'same'}),
                             ('failure_artifacts',{'bad-role':'diagnostic'})]:
            mutated={**failed,field:value}
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):failure_artifacts(mutated)
