"""Terminal custody outranks profile diagnostics during experimental integration."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.loaders import UndrainedDeviceWork
from open_cake_ir.tasks import evaluate as worker
from tests.contracts import test_metax_program_profile as fixtures


class QueuedWorkerIntegration(unittest.TestCase):
    def test_malformed_profile_cannot_replace_terminal_close_with_normal_failure(self):
        fixtures.ProgramProfile.setUpClass()
        fixture=fixtures.ProgramProfile('runTest')
        error=UndrainedDeviceWork('profile close could not drain')
        loaded=NS(module_count=4,loaded=NS(launch_calls=8,resources={},closed=False),
            close=Mock(side_effect=[error,AssertionError('unsafe second close')]),
            fresh_argument_sets=Mock(return_value=[object()]),snapshot=Mock(return_value=({},{})),launch=Mock())
        metrics=dict(output_mismatches=0,max_abs_error=0.,inputs_unchanged=True)
        receipt=NS(correctness_passed=True,correctness=metrics,
                   artifact_payloads={'launch_receipt':json.dumps({'kernel_calls':4}).encode()})
        with tempfile.TemporaryDirectory() as directory:
            authority=NS(workload=fixture.workload,case_id='primary',candidate=fixture.candidate,
                manifest=fixture.manifest,request={'purpose':'attribution','evaluation_protocol':{}},
                request_root=Path(directory))
            result=worker._base_result(fixture.admission.broker_job_id)
            with patch.object(worker,'_inputs_for',return_value={}), \
                 patch.object(worker,'_correctness_preparation',return_value={}), \
                 patch.object(worker,'_reference_for',return_value={}), \
                 patch.object(worker,'LoadedTorchTensorCandidate',return_value=loaded), \
                 patch.object(worker,'evaluate_tile_workload',return_value=receipt), \
                 patch.object(worker,'compare_tile_outputs',return_value=(True,metrics)), \
                 self.assertRaises(UndrainedDeviceWork) as caught:
                worker._evaluate_tile_candidate(authority,result,None,fixture.admission,False,
                    route_calls_per_cohort=None,profile_source=Mock(return_value={'malformed':object()}),
                    profile_format=NS(kind='fixture',summary=Mock()))
            self.assertIs(caught.exception,error)
            loaded.close.assert_called_once_with()
            self.assertTrue(any(owner is loaded for owner in error._owners))
            self.assertIsNone(result['receipt'])


if __name__=='__main__':unittest.main()
