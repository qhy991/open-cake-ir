"""The reference entry delegates its verdict and never runs without a device lease."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import check_c550_bench_reference as entry


@dataclass
class Admission:
    target: str = 'xcore1002'
    pci_bus_id: str = '0000:0f:00'


class BenchReferenceEntry(unittest.TestCase):
    def test_original_all_case_cli_and_exit_status_survive_the_lease_wrapper(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);out=root/'output'
                problem=SimpleNamespace(root=root,api=SimpleNamespace(document=lambda name:{'seed':200}))
                events=[]
                def run(path, *, run_name):
                    events.append('original_cli')
                    self.assertEqual(run_name,'__main__')
                    args=sys.argv
                    self.assertIn('--reference-selfcheck',args)
                    self.assertNotIn('--candidate',args)
                    self.assertEqual(args[args.index('--workloads')+1],'all')
                    self.assertEqual(args[args.index('--rounds')+1],'10')
                    self.assertEqual(args[args.index('--seed')+1],'200')
                    raise SystemExit(exit_code)
                old_argv=sys.argv
                with patch.object(entry,'checkout_commit',return_value='fixture'), \
                     patch.object(entry.BenchProblem,'open',return_value=problem), \
                     patch.object(entry.ExecutorRevision,'for_target',return_value=SimpleNamespace(admit_host=lambda:{'runtime_library':'/sdk/lib.so'})), \
                     patch.object(entry,'admit_local_job',side_effect=lambda *a,**k:events.append('lease')), \
                     patch.object(entry,'observe_local_metax',side_effect=lambda *a,**k:(events.append('observe') or Admission())), \
                     patch.object(entry.runpy,'run_path',side_effect=run):
                    with self.assertRaises(SystemExit) as caught:
                        entry.main(['--bench-root',str(root),'--task','original-task','--output',str(out),
                            '--physical-device','0','--runtime-device','0','--expected-pci','0000:0f:00'])
                    self.assertEqual(caught.exception.code,exit_code)
                self.assertEqual(events,['lease','observe','original_cli'])
                self.assertIs(sys.argv,old_argv)
                self.assertFalse(json.loads((out/'authority.json').read_text())['candidate_qualified'])

    def test_busy_lease_never_reaches_reference_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output'
            problem=SimpleNamespace(root=root,api=SimpleNamespace(document=lambda name:{'seed':200}))
            with patch.object(entry,'checkout_commit',return_value='fixture'), \
                 patch.object(entry.BenchProblem,'open',return_value=problem), \
                 patch.object(entry.ExecutorRevision,'for_target',return_value=SimpleNamespace(admit_host=lambda:{'runtime_library':'/sdk/lib.so'})), \
                 patch.object(entry,'admit_local_job',side_effect=BlockingIOError('busy')), \
                 patch.object(entry,'observe_local_metax') as observe, \
                 patch.object(entry.runpy,'run_path') as run:
                with self.assertRaises(BlockingIOError):
                    entry.main(['--bench-root',str(root),'--task','original-task','--output',str(out),
                        '--physical-device','0','--runtime-device','0','--expected-pci','0000:0f:00'])
                observe.assert_not_called();run.assert_not_called()
