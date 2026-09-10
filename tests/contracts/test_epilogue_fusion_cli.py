"""CLI uses the actual Compiler public result; a loader fixture permits draft checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler
from tools import apply_epilogue_fusion as cli

ROOT = Path(__file__).resolve().parents[2]


class EpilogueFusionCliTests(unittest.TestCase):
    def invoke(self, output, private):
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        args=['apply_epilogue_fusion.py','--producer',str(ROOT/'examples/python/epilogue_producer.py'),
              '--epilogue',str(ROOT/'examples/python/epilogue_consumer.py'),'--private-intermediate',private,
              '--schedule-id','cli_composition','--entry-point','cli_composition','--output',str(output)]
        with patch.object(cli.Compiler,'load',return_value=compiler), patch('sys.argv',args), contextlib.redirect_stdout(io.StringIO()):
            return cli.main()

    def test_applied_cli_uses_assessment_identity_and_writes_complete_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root).resolve()/'candidate'
            self.assertEqual(self.invoke(out,'mid'),0)
            report=json.loads((out/'result.json').read_text())
            self.assertTrue(report['applied'])
            self.assertIn('compiler_revision_id',report)
            self.assertTrue((out/'schedule.json').is_file())
            self.assertTrue((out/'lowered.py').is_file())
            original=(out/'result.json').read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(out,'mid')
            self.assertEqual((out/'result.json').read_bytes(),original)

    def test_refused_cli_retains_reason_without_a_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root).resolve()/'refused'
            self.assertEqual(self.invoke(out,'not_private'),2)
            report=json.loads((out/'result.json').read_text())
            self.assertFalse(report['applied'])
            self.assertEqual(report['reason'],'composition_boundary')
            self.assertEqual({p.name for p in out.iterdir()},{'result.json'})
