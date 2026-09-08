"""Offline compile-gate selection and source-root wiring; no Triton or GPU needed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from open_cake_ir.compiler import Compiler
from tools import check_triton_topk_compile as gate

ROOT=Path(__file__).resolve().parents[2]


class OfflineTopkGateTests(unittest.TestCase):
    def test_gate_selects_only_accepted_loop_carried_corpus_cases(self):
        compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
        selected,refused=gate.select_cases(compiler,ROOT)
        self.assertEqual(len(selected),5)
        self.assertTrue(any('qsa-score-topk' in case['case_id'] for case in selected))
        self.assertTrue(any('int32' in case['case_id'] for case in refused))
        self.assertFalse({case['case_id'] for case in selected}&{case['case_id'] for case in refused})

    def test_child_uses_isolated_python_and_explicit_tree_binding(self):
        command=gate.child_command(ROOT,'compiler/revision.json',Path('/external-output'),'case')
        self.assertEqual(command[:2],[sys.executable,'-I'])
        self.assertEqual(command[command.index('--project-root')+1],str(ROOT))
        self.assertIn('--worker-case',command)
        with tempfile.TemporaryDirectory() as directory:
            # An already imported head Compiler must never masquerade as another tree.
            with self.assertRaisesRegex(ValueError,'import differs'):
                gate.load_compiler(Path(directory),'compiler/revision.json')

    def test_real_list_process_uses_the_requested_compiler_and_toolchain_modules(self):
        result=subprocess.run([sys.executable,'-I',str(ROOT/'tools/check_triton_topk_compile.py'),
            '--project-root',str(ROOT),'--revision','compiler/revision.json','--list'],
            cwd='/',capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(Path(report['compiler_module']),ROOT/'src/open_cake_ir/compiler/__init__.py')
        self.assertEqual(Path(report['toolchain_module']),ROOT/'src/open_cake_ir/compiler/toolchain.py')
        self.assertEqual(len(report['selected']),5)
