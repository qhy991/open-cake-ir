"""Naming a broadcast marker preserves the canonical inline operation."""
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler, frontend

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="broadcast", target="sm_103a", backend="triton", entry_point="broadcast")
def candidate(lm, x: cake.Tensor((2,32),"fp32"), w: cake.Tensor((32,),"fp32"), out: cake.Tensor((2,32),"fp32",mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(x,axis=0,dimension=0,tile=2)
    col=lm.program(x,axis=1,dimension=1,tile=32)
    with compute:
        values=lm.load(x[row,col])
        weights=lm.load(w[col])
        result=values * lm.broadcast(weights,axis=1)
        lm.store(out[row,col],result)
'''


class NamedBroadcast(unittest.TestCase):
    def test_named_marker_is_erased_to_the_inline_canonical_form(self):
        named=SOURCE.replace('result=values * lm.broadcast(weights,axis=1)',
                             'expanded=lm.broadcast(weights,axis=1)\n        result=values * expanded')
        inline=frontend.parse(SOURCE).document
        self.assertEqual(frontend.parse(named).document, inline)
        compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
        self.assertTrue(compiler.assess(inline).lowering_eligible)
        invalid=frontend.parse(named.replace('expanded=lm.broadcast(weights,axis=1)',
                                            'expanded=lm.broadcast(weights,axis=0)')).document
        codes={f.code for f in compiler.assess(invalid).findings if f.blocks_acceptance}
        self.assertIn('ELEMENTWISE_BROADCAST',codes)
        with self.assertRaises(frontend.FrontendError):
            frontend.parse(named.replace('lm.store(out[row,col],result)', 'lm.store(out[row,col],expanded)'))
