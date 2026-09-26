"""One frozen EP4 Workload and Compiler source bind a single rank plan."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule
from open_cake_ir.evaluation.ranked_tile_manifest import (
    RankedTileCandidate, RankedTileLaunchManifest,
    prepare_sealed_ranked_tile_case,
)
from open_cake_ir.lab.ranked_tile_build import (
    RankedTileBuildError, compile_ranked_tile_candidate,
)
from open_cake_ir.tasks.workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]


class RankedTileManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=Compiler.load(ROOT)
        if compiler.commit is None:
            raise unittest.SkipTest('ranked seal needs a clean fixed Compiler')
        program=Program.from_dict(json.loads((ROOT/'examples/programs/'
            'weave-model-local-expert-ffn-native-b300.json').read_text()))
        effects=RankedTileEffects.from_dict(json.loads((ROOT/'examples/programs/'
            'weave-model-ranked-tile-effects-b300.json').read_text()))
        cls.combine_document=json.loads((ROOT/'examples/schedules/native/'
            'weave-model-weighted-combine-rank512-b300.json').read_text())
        combine=Schedule.from_dict(cls.combine_document)
        cls.lowered=compiler.lower_ranked_tiles(effects,program,combine)
        cls.workload=load_workload(ROOT/'contracts/workloads/'
            'weave-model-ep4-bf16-moe-b300-v1.json')

    def plans(self, chunks=2):
        return {rank:{'communication_ctas':1,'chunks':chunks,
                      'steal_budget':5888} for rank in range(4)}

    def manifest(self):
        return RankedTileLaunchManifest.from_lowered(
            self.lowered,combine_document=self.combine_document,
            workload=self.workload,
            case_id='mixed_full_early_terminal',plans=self.plans())

    def test_complete_typed_source_and_workload_round_trip(self):
        manifest=self.manifest()
        restored=RankedTileLaunchManifest.from_dict(manifest.as_dict())
        self.assertEqual(restored,manifest)
        self.assertEqual(restored.as_dict()['abi'],'ranked_tile_b300_pointer_v4')
        self.assertEqual(restored.plans(),self.plans())
        restored.check_lowered(self.lowered)
        restored.check_workload(self.workload,'mixed_full_early_terminal',
                                self.lowered)
        changed=manifest.as_dict()
        changed['plans'][2]['chunks']=1
        with self.assertRaisesRegex(ValueError,'chunks must agree'):
            RankedTileLaunchManifest.from_dict(changed)
        changed=manifest.as_dict()
        changed['source_map']['effect.bin.reserve']=[0,0]
        with self.assertRaisesRegex(ValueError,'source map'):
            RankedTileLaunchManifest.from_dict(changed)
        older=replace(self.lowered,source=self.lowered.source.replace(
            '_abi_version() { return 4; }','_abi_version() { return 3; }'))
        with self.assertRaisesRegex(ValueError,'emitted ABI source'):
            restored.check_lowered(older)

    def test_compiled_bytes_and_case_plan_refused_before_device_binding(self):
        manifest=self.manifest()
        candidate=RankedTileCandidate.seal(
            self.lowered,manifest,workload=self.workload,
            case_id='mixed_full_early_terminal',library=b'\x7fELFtest')
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/'ranked.so'
            library.write_bytes(b'\x7fELFtest')
            candidate.check(self.lowered,self.workload,
                            'mixed_full_early_terminal',self.plans(),library)
            with self.assertRaisesRegex(ValueError,'frozen Workload'):
                candidate.check(self.lowered,self.workload,'fanin_v2',
                                self.plans(),library)
            with self.assertRaisesRegex(ValueError,'sealed source, plan or library'):
                candidate.check(self.lowered,self.workload,
                                'mixed_full_early_terminal',self.plans(1),library)
            with self.assertRaisesRegex(ValueError,'isolated process'):
                prepare_sealed_ranked_tile_case(
                    candidate,self.lowered,self.workload,
                    'mixed_full_early_terminal',{}, {},self.plans(),
                    library_path=library,pointer_of=lambda _:0,
                    isolated_process=False,
                    check_tensor=lambda *_:None,
                    storage_span=lambda _:None,
                    execution_context=lambda _:None)
            library.write_bytes(b'\x7fELFother')
            with self.assertRaisesRegex(ValueError,'sealed source, plan or library'):
                prepare_sealed_ranked_tile_case(
                    candidate,self.lowered,self.workload,
                    'mixed_full_early_terminal',{}, {},self.plans(),
                    library_path=library,pointer_of=lambda _:0,
                    isolated_process=True,
                    check_tensor=lambda *_:None,
                    storage_span=lambda _:None,
                    execution_context=lambda _:None)

    def test_cpu_build_retains_source_library_and_failure(self):
        manifest=self.manifest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            nvcc=root/'fake_nvcc'
            nvcc.write_text('#!/usr/bin/env python3\n'
                'import pathlib,sys\n'
                'p=pathlib.Path(sys.argv[sys.argv.index("-o")+1])\n'
                'p.write_bytes(b"\\x7fELFtest")\n')
            nvcc.chmod(0o755)
            output=root/'candidate'
            products=compile_ranked_tile_candidate(
                self.lowered,manifest,self.workload,
                'mixed_full_early_terminal',nvcc=nvcc,output_dir=output)
            self.assertEqual(products.source.read_bytes(),
                             self.lowered.source.encode('utf-8'))
            products.candidate.check(self.lowered,self.workload,
                'mixed_full_early_terminal',self.plans(),products.library)
            with self.assertRaisesRegex(ValueError,'create-only'):
                compile_ranked_tile_candidate(
                    self.lowered,manifest,self.workload,
                    'mixed_full_early_terminal',nvcc=nvcc,output_dir=output)
            nvcc.write_text('#!/usr/bin/env python3\nimport sys\n'
                            'print("build refused",file=sys.stderr)\n'
                            'raise SystemExit(7)\n')
            with self.assertRaisesRegex(RankedTileBuildError,'rejected source'):
                compile_ranked_tile_candidate(
                    self.lowered,manifest,self.workload,
                    'mixed_full_early_terminal',nvcc=nvcc,
                    output_dir=root/'failed')
            self.assertTrue((root/'failed/build_failure.json').is_file())
            self.assertIn('build refused',
                          (root/'failed/compile.log').read_text())


if __name__=='__main__':
    unittest.main()
