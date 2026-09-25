"""CPU ranked nvcc handoff retains both compiled products and failures."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program, RankedMailboxEffects, Schedule
from open_cake_ir.evaluation.ranked_manifest import RankedMailboxLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.ranked_build import (RankedBuildError,
    compile_ranked_development)
from open_cake_ir.tasks.weave_ep.workload import validate_contract


ROOT = Path(__file__).resolve().parents[2]


def material():
    compiler = Compiler.load(ROOT)
    effects = RankedMailboxEffects.from_dict(json.loads((ROOT / 'examples/programs/'
        'weave-ranked-mailbox-effects-b300.json').read_text()))
    local = Program.from_dict(json.loads((ROOT / 'examples/programs/'
        'weave-local-expert-ffn-native-b300.json').read_text()))
    combine_document = json.loads((ROOT / 'examples/schedules/native/'
        'weave-weighted-combine-t7-h16.json').read_text())
    lowered = compiler.lower_ranked_mailbox(effects, local,
                                             Schedule.from_dict(combine_document))
    contract_path = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'
    workload = WorkloadContract.from_document(json.loads(contract_path.read_text()),
        contract_path, validate=validate_contract)
    bindings = {'hidden_states': 'hidden', 'expert_ids': 'expert_ids',
                'route_weights': 'route_weights', 'w_up_gate': 'w_up_gate',
                'w_down': 'w_down'}
    plans = [{'rank': rank, 'communication_ctas': (147, 12, 36, 72)[rank],
              'chunks': (2, 3, 7, 1)[rank],
              'steal_budget': 14 if rank == 0 else 0}
             for rank in range(4)]
    manifest = RankedMailboxLaunchManifest.from_lowered(
        lowered, combine_document=combine_document, workload=workload,
        case_id='tail_tokens', tensor_bindings=bindings, plans=plans)
    return lowered, manifest, workload


def fake_nvcc(path: Path):
    path.write_text('''#!/usr/bin/env python3
import os, pathlib, sys, time
args = sys.argv[1:]
output = pathlib.Path(args[args.index('-o') + 1])
if os.environ.get('CAKE_FAKE_SLEEP_CUBIN') == '1' and '--cubin' in args:
    time.sleep(2)
if os.environ.get('CAKE_FAKE_FAIL_CUBIN') == '1' and '--cubin' in args:
    print('intentional cubin rejection', file=sys.stderr)
    raise SystemExit(7)
output.write_bytes(b'\\x7fELF' + (b'CUBIN' if '--cubin' in args else b'HOST'))
print('fake nvcc accepted', output.name)
''')
    path.chmod(0o755)


class RankedBuild(unittest.TestCase):
    def test_exact_source_compiles_two_create_only_elf_products(self):
        lowered, manifest, workload = material()
        with tempfile.TemporaryDirectory(prefix='cake-ranked-build-') as directory:
            root = Path(directory)
            compiler = root / 'fake-nvcc'
            fake_nvcc(compiler)
            output = root / 'first'
            products = compile_ranked_development(
                lowered, manifest, workload, 'tail_tokens',
                nvcc=compiler, output_dir=output)
            self.assertEqual(products.output_dir, output)
            self.assertTrue(products.host_wrapper.is_file())
            self.assertTrue(products.cubin.is_file())
            self.assertEqual(products.source.read_text(), lowered.source)
            plan = json.loads((output / 'build_plan.json').read_text())
            host = plan['commands']['host_wrapper']
            cubin = plan['commands']['cubin']
            self.assertIn('--gpu-architecture=compute_103a', host)
            self.assertIn('--gpu-code=sm_103a', cubin)
            self.assertIn('-lcudart', host)
            self.assertIn('--cubin', cubin)
            self.assertEqual(host[host.index('-o') + 1],
                             str(products.host_wrapper))
            self.assertEqual(json.loads(products.report.read_text())['scope'],
                             'CPU nvcc build only; no GPU or candidate qualification')
            with self.assertRaisesRegex(ValueError, 'create-only'):
                compile_ranked_development(lowered, manifest, workload,
                    'tail_tokens', nvcc=compiler, output_dir=output)

    def test_compiler_failure_and_manifest_drift_retain_or_refuse(self):
        lowered, manifest, workload = material()
        with tempfile.TemporaryDirectory(prefix='cake-ranked-build-') as directory:
            root = Path(directory)
            compiler = root / 'fake-nvcc'
            fake_nvcc(compiler)
            failure = root / 'failure'
            with patch.dict(os.environ, {'CAKE_FAKE_FAIL_CUBIN': '1'}):
                with self.assertRaisesRegex(RankedBuildError, 'cubin nvcc rejected'):
                    compile_ranked_development(lowered, manifest, workload,
                        'tail_tokens', nvcc=compiler, output_dir=failure)
            self.assertTrue((failure / 'ranked.so').is_file())
            self.assertEqual(json.loads((failure / 'build_failure.json').read_text())['stage'],
                             'cubin')
            self.assertIn('intentional cubin rejection',
                          (failure / 'cubin.compile.log').read_text())
            timeout = root / 'timeout'
            with patch.dict(os.environ, {'CAKE_FAKE_SLEEP_CUBIN': '1'}):
                with self.assertRaisesRegex(RankedBuildError, 'cubin nvcc timed out'):
                    compile_ranked_development(lowered, manifest, workload,
                        'tail_tokens', nvcc=compiler, output_dir=timeout,
                        timeout_seconds=1)
            self.assertTrue((timeout / 'ranked.so').is_file())
            self.assertEqual(json.loads((timeout / 'build_failure.json').read_text())['reason'],
                             'timeout')
            wrong = root / 'wrong-case'
            with self.assertRaisesRegex(ValueError, 'Workload'):
                compile_ranked_development(lowered, manifest, workload,
                    'skew_to_rank0', nvcc=compiler, output_dir=wrong)
            self.assertFalse(wrong.exists())
            wrong_req = dict(lowered.toolchain_requirements)
            wrong_req['nvcc_flags'] = [flag.replace('sm_103a', 'sm_100a')
                                      for flag in wrong_req['nvcc_flags']]
            with self.assertRaisesRegex(ValueError, 'exact architecture'):
                compile_ranked_development(replace(lowered,
                    toolchain_requirements=wrong_req), manifest, workload,
                    'tail_tokens', nvcc=compiler, output_dir=root / 'wrong-target')


if __name__ == '__main__':
    unittest.main()
