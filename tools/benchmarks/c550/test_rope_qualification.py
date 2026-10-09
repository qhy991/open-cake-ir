"""Same-source bounded RoPE artifacts delegate to the shared original Bench loop."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import frontend
from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
from open_cake_ir.tasks.c550_bench.binding import BenchProblem
from tests.contracts.test_c550_bench_binding import ORACLE_NUMERICS
from tests.contracts.test_portable_program_evaluation import NativeCompiler
from tools.benchmarks.c550.test_rope import original_problem

spec = importlib.util.spec_from_file_location('rope_qualification', Path(__file__).with_name('qualify_rope.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ArtifactBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        original = original_problem()
        rows = tuple(SimpleNamespace(uuid=f'fixture-{i}', axes={'batch_size': 2, 'seq_len': i + 2},
            tolerance=original.workloads[0].tolerance) for i in range(16))
        raw = tuple({**deepcopy(original.raw_workloads[0]), 'uuid': row.uuid} for row in rows)
        self.problem = replace(original, workloads=rows, raw_workloads=raw)
        opening = patch.object(BenchProblem, 'open', return_value=self.problem)
        opening.start(); self.addCleanup(opening.stop)
        self.commit = probe.checkout_commit(probe.ROOT)
        index = {'passed': True, 'source_commit': self.commit, 'bench_commit': probe.BENCH_COMMIT,
                 'task': probe.TASK, 'target': 'xcore1002', 'oracle_numerics': deepcopy(ORACLE_NUMERICS), 'cases': []}
        for row in rows:
            folder = self.root / row.uuid; folder.mkdir()
            document = self.problem.workload_document(row.uuid, oracle_numerics=ORACLE_NUMERICS)
            workload = probe.BenchWorkload(document)
            candidate = self.candidate(workload)
            self.save(folder, candidate)
            (folder / 'workload.json').write_bytes(probe.canonical_json_bytes(document))
            index['cases'].append({'uuid': row.uuid, 'shape': list(workload.tensor_abi('primary')[0].shape)})
        (self.root / 'result.json').write_text(json.dumps(index))
        self.last, self.folder, self.workload = candidate, folder, workload
        self.args = SimpleNamespace(built=self.root, bench_root=self.root,
            oracle_numerics=Path('/fixture/retained-policy.json'), output=self.root / 'output',
            physical_device=7, runtime_device=0, expected_pci='0000:34:00')

    def candidate(self, workload):
        raw, emission = probe.emission_for(workload)
        payload = emission.source.encode()
        requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, probe.canonical_json_bytes(raw))
        request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                               'xcore1002', requirements['kernel_entry_point'], requirements)
        abi = tuple((a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi('primary'))
        return TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=NativeCompiler()).build_stage(request, abi)

    @staticmethod
    def save(folder, candidate):
        (folder / 'candidate.json').write_bytes(probe.canonical_json_bytes(candidate_identity(candidate)))
        for role, payload in candidate.artifact_payloads.items():
            (folder / (role + '.bin')).write_bytes(payload)

    def refuses(self, message):
        with patch.object(probe, 'read_oracle_numerics', return_value=deepcopy(ORACLE_NUMERICS)), \
             patch.object(probe, 'evaluate_original') as original_loop, \
             patch('open_cake_ir.evaluation.local_broker.admit_local_job') as allocate, \
             self.assertRaisesRegex(ValueError, message):
            probe.evaluate(self.args, {'source_commit': self.commit})
        original_loop.assert_not_called(); allocate.assert_not_called()

    def test_valid_original_case_set_binds_all_policies_and_native_abis(self):
        problem, cases = probe.bind_original_build(self.root, self.root, self.commit, oracle_numerics=ORACLE_NUMERICS)
        self.assertIs(problem, self.problem)
        self.assertEqual([case.uuid for case in cases], [row.uuid for row in self.problem.workloads])
        self.assertTrue(all(case.workload.document['semantics']['oracle_numerics'] == ORACLE_NUMERICS for case in cases))
        self.assertTrue(all(case.manifest.kernels_per_call == 1 for case in cases))

    def test_wrong_candidate_seal_and_entry_refuse_before_gpu(self):
        c = self.last
        self.save(self.folder, LaunchableCandidate(c.candidate_sha256, c.target, c.entry_point,
                                                  c.artifact_roles, 'f' * 64, c.artifact_payloads))
        self.refuses('launch authority differs')
        payloads = dict(c.artifact_payloads)
        record = json.loads(payloads['stage_compilation']); record['kernel_name'] = 'different_entry'
        payloads['stage_compilation'] = probe.canonical_json_bytes(record)
        changed = LaunchableCandidate(c.candidate_sha256, c.target, 'different_entry',
            {role: sha256(payload).hexdigest() for role, payload in payloads.items()}, c.launch_spec_sha256, payloads)
        self.save(self.folder, changed); self.refuses('launch authority differs')

    def test_case_directory_cannot_escape_the_build_root(self):
        with tempfile.TemporaryDirectory() as external:
            moved = Path(external) / self.folder.name
            self.folder.rename(moved); self.folder.symlink_to(moved, target_is_directory=True)
            self.refuses('case directory escapes')

    def test_changed_source_or_policy_cannot_reuse_the_build(self):
        path = self.root / 'result.json'; original = json.loads(path.read_text())
        for change in (
            lambda d: d.update(source_commit='other'),
            lambda d: d.pop('oracle_numerics'),
            lambda d: d['oracle_numerics'].update(allow_bf16_reduced_precision_reduction=False),
            lambda d: d['cases'].reverse(),
        ):
            value = deepcopy(original); change(value); path.write_text(json.dumps(value))
            self.refuses('this source and all ordered original workloads')
        path.write_text(json.dumps(original))
        document = json.loads((self.folder / 'workload.json').read_text())
        document['validation']['effective_tolerance']['max_rtol'] = 10.
        (self.folder / 'workload.json').write_text(json.dumps(document))
        self.refuses('retained RoPE original Workload')

    def test_old_ieee_starter_is_not_the_bounded_successor(self):
        from benchmarks.c550.rope import source_for
        batch, sequence = self.workload.tensor_abi('primary')[0].shape
        with patch.object(probe, 'source_for_workload', return_value=source_for(batch, sequence)):
            candidate = self.candidate(self.workload)
        self.save(self.folder, candidate)
        self.refuses('artifact or unique original shape')

    def test_self_consistent_wrong_pointer_count_is_refused_before_lease(self):
        c = self.last; payloads = dict(c.artifact_payloads)
        for role in ('launch_manifest', 'stage_compilation'):
            value = json.loads(payloads[role]); value['hidden_null_pointer_parameters'] = 2
            payloads[role] = probe.canonical_json_bytes(value)
        candidate = LaunchableCandidate(c.candidate_sha256, c.target, c.entry_point,
            {role: sha256(payload).hexdigest() for role, payload in payloads.items()},
            sha256(payloads['launch_manifest']).hexdigest(), payloads)
        self.save(self.folder, candidate)
        self.refuses('native pointer ABI')

    def test_original_loop_owns_native_callbacks_policy_boundaries_and_verdict(self):
        for passed in (True, False):
            result = {'source_commit': self.commit}
            outcome = {'passed': passed, 'full_device_correctness': passed,
                       'original_checks': 160, 'performance': 'not_measured'}
            with patch.object(probe, 'read_oracle_numerics', return_value=deepcopy(ORACLE_NUMERICS)), \
                 patch.object(probe, 'evaluate_original', return_value=outcome) as shared:
                probe.evaluate(self.args, result)
            shared.assert_called_once()
            args, kwargs = shared.call_args
            self.assertIs(args[0], self.problem); self.assertEqual(len(args[1]), 16)
            self.assertEqual(args[2], self.args.output)
            self.assertEqual(kwargs, {'physical_device': 7, 'runtime_device': 0, 'expected_pci': '0000:34:00'})
            self.assertEqual(result['passed'], passed)
            self.assertFalse(result['target_admission_changed'])
            self.assertEqual(result['oracle_numerics'], ORACLE_NUMERICS)

    def test_shared_original_loop_failure_remains_a_failure(self):
        result = {'source_commit': self.commit, 'passed': False}
        with patch.object(probe, 'read_oracle_numerics', return_value=deepcopy(ORACLE_NUMERICS)), \
             patch.object(probe, 'evaluate_original', side_effect=ValueError('original loop policy drift')):
            with self.assertRaisesRegex(ValueError, 'original loop policy drift'):
                probe.evaluate(self.args, result)
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
