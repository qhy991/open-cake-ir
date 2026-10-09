"""Native return-value bridge controls; original Bench remains the oracle owner."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rope_qualification', Path(__file__).with_name('qualify_rope.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tensor:
    def __init__(self, shape, data): self.shape, self.data = shape, list(data)
    def view(self, dtype): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.shape, self.data)


class ReturnValueBridge(unittest.TestCase):
    def test_only_a_native_launch_supplies_output_and_every_path_closes(self):
        torch = SimpleNamespace(uint8='uint8', bfloat16='bf16',
            full=lambda shape, value, **kwargs: Tensor(shape, [value]),
            equal=lambda a, b: a.data == b.data,
            cuda=SimpleNamespace(synchronize=lambda: None, current_stream=lambda: SimpleNamespace(cuda_stream=7)))
        for mutate in (False, True):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                runtime = probe.NativeRope.__new__(probe.NativeRope)
                runtime.root, runtime.trace = Path(directory), Path(directory) / 'calls.jsonl'
                runtime.trace.touch()
                runtime.cases, runtime.admission = {(1, 2): 'original'}, object()
                positions, frequency = Tensor((1, 2), [3, 9]), Tensor((64,), [1, 2])
                seen = []
                class Loaded:
                    launch_calls = 0
                    closed = False
                    def launch(self, args, *, tensor_contract, stream):
                        self.launch_calls += 1
                        seen.append((args, stream))
                        args[-1].data[:] = ['native result']
                        if mutate: args[0].data[0] = -1
                    def close(self, *, synchronize, primary=None):
                        synchronize(); self.closed = True
                        if primary: raise primary
                loaded = Loaded()
                with patch.dict('sys.modules', {'torch': torch}), \
                     patch.object(probe, 'read_candidate', return_value=(object(), object())), \
                     patch('open_cake_ir.evaluation.metax_driver.LoadedMetaxCandidate.load', return_value=loaded):
                    if mutate:
                        with self.assertRaisesRegex(ValueError, 'changed an original input'):
                            runtime.run(positions, frequency, 1.0)
                    else:
                        out = runtime.run(positions, frequency, 1.0)
                        self.assertEqual(out.data, ['native result'])
                self.assertIs(seen[0][0][0], positions)
                self.assertIs(seen[0][0][1], frequency)
                self.assertEqual(seen[0][1], 7)
                record = json.loads(runtime.trace.read_text())
                self.assertTrue(record['module_closed'])
                self.assertEqual(record['kernel_calls'], 1)
                self.assertEqual(record['input_unchanged'], not mutate)

    def test_original_shape_and_scalar_cannot_select_another_variant(self):
        runtime = probe.NativeRope.__new__(probe.NativeRope)
        runtime.cases = {(1, 2): 'original'}
        with patch.dict('sys.modules', {'torch': SimpleNamespace()}), \
             patch.object(probe, 'read_candidate') as load:
            for shape, scalar in (((2, 1), 1.0), ((1, 2), 2.0), ((1, 2), True)):
                with self.assertRaisesRegex(ValueError, 'original scalar'):
                    runtime.run(Tensor(shape, []), Tensor((64,), []), scalar)
            load.assert_not_called()


class ArtifactBinding(unittest.TestCase):
    def setUp(self):
        from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
        from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
        from tests.contracts.test_portable_program_evaluation import NativeCompiler
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        rows = [SimpleNamespace(uuid=f'fixture-{i}') for i in range(16)]
        self.problem = SimpleNamespace(workloads=rows, workload_document=lambda uuid: self.document(int(uuid.split('-')[1])))
        index = {'passed': True, 'source_commit': 'fixture-source', 'bench_commit': probe.BENCH_COMMIT,
                 'task': probe.TASK, 'cases': []}
        for i, row in enumerate(rows):
            folder = self.root / row.uuid
            folder.mkdir()
            workload = probe.BenchWorkload(self.document(i))
            raw, emission = probe.emission_for(workload)
            payload = emission.source.encode()
            requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
            submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, probe.canonical_json_bytes(raw))
            request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                   'xcore1002', requirements['kernel_entry_point'], requirements)
            abi = tuple((a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi('primary'))
            candidate = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=NativeCompiler()).build_stage(request, abi)
            self.save(folder, candidate)
            index['cases'].append({'uuid': row.uuid, 'shape': list(abi[0][1])})
        (self.root / 'result.json').write_text(json.dumps(index))
        self.last, self.folder = candidate, folder

    @staticmethod
    def document(index):
        return {'workload_id': f'rope-fixture-{index}', 'operator': 'fixture',
            'cases': [{'case_id': 'primary', 'shape': {'B': index + 1, 'S': 2}, 'seed': 401, 'mode': 'fixture'}],
            'tensors': {'position_ids': {'shape': ['B','S'], 'dtype': 'int64', 'layout': 'contiguous_row_major'},
                       'inv_freq': {'shape': [64], 'dtype': 'fp32', 'layout': 'contiguous_row_major'},
                       'cos_sin': {'shape': ['B','S',128,2], 'dtype': 'bf16', 'layout': 'contiguous_row_major'}},
            'semantics': {'target': 'xcore1002', 'candidate_abi': {'inputs': ['position_ids','inv_freq'], 'outputs': ['cos_sin']},
                         'fixed_scalar_inputs': {'attention_scaling': {'dtype': 'float32', 'value': 1.0, 'binding': 'original_factory_literal'}}},
            'validation': {'primary_case': 'primary', 'all_cases_required': True, 'atol': 1e-5, 'rtol': .05}}

    @staticmethod
    def save(folder, candidate):
        from open_cake_ir.evaluation.paired import candidate_identity
        (folder / 'candidate.json').write_bytes(probe.canonical_json_bytes(candidate_identity(candidate)))
        for role, payload in candidate.artifact_payloads.items():
            (folder / (role + '.bin')).write_bytes(payload)

    def refuses(self, message):
        with patch.object(probe.BenchProblem, 'open', return_value=self.problem), \
             patch('open_cake_ir.evaluation.local_broker.admit_local_job') as allocate, \
             self.assertRaisesRegex(ValueError, message):
            probe.evaluate(SimpleNamespace(built=self.root, bench_root=self.root), {'source_commit': 'fixture-source'})
        allocate.assert_not_called()

    def test_valid_original_case_set_binds(self):
        with patch.object(probe.BenchProblem, 'open', return_value=self.problem):
            self.assertIs(probe.bind_original_build(self.root, self.root, 'fixture-source'), self.problem)

    def test_wrong_candidate_seal_and_entry_refuse_before_gpu(self):
        from open_cake_ir.evaluation.core import LaunchableCandidate
        c = self.last
        self.save(self.folder, LaunchableCandidate(c.candidate_sha256, c.target, c.entry_point,
                                                  c.artifact_roles, 'f' * 64, c.artifact_payloads))
        self.refuses('launch authority differs')
        payloads = dict(c.artifact_payloads)
        record = json.loads(payloads['stage_compilation'])
        record['kernel_name'] = 'different_entry'
        payloads['stage_compilation'] = probe.canonical_json_bytes(record)
        changed = LaunchableCandidate(c.candidate_sha256, c.target, 'different_entry',
            {role: sha256(payload).hexdigest() for role, payload in payloads.items()}, c.launch_spec_sha256, payloads)
        self.save(self.folder, changed)
        self.refuses('launch authority differs')

    def test_case_directory_cannot_escape_the_build_root(self):
        with tempfile.TemporaryDirectory() as external:
            moved = Path(external) / self.folder.name
            self.folder.rename(moved)
            self.folder.symlink_to(moved, target_is_directory=True)
            self.refuses('case directory escapes')


if __name__ == '__main__':
    unittest.main()
