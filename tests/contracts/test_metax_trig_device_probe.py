"""A bounded math probe retains exact inputs and never opens production admission."""
import ast
from hashlib import sha256
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts.test_portable_program_evaluation import NativeCompiler
from tests.contracts.test_triton_loop_scopes import _Pointer, _TL, _Tile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('trig_device_probe', ROOT / 'tools/qualify_metax_trig_device.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def math_value(op, value):
    return probe.fp32(getattr(math, op)(value)) if math.isfinite(value) else float('nan')


def expected_bytes(op, raw):
    values = struct.unpack('<' + 'f' * probe.COUNT, raw)
    return struct.pack('<' + 'f' * probe.COUNT, *(math_value(op, value) for value in values))


class NumericContract(unittest.TestCase):
    def test_fixed_input_bits_cover_nonperiodic_rope_and_special_values(self):
        patterns = probe.input_patterns()
        self.assertEqual(len(patterns), 5)
        values = [value for _, raw in patterns for value in struct.unpack('<' + 'f' * probe.COUNT, raw)]
        finite = [value for value in values if math.isfinite(value)]
        self.assertEqual((min(finite), max(finite)), (-65536., 65536.))
        first = probe.words(patterns[0][1])
        self.assertEqual(first[:6], (0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321))
        for _, raw in patterns:
            rows = [raw[i * 1024:(i + 1) * 1024] for i in range(4)]
            self.assertEqual(len(set(rows)), 4)
        self.assertGreater(len(set(probe.words(patterns[1][1]))), 1000)

    def test_special_classification_swaps_and_missing_stores_are_observable(self):
        for op in probe.OPERATIONS:
            for _, raw in probe.input_patterns():
                good = expected_bytes(op, raw)
                self.assertTrue(probe.check_outputs(op, raw, good)['passed'])
                other = 'sin' if op == 'cos' else 'cos'
                self.assertFalse(probe.check_outputs(op, raw, expected_bytes(other, raw))['passed'])
                for poison in (2., -2.):
                    bad = bytearray(good)
                    bad[-4:] = struct.pack('<f', poison)
                    self.assertFalse(probe.check_outputs(op, raw, bad)['passed'])
            special = probe.input_patterns()[0][1]
            for index in (0, 1, 2, 3, 4, 5):
                bad = bytearray(expected_bytes(op, special))
                bad[index * 4:(index + 1) * 4] = struct.pack('<f', 2.)
                self.assertFalse(probe.check_outputs(op, special, bad)['passed'])
        signed = bytearray(expected_bytes('sin', special))
        signed[4:8] = struct.pack('<I', 0)
        self.assertFalse(probe.check_outputs('sin', special, signed)['passed'])

    def test_actual_emission_selects_operation_and_writes_every_coordinate(self):
        compiler = Compiler.load(ROOT)
        for op in probe.OPERATIONS:
            document, emission, _ = probe.probe_emission(op)
            self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', {f.code for f in compiler.assess(document).findings})
            tree = ast.parse(emission.source)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
            function.decorator_list = []
            for arg in function.args.args:
                arg.annotation = None
            for _, raw in probe.input_patterns():
                values = list(struct.unpack('<' + 'f' * probe.COUNT, raw))
                memory = {'x': values, 'out': [2.] * probe.COUNT}
                tl = _TL()
                libdevice = SimpleNamespace(**{name: (lambda tile, name=name:
                    _Tile(tile.shape, [math_value(name, value) for value in tile.values])) for name in probe.OPERATIONS})
                scope = {'tl': tl, 'libdevice': libdevice}
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<trig-emission>', 'exec'), scope)
                for pid in itertools.product(*(range(n) for n in emission.toolchain['grid'])):
                    tl.program = pid
                    scope[function.name](**{name: _Pointer(data) for name, data in memory.items()},
                                         **emission.toolchain['compile_constants'])
                observed = struct.pack('<' + 'f' * probe.COUNT, *memory['out'])
                self.assertTrue(probe.check_outputs(op, raw, observed)['passed'])
                self.assertEqual(len(tl.stores), probe.COUNT)
                self.assertEqual(set(tl.stores.values()), {1})


class ArtifactBinding(unittest.TestCase):
    def setUp(self):
        from open_cake_ir.evaluation.paired import candidate_identity
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.commit = '1' * 40
        self.result = {'source_commit': self.commit, 'target': 'xcore1002', 'passed': True,
                       'operations': [{'operation': op, 'native_compiled': True} for op in probe.OPERATIONS]}
        probe.write(self.folder / 'result.json', self.result)
        probe.write(self.folder / 'numeric-contract.json', probe.numeric_contract())
        for name, raw in probe.input_patterns():
            (self.folder / (name + '.fp32')).write_bytes(raw)
        for op in probe.OPERATIONS:
            directory = self.folder / op
            directory.mkdir()
            document, emission, _ = probe.probe_emission(op)
            payload = emission.source.encode()
            requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
            probe.write(directory / 'requirements.json', requirements)
            workload = WorkloadContract(probe.workload_document(op))
            submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
            builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=NativeCompiler())
            request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                   'xcore1002', requirements['kernel_entry_point'], requirements)
            abi = tuple((a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi('primary'))
            candidate = builder.build_stage(request, abi)
            for role, data in candidate.artifact_payloads.items():
                (directory / (role + '.bin')).write_bytes(data)
            probe.write(directory / 'candidate.json', candidate_identity(candidate))
            self.last = directory, candidate

    def assert_refused_before_allocation(self, message):
        with patch('open_cake_ir.evaluation.local_broker.admit_local_job') as allocate, \
             self.assertRaisesRegex(ValueError, message):
            probe.evaluate(SimpleNamespace(built=self.folder), {'source_commit': self.commit})
        allocate.assert_not_called()

    def reseal(self, *, source=None, manifest_change=None):
        from open_cake_ir.evaluation.core import LaunchableCandidate
        from open_cake_ir.evaluation.paired import candidate_identity
        directory, candidate = self.last
        payloads = dict(candidate.artifact_payloads)
        if source is not None:
            payloads['lowered_source'] = source
        if manifest_change:
            manifest = json.loads(payloads['launch_manifest'])
            manifest.update(manifest_change)
            payloads['launch_manifest'] = canonical_json_bytes(manifest)
        sealed = LaunchableCandidate(candidate.candidate_sha256, candidate.target, candidate.entry_point,
            {role: sha256(data).hexdigest() for role, data in payloads.items()},
            sha256(payloads['launch_manifest']).hexdigest(), payloads)
        for role, data in payloads.items():
            (directory / (role + '.bin')).write_bytes(data)
        (directory / 'candidate.json').write_bytes(canonical_json_bytes(candidate_identity(sealed)))

    def test_complete_fixture_binds_without_gpu(self):
        self.assertEqual([op for op, _, _ in probe.bound_candidates(self.folder, self.commit)], list(probe.OPERATIONS))

    def test_stale_build_and_changed_input_refuse_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'exact source commit'):
            probe.bound_candidates(self.folder, '2' * 40)
        path = self.folder / (probe.input_patterns()[0][0] + '.fp32')
        path.write_bytes(bytes(probe.COUNT * 4))
        self.assert_refused_before_allocation('input bits')

    def test_resealed_wrong_source_refuses_before_allocation(self):
        self.reseal(source=self.last[1].artifact_payloads['lowered_source'] + b'\n# different\n')
        self.assert_refused_before_allocation('current fixed probe emission')

    def test_resealed_wrong_workload_refuses_before_allocation(self):
        self.reseal(manifest_change={'workload_sha256': '9' * 64})
        self.assert_refused_before_allocation('Workload|workload')

    def test_resealed_wrong_block_refuses_before_allocation(self):
        self.reseal(manifest_change={'block': [128, 1, 1]})
        self.assert_refused_before_allocation('current fixed probe emission')


if __name__ == '__main__':
    unittest.main()
