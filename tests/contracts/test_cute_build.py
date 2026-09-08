"""CuTe build boundary tests with CPU-only compiler receipts and process fixtures."""
from __future__ import annotations

import base64
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.lab.cute_build import IsolatedCuTeCompiler, CuTeToolchainBuilder, _compile_failure, _worker
from open_cake_ir.lab.environments import BuildRequest
from open_cake_ir.lab.faults import CandidateCompileRejected, RunProtocolFault
from open_cake_ir.lab.process import SupervisedProcessTimeout, SupervisedProcessOutputLimit
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts.test_cute_toolchain import SOURCE, NAME, compilation_fixture, requirements

ROOT = Path(__file__).resolve().parents[2]


def receipt(compilation):
    return {"target": compilation.target, "entry_point": compilation.entry_point,
        "compiler_version": compilation.compiler_version, "threads_per_cta": compilation.threads_per_cta,
        "dynamic_shared_bytes": compilation.dynamic_shared_bytes,
        "artifacts": {key: base64.b64encode(value).decode() for key, value in compilation.artifacts.items()}}


def compiler_fixture(root, extra_roots=()):
    runtime = root / 'runtime'; runtime.mkdir()
    for name in ('python', 'cuobjdump'):
        (runtime / name).write_bytes(b'CPU fixture')
    bwrap = root / 'bwrap'; bwrap.write_bytes(b'CPU isolation fixture')
    with mock.patch('open_cake_ir.lab.cute_build.sys.platform', 'linux'):
        return IsolatedCuTeCompiler(python=str(runtime / 'python'), bubblewrap=str(bwrap),
            runtime_roots=[str(runtime), *map(str, extra_roots)],
            cuobjdump=str(runtime / 'cuobjdump'), cutlass_version='4.5.2')


class IsolatedCuTeBuildTests(unittest.TestCase):
    def test_bwrap_uses_readonly_runtime_blank_environment_and_private_device_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = compiler_fixture(Path(directory))
            def supervise(argv, **kwargs):
                Path(kwargs['cwd'], 'compilation.json').write_text(json.dumps(receipt(compilation_fixture())))
                return subprocess.CompletedProcess(argv, 0, b'', b'')
            with mock.patch('open_cake_ir.lab.cute_build.run_supervised', side_effect=supervise) as run:
                compiled = compiler.compile(SOURCE, requirements())
            self.assertEqual(compiled.entry_point, NAME)
            argv = run.call_args.args[0]
            self.assertIn('--unshare-all', argv); self.assertIn('--clearenv', argv)
            self.assertIn('--dev', argv); self.assertNotIn('/dev/nvidia0', argv)
            self.assertNotIn(str(Path.home()), argv); self.assertNotIn(str(ROOT), argv)
            self.assertEqual(argv.count('--bind'), 1)
            self.assertEqual(argv[argv.index('--dev') + 1], '/dev')
            self.assertEqual(run.call_args.kwargs['environment'], {})
            env = {argv[i + 1]: argv[i + 2] for i, arg in enumerate(argv) if arg == '--setenv'}
            self.assertEqual(env['CUTE_DSL_ARCH'], 'sm_103a')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')

    def test_loader_alias_preserves_declared_guest_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            libraries = root / 'libraries'; libraries.mkdir()
            alias = root / 'lib64'; alias.symlink_to(libraries, target_is_directory=True)
            compiler = compiler_fixture(root, [alias])
            with mock.patch('open_cake_ir.lab.cute_build.run_supervised',
                            return_value=subprocess.CompletedProcess([], 1, b'', b'stop')) as run:
                with self.assertRaises(RunProtocolFault):
                    compiler.compile(SOURCE, requirements())
            argv = run.call_args.args[0]
            mounts = [argv[i+1:i+3] for i, arg in enumerate(argv) if arg == '--ro-bind']
            self.assertIn([str(libraries), str(alias)], mounts)

    def test_no_native_source_is_executed_before_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = compiler_fixture(Path(directory))
            with mock.patch('open_cake_ir.lab.cute_build.run_supervised') as run:
                with self.assertRaises(ValueError):
                    compiler.compile(SOURCE + b'\nprint("side effect")\n', requirements())
            run.assert_not_called()

    def test_unavailable_isolation_timeout_and_output_limit_are_harness_faults_without_fallback(self):
        errors = [FileNotFoundError('bwrap missing'), SupervisedProcessTimeout(b'out', b'err'),
                  SupervisedProcessOutputLimit(b'out', b'err')]
        with tempfile.TemporaryDirectory() as directory:
            compiler = compiler_fixture(Path(directory))
            for error in errors:
                with self.subTest(error=type(error).__name__), mock.patch(
                    'open_cake_ir.lab.cute_build.run_supervised', side_effect=error), mock.patch(
                    'open_cake_ir.lab.cute_build.compile_cute') as compile_:
                    with self.assertRaises(RunProtocolFault) as caught:
                        compiler.compile(SOURCE, requirements())
                    self.assertEqual(caught.exception.protocol_adherence, 'harness_fault')
                    self.assertIn('toolchain_stderr', caught.exception.artifact_payloads)
                    compile_.assert_not_called()

    def test_candidate_rejection_retains_separate_cuda_attempt_record(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = compiler_fixture(Path(directory))
            def fail(argv, **kwargs):
                path = Path(kwargs['cwd'], 'artifacts'); path.mkdir()
                (path / 'cuda_call_guard.json').write_text('{"denied_cuda_calls": ["cuda.bindings.driver.cuInit"]}')
                return subprocess.CompletedProcess(argv, 2, b'compile stdout', b'candidate type error')
            with mock.patch('open_cake_ir.lab.cute_build.run_supervised', side_effect=fail):
                with self.assertRaises(CandidateCompileRejected) as caught:
                    compiler.compile(SOURCE, requirements())
            payloads = caught.exception.artifact_payloads
            self.assertEqual(payloads['toolchain_stderr'], b'candidate type error')
            self.assertEqual(json.loads(payloads['toolchain_cuda_call_guard'])['denied_cuda_calls'],
                             ['cuda.bindings.driver.cuInit'])

    def test_malformed_or_mismatched_success_receipts_are_harness_faults(self):
        good = receipt(compilation_fixture())
        variants = ['not json', json.dumps({**good, 'compiler_version': '4.5.1'}),
            json.dumps({**good, 'artifacts': []}),
            json.dumps({**good, 'target': 'sm_100a'}), json.dumps({**good, 'entry_point': 'wrong'}),
            json.dumps({**good, 'threads_per_cta': 64}),
            json.dumps({**good, 'artifacts': {**good['artifacts'], 'cubin': '@@'}}),
            json.dumps({**good, 'extra': True})]
        with tempfile.TemporaryDirectory() as directory:
            compiler = compiler_fixture(Path(directory))
            for record in variants:
                def supervise(argv, **kwargs):
                    Path(kwargs['cwd'], 'compilation.json').write_text(record)
                    return subprocess.CompletedProcess(argv, 0, b'', b'')
                with self.subTest(record=record[:80]), mock.patch(
                    'open_cake_ir.lab.cute_build.run_supervised', side_effect=supervise):
                    with self.assertRaisesRegex(RunProtocolFault, 'invalid CuTe compilation receipt'):
                        compiler.compile(SOURCE, requirements())

    def test_runtime_contract_refuses_unpinned_sdk_and_exposed_workspaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); compiler = compiler_fixture(root)
            packages = {name: '4.5.2' for name in ('nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base', 'nvidia-cutlass-dsl-libs-cu13')}
            host = {'python': {'invocation_path': str(compiler.python)}, 'packages': packages}
            executor = SimpleNamespace(document={'host_environment': host})
            compiler.check_executor(executor, author_workspace=root / 'author')
            with self.assertRaisesRegex(ValueError, 'author workspace'):
                compiler.check_executor(executor, author_workspace=root / 'runtime' / 'author')
            packages['nvidia-cutlass-dsl-libs-cu13'] = '4.5.1'
            with self.assertRaisesRegex(ValueError, 'frozen Executor'):
                compiler.check_executor(executor, author_workspace=root / 'author')
            with mock.patch('open_cake_ir.lab.cute_build.sys.platform', 'linux'):
                for extra in (str(Path.home()), '/dev', str(ROOT)):
                    with self.subTest(root=extra), self.assertRaises(ValueError):
                        IsolatedCuTeCompiler(python=str(compiler.python), bubblewrap=str(compiler.bubblewrap),
                            cuobjdump=str(compiler.cuobjdump), runtime_roots=[str(root / 'runtime'), extra],
                            cutlass_version='4.5.2')

    def test_sdk_user_error_with_nested_environment_failure_stays_harness_fault(self):
        sdk_error = type('DSLUserCodeError', (Exception,), {'__module__': 'cutlass.base_dsl.common'})
        candidate = sdk_error('invalid tensor shape')
        self.assertTrue(_compile_failure(candidate)[0])
        candidate.cause = FileNotFoundError('missing ptxas')
        rejected, diagnostic = _compile_failure(candidate)
        self.assertFalse(rejected); self.assertIn('missing ptxas', diagnostic)
        self.assertFalse(_compile_failure(RuntimeError('unknown compiler crash'))[0])

    def test_sdk_op_error_returns_candidate_rejection_from_the_compile_worker(self):
        # SDK 4.5.2 raises this class for an invalid MmaF16BF16Op shape_mnk.
        sdk_error = type('OpError', (Exception,), {'__module__': 'cutlass.cute.nvgpu.common'})
        error = sdk_error("expects the 'shape_mnk' Op parameter to be one of (16,8,8) or (16,8,16)")
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.json'
            request.write_text(json.dumps({'source': SOURCE.decode(), 'requirements': requirements(),
                'cutlass_version': '4.5.2', 'cuobjdump': '/admitted/cuda/bin/cuobjdump'}))
            with mock.patch('open_cake_ir.lab.cute_build.importlib.metadata.version', return_value='4.5.2'), \
                 mock.patch('open_cake_ir.lab.cute_build.compile_cute', side_effect=error) as compile_, \
                 mock.patch('open_cake_ir.lab.cute_build.sys.stderr', new_callable=io.StringIO) as diagnostic:
                self.assertEqual(_worker(str(request)), 2)
            compile_.assert_called_once()
            self.assertIn("OpError: expects the 'shape_mnk'", diagnostic.getvalue())

    def test_sdk_op_error_allowance_requires_the_exact_class_and_module(self):
        for module, name in (('cutlass.cute.nvgpu', 'OpError'),
                             ('cutlass.cute.nvgpu.common.extra', 'OpError'),
                             ('other.compiler', 'OpError'),
                             ('cutlass.cute.nvgpu.common', 'OtherError')):
            with self.subTest(module=module, name=name):
                error = type(name, (Exception,), {'__module__': module})('invalid shape')
                self.assertFalse(_compile_failure(error)[0])
        self.assertFalse(_compile_failure(RuntimeError('OpError: invalid shape'))[0])

    def test_sdk_op_error_with_an_infrastructure_cause_remains_a_harness_fault(self):
        sdk_error = type('OpError', (Exception,), {'__module__': 'cutlass.cute.nvgpu.common'})
        for link in ('__cause__', 'cause'):
            for cause in (FileNotFoundError('missing ptxas'), ImportError('missing CUDA bindings')):
                with self.subTest(link=link, cause=type(cause).__name__):
                    error = sdk_error('operation construction failed')
                    setattr(error, link, cause)
                    rejected, diagnostic = _compile_failure(error)
                    self.assertFalse(rejected)
                    self.assertIn(str(cause), diagnostic)


class CuTeSealedBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = load_workload(ROOT / 'contracts/workloads/gemm-bias-bf16-fp32-v2.json')

    def request(self, source_role='authored_source', req=None):
        return BuildRequest('a' * 64, SOURCE, source_role, sha256(SOURCE).hexdigest(),
                            'sm_103a', 'fixture', requirements() if req is None else req)

    def test_both_authoring_representations_use_same_compiler_and_zero_hidden_pointers(self):
        isolated = SimpleNamespace(compile=mock.Mock(return_value=compilation_fixture()))
        builder = CuTeToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=isolated)
        candidates = []
        for role in ('authored_source', 'lowered_source'):
            candidate = builder.build(self.request(role))
            manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
            manifest.check_workload(self.workload, 'primary')
            self.assertEqual(manifest.hidden_null_pointer_parameters, 0)
            self.assertEqual([row[0] for row in manifest.tensor_abi], ['a', 'b', 'bias', 'c'])
            self.assertEqual(candidate.artifact_payloads[role], SOURCE)
            self.assertNotEqual(candidate.artifact_payloads['compiler_expanded_source'], SOURCE)
            self.assertEqual(manifest.kernel_name, NAME)
            candidates.append(candidate)
        self.assertEqual(candidates[0].artifact_payloads['cubin'], candidates[1].artifact_payloads['cubin'])
        self.assertEqual(isolated.compile.call_count, 2)

    def test_builder_rejects_workload_pointer_order_before_compile(self):
        isolated = SimpleNamespace(compile=mock.Mock())
        builder = CuTeToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=isolated)
        request = requirements()
        request['signature'][0]['name'] = 'renamed'
        source = SOURCE.replace(b'fixture(a:', b'fixture(renamed:')
        build = BuildRequest('a' * 64, source, 'authored_source', sha256(source).hexdigest(),
                             'sm_103a', 'fixture', request)
        with self.assertRaisesRegex(ValueError, 'Workload tensor ABI'):
            builder.build(build)
        isolated.compile.assert_not_called()

    def test_missing_isolation_and_wrong_toolchain_output_cannot_seal(self):
        builder = CuTeToolchainBuilder(workload=self.workload, case_id='primary')
        with self.assertRaisesRegex(RunProtocolFault, 'filesystem-isolated'):
            builder.build(self.request())
        isolated = SimpleNamespace(compile=lambda *args: replace(compilation_fixture(), dynamic_shared_bytes=16))
        builder = CuTeToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=isolated)
        with self.assertRaises(ValueError):
            builder.build(self.request())


if __name__ == '__main__':
    unittest.main()
