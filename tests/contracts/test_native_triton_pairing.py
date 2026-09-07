"""CPU contract fixtures only; no fixture here is a real GPU or isolation qualification."""
from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
from hashlib import sha256
import json
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.toolchain import TritonCompilation, project_triton_kernel, validate_triton_kernel
from open_cake_ir.evaluation.core import EvaluationProtocol, TensorLaunchManifest, evaluate_tile_workload, parse_launch_manifest
from open_cake_ir.evaluation.tile_workloads import reference_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, Lab, NativeTritonEnvironment, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.pairing import bind_baseline, native_baseline, triton_optimization_analysis_plan
from open_cake_ir.lab.providers import _project_candidate_submission, CANDIDATE_SET_ENVELOPE_V1
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from tests.contracts.test_lab import FakeProvider, FakeEvaluator

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class DraftCompilerFixture:
    """Exercise released-only adapters with real draft compilation in this test only."""
    state = 'released'

    def __init__(self):
        self.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def __getattr__(self, name):
        return getattr(self.compiler, name)


class CompilationFixture:
    def __init__(self):
        self.requests = []

    def compile(self, source, requirements):
        validate_triton_kernel(source, requirements)
        self.requests.append((source, requirements))
        return TritonCompilation(source, 'sm_100a', requirements['kernel_entry_point'], {
            'source': b'# fixture-expanded\n' + source, 'ttir': b'fixture ttir',
            'ttgir': b'fixture ttgir', 'llir': b'fixture llir', 'ptx': b'.target sm_100a\n',
            'cubin': b'\x7fELFfixture_not_launchable_on_GPU',
        }, requirements['compile_options']['num_warps'] * 32, 0, 'fixture')


def baseline(workload, case='primary'):
    provenance = next(p for p in workload.document['provenance'] if p.get('kind') == 'source_schedule')
    return bind_baseline(json.loads((ROOT / provenance['path']).read_text()), workload, case)


def python_rms(workload):
    return '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="python-rms", target="sm_100a", backend="triton", entry_point="cake_rmsnorm_b8_smoke",
               metadata={"workload_contract_sha256": "DIGEST"},
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 96})
def rms(lm, x: cake.Tensor((8,512,128), "fp32"), gamma: cake.Tensor((128,), "fp32"),
        y: cake.Tensor((8,512,128), "fp32", mode="output")):
    compute = lm.role(warps=[0,1,2,3])
    row_block = lm.program(x, axis=0, dimension=1, tile=64)
    batch = lm.program(x, axis=1, dimension=0, tile=1)
    with compute:
        x_tile = lm.load(x[batch, row_block, :], reuse="reused", id="load_x")
        sq = lm.mul(x_tile, x_tile, id="square")
        sumsq = lm.reduce(sq, op="sum", axis=1, scope="cta", id="sum_sq")
        meansq = lm.mul(sumsq, 0.0078125, id="mean")
        shifted = lm.add(meansq, 1e-6, id="shift")
        inv_rms = lm.rsqrt(shifted, id="rsqrt")
        gamma_tile = lm.load(gamma[:], id="load_gamma")
        normed = lm.mul(x_tile, lm.broadcast(inv_rms, axis=0), id="scale")
        weighted = lm.mul(normed, lm.broadcast(gamma_tile, axis=1), id="weight")
        lm.store(y[batch, row_block, :], weighted, id="store_y")
'''.replace('DIGEST', workload.canonical_sha256)


class NativePairingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = DraftCompilerFixture()
        cls.workload = WorkloadContract.load(ROOT / 'contracts/workloads/rmsnorm-fp32-v1.json')
        cls.schedule = baseline(cls.workload)
        cls.lowering = cls.compiler.lower(cls.compiler.assess(cls.schedule))
        cls.native = native_baseline(cls.lowering)

    def environments(self):
        fixture = CompilationFixture()
        builder = TritonToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=fixture)
        open_env = OpenCakeEnvironment(self.compiler, builder, authority_document={
            'lowering_route': self.schedule['lowering'], 'input_format': 'schedule_or_python_v1',
        }, workload=self.workload, case_id='primary')
        native_env = NativeTritonEnvironment(builder, toolchain_requirements=self.lowering.toolchain_requirements,
            authority_document={'environment_kind': 'native_triton'}, workload=self.workload, case_id='primary')
        return open_env, native_env, fixture

    def test_three_primary_baselines_project_only_the_exact_trusted_kernel(self):
        for name in ('rmsnorm-fp32-v1', 'gemm-bias-bf16-fp32-v1', 'indexed-gather-bf16-v1'):
            with self.subTest(workload=name):
                workload = WorkloadContract.load(ROOT / f'contracts/workloads/{name}.json')
                lowering = self.compiler.lower(self.compiler.assess(baseline(workload)))
                projected = native_baseline(lowering)
                tree = ast.parse(projected['kernel_source'])
                original = next(n for n in ast.parse(lowering.source).body if isinstance(n, ast.FunctionDef)
                                and n.name == lowering.toolchain_requirements['kernel_entry_point'])
                self.assertEqual(ast.dump(tree.body[-1]), ast.dump(original))
                self.assertNotIn('import torch', projected['kernel_source'])
                self.assertEqual(projected['compile_options'], lowering.toolchain_requirements['compile_options'])
                self.assertEqual(projected['grid'], lowering.toolchain_requirements['grid'])

    def test_native_environment_accepts_the_compiler_pointer_types_for_every_workload(self):
        for name in ('rmsnorm-fp32-v1', 'gemm-bias-bf16-fp32-v1', 'indexed-gather-bf16-v1'):
            with self.subTest(workload=name):
                workload = WorkloadContract.load(ROOT / f'contracts/workloads/{name}.json')
                lowering = self.compiler.lower(self.compiler.assess(baseline(workload)))
                fixture = CompilationFixture()
                builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=fixture)
                environment = NativeTritonEnvironment(builder, workload=workload, case_id='primary',
                    toolchain_requirements=lowering.toolchain_requirements, authority_document={})
                result = environment.build(CandidateSubmission.seal(environment.media_type, encoded(native_baseline(lowering))))
                self.assertEqual(result.disposition, 'launchable')
                self.assertEqual(fixture.requests[0][1]['signature'], lowering.toolchain_requirements['signature'])

    def test_both_submission_paths_share_manifest_and_compilation_contract(self):
        open_env, native_env, fixture = self.environments()
        ir = open_env.build(CandidateSubmission.seal(open_env.media_type, encoded(self.schedule)))
        native = native_env.build(CandidateSubmission.seal(native_env.media_type, encoded(self.native)))
        self.assertEqual((ir.disposition, native.disposition), ('launchable', 'launchable'))
        self.assertEqual(fixture.requests[0], fixture.requests[1])
        self.assertEqual(ir.launchable.launch_spec_sha256, native.launchable.launch_spec_sha256)
        manifest = parse_launch_manifest(json.loads(native.launchable.artifact_payloads['launch_manifest']))
        self.assertEqual([t[0] for t in manifest.tensor_abi], ['x', 'gamma', 'y'])
        manifest.check_workload(self.workload, 'primary')
        with self.assertRaises(ValueError):
            manifest.check_workload(self.workload, 'tiny')

    def test_python_ir_real_submission_canonicalizes_and_keeps_diagnostics(self):
        open_env, _, fixture = self.environments()
        source = python_rms(self.workload)
        result = open_env.build(CandidateSubmission.seal(open_env.media_type, encoded({'python_source': source})))
        self.assertEqual(result.disposition, 'launchable', result.feedback)
        canonical = parse(source).document
        json_result = open_env.build(CandidateSubmission.seal(open_env.media_type, encoded(canonical)))
        self.assertEqual(result.semantic_sha256, json_result.semantic_sha256)
        self.assertEqual(fixture.requests[0][0], fixture.requests[1][0])
        bad = source.replace('1e-6', 'open("forbidden")')
        rejected = open_env.build(CandidateSubmission.seal(open_env.media_type, encoded({'python_source': bad})))
        self.assertEqual(rejected.disposition, 'rejected')
        self.assertEqual(rejected.feedback['source_location']['filename'], 'candidate.ir.py')
        self.assertGreater(rejected.feedback['source_location']['line'], 1)

    def test_native_host_side_effects_never_reach_builder(self):
        _, environment, fixture = self.environments()
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'MUST_NOT_EXIST'
            payloads = [
                f'open({str(marker)!r}, "w").write("side effect")\n' + self.native['kernel_source'],
                self.native['kernel_source'].replace('@triton.jit', f'@open({str(marker)!r}, "w")'),
                self.native['kernel_source'].replace('    x,', f'    x=open({str(marker)!r}, "w"),'),
                self.native['kernel_source'].replace('    x,', f'    x: open({str(marker)!r}, "w"),'),
                self.native['kernel_source'] + f'\nopen({str(marker)!r}, "w")\n',
                self.native['kernel_source'].replace('    row_block =', f'    open({str(marker)!r}, "w")\n    row_block ='),
                self.native['kernel_source'].replace('    row_block =', '    import os\n    row_block ='),
                self.native['kernel_source'].replace('tl.program_id(0)', 'tl.__dict__["program_id"](0)'),
                self.native['kernel_source'].replace('    row_block =', '    while True:\n        pass\n    row_block ='),
            ]
            for source in payloads:
                with self.subTest(source=source[:65]):
                    result = environment.build(CandidateSubmission.seal(environment.media_type, encoded({**self.native, 'kernel_source': source})))
                    self.assertEqual(result.disposition, 'rejected', result.feedback)
                    self.assertFalse(marker.exists())
            self.assertFalse(fixture.requests)

    def test_reversed_runtime_pointer_order_is_rejected_at_both_boundaries(self):
        source = b'import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(b, a):\n    x = tl.load(a)\n    tl.store(b, x)\n'
        with self.assertRaisesRegex(ValueError, 'positional signature'):
            validate_triton_kernel(source, {'kernel_entry_point': 'kernel', 'signature': {'a': '*fp32', 'b': '*fp32'}, 'compile_constants': {}})
        _, environment, fixture = self.environments()
        reversed_source = self.native['kernel_source'].replace('    x,\n    gamma,', '    gamma,\n    x,')
        result = environment.build(CandidateSubmission.seal(environment.media_type, encoded({**self.native, 'kernel_source': reversed_source})))
        self.assertEqual(result.disposition, 'rejected')
        self.assertFalse(fixture.requests)

    def test_signature_order_is_restored_from_workload_not_sorted_json_keys(self):
        _, _, fixture = self.environments()
        builder = TritonToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=fixture)
        requirements = json.loads(encoded(dict(self.lowering.toolchain_requirements)))
        self.assertEqual(list(requirements['signature']), ['gamma', 'x', 'y'])
        environment = NativeTritonEnvironment(builder, toolchain_requirements=requirements,
            authority_document={}, workload=self.workload, case_id='primary')
        result = environment.build(CandidateSubmission.seal(environment.media_type, encoded(self.native)))
        self.assertEqual(result.disposition, 'launchable', result.feedback)
        self.assertEqual(list(fixture.requests[0][1]['signature']), ['x', 'gamma', 'y'])

    def test_no_implicit_backend_or_extra_launch_knobs(self):
        _, env, fixture = self.environments()
        for field, value in [('backend', 'cuda'), ('signature', {}), ('launcher', 'evil()')]:
            result = env.build(CandidateSubmission.seal(env.media_type, encoded({**self.native, field: value})))
            self.assertEqual(result.disposition, 'rejected')
        self.assertFalse(fixture.requests)

    def test_native_cannot_use_historical_unsandboxed_builder(self):
        env = NativeTritonEnvironment(TritonToolchainBuilder(), toolchain_requirements=self.lowering.toolchain_requirements,
            authority_document={}, workload=self.workload, case_id='primary')
        with self.assertRaisesRegex(ValueError, 'Compiler lowering only'), mock.patch('open_cake_ir.lab.environments.compile_triton') as compile_:
            env.build(CandidateSubmission.seal(env.media_type, encoded(self.native)))
        compile_.assert_not_called()

    def test_missing_isolation_does_not_fall_back_to_inprocess_compilation(self):
        builder = TritonToolchainBuilder(workload=self.workload, case_id='primary')
        env = NativeTritonEnvironment(builder, toolchain_requirements=self.lowering.toolchain_requirements,
            authority_document={}, workload=self.workload, case_id='primary')
        with self.assertRaisesRegex(RunProtocolFault, 'filesystem-isolated'), mock.patch('open_cake_ir.lab.environments.compile_triton') as compile_:
            env.build(CandidateSubmission.seal(env.media_type, encoded(self.native)))
        compile_.assert_not_called()

    def test_supervised_isolation_command_has_no_workspace_or_home_mount(self):
        # Command construction only. A real Linux isolation canary is R2 pending.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'fixture runtime')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'fixture bwrap')
            with mock.patch('open_cake_ir.lab.triton_build.sys.platform', 'linux'):
                compiler = IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap), runtime_roots=[str(runtime)], triton_version='fixture')
            commands = []
            def supervise(argv, **kwargs):
                commands.append((argv, kwargs))
                Path(kwargs['cwd'], 'compilation.json').write_text(json.dumps({
                    'target': 'sm_100a', 'entry_point': 'fixture', 'artifacts': {'cubin': 'eA=='},
                    'threads_per_cta': 128, 'dynamic_shared_bytes': 0, 'compiler_version': 'fixture'}))
                return subprocess.CompletedProcess(argv, 0, b'', b'')
            with mock.patch('open_cake_ir.lab.triton_build.run_supervised', side_effect=supervise):
                compiler.compile(self.native['kernel_source'].encode(), self.lowering.toolchain_requirements)
            argv, kwargs = commands[0]
            self.assertIn('--unshare-all', argv)
            self.assertIn('--clearenv', argv)
            self.assertNotIn(str(Path.home()), argv)
            self.assertNotIn(str(ROOT), argv)
            self.assertEqual(kwargs['environment'], {})
            self.assertEqual(argv.count('--bind'), 1)
            self.assertIn('/compiler-src', argv)

    def test_runtime_loader_alias_keeps_its_guest_mount_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'fixture python')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'fixture bwrap')
            libraries = root / 'usr-lib64'; libraries.mkdir()
            alias = root / 'lib64'; alias.symlink_to(libraries, target_is_directory=True)
            with mock.patch('open_cake_ir.lab.triton_build.sys.platform', 'linux'):
                compiler = IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap),
                    runtime_roots=[str(runtime), str(alias)], triton_version='fixture')
            with mock.patch('open_cake_ir.lab.triton_build.run_supervised',
                            return_value=subprocess.CompletedProcess([], 1, b'', b'fixture stop')) as run:
                with self.assertRaises(RunProtocolFault):
                    compiler.compile(self.native['kernel_source'].encode(), self.lowering.toolchain_requirements)
            argv = run.call_args.args[0]
            mounts = [argv[i + 1:i + 3] for i, value in enumerate(argv) if value == '--ro-bind']
            self.assertIn([str(libraries), str(alias)], mounts)
            self.assertNotIn([str(libraries), str(libraries)], mounts)

    def test_runtime_alias_retarget_cannot_change_the_checked_host_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'fixture python')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'fixture bwrap')
            libraries = root / 'libraries'; libraries.mkdir()
            workspace = root / 'author'; workspace.mkdir()
            alias = workspace / 'runtime-link'; alias.symlink_to(libraries, target_is_directory=True)
            with mock.patch('open_cake_ir.lab.triton_build.sys.platform', 'linux'):
                compiler = IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap),
                    runtime_roots=[str(runtime), str(alias)], triton_version='fixture')
            executor = SimpleNamespace(document={'host_environment': {
                'python': {'invocation_path': str(python)}, 'packages': {'triton': 'fixture'}}})
            compiler.check_executor(executor, author_workspace=workspace)
            identity = compiler.identity
            alias.unlink(); alias.symlink_to(workspace, target_is_directory=True)
            with mock.patch('open_cake_ir.lab.triton_build.run_supervised',
                            return_value=subprocess.CompletedProcess([], 1, b'', b'fixture stop')) as run:
                with self.assertRaises(RunProtocolFault):
                    compiler.compile(self.native['kernel_source'].encode(), self.lowering.toolchain_requirements)
            argv = run.call_args.args[0]
            mounts = [argv[i + 1:i + 3] for i, value in enumerate(argv) if value == '--ro-bind']
            self.assertIn([str(libraries), str(alias)], mounts)
            self.assertNotIn(str(workspace), [source for source, _ in mounts])
            self.assertEqual(compiler.identity, identity)

    def test_runtime_aliases_cannot_hide_home_or_author_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'fixture python')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'fixture bwrap')
            alias = root / 'alias'; alias.symlink_to(Path.home(), target_is_directory=True)
            with mock.patch('open_cake_ir.lab.triton_build.sys.platform', 'linux'):
                with self.assertRaisesRegex(ValueError, 'runtime mount contract'):
                    IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap),
                        runtime_roots=[str(runtime), str(alias)], triton_version='fixture')
                alias.unlink()
                workspace = root / 'author'; workspace.mkdir()
                alias.symlink_to(workspace, target_is_directory=True)
                compiler = IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap),
                    runtime_roots=[str(runtime), str(alias)], triton_version='fixture')
            executor = SimpleNamespace(document={'host_environment': {
                'python': {'invocation_path': str(python)}, 'packages': {'triton': 'fixture'}}})
            with self.assertRaisesRegex(ValueError, 'must not expose author workspace'):
                compiler.check_executor(executor, author_workspace=workspace)

    def test_canonical_composition_rejects_an_unpinned_isolated_runtime_before_build(self):
        from open_cake_ir.lab.compose import execute_matched_from_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'fixture python')
            other = runtime / 'other'; other.write_bytes(b'fixture other python')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'fixture isolation binary')
            executable = root / 'provider'; executable.write_bytes(b'fixture provider')
            output_schema = {'path':'unused.json', 'sha256':'f'*64}
            authority = {'qualification':{'path':'fixture.json','canonical_sha256':'a'*64},
                         'output_schema':output_schema, 'executable_sha256':sha256(executable.read_bytes()).hexdigest()}
            arm = {'provider':authority, 'toolchain_sha256':'b'*64}
            lock = SimpleNamespace(study_kind='matched_search', claim_scope='scientific_matched_search',
                document={'resolved_inputs':{'arm_environments':{'open_cake':arm,'native_triton':arm}, 'budget':{}}})
            executor = SimpleNamespace(document={'host_environment':{
                'python':{'invocation_path':str(python)},'packages':{'triton':'fixture'}}})
            qualification = SimpleNamespace(scope='live_two_turn_current_provider',canonical_sha256='a'*64)
            (root / 'fixture.json').write_text('{}')
            config = {'schema_version':1, 'provider':{'executable':str(executable),'workspace_root':str(root/'author')},
                'toolchain':{'python':str(other),'bubblewrap':str(bwrap),'runtime_roots':[str(runtime)],
                             'triton_version':'fixture','timeout_seconds':1},
                'broker':{'command':['unused'],'cwd':str(root),'timeout_seconds':1,'service_user':'fixture','service_group':'fixture'}}
            path = root/'runtime.json'; path.write_bytes(encoded(config))
            with mock.patch('open_cake_ir.lab.compose._admit_executor',return_value=(executor,None)), \
                 mock.patch('open_cake_ir.lab.compose.ProviderQualificationReceipt.load',return_value=qualification), \
                 mock.patch('open_cake_ir.lab.triton_build.sys.platform','linux'), \
                 mock.patch('open_cake_ir.lab.triton_build.run_supervised') as run:
                with self.assertRaisesRegex(ValueError,'runtime differs from the frozen Executor'):
                    execute_matched_from_config(root,lock,path,root/'evidence')
                run.assert_not_called()
                config['toolchain']['python'] = str(python)
                config['toolchain']['triton_version'] = 'other-version'
                path.write_bytes(encoded(config))
                with self.assertRaisesRegex(ValueError,'runtime differs from the frozen Executor'):
                    execute_matched_from_config(root,lock,path,root/'evidence')
                run.assert_not_called()

    def test_worker_runtime_faults_propagate_through_both_authoring_environments(self):
        from open_cake_ir.lab import triton_build
        wrapped = ValueError('compile wrapper failed')
        wrapped.__cause__ = ImportError('fixture missing compiler dependency')
        middle = RuntimeError('backend loading wrapper')
        middle.__cause__ = ImportError('fixture missing compiler dependency')
        deep = ValueError('deep compile wrapper failed')
        deep.__cause__ = middle
        try:
            try:
                raise ImportError('fixture missing compiler dependency')
            except ImportError:
                raise ValueError('implicit compile wrapper failed')
        except ValueError as caught:
            implicit = caught
        cases = (
            (ImportError('libtriton.so: libstdc++.so.6: cannot open shared object file'), False),
            (FileNotFoundError(2, 'No such file or directory', '/runtime/bin/ptxas'), False),
            (subprocess.CalledProcessError(127, ['ptxas']), False),
            (RuntimeError('compiler internal failure'), False),
            (wrapped, False),
            (deep, False),
            (implicit, False),
            (ValueError('fixture candidate compile rejection'), True),
            (SyntaxError('fixture candidate syntax rejection'), True),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / 'runtime'; runtime.mkdir()
            python = runtime / 'python'; python.write_bytes(b'non-executable fixture runtime')
            bwrap = root / 'bwrap'; bwrap.write_bytes(b'non-executable fixture isolation')
            with mock.patch.object(triton_build.sys, 'platform', 'linux'):
                compiler = IsolatedTritonCompiler(python=str(python), bubblewrap=str(bwrap),
                    runtime_roots=[str(runtime)], triton_version='fixture')
            builder = TritonToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=compiler)
            environments = (
                OpenCakeEnvironment(self.compiler, builder, authority_document={
                    'lowering_route': self.schedule['lowering'], 'input_format':'schedule_or_python_v1'},
                    workload=self.workload, case_id='primary'),
                NativeTritonEnvironment(builder, toolchain_requirements=self.lowering.toolchain_requirements,
                    authority_document={}, workload=self.workload, case_id='primary'),
            )
            for environment, payload in zip(environments, (self.schedule, self.native), strict=True):
                for error, candidate_rejection in cases:
                    with self.subTest(environment=type(environment).__name__, failure=type(error).__name__, message=str(error)):
                        returncodes = []
                        def supervise(argv, **kwargs):
                            stderr = StringIO()
                            with mock.patch('importlib.metadata.version', return_value='fixture'), \
                                 mock.patch.object(triton_build, 'compile_triton', side_effect=error), \
                                 contextlib.redirect_stderr(stderr):
                                code = triton_build._worker(str(Path(kwargs['cwd']) / 'request.json'))
                            returncodes.append(code)
                            return subprocess.CompletedProcess(argv, code, b'fixture build stdout\n', stderr.getvalue().encode())
                        submission = CandidateSubmission.seal(environment.media_type, encoded(payload))
                        with mock.patch.object(triton_build, 'run_supervised', side_effect=supervise):
                            if candidate_rejection:
                                result = environment.build(submission)
                                self.assertEqual(result.disposition, 'rejected')
                                self.assertEqual(result.feedback['stage'], 'compile')
                                artifacts = result.artifact_payloads
                            else:
                                with self.assertRaises(RunProtocolFault) as caught:
                                    environment.build(submission)
                                self.assertEqual(caught.exception.protocol_adherence, 'harness_fault')
                                artifacts = caught.exception.artifact_payloads
                        self.assertEqual(returncodes, [2 if candidate_rejection else 1])
                        self.assertEqual(artifacts['toolchain_stdout'], b'fixture build stdout\n')
                        self.assertIn(str(error).encode(), artifacts['toolchain_stderr'])
                        if error in (wrapped, deep, implicit):
                            self.assertIn(b'ImportError: fixture missing compiler dependency', artifacts['toolchain_stderr'])
                with mock.patch.object(triton_build, 'run_supervised', side_effect=FileNotFoundError('fixture bwrap disappeared')):
                    with self.assertRaises(RunProtocolFault) as caught:
                        environment.build(CandidateSubmission.seal(environment.media_type, encoded(payload)))
                    self.assertEqual(caught.exception.protocol_adherence, 'harness_fault')
                    self.assertIn(b'fixture bwrap disappeared', caught.exception.artifact_payloads['toolchain_stderr'])

    def test_compile_failure_follows_both_links_and_terminates_on_a_cycle(self):
        from open_cake_ir.lab.triton_build import _compile_failure
        outer = ValueError('cyclic outer wrapper')
        outer.__cause__ = ValueError('candidate-looking cause')
        outer.__context__ = ImportError('cyclic runtime dependency')
        outer.__context__.__cause__ = outer
        rejected, diagnostic = _compile_failure(outer)
        self.assertFalse(rejected)
        self.assertEqual(diagnostic.count('ValueError: cyclic outer wrapper'), 1)
        self.assertEqual(diagnostic.count('ValueError: candidate-looking cause'), 1)
        self.assertEqual(diagnostic.count('ImportError: cyclic runtime dependency'), 1)

    def test_provider_projects_native_and_python_members_without_executing_source(self):
        for arm, member in [('native_triton', self.native), ('open_cake', {'python_source': 'not executed'})]:
            envelope = encoded({'schema_version': 1, 'arm': arm, 'candidates': [member]}) + b'\n'
            self.assertEqual(_project_candidate_submission(envelope, submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm=arm, maximum_candidates_per_turn=3), (encoded(member),))

    def test_common_correctness_uses_workload_oracle_and_detects_mutated_inputs(self):
        _, env, _ = self.environments()
        candidate = env.build(CandidateSubmission.seal(env.media_type, encoded(self.native))).launchable
        # Use the tiny case here, preserving actual source-bound ABI via a new fixture manifest.
        workload = self.workload
        primary = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
        tiny = TensorLaunchManifest.for_workload(workload, 'tiny', target='sm_100a', kernel_name=primary.kernel_name,
            grid=[1,1,1], block=[128,1,1], dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)
        payloads = dict(candidate.artifact_payloads); payloads['launch_manifest'] = encoded(tiny.as_dict())
        candidate = dataclasses.replace(candidate, artifact_payloads=payloads,
            artifact_roles={k: sha256(v).hexdigest() for k,v in payloads.items()}, launch_spec_sha256=tiny.canonical_sha256)
        protocol = EvaluationProtocol('fixture', 'search', workload.canonical_sha256, 'tiny', 'none')
        class Launcher:
            def launch_tensors(self, candidate, manifest, inputs):
                output = reference_outputs(workload, 'tiny', inputs)
                after = copy.deepcopy(inputs)
                if getattr(self, 'mutate', False): after['x'][0] += 1
                if getattr(self, 'wrong', False): output['y'][-1] += 1
                return output, after, {'candidate_sha256': candidate.candidate_sha256, 'kernel_calls':1, 'fallback_calls':0}
        launcher = Launcher()
        self.assertTrue(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)
        launcher.mutate = True
        self.assertFalse(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)
        launcher.mutate = False; launcher.wrong = True
        self.assertFalse(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)


class PairedLabFixtureTests(unittest.TestCase):
    def test_real_matched_loop_preflight_submission_confirmation_replay_and_threshold_view(self):
        draft = DraftCompilerFixture()
        workload = WorkloadContract.load(ROOT / 'contracts/workloads/rmsnorm-fp32-v1.json')
        schedule = baseline(workload)
        lowering = draft.lower(draft.assess(schedule))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory) / 'project'; root.mkdir()
            for name in ('contracts', 'corpus', 'compiler', 'src', 'docs'):
                shutil.copytree(ROOT / name, root / name)
            document = json.loads((root / 'contracts/studies/matched-search-triton-optimization-template.json').read_text())
            document['budget'].update({'unit':'provider_tokens', 'limit':80000, 'checkpoints':[80000], 'maximum_turns':1, 'maximum_candidates_per_turn':1})
            document['evaluation_protocol'] = {'case_id':'primary', 'search_evaluation':'correctness_then_paired_cupti',
                'confirmatory_evaluation':'fresh_fixed_candidate_correctness_then_paired_cupti'}
            configuration = {**FakeProvider.configuration, 'output_schema_sha256': document['arms']['open_cake']['provider']['output_schema']['sha256']}
            qualification = json.loads((root / 'contracts/providers/fixture-provider-candidate-set-ralph-v1.json').read_text())
            qualification['configuration_sha256'] = sha256(encoded(configuration)).hexdigest()
            qualification['provider_revision'] = 'native-ralph-pairing-cpu-fixture'
            qp = root / 'contracts/providers/native-fixture.json'; qp.write_bytes(encoded(qualification))
            for arm, value in document['arms'].items():
                value['provider']['disabled_features'] = configuration['disabled_features']
                value['provider']['revision'] = qualification['provider_revision']
                value['provider']['qualification'] = {'path':'contracts/providers/native-fixture.json', 'canonical_sha256':sha256(encoded(qualification)).hexdigest()}
                value['feedback'] = (['findings'] if arm == 'open_cake' else ['compile']) + ['correctness','qualified_timing']
            study = root / 'fixture-study.json'; study.write_bytes(encoded(document))
            gate = SimpleNamespace(compiler_revision_id='compiler-fixture', compiler_revision_sha256='a'*64, passed=True)
            reference = {'revision_id':'compiler-fixture','path':'compiler/revision.lock.json','canonical_sha256':'a'*64}
            executor = {'executor_id':'open-cake-ir-b200-v9000','path':'runtime/executors/fixture.json','canonical_sha256':'e'*64}
            stack.enter_context(mock.patch('open_cake_ir.lab.core._resolve_compiler_reference', return_value=(gate, reference['path'], reference)))
            stack.enter_context(mock.patch('open_cake_ir.lab.core._resolve_executor_reference', return_value=executor))
            stack.enter_context(mock.patch('open_cake_ir.lab.core._validate_executor_revision'))
            stack.enter_context(mock.patch('open_cake_ir.lab.core.Compiler.load', return_value=draft))
            lab = Lab(root)
            lock = lab.preflight(study)
            self.assertEqual(lock.analysis_plan, triton_optimization_analysis_plan())
            fixture = CompilationFixture()
            builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=fixture)
            arms = lock.document['resolved_inputs']['arm_environments']
            environments = {
                'open_cake':OpenCakeEnvironment(draft,builder,authority_document=arms['open_cake'],workload=workload,case_id='primary'),
                'native_triton':NativeTritonEnvironment(builder,toolchain_requirements=lowering.toolchain_requirements,
                    authority_document=arms['native_triton'],workload=workload,case_id='primary')}
            class Provider(FakeProvider):
                def turn(self, request):
                    original = super().turn(request)
                    member = schedule if request.arm == 'open_cake' else native_baseline(lowering)
                    # Retain one real candidate rejection as an observed failure.
                    if request.run_id == 'native_triton-3': member = {**member, 'host_launcher':'forbidden'}
                    payload = encoded(member)
                    return dataclasses.replace(original,candidates=(payload,),candidate_sha256s=(sha256(payload).hexdigest(),))
            provider = Provider(); provider.configuration = configuration
            provider.provider_revision = qualification['provider_revision']
            from open_cake_ir.lab import render_task_package
            provider.packages = {run_id: render_task_package(root, lock, run_id) for run_id in lock.run_order}
            provider.qualification_sha256 = sha256(encoded(qualification)).hexdigest()
            class Evaluator(FakeEvaluator):
                def evaluate(self,candidate,*,case_id,purpose):
                    # Existing sealed broker-receipt fixture; these are never live measurements.
                    arm = 'open_cake' if 'lowered_source' in candidate.artifact_roles else 'direct_cuda'
                    virtual_name = 'open_cake_turn_10' if purpose == 'search' else arm + '_turn_1'
                    return super().evaluate(dataclasses.replace(candidate,entry_point=virtual_name),case_id=case_id,purpose=purpose)
            evaluator = Evaluator(lock.document['evaluation_protocol'],sha256(encoded(lock.document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256)
            campaign = lab.execute(lock, Path(directory) / 'evidence',provider=provider,environments=environments,evaluator=evaluator)
            report = lab.audit(campaign)
            self.assertTrue(report.archive_integrity_passed)
            self.assertTrue(report.semantic_replay_passed)
            self.assertEqual(len(report.descriptive['paired_runs']),3)
            self.assertEqual(report.descriptive['paired_runs'][2]['native_triton_outcome'],'no_qualified_candidate')
            view = lab.threshold_view(campaign,0.5)
            # Search fixture latencies are 0.1ms, but fresh confirmations are 1/2ms.
            self.assertEqual(len(view['runs']),6)
            self.assertTrue(all(row['first_confirmation_turn'] is None for row in view['runs']))
            self.assertTrue(report.filesystem_custody_verified, 'new Evidence fixture requires a custody-capable temporary filesystem')
            reached = lab.threshold_view(campaign,1.5)
            self.assertEqual(sum(row['status']=='reached_by_fresh_confirmation' for row in reached['runs']),3)
            self.assertTrue(all(row['elapsed_wall_seconds'] is not None for row in reached['runs'] if row['first_confirmation_turn']))


if __name__ == '__main__':
    unittest.main()
