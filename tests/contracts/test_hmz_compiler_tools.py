"""Exercise the external author boundary without hmz, a model, or a device."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'tools/hmz_compiler_tools.py'


class ExternalCompilerToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'campaign/candidates').mkdir(parents=True)
        (self.root / '.deps').mkdir()
        (self.root / '.deps/cake-ir').symlink_to(ROOT, target_is_directory=True)
        self.pin = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        self.write('campaign/binding.json', dict(compiler_commit=self.pin))
        (self.root / '.gitignore').write_text('.deps/\ncampaign/candidates/\ncampaign/compiler-actions/\ncampaign/deadline.json\ncampaign/intake.json\n')
        self.git('init', '-q')
        self.git('config', 'user.name', 'Author fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.tool('freeze')
        self.git('add', '.')
        self.git('commit', '-qm', 'Prepared external workspace')
        self.prepared = self.git('rev-parse', 'HEAD')
        self.write('campaign/intake.json', dict(source_commit=self.prepared))
        self.write('campaign/deadline.json', dict(search_stop_at_epoch=time.time() + 600))
        self.process('''
from pathlib import Path
from open_cake_ir.tasks.workloads import create_task
for backend in ('metal-m2', 'triton-gfx1151', 'triton-dcu', 'triton-b300', 'triton-metax'):
    _, source = create_task('silu', backend=backend, rows=32, columns=64)
    Path('campaign/candidates/' + backend + '.py').write_text(source)
_, source = create_task('gemm', backend='triton-dcu', rows=32, columns=64, depth=128)
Path('campaign/candidates/gemm.py').write_text(source)
''')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def process(self, source):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        result = subprocess.run([sys.executable, '-c', source], cwd=self.root, env=environment,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def tool(self, *args, success=True):
        result = subprocess.run([sys.executable, str(TOOL), *args], cwd=self.root,
                                text=True, capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def request(self, name='gemm'):
        parent = 'campaign/candidates/' + name + '.py'
        stage = self.tool('inspect', '--parent', parent)['stages'][0]['name']
        return dict(action='transform', parent=parent, transformation='specialize_fp32_contraction',
                    parameters=dict(stage=stage, row_tile=16, column_tile=32, k_tile=32,
                                    num_warps=4, num_stages=1, schedule_id='mma', entry_point='mma'))

    def apply(self, identity, request):
        self.write('campaign/candidates/request.json', request)
        return self.tool('transform', '--id', identity, '--request', 'campaign/candidates/request.json')

    def test_catalog_and_own_source_inspection_cover_five_vendor_routes(self):
        api = self.tool('catalog')
        self.assertEqual(api['compiler_commit'], self.pin)
        names = {item['name'] for item in api['transformations']}
        self.assertIn('specialize_fp32_contraction', names)
        for backend in ('metal-m2', 'triton-gfx1151', 'triton-dcu', 'triton-b300', 'triton-metax'):
            with self.subTest(backend=backend):
                inspected = self.tool('inspect', '--parent', 'campaign/candidates/' + backend + '.py')
                self.assertTrue(inspected['stages'])
        for path in ('../foreign.py', '.deps/cake-ir/examples/python/fma.py'):
            self.tool('inspect', '--parent', path, success=False)
        (self.root / 'campaign/candidates/foreign.py').symlink_to(ROOT / 'tools/hmz_compiler_tools.py')
        self.tool('inspect', '--parent', 'campaign/candidates/foreign.py', success=False)

    def test_complete_transform_replays_retained_parent_and_detects_stage_edits(self):
        self.assertEqual(self.apply('mma', self.request())['reason'], 'applied')
        self.assertEqual(self.tool('verify', '--id', 'mma'),
                         dict(id='mma', replay='matched', device_qualification=False))
        stage = 'campaign/compiler-actions/mma/stage-000.json'
        self.assertTrue(self.tool('inspect', '--parent', stage)['stages'])
        (self.root / 'campaign/candidates/gemm.py').write_text('# next proposal')
        self.assertEqual(self.tool('verify', '--id', 'mma')['replay'], 'matched')
        self.write(stage, {})
        self.tool('verify', '--id', 'mma', success=False)
        self.assertFalse((self.root / 'campaign/DONE.json').exists())

    def test_refusal_and_unknown_action_are_retained_and_ids_cannot_be_reused(self):
        self.assertNotEqual(self.apply('refused', self.request('triton-dcu'))['reason'], 'applied')
        self.assertEqual(self.tool('verify', '--id', 'refused')['replay'], 'matched')
        request = self.request()
        request['transformation'] = 'not_declared'
        self.assertEqual(self.apply('unknown', request)['reason'], 'transform_not_granted')
        record = self.root / 'campaign/compiler-actions/unknown/result.json'
        original = record.read_bytes()
        self.tool('transform', '--id', 'unknown', '--request', 'campaign/candidates/request.json', success=False)
        self.assertEqual(record.read_bytes(), original)
        self.assertEqual(self.tool('verify', '--id', 'unknown')['replay'], 'matched')
        self.assertFalse(record.with_name('program.json').exists())

    def test_frozen_catalog_binding_and_search_closure_are_enforced(self):
        self.apply('before-close', self.request())
        self.write('campaign/deadline.json', dict(search_stop_at_epoch=0))
        self.tool('transform', '--id', 'late', '--request', 'campaign/candidates/request.json', success=False)
        self.assertFalse((self.root / 'campaign/compiler-actions/late').exists())
        (self.root / 'campaign/binding.json').unlink()
        self.write('campaign/development-binding.json', dict(compiler=self.pin))
        self.write('campaign/deadline.json', dict(source_commit=self.prepared, search_stop_at_epoch=time.time() + 600))
        self.write('campaign/search-close-owner.json', dict(decision='approved'))
        self.tool('transform', '--id', 'closed', '--request', 'campaign/candidates/request.json', success=False)
        self.assertEqual(self.tool('verify', '--id', 'before-close')['replay'], 'matched')
        self.tool('freeze', success=False)
        original = (self.root / 'campaign/compiler-api.json').read_bytes()
        self.write('campaign/compiler-api.json', dict(compiler_commit=self.pin, transformations=[]))
        self.tool('catalog', success=False)
        (self.root / 'campaign/compiler-api.json').write_bytes(original)
        self.write('campaign/development-binding.json', dict(compiler='0' * 40))
        self.tool('catalog', success=False)

    def test_loaded_foreign_compiler_is_refused_before_using_its_registry(self):
        self.process(f'''
import runpy, sys, types
from pathlib import Path
module = runpy.run_path({str(TOOL)!r})
foreign = types.ModuleType('open_cake_ir')
foreign.__file__ = '/another-condition/src/open_cake_ir/__init__.py'
sys.modules['open_cake_ir'] = foreign
try:
    module['compiler'](Path.cwd())
except ValueError as error:
    assert 'fresh process' in str(error)
else:
    raise AssertionError('A foreign loaded Compiler was accepted')
''')
