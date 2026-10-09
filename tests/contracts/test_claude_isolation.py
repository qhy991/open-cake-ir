"""The actual published launcher and fail-closed probe admission."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from open_cake_ir.lab.claude_isolation import publish_launcher,probe_launcher,validate_probe_observation

class ClaudeIsolationTests(unittest.TestCase):
    def test_launcher_is_self_contained_create_only_and_does_not_embed_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            config={name:str(root/name) for name in ('bubblewrap','native_executable','credential_source')}
            for path in config.values():Path(path).write_text('never-execute-test-secret')
            (root/'homes').mkdir()
            config.update(endpoint='https://example.invalid',model='glm-5.3',home_root=str(root/'homes'))
            path=publish_launcher(root/'launcher',config)
            source=path.read_text();compile(source,str(path),'exec')
            self.assertNotIn('never-execute-test-secret',source)
            self.assertNotIn('open_cake_ir',source)
            with self.assertRaises(FileExistsError):publish_launcher(path,config)
            namespace={'CONFIG':config,'__name__':'test'}
            exec(compile(source,str(path),'exec'),namespace)
            with patch.dict('os.environ',{'ANTHROPIC_BASE_URL':'https://example.invalid',
                'ANTHROPIC_MODEL':'glm-5.3','ANTHROPIC_DEFAULT_OPUS_MODEL':'glm-5.3',
                'ANTHROPIC_DEFAULT_SONNET_MODEL':'glm-5.3','ANTHROPIC_DEFAULT_HAIKU_MODEL':'glm-5.3'}):
                argv=namespace['command'](root,['/provider/claude','--help'])
            self.assertIn('--unshare-all',argv);self.assertIn('--cap-drop',argv)
            self.assertNotIn(config['credential_source'],argv)
            shim=argv[argv.index('-c')+1]
            compile(shim,'shim','exec')
            self.assertIn('rstrip(b"\\n")',shim)
            import subprocess
            workspace=root/'qualification'/'arm';workspace.mkdir(parents=True)
            with patch('sys.argv',['launcher','--cake-isolation-probe']), patch('pathlib.Path.cwd',return_value=workspace), patch('subprocess.run',return_value=subprocess.CompletedProcess([],0)):
                self.assertEqual(namespace['main'](),0)
            home=Path(config['home_root'])/workspace.relative_to('/')
            self.assertTrue(home.is_dir())
            self.assertEqual(list(workspace.iterdir()),[])
            self.assertEqual(list((root/'qualification').iterdir()),[workspace])

    def test_probe_requires_every_boundary_and_rejects_partial_or_false_observations(self):
        checks={name:True for name in ('ambient_home','project','experiments','host_root','gpu_nodes','escape',
                                      'home_is_local','gpu_device_nodes_absent','workspace_writable')}
        value={'policy':'linux_claude_workspace_v1','checks':checks}
        validate_probe_observation(value)
        checks['escape']=False
        with self.assertRaisesRegex(ValueError,'isolation probe failed'):validate_probe_observation(value)
        del checks['escape']
        with self.assertRaises(ValueError):validate_probe_observation(value)
